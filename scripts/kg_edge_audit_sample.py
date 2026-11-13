"""Seeded sample of KG edges for labelling, with call-site evidence (#239).

kg_edge_audit.py finds edges that contradict the code's structure; it cannot
say how many of the other edges are right. This draws a fixed sample for a
person to label, from two kinds of stratum per repository:

  random   N edges from each (relation, confidence) cell, e.g. calls/exact,
           calls/name_only, accesses/qualified: an estimate of precision per
           cell, weighted back by the cell sizes in the key file
  flagged  M edges from each rule kg_edge_audit.py flagged (self_loop,
           self_call_class, ...): how many flagged cases are real defects

An edge drawn by both is labelled once. The labelling sheet does not show the
stratum, the confidence or the rule, so the label is not led by them; those
are in the key file, joined back by item number after labelling.

    PYTHONPATH=src .venv/bin/python scripts/kg_edge_audit_sample.py \\
        --kg-json kgs/psf__requests.json kgs/django__django.json \\
        --per-cell 10 --flagged-per-rule 10 --seed 20261114 --out edge_sample

writes edge_sample.md (one section per item: the call site lines with their
line numbers, the caller's import of the name, the target's definition, and
other definitions with the same name), edge_sample.csv (item, repo, blank
`label` and `reason`) and edge_sample.key.csv (item, stratum, relation,
confidence, rules, ids, cell size). Labels: `correct` (at runtime this site
can call this target), `wrong`, or `unclear` with a reason.

The KGs come from kg_edge_audit.py --save-dir, so the sample and the audit
table describe the same build. Same KGs, same seed and same arguments give
the same sample.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kg_edge_audit as audit  # noqa: E402

CONTEXT = 1
MAX_ALTERNATIVES = 5


def _meta(n):
    return n.get("metadata") or {}


def _where(n):
    m = _meta(n)
    return f"{m.get('filepath') or m.get('path') or '?'}:{m.get('lineno', '?')}"


def _name(n):
    return _meta(n).get("qualname") or n["label"]


class Repo:
    def __init__(self, name, kg):
        self.name, self.kg = name, kg
        self.by_id = {n["id"]: n for n in kg["nodes"]}
        self.container = {e["target"]: e["source"] for e in kg["edges"] if e["relation"] == "contains"}
        self.modules = audit.ModuleMap(kg["nodes"], kg["edges"])
        self.by_label = defaultdict(list)
        for n in kg["nodes"]:
            if n["type"] in audit.DEFINED:
                self.by_label[n["label"]].append(n["id"])

    def file_of(self, nid):
        seen = set()
        while nid in self.container and nid not in seen:
            seen.add(nid)
            nid = self.container[nid]
        return nid if self.by_id.get(nid, {}).get("type") in audit.FILES else None

    def local_names(self, caller, target_label):
        """The names the caller's file binds to target_label (itself and any
        import alias), with each import's origin."""
        f = self.file_of(caller)
        names = {target_label: None}
        for local, origin in self.modules.imports.get(f, {}).items():
            if local == target_label or origin.split(".")[-1] == target_label:
                names[local] = origin
        return names

    def evidence(self, rel, src, tgt):
        s, t = self.by_id[src], self.by_id[tgt]
        names = self.local_names(src, t["label"])
        source = _meta(s).get("source_code") or ""
        lines = source.splitlines()
        start = _meta(s).get("lineno") or 1
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        keep = sorted({j for i in hits for j in range(i - CONTEXT, i + CONTEXT + 1) if 0 <= j < len(lines)})
        shown, last = [], None
        for j in keep:
            if last is not None and j != last + 1:
                shown.append("    ...")
            shown.append(f"{start + j:>6}  {lines[j]}")
            last = j
        decs = _meta(s).get("decorators") or []
        dec_hits = [d for d in decs if pattern.search(str(d))]
        alts = [a for a in self.by_label.get(t["label"], []) if a != tgt]
        imports = {k: v for k, v in names.items() if v}
        return {
            "caller": f"{s['type']} {_name(s)} ({_where(s)})",
            "site": "\n".join(shown) if shown else "(the name does not appear in the caller's source)",
            "decorators": dec_hits,
            "imports": imports,
            "target": f"{t['type']} {_name(t)} ({_where(t)})",
            "signature": (_meta(t).get("signature") or "").strip(),
            "alternatives": [f"{self.by_id[a]['type']} {_name(self.by_id[a])} ({_where(self.by_id[a])})"
                             for a in alts[:MAX_ALTERNATIVES]],
            "alternatives_total": len(alts),
        }


def draw(repos, relations, per_cell, flagged_per_rule, seed):
    """Items in a fixed, shuffled order, each with its key."""
    rng = random.Random(seed)
    items = {}   # (repo, relation, source, target) -> key

    def add(repo, e_key, stratum, cell, cell_size, rule=None):
        k = (repo.name,) + e_key
        if k not in items:
            rel, src, tgt = e_key
            edge = repo.edge_meta.get(e_key, {})
            items[k] = {"repo": repo.name, "relation": rel, "source": src, "target": tgt,
                        "confidence": edge.get("confidence", ""), "strata": [], "rules": [],
                        "cells": {}}
        items[k]["strata"].append(stratum)
        items[k]["cells"][stratum] = (cell, cell_size)
        if rule:
            items[k]["rules"].append(rule)

    for repo in repos:
        repo.edge_meta = {(e["relation"], e["source"], e["target"]): (e.get("metadata") or {})
                          for e in repo.kg["edges"]}
        cells = defaultdict(list)
        for e in repo.kg["edges"]:
            if e["relation"] in relations:
                cells[(e["relation"], (e.get("metadata") or {}).get("confidence", ""))].append(
                    (e["relation"], e["source"], e["target"]))
        for cell in sorted(cells):
            pool = sorted(cells[cell])
            for e_key in rng.sample(pool, min(per_cell, len(pool))):
                add(repo, e_key, "random", "/".join(cell), len(pool))
        flagged = {}
        audit.audit(repo.kg, flagged)
        for rule in sorted(flagged):
            pool = sorted(set(k for k in flagged[rule] if k[0] in relations))
            for e_key in rng.sample(pool, min(flagged_per_rule, len(pool))):
                add(repo, e_key, "flagged", rule, len(pool), rule)

    order = sorted(items)
    rng.shuffle(order)
    return [dict(items[k], item=i) for i, k in enumerate(order, 1)]


def write(repos, items, out):
    by_name = {r.name: r for r in repos}
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "repo", "label", "reason"])
        for it in items:
            w.writerow([it["item"], it["repo"], "", ""])
    with open(out + ".key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "repo", "relation", "confidence", "strata", "cells", "cell_sizes",
                    "rules", "source_id", "target_id"])
        for it in items:
            st = sorted(it["cells"])
            w.writerow([it["item"], it["repo"], it["relation"], it["confidence"], ";".join(st),
                        ";".join(it["cells"][s][0] for s in st), ";".join(str(it["cells"][s][1]) for s in st),
                        ";".join(it["rules"]), it["source"], it["target"]])
    with open(out + ".md", "w") as f:
        f.write("# KG edge sample: labelling sheet\n\n"
                "For each item: at runtime, can this call site reach this target? Label it in the CSV "
                "as `correct`, `wrong` or `unclear`, with a short reason (for `wrong`, what the site "
                "really calls if you can tell). Read the repository's code when the excerpt is not "
                "enough. Do not open the key file until every item is labelled.\n\n")
        for it in items:
            ev = by_name[it["repo"]].evidence(it["relation"], it["source"], it["target"])
            f.write(f"## {it['item']}. {it['repo']}: {it['relation']}\n\n"
                    f"**From** {ev['caller']}\n\n```python\n{ev['site']}\n```\n\n")
            if ev["decorators"]:
                f.write("In its decorators: " + "; ".join(f"`@{d}`" for d in ev["decorators"]) + "\n\n")
            if ev["imports"]:
                f.write("Its file imports: " + "; ".join(f"`{k}` from `{v}`" for k, v in ev["imports"].items()) + "\n\n")
            f.write(f"**To** {ev['target']}\n\n")
            if ev["signature"]:
                f.write(f"```python\n{ev['signature']}\n```\n\n")
            if ev["alternatives_total"]:
                more = ev["alternatives_total"] - len(ev["alternatives"])
                f.write(f"Other definitions named `{ev['target'].split(' ')[1].split('.')[-1]}` "
                        f"({ev['alternatives_total']}):\n\n"
                        + "".join(f"- {a}\n" for a in ev["alternatives"])
                        + (f"- and {more} more\n" if more else "") + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-json", nargs="+", required=True, help="KGs saved by kg_edge_audit.py --save-dir")
    ap.add_argument("--relations", nargs="+", default=["calls", "accesses"])
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--flagged-per-rule", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out", required=True, help="output stem: <stem>.md, <stem>.csv, <stem>.key.csv")
    args = ap.parse_args(argv)
    repos = [Repo(Path(p).stem, json.load(open(p))) for p in args.kg_json]
    items = draw(repos, set(args.relations), args.per_cell, args.flagged_per_rule, args.seed)
    write(repos, items, args.out)
    print(f"{len(items)} items -> {args.out}.md, {args.out}.csv, {args.out}.key.csv")
    return items


if __name__ == "__main__":
    main()
