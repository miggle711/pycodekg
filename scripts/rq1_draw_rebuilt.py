"""Draw RQ1's new edges from the rebuilt KGs (miggle711/testgeneval#70).

rq1_recheck_labels.py found that the rebuild (pycodekg#169, 11 Nov) removed
almost all ambiguous edges: 26 of the 30 sampled resolved_ambiguous edges and
all 10 sampled test_resolved_ambiguous edges are gone, and resolved_ambiguous
has 8 edges left in the 11 KGs. The kept labels carry over; this script draws
the edges that replace the gone ones (fyp-paper#5, decision of 21 Nov):

  --census STRATUM     every edge of the stratum (resolved_ambiguous: 8)
  --draw STRATUM=N     N edges at random (test_resolved_ambiguous=10)
  --equal-chance       draw them as one simple random sample of the stratum
                       (decision of 22 Nov: with 10 draws, the per-cell
                       allocation gave about 2,000 of the 28,474 edges no
                       chance of selection, so 10 labels could not stand
                       for the stratum)

Strata, the test-code rule and the per-cell draw are rq1_sample_stratified.py's
(11 Oct): edges go to (repository, relation) cells by the largest remainder
method and each cell is sampled with random.Random(f"{seed}|{stratum}|{repo}|{rel}").
Edges already in a sample (--sample, matched as rq1_recheck_labels.py does) are
never drawn again.

Usage (on M3, in a Slurm job; each KG is up to a few GB in memory):

    python scripts/rq1_draw_rebuilt.py --kg-dir <rebuilt kg_output> \\
        --sample <rq1/v3/rq1_sample.csv> --sample <rq1_test/rq1_test_sample.csv> \\
        --census resolved_ambiguous --draw test_resolved_ambiguous=10 \\
        --equal-chance --seed 20261122 --first-id 111 --out rq1_new_sample.csv
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_recheck_labels as rc  # noqa: E402

COLUMNS = ["sample_id", "annotator", "stratum", "repo", "relation", "confidence", "rendered_current",
           "source_type", "source_name", "source_file", "source_line", "source_link", "source_snippet",
           "target_type", "target_name", "target_file", "target_line", "target_link", "target_snippet",
           "edge_metadata", "commit"]


def largest_remainder(weights, total):
    """The sampler's allocation: integers proportional to weights, summing to total."""
    s = sum(weights.values())
    if s == 0 or total == 0:
        return {k: 0 for k in weights}
    quotas = {k: total * w / s for k, w in weights.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    left = total - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), str(k)))[:left]:
        alloc[k] += 1
    return alloc


def rendered_current(edge, src, tgt):
    """The sampler's rendered(exact_only=True): can the edge type reach a kg_only prompt."""
    rel = edge["relation"]
    if rel == "calls":
        return True
    if rel in ("inherits", "uses"):
        return (edge.get("metadata") or {}).get("confidence") == "exact"
    if rel == "contains":
        return src.get("type") == "class" and tgt.get("type") in ("method", "function")
    return False


def snippet(node, lines=3):
    md = node.get("metadata") or {}
    if md.get("signature"):
        return md["signature"]
    src = md.get("source_code") or ""
    return "\n".join(src.splitlines()[:lines]) if src else rc.node_path(node)


def link(repo, commit, path, line):
    if not path:
        return ""
    url = f"https://github.com/{repo}/blob/{commit}/{path}"
    return f"{url}#L{line}" if line else url


def edge_row(stratum, repo, commit, e, s, t):
    em = e.get("metadata") or {}
    sp, tp = rc.node_path(s), rc.node_path(t)
    sl, tl = (s.get("metadata") or {}).get("lineno"), (t.get("metadata") or {}).get("lineno")
    call_line = em.get("lineno") or em.get("line")
    return {"stratum": stratum, "repo": repo, "commit": commit, "relation": e["relation"],
            "confidence": em.get("confidence", ""),
            "rendered_current": "yes" if rendered_current(e, s, t) else "no",
            "source_type": s.get("type"), "source_name": s.get("label"), "source_file": sp,
            "source_line": sl or "", "source_link": link(repo, commit, sp, call_line or sl),
            "source_snippet": snippet(s),
            "target_type": t.get("type"), "target_name": t.get("label"), "target_file": tp,
            "target_line": tl or "", "target_link": link(repo, commit, tp, tl), "target_snippet": snippet(t),
            "edge_metadata": json.dumps(em, sort_keys=True)[:300]}


def parse_draw(specs):
    out = {}
    for spec in specs:
        stratum, _, n = spec.partition("=")
        if not n.isdigit():
            raise ValueError(f"--draw {spec!r}: expected STRATUM=N")
        out[stratum] = int(n)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-dir", required=True)
    ap.add_argument("--sample", action="append", required=True,
                    help="an existing sample CSV; its repos and commits name the KGs, its edges are excluded")
    ap.add_argument("--census", action="append", default=[], help="take every edge of this stratum")
    ap.add_argument("--draw", action="append", default=[], help="STRATUM=N: draw N edges at random")
    ap.add_argument("--equal-chance", action="store_true",
                    help="draw as one simple random sample of the stratum, so every edge has the same chance; "
                         "the per-cell allocation gives cells rounded to 0 no chance at all")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--first-id", type=int, required=True, help="number of the first new RQ1-NNN id")
    ap.add_argument("--annotator", default="both")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    draw = parse_draw(args.draw)
    wanted = set(args.census) | set(draw)
    if not wanted:
        raise ValueError("nothing to do: give --census and/or --draw")
    if set(args.census) & set(draw):
        raise ValueError("a stratum is given to both --census and --draw")

    old = []
    for path in args.sample:
        old.extend(rc.read_sample(path))
    by_kg = defaultdict(list)
    for r in old:
        by_kg[(r["repo"], r["commit"][:8])].append(r)
    taken_ids = {r["sample_id"] for r in old}

    cells = defaultdict(list)   # (stratum, repo, relation) -> [row]
    excluded = Counter()
    for (repo, commit8), group in sorted(by_kg.items()):
        path = rc.kg_file_for(args.kg_dir, repo, commit8)
        if not path:
            raise FileNotFoundError(f"no rebuilt KG for {repo} {commit8} in {args.kg_dir}")
        with open(path) as f:
            data = json.load(f)
        kg = rc.KG(data)
        commit = data["metadata"].get("base_commit") or max((r["commit"] for r in group), key=len)
        sampled = set()
        for r in group:
            s, _ = kg.find(r["source_type"], r["source_name"], r["source_file"], r.get("source_line"))
            t, _ = kg.find(r["target_type"], r["target_name"], r["target_file"], r.get("target_line"))
            if s and t:
                sampled.add((r["relation"], s, t))
        for e in data["edges"]:
            s, t = kg.nodes.get(e["source"]), kg.nodes.get(e["target"])
            if s is None or t is None:
                continue
            stratum = rc.full_stratum(e["relation"], (e.get("metadata") or {}).get("confidence"), rc.is_test_node(s))
            if stratum not in wanted:
                continue
            if (e["relation"], e["source"], e["target"]) in sampled:
                excluded["already sampled"] += 1
                continue
            cells[(stratum, repo, e["relation"])].append(edge_row(stratum, repo, commit, e, s, t))
        print(f"{repo:28s} {os.path.basename(path)}")
        del data, kg

    rows = []
    for stratum in sorted(wanted):
        sc = {k: v for k, v in cells.items() if k[0] == stratum}
        population = sum(map(len, sc.values()))
        if stratum in draw:
            if draw[stratum] > population:
                raise ValueError(f"{stratum}: asked for {draw[stratum]}, only {population} edges")
            alloc = largest_remainder({k: len(v) for k, v in sc.items()}, draw[stratum])
        else:
            alloc = {k: len(v) for k, v in sc.items()}
        print(f"\n{stratum}: population {population}, taking {sum(alloc.values())}"
              f" ({'census' if stratum not in draw else f'seed {args.seed}'})")
        for key in sorted(sc, key=lambda k: (k[1], k[2])):
            k = alloc[key]
            print(f"   {key[1]:28s} {key[2]:10s} {len(sc[key]):>7}  taken {k}")
            if k == 0:
                continue
            if stratum not in draw:
                rows.extend(sc[key])
            elif not args.equal_chance:
                rng = random.Random(f"{args.seed}|{stratum}|{key[1]}|{key[2]}")
                rows.extend(rng.sample(sc[key], k))
        if stratum in draw and args.equal_chance:
            pool = [r for key in sorted(sc, key=lambda k: (k[1], k[2])) for r in sc[key]]
            picked = random.Random(f"{args.seed}|{stratum}").sample(pool, draw[stratum])
            print(f"   equal chance: {draw[stratum]} of {len(pool)} edges at random; by cell: "
                  f"{dict(Counter((r['repo'], r['relation']) for r in picked))}")
            rows.extend(picked)

    for i, r in enumerate(rows, args.first_id):
        r["sample_id"] = f"RQ1-{i:03d}"
        r["annotator"] = args.annotator
        if r["sample_id"] in taken_ids:
            raise ValueError(f"{r['sample_id']} is already used by an existing sample; pick another --first-id")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in COLUMNS} for r in rows)
    print(f"\nwrote {len(rows)} edges to {args.out} "
          f"({rows[0]['sample_id'] if rows else '-'} to {rows[-1]['sample_id'] if rows else '-'}); "
          f"excluded: {dict(excluded) or 'none'}")
    for r in rows:
        print(f"   {r['sample_id']} {r['stratum']:24s} {r['repo']:22s} {r['relation']:9s} "
              f"{r['source_name']} -> {r['target_name']} ({r['target_file']})")
    return rows


if __name__ == "__main__":
    main()
