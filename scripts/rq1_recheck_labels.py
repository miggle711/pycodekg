"""Check RQ1's labelled edges against rebuilt KGs (miggle711/testgeneval#70).

RQ1's edge sample was drawn in October from one KG per repository (its
latest built commit) and labelled by hand. The KGs are being rebuilt with
the resolver fixes (miggle711/fyp-paper#5, section 3), so each labelled edge
may still be there, may have changed stratum, or may be gone. This script
finds out for every sampled edge, so only the edges that changed need a new
look.

An edge is matched by its relation and its two endpoints. The sample CSV has
no node ids, so an endpoint is found by (type, label, file); when several
nodes share those, the sample's line number picks one. A node whose type
changed (for example test_file to file) is still found by (label, file), and
the change is reported.

Status per sampled edge:

  kept              the edge is in the rebuilt KG, in the same stratum.
                    Its label still applies.
  kept_restratified the edge is there but its confidence changed, so it now
                    belongs to another stratum (for example ambiguous to
                    exact). Its label still applies; its weight does not.
  gone              both endpoints are there and the edge is not. A label of
                    "incorrect" here is an error the rebuild removed.
  endpoint_missing  an endpoint is not in the rebuilt KG. Needs a look.
  endpoint_unclear  more than one node matches an endpoint even with the
                    line number. Needs a look.
  no_kg             no rebuilt KG for the sample's repository and commit.

`source_now_test` marks an edge whose source is test code in the rebuilt KG
but was not in the sample (or the reverse): its stratum moves between the
non-test strata and the test_ strata (rq1_sample_stratified.py's test rule,
copied below). Such an edge is reported as kept_restratified.

The rebuilt strata populations are printed with the same rules as the
sampler, next to the sample's own rq1_strata.csv when given, because the
overall estimate is weighted by them.

Usage (on M3, in a Slurm job: each KG is up to a few GB in memory):

    python scripts/rq1_recheck_labels.py \\
        --kg-dir <rebuilt kg_output> \\
        --sample <rq1/v3/rq1_sample.csv> --sample <rq1_test/rq1_test_sample.csv> \\
        --labels <labelling sheet, Edges tab, as CSV> \\
        --old-strata <rq1/v3/rq1_strata.csv> \\
        --out rq1_recheck.csv

--labels is optional. Without it the statuses are still reported.
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

# The sampler's rules (rq1_sample_stratified.py, testgeneval#70, 11 Oct).
TEST_NODE_TYPES = {"test_file", "test_function", "test_class", "test_method"}
SYNTACTIC = {"contains", "imports", "depends_on", "module_depends_on", "decorated_by"}

STATUSES = ["kept", "kept_restratified", "gone", "endpoint_missing", "endpoint_unclear", "no_kg"]


def node_path(node):
    md = node.get("metadata") or {}
    return md.get("filepath") or md.get("path") or ""


def is_test_path(path):
    parts = path.replace("\\", "/").split("/")
    base = parts[-1]
    return (
        any(p in ("tests", "test", "testing") for p in parts[:-1])
        or base.startswith("test_")
        or base.endswith("_test.py")
        or base == "conftest.py"
    )


def is_test_node(node):
    return node.get("type") in TEST_NODE_TYPES or is_test_path(node_path(node))


def stratum_of(relation, confidence):
    if relation in SYNTACTIC:
        return "syntactic"
    return "resolved_ambiguous" if confidence == "ambiguous" else "resolved_confident"


def full_stratum(relation, confidence, test_source):
    return ("test_" if test_source else "") + stratum_of(relation, confidence)


def kg_file_for(kg_dir, repo, commit):
    """kg_<repo with / and - as _>_<first 8 of commit>.json, as the batch build names them."""
    slug = repo.replace("/", "_").replace("-", "_")
    path = os.path.join(kg_dir, f"kg_{slug}_{commit[:8]}.json")
    if os.path.exists(path):
        return path
    found = glob.glob(os.path.join(kg_dir, f"kg_*_{commit[:8]}.json"))
    return found[0] if len(found) == 1 else None


class KG:
    """One rebuilt KG, indexed for endpoint and edge lookups."""

    def __init__(self, data):
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.by_key = defaultdict(list)       # (type, label, file) -> [id]
        self.by_name = defaultdict(list)      # (label, file) -> [id]
        for n in data["nodes"]:
            self.by_key[(n.get("type"), n.get("label"), node_path(n))].append(n["id"])
            self.by_name[(n.get("label"), node_path(n))].append(n["id"])
        self.edges = defaultdict(list)        # (relation, source, target) -> [edge]
        self.out = defaultdict(list)          # (relation, source) -> [target]
        for e in data["edges"]:
            self.edges[(e["relation"], e["source"], e["target"])].append(e)
            self.out[(e["relation"], e["source"])].append(e["target"])
        self.data = data

    def find(self, typ, label, path, line):
        """(node id, note) for an endpoint. note is '' when the match is plain,
        'type now <t>' when only (label, file) matched, 'missing' or 'unclear'."""
        ids = self.by_key.get((typ, label, path), [])
        note = ""
        if not ids:
            ids = self.by_name.get((label, path), [])
            if ids:
                types = sorted({self.nodes[i].get("type") for i in ids})
                note = f"type now {'/'.join(types)}"
        if not ids:
            return None, "missing"
        if len(ids) > 1 and line:
            ids = [i for i in ids if str((self.nodes[i].get("metadata") or {}).get("lineno") or "") == str(line)] or ids
        if len(ids) > 1:
            return None, "unclear"
        return ids[0], note

    def strata(self):
        """Edges per stratum with the sampler's rules; an edge whose source is
        test code counts under test_<stratum>, as in the test-source sample."""
        out = Counter()
        for e in self.data["edges"]:
            src, tgt = self.nodes.get(e["source"]), self.nodes.get(e["target"])
            if src is None or tgt is None:
                continue
            out[full_stratum(e["relation"], (e.get("metadata") or {}).get("confidence"), is_test_node(src))] += 1
        return out


def recheck(row, kg):
    """The rebuilt KG's view of one sampled edge, as a dict of report columns."""
    out = {"status": "", "confidence_now": "", "stratum_now": "", "source_now_test": "",
           "note": "", "now_targets": ""}
    if kg is None:
        out["status"] = "no_kg"
        return out
    src, s_note = kg.find(row["source_type"], row["source_name"], row["source_file"], row.get("source_line"))
    tgt, t_note = kg.find(row["target_type"], row["target_name"], row["target_file"], row.get("target_line"))
    notes = []
    if s_note:
        notes.append(f"source {s_note}")
    if t_note:
        notes.append(f"target {t_note}")
    out["note"] = "; ".join(notes)
    if "missing" in (s_note, t_note):
        out["status"] = "endpoint_missing"
        return out
    if "unclear" in (s_note, t_note):
        out["status"] = "endpoint_unclear"
        return out
    now_test = is_test_node(kg.nodes[src])
    out["source_now_test"] = "yes" if now_test != row.get("stratum", "").startswith("test_") else ""
    edges = kg.edges.get((row["relation"], src, tgt), [])
    if not edges:
        out["status"] = "gone"
        others = [kg.nodes[t].get("label", "") + " (" + node_path(kg.nodes[t]) + ")"
                  for t in kg.out.get((row["relation"], src), [])]
        out["now_targets"] = "; ".join(others[:3]) + (f"; and {len(others) - 3} more" if len(others) > 3 else "")
        return out
    conf = (edges[0].get("metadata") or {}).get("confidence") or ""
    out["confidence_now"] = conf
    out["stratum_now"] = full_stratum(row["relation"], conf, now_test)
    out["status"] = "kept" if out["stratum_now"] == row["stratum"] else "kept_restratified"
    return out


def read_labels(path, id_col, label_cols, agreed_col):
    """{sample id: label} from the labelling sheet: the agreed label where
    there is one, otherwise the one annotator's label, otherwise 'disagree'
    when the annotators differ and nothing was agreed."""
    labels = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sid = (r.get(id_col) or "").strip()
            if not sid:
                continue
            agreed = (r.get(agreed_col) or "").strip()
            given = [(r.get(c) or "").strip() for c in label_cols]
            given = [g for g in given if g]
            if agreed:
                labels[sid] = agreed
            elif len(set(given)) == 1:
                labels[sid] = given[0]
            elif given:
                labels[sid] = "disagree"
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-dir", required=True, help="directory holding the rebuilt kg_*.json files")
    ap.add_argument("--sample", action="append", required=True, help="a sample CSV; repeat for each")
    ap.add_argument("--labels", help="the labelling sheet's Edges tab exported as CSV")
    ap.add_argument("--id-col", default="Sample")
    ap.add_argument("--label-cols", nargs="+", default=["Mun label", "Joe label"])
    ap.add_argument("--agreed-col", default="Agreed label (both)")
    ap.add_argument("--old-strata", action="append", default=[], help="a sample's rq1_strata.csv; repeat")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = []
    for path in args.sample:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    labels = read_labels(args.labels, args.id_col, args.label_cols, args.agreed_col) if args.labels else {}

    by_kg = defaultdict(list)
    for r in rows:
        by_kg[(r["repo"], r["commit"])].append(r)

    report, strata_now = [], Counter()
    for (repo, commit), group in sorted(by_kg.items()):
        path = kg_file_for(args.kg_dir, repo, commit)
        kg = None
        if path:
            with open(path) as f:
                data = json.load(f)
            kg = KG(data)
            schema = data.get("metadata", {}).get("schema_version")
            strata_now.update(kg.strata())
            print(f"{repo:28s} {commit[:8]}  {os.path.basename(path)}  schema {schema}  "
                  f"{len(data['nodes']):,} nodes  {len(data['edges']):,} edges  {len(group)} sampled")
            del data
        else:
            print(f"{repo:28s} {commit[:8]}  NO REBUILT KG in {args.kg_dir}  {len(group)} sampled")
        for r in group:
            res = recheck(r, kg)
            report.append({"sample_id": r["sample_id"], "stratum": r["stratum"], "repo": repo,
                           "relation": r["relation"], "confidence": r.get("confidence", ""),
                           "label": labels.get(r["sample_id"], ""), **res,
                           "source": f"{r['source_name']} ({r['source_file']})",
                           "target": f"{r['target_name']} ({r['target_file']})"})
        kg = None

    cols = ["sample_id", "stratum", "repo", "relation", "confidence", "label", "status",
            "confidence_now", "stratum_now", "source_now_test", "note", "now_targets", "source", "target"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(report)

    print(f"\n== Sampled edges in the rebuilt KGs ({len(report)})")
    strata = sorted({r["stratum"] for r in report})
    print(f"   {'status':20}" + "".join(f"{s:>22}" for s in strata) + f"{'total':>8}")
    for st in STATUSES:
        n = [sum(1 for r in report if r["status"] == st and r["stratum"] == s) for s in strata]
        if sum(n):
            print(f"   {st:20}" + "".join(f"{x:>22}" for x in n) + f"{sum(n):>8}")
    moved = Counter((r["stratum"], r["stratum_now"]) for r in report if r["status"] == "kept_restratified")
    for (a, b), n in sorted(moved.items()):
        print(f"   restratified: {n} from {a} to {b}")
    now_test = sum(1 for r in report if r["source_now_test"])
    if now_test:
        print(f"   source test status changed (in or out of the non-test frame): {now_test}")

    if labels:
        print("\n== By label (labels from the sheet)")
        lab = sorted({r["label"] or "(unlabelled)" for r in report})
        print(f"   {'status':20}" + "".join(f"{l:>14}" for l in lab))
        for st in STATUSES:
            n = [sum(1 for r in report if r["status"] == st and (r["label"] or "(unlabelled)") == l) for l in lab]
            if sum(n):
                print(f"   {st:20}" + "".join(f"{x:>14}" for x in n))

    print("\n== Strata populations in the rebuilt KGs (the sampler's rules)")
    old = {}
    for path in args.old_strata:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                old[r["stratum"]] = int(r["population_edges"])
    for s in sorted(set(strata_now) | set(old)):
        was = f"{old[s]:>10,}" if s in old else f"{'':>10}"
        print(f"   {s:26} sample's {was}   now {strata_now.get(s, 0):>10,}")
    print(f"\nWrote {args.out}. Edges that are not 'kept' or 'kept_restratified' need a new look; "
          f"weights come from the populations above.")
    return report


if __name__ == "__main__":
    main()
