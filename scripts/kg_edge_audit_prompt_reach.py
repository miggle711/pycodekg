"""Which edges kg_edge_audit.py flags can reach a benchmark kg_only prompt (#239).

LLMSerializer renders a calls edge only when one end is a seed (callers are
edges into it, callees edges out of it), skips the other end when it is test
code, and keeps the first 10 per category; accesses edges are not rendered.
So a flagged calls edge can reach a prompt only if one end is a benchmark
target function and the other is not test code. The count is an upper bound:
an edge past the cap of 10 is still counted, marked `over_cap`.

Two steps, because they need different environments:

  targets  (testgeneval's environment: `datasets`) resolves each instance's
           target functions as testgeneval does, with
           swebench_docker/target_range.py's resolve_target_line_range on the
           post-patch code_src, and names them by file and qualified name:

    python scripts/kg_edge_audit_prompt_reach.py targets \\
        --target-range <testgeneval>/swebench_docker/target_range.py \\
        --repos pytest-dev/pytest django/django sympy/sympy --out bench_targets.json

  reach    (pycodekg's venv) matches those names to nodes of KGs saved by
           kg_edge_audit.py --save-dir, runs the audit, and reports which
           flagged calls edges touch a target:

    PYTHONPATH=src .venv/bin/python scripts/kg_edge_audit_prompt_reach.py reach \\
        --targets bench_targets.json --out reach.json kgs/*.json

The KGs are built at repo_cache HEAD, not at each instance's base commit, so
a target is matched by file and qualified name; unmatched instances are
counted. KG file names must be <owner>__<repo>.json, as --save-dir writes.
"""

import argparse
import ast
import collections
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CAP = 10


def targets(args):
    from datasets import load_dataset
    spec = importlib.util.spec_from_file_location("target_range", args.target_range)
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    out = []
    for r in load_dataset(args.dataset, split="test"):
        if r["repo"] not in args.repos:
            continue
        slug = r["repo"].replace("/", "__")
        try:
            ranges = tr.resolve_target_line_range(r["code_src"], r["patch"], r["code_file"]) or []
            tree = ast.parse(r["code_src"])
        except Exception as e:  # noqa: BLE001 - recorded per instance
            out.append({"id": r["id"], "repo": slug, "file": r["code_file"], "targets": [], "error": str(e)[:80]})
            continue
        names = []

        def walk(node, prefix):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = f"{prefix}.{c.name}" if prefix else c.name
                    if not isinstance(c, ast.ClassDef):
                        start = min([c.lineno] + [d.lineno for d in c.decorator_list])
                        if any((a, b) == (start, c.end_lineno) for a, b in ranges):
                            names.append(q)
                    walk(c, q)
        walk(tree, "")
        out.append({"id": r["id"], "repo": slug, "file": r["code_file"], "targets": sorted(set(names)),
                    "ranges": ranges})
    json.dump(out, open(args.out, "w"))
    print(f"{len(out)} instances -> {args.out}")


def _is_test(node):
    from kg_construction.ast.helpers import _is_test_file
    return node["type"] == "test_function" or _is_test_file((node.get("metadata") or {}).get("filepath", ""))


def reach_one(kg, insts):
    """(rows, summary) for one repository's KG and its instances."""
    import kg_edge_audit as audit
    by_id = {n["id"]: n for n in kg["nodes"]}
    key = {((n.get("metadata") or {}).get("filepath"), (n.get("metadata") or {}).get("qualname")): n["id"]
           for n in kg["nodes"] if n["type"] in ("function", "method")}
    seeds_of = {t["id"]: {key[(t["file"], q)] for q in t["targets"] if (t["file"], q) in key} for t in insts}
    all_seeds = set().union(*seeds_of.values()) if seeds_of else set()
    n_callers, n_callees, reachable = collections.Counter(), collections.Counter(), 0
    for e in kg["edges"]:
        if e["relation"] != "calls":
            continue
        s_test, t_test = _is_test(by_id[e["source"]]), _is_test(by_id[e["target"]])
        if not s_test:
            n_callers[e["target"]] += 1
        if not t_test:
            n_callees[e["source"]] += 1
        if (e["target"] in all_seeds and not s_test) or (e["source"] in all_seeds and not t_test):
            reachable += 1
    flagged = {}
    audit.audit(kg, flagged)

    def name(x):
        m = by_id[x].get("metadata") or {}
        return f"{m.get('qualname') or by_id[x]['label']} ({m.get('filepath')})"

    rows = []
    for rule, keys in sorted(flagged.items()):
        for rel, s, t in sorted(set(keys)):
            if rel != "calls":
                continue
            for iid, seeds in sorted(seeds_of.items()):
                if t in seeds and not _is_test(by_id[s]):
                    role, over = "caller", n_callers[t] > CAP
                elif s in seeds and not _is_test(by_id[t]):
                    role, over = "callee", n_callees[s] > CAP
                else:
                    continue
                rows.append({"rule": rule, "instance": iid, "role": role, "over_cap": over,
                             "source": name(s), "target": name(t), "source_id": s, "target_id": t})
    summary = {
        "instances": len(insts),
        "instances_matched": sum(bool(v) for v in seeds_of.values()),
        "calls_edges_that_can_reach_a_prompt": reachable,
        "flagged_calls_edges": {r: len({k for k in v if k[0] == "calls"}) for r, v in flagged.items()},
        "flagged_reaching_edges": len({(r["source_id"], r["target_id"]) for r in rows}),
        "flagged_reaching_instances": len({r["instance"] for r in rows}),
    }
    return rows, summary


def reach(args):
    tg = json.load(open(args.targets))
    result = {}
    for p in args.kg_json:
        repo = Path(p).stem
        rows, summary = reach_one(json.load(open(p)), [t for t in tg if t["repo"] == repo])
        result[repo] = {**summary, "reaching": rows}
        print(f"{repo}: {summary['instances_matched']} of {summary['instances']} instances matched; "
              f"{summary['calls_edges_that_can_reach_a_prompt']} calls edges can reach a prompt; "
              f"flagged among them: {summary['flagged_reaching_edges']} edges in "
              f"{summary['flagged_reaching_instances']} instances")
        for r in rows:
            print(f"  {r['rule']}: {r['instance']} {r['role']}{' (over cap)' if r['over_cap'] else ''}: "
                  f"{r['source']} -> {r['target']}")
    json.dump(result, open(args.out, "w"), indent=1)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--target-range", required=True, help="testgeneval's swebench_docker/target_range.py")
    t.add_argument("--dataset", default="kjain14/testgeneval")
    t.add_argument("--repos", nargs="+", default=["pytest-dev/pytest", "django/django", "sympy/sympy"])
    t.add_argument("--out", required=True)
    r = sub.add_parser("reach")
    r.add_argument("--targets", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("kg_json", nargs="+")
    args = ap.parse_args(argv)
    return targets(args) if args.cmd == "targets" else reach(args)


if __name__ == "__main__":
    main()
