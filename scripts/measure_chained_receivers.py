"""Measure #118 call-result receiver gaps on benchmark targets, without changing KGs.

Build one KG per supplied bare clone at HEAD. Locate targets with the supplied
version of testgeneval's target_range.py on post-patch dataset code_src, then
match their file/qualified name and property role in the HEAD source. Inspect
the HEAD function body, not patched text against a different revision's edges.
This is a HEAD proxy: missing/ambiguous targets are explicit, not zero gains.

A read-only profiler observes the builder's real _resolve_call results before
function-level edge folding. Candidate probes use the unchanged resolver with
an explicit return annotation, or the original receiver hint for a strict
return-self method. They are hypothetical evidence, never merged into the KG.

Example:
 PYTHONPATH=src python scripts/measure_chained_receivers.py \
   --arrow /path/to/testgeneval-test.arrow --target-range /path/to/target_range.py \
   --cache-dir /path/to/repo_cache --output /tmp/chains.json
"""

import argparse
import ast
import builtins
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tokenize

from kg_construction.ast.helpers import assign_qualnames
from kg_construction.kg.builder import RepoASTParser, SCHEMA_VERSION

REPOS = {"pytest-dev/pytest": "pytest-dev__pytest", "django/django": "django__django",
         "sympy/sympy": "sympy__sympy"}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def owned_nodes(function):
    """Body expressions only; skip nested defs/classes/lambdas and header expressions."""
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        if isinstance(node, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def span(node):
    return min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno


def property_role(node):
    names = [ast.unparse(d) for d in node.decorator_list]
    for role in ("setter", "deleter"):
        if any(n.endswith("." + role) for n in names):
            return role
    return "getter" if any(n.rsplit(".", 1)[-1] in ("property", "cached_property") for n in names) else "function"


def identity(qualname, node):
    return re.sub(r"#\d+", "", qualname), type(node).__name__, property_role(node)


class SourceIndex:
    def __init__(self, root):
        self.root, self.files = root, {}

    def read(self, filepath):
        if filepath not in self.files:
            with tokenize.open(self.root / filepath) as handle:
                source = handle.read()
            tree = ast.parse(source, filename=filepath)
            defs = {q: n for n, q in assign_qualnames(tree) if isinstance(n, FUNCTIONS)}
            groups = defaultdict(list)
            for q, node in defs.items():
                groups[identity(q, node)].append((q, node))
            self.files[filepath] = source, defs, groups
        return self.files[filepath]

    def function(self, node):
        md = node.get("metadata", {})
        try:
            return self.read(md["filepath"])[1][md["qualname"]]
        except (OSError, SyntaxError, UnicodeError, KeyError):
            return None


def target_specs(rows, resolve_targets, sources):
    specs, totals = [], Counter()
    for row in rows:
        totals["dataset_instances"] += 1
        filepath = row["code_file"]
        try:
            tree = ast.parse(row["code_src"])
            ranges = resolve_targets(row["code_src"], row["patch"], filepath) or []
        except (SyntaxError, ValueError) as error:
            specs.append({"instance": row["id"], "status": "invalid_post_patch_source", "error": str(error)})
            continue
        if not ranges:
            totals["instances_without_function_target"] += 1
            continue
        totals["instances_with_function_target"] += 1
        post = [(n, q) for n, q in assign_qualnames(tree) if isinstance(n, FUNCTIONS)]
        for selected_span in ranges:
            totals["target_function_occurrences"] += 1
            matches = [(n, q) for n, q in post if span(n) == tuple(selected_span)]
            spec = {"instance": row["id"], "filepath": filepath, "post_patch_range": selected_span}
            if len(matches) != 1:
                spec["status"] = "post_patch_target_unclear"
            else:
                node, qualname = matches[0]
                spec["post_patch_qualname"] = qualname
                try:
                    head_matches = sources.read(filepath)[2].get(identity(qualname, node), [])
                except (OSError, SyntaxError, UnicodeError) as error:
                    head_matches = []
                    spec["head_source_error"] = str(error)
                if len(head_matches) == 1:
                    spec.update(status="matched", head_qualname=head_matches[0][0])
                else:
                    spec["status"] = "head_target_missing" if not head_matches else "head_target_ambiguous"
            specs.append(spec)
    return specs, dict(totals)


class ObservingParser(RepoASTParser):
    """Observe actual resolver returns; don't infer site resolution from folded edges."""
    def __init__(self, wanted, max_workers=4):
        super().__init__(max_workers=max_workers)
        self.wanted = wanted
        self.call_records = []

    def _resolve_edges(self, nodes, edges, indices):
        self.nodes, self.indices = nodes, indices
        selected = {n["id"] for n in nodes if
                    (n.get("metadata", {}).get("filepath"), n.get("metadata", {}).get("qualname")) in self.wanted}
        self.call_records, result = self.observe(nodes, edges, indices, selected)
        return result

    def observe(self, nodes, edges, indices, selected=None):
        if sys.getprofile() is not None:
            raise RuntimeError("Run this audit without another active Python profiler")
        records, classes = [], {}

        def profile(frame, event, value):
            if event != "return":
                return
            if frame.f_code.co_name == "_class_ref" and frame.f_back.f_code.co_name == "_typed_classes":
                classes[id(frame.f_back.f_locals["meta"])] = list(value or [])
            elif frame.f_code.co_name == "_resolve_call":
                local = frame.f_locals
                if selected is None or local["source_id"] in selected:
                    records.append({"source": local["source_id"], "name": local["callee_name"],
                                    "meta": dict(local["meta"]), "meta_id": id(local["meta"]),
                                    "targets": list(value[0]), "confidence": value[1],
                                    "typed_classes": list(classes.get(id(local["meta"]), []))})
        sys.setprofile(profile)
        try:
            result = super()._resolve_edges(nodes, edges, indices)
        finally:
            sys.setprofile(None)
        return records, result


def receiver(call):
    return ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else None


def generic(record):
    md = record["meta"]
    return (md.get("receiver") is not None and not any(md.get(k) for k in
            ("class_hint", "class_hint_id", "local_type_hint", "import_resolved", "is_super_call")))


def definite_self_return(node):
    if not isinstance(node, ast.FunctionDef) or node.decorator_list:
        return False
    args = node.args.posonlyargs + node.args.args
    if not args or args[0].arg != "self" or not node.body or not isinstance(node.body[-1], ast.Return):
        return False
    body = list(owned_nodes(node))
    returns = [n for n in body if isinstance(n, ast.Return)]
    return (returns and all(isinstance(n.value, ast.Name) and n.value.id == "self" for n in returns)
            and not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in body)
            and not any(isinstance(n, ast.Name) and n.id == "self" and isinstance(n.ctx, (ast.Store, ast.Del)) for n in body))


def simple_annotation(node):
    ann = node.returns
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            ann = ast.parse(ann.value, mode="eval").body
        except SyntaxError:
            return None
    if not isinstance(ann, (ast.Name, ast.Attribute)):
        return None
    name = ast.unparse(ann)
    return None if name in vars(builtins) else name


def collect_sites(parser, kg, specs, sources):
    nodes = {n["id"]: n for n in kg["nodes"]}
    by_function = {(n["metadata"].get("filepath"), n["metadata"].get("qualname")): n
                   for n in kg["nodes"] if n["type"] in ("function", "method", "test_function")}
    records = defaultdict(list)
    for r in parser.call_records:
        if not r["meta"].get("via_closure"):
            records[(r["source"], r["name"])].append(r)
    outgoing = defaultdict(set)
    for e in kg["edges"]:
        if e["relation"] == "calls":
            outgoing[e["source"]].add(nodes[e["target"]]["label"])
    sites, probes = [], []
    seen = set()
    for spec in specs:
        if spec["status"] != "matched":
            continue
        caller = by_function.get((spec["filepath"], spec["head_qualname"]))
        if caller is None:
            spec["status"] = "head_target_not_in_kg"
            continue
        function = sources.function(caller)
        if function is None:
            spec["status"] = "head_source_unavailable"
            continue
        for call in owned_nodes(function):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Call)):
                continue
            key = (spec["instance"], caller["id"], call.lineno, call.col_offset, call.end_lineno, call.end_col_offset)
            if key in seen:
                continue
            seen.add(key)
            inner = call.func.value
            site = {"instance": spec["instance"], "caller_id": caller["id"], "filepath": spec["filepath"],
                    "qualname": spec["head_qualname"], "line": call.lineno, "column": call.col_offset,
                    "end_line": call.end_lineno, "end_column": call.end_col_offset,
                    "call": ast.unparse(call), "outer": call.func.attr,
                    "candidates": [], "notes": []}
            sites.append(site)
            if isinstance(inner.func, ast.Name) and inner.func.id == "super":
                site["status"] = "special_super_excluded"
                continue
            outer = [r for r in records[(caller["id"], call.func.attr)] if generic(r)]
            if len(outer) != 1:
                site["status"] = "resolution_not_observed"
                continue
            site["status"] = "resolved" if outer[0]["targets"] else "unresolved"
            site["current_targets"] = outer[0]["targets"]
            site["current_confidence"] = outer[0]["confidence"]
            site["same_name_edge_elsewhere_in_caller"] = call.func.attr in outgoing[caller["id"]]
            if site["status"] != "unresolved":
                continue
            inner_name = inner.func.attr if isinstance(inner.func, ast.Attribute) else (
                inner.func.id if isinstance(inner.func, ast.Name) else None)
            choices = records[(caller["id"], inner_name)]
            exact_receiver = [r for r in choices if r["meta"].get("receiver") == receiver(inner)]
            if exact_receiver:
                choices = exact_receiver
            if not choices or any(len(r["targets"]) != 1 or r["confidence"] not in ("exact", "qualified") for r in choices):
                site["notes"].append("inner call not uniquely anchored by current resolver")
                continue
            targets = {r["targets"][0] for r in choices}
            if len(targets) != 1:
                site["notes"].append("inner receiver has multiple possible targets")
                continue
            inner_node = nodes[targets.pop()]
            site["inner_target"] = {"id": inner_node["id"], **{k: inner_node.get("metadata", {}).get(k)
                                      for k in ("filepath", "qualname", "lineno", "returns", "decorators")}}
            if inner_node["type"] == "class":
                site["notes"].append("constructor receiver; not counted as annotation or return-self evidence")
                continue
            inner_def = sources.function(inner_node)
            if inner_def is None:
                site["notes"].append("inner definition unavailable")
                continue
            if isinstance(inner_def, ast.AsyncFunctionDef) or any(
                    isinstance(n, (ast.Yield, ast.YieldFrom)) for n in owned_nodes(inner_def)):
                site["notes"].append("inner call returns a coroutine/generator, not the annotated object directly")
                continue
            annotation = simple_annotation(inner_def)
            if annotation:
                site["return_annotation"] = annotation
                if annotation not in ("Self", "typing.Self", "typing_extensions.Self"):
                    probes.append((site, "return_annotation", inner_node["id"],
                                   {"receiver": "<return-value>", "local_type_hint": annotation, "unresolved": True}))
            self_evidence = definite_self_return(inner_def)
            self_annotation = annotation in ("Self", "typing.Self", "typing_extensions.Self")
            if self_evidence or self_annotation:
                # Keep the original receiver's class, including a subclass;
                # the defining method's owner alone would lose overrides.
                anchored = [r for r in choices if r["meta"].get("class_hint_id") or r["meta"].get("local_type_hint")]
                if anchored and len(anchored) == len(choices):
                    hint_signatures = {json.dumps(r["meta"], sort_keys=True) for r in anchored}
                    if len(hint_signatures) == 1:
                        md = dict(anchored[0]["meta"])
                        probes.append((site, "return_annotation" if self_annotation else "return_self", caller["id"], md))
    structural = [e for e in kg["edges"] if e["relation"] != "calls" and e["relation"] != "tests"]
    probe_edges = []
    metadata_to_probe = {}
    for site, evidence, source, md in probes:
        md = dict(md, unresolved=True)
        probe_edges.append({"source": source, "target": site["outer"], "relation": "calls", "metadata": md})
        metadata_to_probe[id(md)] = site, evidence
    if probe_edges:
        observed, _ = parser.observe(parser.nodes, structural + probe_edges, parser.indices)
        for result in observed:
            pair = metadata_to_probe.get(result["meta_id"])
            if pair is None:
                continue
            site, evidence = pair
            class_ids = result["typed_classes"] or ([result["meta"]["class_hint_id"]] if result["meta"].get("class_hint_id") else [])
            if (len(class_ids) == 1 and len(result["targets"]) == 1 and result["confidence"] == "qualified"
                    and nodes[result["targets"][0]]["type"] in ("method", "function")):
                target = nodes[result["targets"][0]]
                site["candidates"].append({"evidence": evidence, "target_id": target["id"],
                                           "file": target["metadata"].get("filepath"),
                                           "qualname": target["metadata"].get("qualname"),
                                           "class_id": class_ids[0]})
    return sites


def summarize(specs, totals, sites):
    counts = Counter(s["status"] for s in specs)
    matched = [s for s in specs if s["status"] == "matched"]
    unresolved = [s for s in sites if s["status"] == "unresolved"]
    candidate = [s for s in unresolved if s["candidates"]]
    return {**totals, "target_statuses": dict(sorted(counts.items())),
            "instances_with_matched_targets": len({s["instance"] for s in matched}),
            "unique_matched_head_functions": len({(s["filepath"], s["head_qualname"]) for s in matched}),
            "site_statuses": dict(sorted(Counter(s["status"] for s in sites).items())),
            "chained_call_site_occurrences": len(sites), "unresolved_outer_site_occurrences": len(unresolved),
            "unresolved_unique_head_sites": len({(s["caller_id"], s["line"], s["column"], s["end_line"], s["end_column"]) for s in unresolved}),
            "instances_with_unresolved_outer_sites": len({s["instance"] for s in unresolved}),
            "annotation_candidate_occurrences": sum(any(c["evidence"] == "return_annotation" for c in s["candidates"]) for s in unresolved),
            "return_self_candidate_occurrences": sum(any(c["evidence"] == "return_self" for c in s["candidates"]) for s in unresolved),
            "either_candidate_occurrences": len(candidate),
            "either_candidate_unique_head_sites": len({(s["caller_id"], s["line"], s["column"], s["end_line"], s["end_column"]) for s in candidate}),
            "instances_with_candidates": len({s["instance"] for s in candidate})}


def measure(rows, clones, resolve_targets, max_workers):
    results = []
    for repo, clone in clones.items():
        commit = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
        print(f"Building {repo} @ {commit[:10]} ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="chain-audit-") as tmp:
            root = Path(tmp)
            export = subprocess.Popen(["git", "-C", str(clone), "archive", commit], stdout=subprocess.PIPE)
            unpack = subprocess.run(["tar", "-x", "-C", str(root)], stdin=export.stdout)
            export.stdout.close()
            if export.wait() or unpack.returncode:
                raise RuntimeError(f"Could not export {repo}")
            sources = SourceIndex(root)
            subset = [r for r in rows if r["repo"] == repo]
            specs, totals = target_specs(subset, resolve_targets, sources)
            wanted = {(s["filepath"], s["head_qualname"]) for s in specs if s["status"] == "matched"}
            parser = ObservingParser(wanted, max_workers)
            kg = parser.parse_repo(repo, root)
            sites = collect_sites(parser, kg, specs, sources)
            result = {"repo": repo, "commit": commit, "summary": summarize(specs, totals, sites),
                      "targets": specs, "sites": sites}
            results.append(result)
            print(json.dumps(result["summary"], sort_keys=True), flush=True)
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--arrow", type=Path)
    source.add_argument("--dataset-path", type=Path)
    cli.add_argument("--cache-dir", type=Path, required=True)
    cli.add_argument("--target-range", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--max-workers", type=int, default=4)
    args = cli.parse_args()
    from datasets import Dataset, load_from_disk
    data = Dataset.from_file(str(args.arrow)) if args.arrow else load_from_disk(str(args.dataset_path))["test"]
    rows = [dict(r) for r in data]
    spec = importlib.util.spec_from_file_location("chain_target_range", args.target_range)
    target_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target_module)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps({k: row[k] for k in ("id", "repo", "base_commit", "code_file", "patch", "code_src")},
                                 sort_keys=True, ensure_ascii=False).encode() + b"\n")
    report = {"input_instances": len(rows), "dataset_fingerprint": data._fingerprint,
              "input_sha256": digest.hexdigest(), "target_range_sha256": hashlib.sha256(args.target_range.read_bytes()).hexdigest(),
              "builder_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip(),
              "schema_version": SCHEMA_VERSION,
              "scope": "Matched HEAD function bodies, excluding nested defs/classes/lambdas and headers; call-result receivers, not general property-chain census.",
              "limitations": ["HEAD proxy, not instance-commit/post-patch KG measurement.",
                              "Candidates are conservative static evidence, not verified correct edges or a proposed production implementation.",
                              "Only one-hop evidence from an already anchored inner call is counted; hypothetical new edges are not propagated through further chain steps.",
                              "Only simple non-builtin return class annotations/Self or strict undecorated synchronous return-self methods qualify.",
                              "Return-self requires every explicit return to return the unrebound self and a final direct return; this is not full control-flow proof.",
                              "Unknown/import-external/union/generator/async returns are not counted as candidates; annotation candidates do not prove decorator/runtime contracts.",
                              "Counts include repeated benchmark occurrences; unique HEAD-site counts are reported separately."],
              "results": measure(rows, {r: args.cache_dir / c for r, c in REPOS.items()}, target_module.resolve_target_line_range, args.max_workers)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("repos: " + ", ".join(f"{r['repo']} @ {r['commit'][:10]}" for r in report["results"]))


if __name__ == "__main__":
    main()
