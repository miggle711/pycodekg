"""Measure referenced module constants and stale KG values without building KGs.

TestGenEval's code_src is POST-patch (creation/transform_swebench.py reads it
after applying the patch). Reverse the focal-file hunks to recover the base,
then reapply and verify an exact normalized-source round trip. New files
start from an empty base. Any failure is reported, not silently excluded.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/analyze_module_constants.py \
    --dataset-path /path/to/saved/dataset --output /tmp/constants.json
  # Without --dataset-path, load kjain14/testgeneval from Hugging Face.
  # --arrow can read an existing cached testgeneval-test.arrow offline.

Target ranges come from the supplied testgeneval checkout's actual
swebench_docker/target_range.py; its hash is recorded in the output. The
serializer's symtable helper and context renderer are reused directly.
The stale-value counts compare base (pre-patch) constants with post-patch
ones: they describe what a KG-only lookup would show, and stay as measured.
Separately, `post_patch_rendering` renders each instance the way prompts are
now built (TestContext.post_patch_constants overlaying the KG's pre-patch file
node) and checks every rendered `NAME = value` line against the post-patch
file: any stale, missing or extra assignment there is counted.
"""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

from kg_construction.ast.helpers import _collect_file_level_info
from kg_construction.extraction.context import post_patch_file_constants
from kg_construction.extraction.patch import (
    _iter_patch_lines, is_newly_created_file, reconstruct_post_patch_source,
)
from kg_construction.llm.llm_serializer import (
    LLMSerializer, MAX_CONSTANT_VALUE_CHARS, referenced_module_globals,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kg_prompts import _build_context  # noqa: E402


def normalize(source):
    return source.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def reverse_focal_patch(patch, code_file):
    """Reverse only hunk content, using the count-aware shared diff reader.

    Keep a normal focal-file header: it identifies the path for the replay
    helper, while swapping old/new hunk coordinates and additions/removals.
    """
    out = [f"--- a/{code_file}", f"+++ b/{code_file}"]
    # split after newline only, matching the shared parser's source rules.
    lines = normalize(patch).split("\n")
    if lines[-1:] == [""]:
        lines.pop()
    for file, kind, line in _iter_patch_lines(lines):
        if file != code_file:
            continue
        if kind == "@@":
            line = re.sub(r"^@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@",
                          lambda m: f"@@ -{m[2]} +{m[1]} @@", line)
        elif kind in ("+", "-"):
            line = ("-" if kind == "+" else "+") + line[1:]
        out.append(line)
    return "\n".join(out) + "\n"


def apply_with_offsets(source, patch, path):
    """Apply exact hunk content with git's offset search, without fuzzing.

    Some dataset patches have shifted hunk coordinates. Git can relocate an
    exact context match; no lines are ignored or whitespace relaxed. All
    writes are restricted to the focal file in a temporary directory.
    """
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("focal file must be a repository-relative path")
    with tempfile.TemporaryDirectory(prefix="constants_patch_") as tmp:
        file = Path(tmp) / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source)
        result = subprocess.run(["git", "apply", "--unidiff-zero", "--whitespace=nowarn",
                                 f"--include={path}", "-"], input=patch, text=True,
                                capture_output=True, cwd=tmp)
        if result.returncode:
            return None
        return file.read_text() if file.exists() else ""


def source_pair(row, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    post = normalize(row["code_src"])
    patch, path = row["patch"], row["code_file"]
    if is_newly_created_file(patch, path):
        before = ""
    else:
        reverse = reverse_focal_patch(patch, path)
        before = reconstruct_post_patch_source(post, reverse, path)
        if before is None:
            before = apply_with_offsets(post, reverse, path)
            diagnostics["reverse_offset_relocation"] = True
    if before is None:
        raise ValueError("reverse patch did not apply to dataset post-patch code_src")
    reconstructed = reconstruct_post_patch_source(before, patch, path)
    if reconstructed is None:
        reconstructed = apply_with_offsets(before, patch, path)
        diagnostics["forward_offset_relocation"] = True
    if reconstructed != post:
        raise ValueError("forward patch round trip differs from dataset code_src")
    return before, reconstructed


def constants(source):
    return _collect_file_level_info(ast.parse(source))[2]


def analyze_row(row, resolve_targets):
    diagnostics = {}
    before, post = source_pair(row, diagnostics)
    path = row["code_file"]
    old, new = constants(before), constants(post)
    ranges = resolve_targets(post, row["patch"], path) or []
    lines = post.split("\n")
    seeds, references = [], set()
    for i, (start, end) in enumerate(ranges):
        source = "\n".join(lines[start - 1:end])
        references.update(referenced_module_globals(source, path))
        seeds.append({"id": str(i), "label": f"target_{i}", "type": "function",
                      "metadata": {"filepath": path, "source_code": source}})

    def rendered_lines(serialized):
        section = serialized["context"]["module_constants"]
        return [line for item in section for line in item["source_code"].split("\n")]

    def expected_line(name, value):
        if len(value) > MAX_CONSTANT_VALUE_CHARS:
            value = value[:MAX_CONSTANT_VALUE_CHARS] + " ... # value truncated"
        return f"{name} = {value}"

    # The prompt as now built: the post-patch overlay, with the pre-patch
    # file node also present, so a fallback to it would show up as stale.
    overlay, overlay_error = post_patch_file_constants(post)
    rendered = LLMSerializer().serialize({
        "seeds": seeds, "file_nodes": [{"id": "file", "type": "file",
                                        "metadata": {"path": path, "constants": old}}],
        "post_patch_constants": {path: overlay} if overlay is not None else {},
        "constants_unavailable": {path: overlay_error} if overlay_error else {}})
    shown = rendered_lines(rendered)
    want = [expected_line(n, new[n]) for n in sorted(references & new.keys())]
    post_patch_rendering = {
        "unavailable": overlay_error,
        "has_section": bool(shown),
        "stale_or_wrong": sorted(set(shown) - set(want)),
        "missing": sorted(set(want) - set(shown)),
    }

    def rendered_growth(values):
        context = {"seeds": seeds, "file_nodes": [{"id": "file", "type": "file",
                   "metadata": {"path": path, "constants": values}}]}
        serialized = LLMSerializer().serialize(context)
        with_constants = _build_context(serialized)
        serialized["context"]["module_constants"] = []
        return len(with_constants) - len(_build_context(serialized))

    relevant = sorted(references & (old.keys() | new.keys()))
    changes = [{"name": n, "kind": "added" if n not in old else "removed" if n not in new else "changed",
                "before": old.get(n), "after": new.get(n)}
               for n in relevant if old.get(n) != new.get(n)]
    referenced_new = sorted(references & new.keys())
    return {"id": row["id"], "repo": row["repo"], "base_commit": row["base_commit"],
            "code_file": path, "target_ranges": ranges, "referenced_constants": referenced_new,
            "reconstruction": diagnostics or {"exact_coordinates": True},
            "base_kg_referenced_constants": sorted(references & old.keys()),
            "added_characters": rendered_growth(new),
            "base_kg_added_characters": rendered_growth(old),
            "truncated_values": [n for n in referenced_new if len(new[n]) > MAX_CONSTANT_VALUE_CHARS],
            "base_kg_truncated_values": [n for n in sorted(references & old.keys())
                                         if len(old[n]) > MAX_CONSTANT_VALUE_CHARS],
            "stale_constants": changes,
            "post_patch_rendering": post_patch_rendering}


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": len(values), "median": statistics.median(values),
            "p90": values[math.ceil(.9 * len(values)) - 1], "max": values[-1]}


def summarize(rows, field="referenced_constants", growth="added_characters"):
    affected = [r for r in rows if r[field]]
    return {"instances": len(rows), "with_constants": len(affected),
            "percent": 100 * len(affected) / len(rows) if rows else 0,
            "added_characters_all": distribution([r[growth] for r in rows]),
            "added_characters_affected": distribution([r[growth] for r in affected])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-path")
    source.add_argument("--arrow", type=Path)
    parser.add_argument("--dataset", default="kjain14/testgeneval")
    parser.add_argument("--target-range", type=Path, default=Path(__file__).resolve().parents[2]
                        / "testgeneval/swebench_docker/target_range.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import Dataset, load_dataset, load_from_disk
    if args.arrow:
        data = Dataset.from_file(str(args.arrow), in_memory=True)
    elif args.dataset_path:
        data = load_from_disk(args.dataset_path)["test"]
    else:
        data = load_dataset(args.dataset)["test"]
    spec = importlib.util.spec_from_file_location("testgeneval_target_range", args.target_range)
    target_range = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target_range)
    rows, failures = [], []
    input_digest = hashlib.sha256()
    for row in data:
        input_digest.update((json.dumps({k: row[k] for k in
                             ("id", "repo", "base_commit", "code_file", "patch", "code_src")},
                             sort_keys=True) + "\n").encode("utf-8"))
        try:
            rows.append(analyze_row(row, target_range.resolve_target_line_range))
        except (SyntaxError, ValueError) as e:
            failures.append({"id": row["id"], "repo": row["repo"], "error": str(e)})
    by_repo = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)
    result = {"dataset": args.dataset, "input_instances": len(data),
              "dataset_fingerprint": data._fingerprint,
              "input_sha256": input_digest.hexdigest(),
              "target_range_sha256": hashlib.sha256(args.target_range.read_bytes()).hexdigest(),
              "percentile_method": "nearest rank", "value_limit": MAX_CONSTANT_VALUE_CHARS,
              "overall": summarize(rows),
              "per_repo": {repo: summarize(rs) for repo, rs in sorted(by_repo.items())},
              "current_base_kg_values": summarize(rows, "base_kg_referenced_constants", "base_kg_added_characters"),
              "no_target_function_instances": sum(not r["target_ranges"] for r in rows),
              "instances_with_relocated_hunk_offsets": sum("reverse_offset_relocation" in r["reconstruction"]
                                                          for r in rows),
              "truncated_value_occurrences": sum(len(r["truncated_values"]) for r in rows),
              "instances_with_truncation": sum(bool(r["truncated_values"]) for r in rows),
              "base_kg_truncated_value_occurrences": sum(len(r["base_kg_truncated_values"]) for r in rows),
              "stale_instances": sum(bool(r["stale_constants"]) for r in rows),
              "post_patch_rendering": {
                  "instances_with_section": sum(r["post_patch_rendering"]["has_section"] for r in rows),
                  "unavailable": sum(bool(r["post_patch_rendering"]["unavailable"]) for r in rows),
                  "instances_with_stale_or_wrong_lines": sum(bool(r["post_patch_rendering"]["stale_or_wrong"]) for r in rows),
                  "instances_with_missing_lines": sum(bool(r["post_patch_rendering"]["missing"]) for r in rows),
              },
              "stale_value_occurrences": dict(Counter(c["kind"] for r in rows for c in r["stale_constants"])),
              "failures": failures, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if failures:
        raise SystemExit("Some rows failed; see failures in output before interpreting prevalence.")


if __name__ == "__main__":
    main()
