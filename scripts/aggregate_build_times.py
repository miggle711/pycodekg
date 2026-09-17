# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Aggregate per-repository KG build timing (build_time_parsing_s /
build_time_resolution_s), real input-size normalization
(total_lines_of_code, build_time_total_s_per_kloc), and real
output-size normalization (node_count/edge_count,
build_time_total_s_per_1k_nodes/_edges) out of a kg_output/ directory
of already-built kg_<repo>_<commit>.json files, for RQ4's one-time KG
construction cost (docs/EXPERIMENT_PLAN.md's RQ4 instrumentation
section).

RepoASTParser.parse_repo() (kg/builder.py) writes these fields into
every KG's own metadata as it's built -- nothing is silently discarded
the way TestContext.retrieval_time_s was before build_kg_prompts.py was
fixed to capture it. This script just needs to walk kg_output/ and pull
the numbers together, no new instrumentation required.

Usage:
    python scripts/aggregate_build_times.py --kg-dir kg_output --out build_times.csv

Files built before this instrumentation existed have no
build_time_parsing_s/build_time_resolution_s keys in their metadata at
all -- skipped, counted, and reported separately, not silently averaged
in as zero.
"""

import argparse
import csv
import glob
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kg-dir", required=True, help="Directory of kg_<repo>_<commit>.json files."
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.kg_dir, "kg_*.json")))
    if not files:
        sys.exit(f"No kg_*.json files found in {args.kg_dir}")

    rows = []
    skipped = []
    for path in files:
        with open(path) as f:
            kg = json.load(f)
        metadata = kg.get("metadata", {})
        parsing_s = metadata.get("build_time_parsing_s")
        resolution_s = metadata.get("build_time_resolution_s")
        if parsing_s is None or resolution_s is None:
            skipped.append(os.path.basename(path))
            continue
        total_s = parsing_s + resolution_s
        loc = metadata.get("total_lines_of_code")
        node_count = metadata.get("node_count")
        edge_count = metadata.get("edge_count")
        rows.append(
            {
                "file": os.path.basename(path),
                "repo": metadata.get("repo", ""),
                "base_commit": metadata.get("base_commit", ""),
                "file_count": metadata.get("file_count", ""),
                "total_lines_of_code": loc if loc is not None else "",
                "node_count": node_count if node_count is not None else "",
                "edge_count": edge_count if edge_count is not None else "",
                "build_time_parsing_s": parsing_s,
                "build_time_resolution_s": resolution_s,
                "build_time_total_s": round(total_s, 3),
                # Real code-size normalization (RQ4): raw seconds alone
                # can't distinguish "this repo is slow because it's huge"
                # from "this repo is slow because something's inefficient".
                # Blank rather than 0 for KGs built before total_lines_of_code
                # existed, so it isn't silently misread as "0s per kLOC".
                "build_time_total_s_per_kloc": (
                    round(total_s / (loc / 1000), 4) if loc else ""
                ),
                # Output-size normalization: distinguishes "this repo has
                # a lot of source" from "this repo produces a lot of
                # graph" -- a repo can be large in one and modest in the
                # other. Blank, not 0, for the same reason as above.
                "build_time_total_s_per_1k_nodes": (
                    round(total_s / (node_count / 1000), 4) if node_count else ""
                ),
                "build_time_total_s_per_1k_edges": (
                    round(total_s / (edge_count / 1000), 4) if edge_count else ""
                ),
            }
        )

    if not rows:
        sys.exit(
            f"None of the {len(files)} KG file(s) in {args.kg_dir} have "
            f"build_time_parsing_s/build_time_resolution_s -- all built "
            f"before this instrumentation existed. Re-run m3_build_kgs.slurm "
            f"to get real numbers."
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    parsing_vals = [r["build_time_parsing_s"] for r in rows]
    resolution_vals = [r["build_time_resolution_s"] for r in rows]
    print(f"{len(rows)} KG(s) -> {args.out}")
    print(
        f"  parsing:    mean {sum(parsing_vals) / len(parsing_vals):.3f}s  "
        f"min {min(parsing_vals):.3f}s  max {max(parsing_vals):.3f}s"
    )
    print(
        f"  resolution: mean {sum(resolution_vals) / len(resolution_vals):.3f}s  "
        f"min {min(resolution_vals):.3f}s  max {max(resolution_vals):.3f}s"
    )

    def _print_normalized_summary(field: str, label: str, unit: str, source_field: str, source_label: str):
        vals = [r[field] for r in rows if r[field] != ""]
        if vals:
            print(
                f"  {label}: mean {sum(vals) / len(vals):.4f}{unit}  "
                f"min {min(vals):.4f}{unit}  max {max(vals):.4f}{unit}"
            )
        missing = len(rows) - len(vals)
        if missing:
            print(
                f"    {missing} KG(s) have timing but no {source_field} "
                f"(built before that field existed) -- excluded from {source_label} stats above."
            )

    _print_normalized_summary(
        "build_time_total_s_per_kloc", "per kLOC   ", "s/kLOC", "total_lines_of_code", "the per-kLOC"
    )
    _print_normalized_summary(
        "build_time_total_s_per_1k_nodes", "per 1k nodes", "s/1k nodes", "node_count", "the per-node"
    )
    _print_normalized_summary(
        "build_time_total_s_per_1k_edges", "per 1k edges", "s/1k edges", "edge_count", "the per-edge"
    )
    if skipped:
        print(
            f"  {len(skipped)} file(s) skipped (built before this "
            f"instrumentation existed, no timing metadata): "
            f"{', '.join(skipped[:5])}"
            f"{', ...' if len(skipped) > 5 else ''}"
        )


if __name__ == "__main__":
    main()
