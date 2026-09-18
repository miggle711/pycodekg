"""
build_kgs_for_instances.py

Builds and saves a full repo KG for each (repo, base_commit) pair used by a
set of TestGenEval dataset instances, one KG per unique commit (a dataset
can have many instances sharing the same commit, or -- as with the 66
django/django instances checked in this project -- every instance can have
its own distinct commit).

Meant to run standalone (e.g. via GitHub Actions) and upload kg_output/ as
an artifact, decoupled from local-model inference or evaluation runs.

Usage:
    python scripts/build_kgs_for_instances.py --instances a,b,c
    python scripts/build_kgs_for_instances.py --dataset-path data/testgenevallite_django
"""

import argparse
import sys
import time
from pathlib import Path

from kg_construction.kg.builder import RepoKGBuilder


def _load_instances(instances: str, dataset_path: str, dataset_name: str) -> list:
    from datasets import load_dataset, load_from_disk

    if dataset_path:
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset(dataset_name)
    rows = list(ds["test"])

    if instances:
        wanted = set(instances.split(","))
        rows = [r for r in rows if r["id"] in wanted]
        missing = wanted - {r["id"] for r in rows}
        if missing:
            print(f"WARNING: ids not found in dataset: {sorted(missing)}", file=sys.stderr)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=str, default=None,
                         help="Comma-separated dataset 'id' values to build KGs for. "
                              "Omit to build for every row in the dataset.")
    parser.add_argument("--dataset-path", type=str, default=None,
                         help="Local disk path to a saved dataset (load_from_disk). "
                              "Takes precedence over --dataset.")
    parser.add_argument("--dataset", type=str, default="kjain14/testgenevallite",
                         help="HuggingFace dataset name, used if --dataset-path is not given.")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    rows = _load_instances(args.instances, args.dataset_path, args.dataset)

    # Multiple instances can share a (repo, base_commit) -- build each real
    # commit once, not once per instance.
    seen = {}
    for row in rows:
        seen[(row["repo"], row["base_commit"])] = row["id"]

    print(f"{len(rows)} instances -> {len(seen)} unique (repo, commit) builds")

    builder = RepoKGBuilder(max_workers=args.max_workers)
    failures = []
    durations = []  # (repo, commit, seconds) for each real (non-cached) build
    for (repo, commit), example_id in seen.items():
        if builder.load(repo, commit) is not None:
            print(f"  cached: {repo}@{commit[:8]} (from instance {example_id})")
            continue
        print(f"  building: {repo}@{commit[:8]} (from instance {example_id})")
        start = time.monotonic()
        try:
            kg = builder.build(repo, commit)
            builder.save(repo, kg)
        except Exception as e:
            print(f"  FAILED: {repo}@{commit[:8]}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((repo, commit, str(e)))
            continue
        elapsed = time.monotonic() - start
        durations.append((repo, commit, elapsed))
        print(f"    done in {elapsed:.1f}s")

    print(f"\n{len(seen) - len(failures)}/{len(seen)} builds succeeded")
    if failures:
        for repo, commit, err in failures:
            print(f"  FAIL {repo}@{commit[:8]}: {err}")

    if durations:
        total = sum(d for _, _, d in durations)
        slowest = sorted(durations, key=lambda x: -x[2])[:5]
        print(f"\n{len(durations)} real builds, {total:.1f}s total, "
              f"{total / len(durations):.1f}s average")
        print("Slowest 5:")
        for repo, commit, elapsed in slowest:
            print(f"  {elapsed:.1f}s  {repo}@{commit[:8]}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
