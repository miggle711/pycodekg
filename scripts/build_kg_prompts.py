"""
build_kg_prompts.py

Pre-computes KG-only test-generation prompts for a set of TestGenEval
instances, targeting TestGenEval's `full` setting (generate a complete
test file from scratch, no existing test content shown to either arm).

For each instance, surfaces what pycodekg's TestContextExtractor +
LLMSerializer retrieve for every seed function/class the instance's patch
touches (a patch can change more than one -- 6/66 real django instances
do): each seed's own source, docstring, structural metadata (module,
class), callers/callees/siblings, and related classes it inherits from
or instantiates.

Also writes each seed's function name (and enclosing class, if any) as a
separate field per instance, meant to be merged into the `instruct` arm's
dataset construction on the testgeneval side so both arms can be told to
focus on the same changed function(s) -- without which the two arms
answer different tasks (kg_only implicitly focuses on the seed(s);
instruct's generic `full`-setting prompt does not name any function at
all). See miggle711/pycodekg#125 and miggle711/testgeneval#6.

Run from repo-kg-construction's own environment (not testgeneval's --
pycodekg isn't installed there by design, to keep the two projects'
dependency stacks decoupled). Writes one JSON file mapping instance id ->
{prompt, target_functions, target_classes} (both lists, one entry per
seed), meant to be merged into the dataset on the testgeneval side before
running inference.

Usage:
    python scripts/build_kg_prompts.py \
        --kg-dir /tmp/kg_66_instances/kg-outputs \
        --dataset-path /path/to/testgenevallite_django \
        --output kg_prompts.json
"""

import argparse
import json
import sys
from pathlib import Path

from kg_construction.kg.query import KGQueryEngine
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.llm.llm_serializer import LLMSerializer

SYSTEM_MESSAGE = (
    "You are an expert Python software testing assistant. Your job is to "
    "generate a complete test file for the given code, using structural "
    "context about the function under test (no full source file is "
    "provided -- work from the function's own source and its "
    "callers/callees/related tests)."
)

SEED_BLOCK_TEMPLATE = """Function under test: {function_name}
Module: {module}
Class: {class_name}
Docstring: {docstring}

Source:
```python
{source_code}
```

Declared exceptions: {exceptions}
"""

PROMPT_TEMPLATE = """{seed_blocks}
{sections}
Your job is to output a corresponding unit test file for {function_or_functions} that obtains
high coverage, including error cases and boundary conditions.

Each unit test must be a function starting with test_. Include all your test imports and setup
before your first test. Do not run the tests in the file, just output a series of tests. Do not
include a main method to run the tests.

Only output the unit test Python file in this format:

```python
Unit test Python code (file level)
```
"""


def _snippet_section(title: str, items: list) -> str:
    if not items:
        return ""
    parts = [f"{title}:"]
    for item in items:
        name = item.get("name", "?")
        parts.append(f"```python\n{item.get('source_code', '')}\n```  # {name}")
    return "\n".join(parts) + "\n"


def _related_section(items: list) -> str:
    # related has two entry shapes (parent_class has source_code,
    # instantiation doesn't), so it needs its own renderer instead of
    # _snippet_section's single generic shape.
    if not items:
        return ""
    parts = ["Related classes:"]
    for item in items:
        name = item.get("name", "?")
        module = item.get("module", "")
        via = "the seed" if item.get("source") == "seed" else "the seed's class"
        if item.get("type") == "parent_class":
            parts.append(
                f"- {name} ({module}), parent class of {via}:\n"
                f"```python\n{item.get('source_code', '')}\n```"
            )
        else:
            parts.append(f"- {name} ({module}), instantiated by {via}")
    return "\n".join(parts) + "\n"


def _build_seed_block(seed: dict) -> str:
    return SEED_BLOCK_TEMPLATE.format(
        function_name=seed.get("function_name", ""),
        module=seed.get("module", ""),
        class_name=seed.get("class_name", "") or "(none -- top-level function)",
        docstring=seed.get("docstring") or "(none)",
        source_code=seed.get("source_code", ""),
        exceptions=", ".join(seed.get("exceptions", [])) or "(none declared)",
    )


def _build_prompt(serialized: dict) -> str:
    # serialized["seed"] is a list -- a patch can change more than one
    # function/class in the same file (6/66 real django instances do),
    # so every seed gets its own block, not just the first.
    seeds = serialized["seed"]
    context = serialized["context"]

    seed_blocks = "\n".join(_build_seed_block(seed) for seed in seeds)

    sections = "\n".join(filter(None, [
        _snippet_section("Callers", context.get("callers", [])),
        _snippet_section("Callees", context.get("callees", [])),
        _snippet_section("Sibling methods", context.get("sibling_methods", [])),
        _related_section(context.get("related", [])),
    ]))

    return PROMPT_TEMPLATE.format(
        seed_blocks=seed_blocks,
        sections=sections,
        function_or_functions="this function" if len(seeds) == 1 else "these functions",
    )


def _repo_slug(repo: str) -> str:
    """Sanitize a repo name for use in a kg_<slug>_<commit>.json filename.

    Must match RepoKGBuilder._cache_path's sanitization exactly
    (kg/builder.py), which also replaces "-" and "." -- a repo like
    scikit-learn/scikit-learn produced kg files this script could never
    find when only "/" was being replaced here (kg_construction#141).
    """
    return repo.replace("/", "_").replace("-", "_").replace(".", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-dir", type=str, required=True,
                         help="Directory of kg_<repo>_<commit>.json files (e.g. the "
                              "build-kgs.yml artifact, downloaded via gh run download).")
    parser.add_argument("--dataset-path", type=str, required=True,
                         help="Local disk path to a saved dataset (load_from_disk).")
    parser.add_argument("--output", type=str, default="kg_prompts.json")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.dataset_path)
    rows = list(ds["test"])

    kg_dir = Path(args.kg_dir)
    prompts = {}
    failures = []
    stale_seed_instances = []
    multi_seed_instances = []
    # RQ4 instrumentation (docs/EXPERIMENT_PLAN.md's RQ4 instrumentation
    # section, per-instance retrieval overhead): TestContext.retrieval_time_s
    # is computed by extractor.extract() below regardless of what happens
    # afterward, so it's recorded for any instance where extract() itself
    # completed, even if serialization later fails that instance out of
    # `prompts` -- retrieval work still genuinely happened and took real
    # time. Written to a sidecar file, not merged into `prompts` itself, so
    # nothing downstream that reads the main kg_prompts_depthN.json schema
    # needs to change.
    retrieval_times = {}

    for row in rows:
        repo_slug = _repo_slug(row["repo"])
        commit = row["base_commit"]
        kg_path = kg_dir / f"kg_{repo_slug}_{commit[:8]}.json"
        if not kg_path.exists():
            failures.append((row["id"], f"no KG file at {kg_path}"))
            continue

        try:
            with open(kg_path) as f:
                kg = json.load(f)
            engine = KGQueryEngine(kg)
            extractor = TestContextExtractor(engine)
            instance = {
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "patch": row["patch"],
                "code_file": row["code_file"],
                "test_file": row["test_file"],
            }
            context = extractor.extract(instance, depth=args.depth)
            retrieval_times[row["id"]] = context.retrieval_time_s
            if context.stale_seed_labels:
                stale_seed_instances.append((row["id"], context.stale_seed_labels))
            context_dict = {
                "seeds": context.seeds,
                "context_nodes": context.context_nodes,
                "edges": context.edges,
                "test_nodes": context.test_nodes,
            }
            serialized = LLMSerializer(repo=row["repo"]).serialize(context_dict)
            if not serialized.get("seed"):
                failures.append((row["id"], "no seed extracted"))
                continue

            seeds = serialized["seed"]
            if len(seeds) > 1:
                multi_seed_instances.append((row["id"], [s.get("function_name") for s in seeds]))
            prompt_text = _build_prompt(serialized)
            prompts[row["id"]] = {
                "prompt": prompt_text,
                "target_functions": [s.get("function_name", "") for s in seeds],
                "target_classes": [s.get("class_name") or None for s in seeds],
            }
            seed_names = [s.get("function_name") for s in seeds]
            print(f"  OK: {row['id']} (seed{'s' if len(seeds) > 1 else ''}: {seed_names})")
        except Exception as e:
            failures.append((row["id"], f"{type(e).__name__}: {e}"))
            print(f"  FAILED: {row['id']}: {type(e).__name__}: {e}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(prompts, f, indent=2)

    retrieval_times_path = Path(args.output).with_name(
        Path(args.output).stem + "_retrieval_times.json"
    )
    with open(retrieval_times_path, "w") as f:
        json.dump(retrieval_times, f, indent=2)

    print(f"\n{len(prompts)}/{len(rows)} prompts built -> {args.output}")
    if retrieval_times:
        values = list(retrieval_times.values())
        print(
            f"{len(retrieval_times)} retrieval times -> {retrieval_times_path} "
            f"(mean {sum(values) / len(values):.4f}s, "
            f"min {min(values):.4f}s, max {max(values):.4f}s)"
        )
    if failures:
        print(f"{len(failures)} failures:")
        for instance_id, err in failures:
            print(f"  {instance_id}: {err}")
    if stale_seed_instances:
        # A seed staying on pre-patch source (kg_construction#124's fix
        # couldn't find it in the post-patch AST) should be rare, a
        # growing count here is worth investigating, not routine.
        print(f"{len(stale_seed_instances)} instance(s) with a stale (pre-patch) seed:")
        for instance_id, labels in stale_seed_instances:
            print(f"  {instance_id}: {labels}")
    if multi_seed_instances:
        # Real, not rare (6/66 real django instances) -- visible here so
        # a run's actual seed count per instance isn't only discoverable
        # by reading kg_prompts.json directly.
        print(f"{len(multi_seed_instances)} instance(s) with multiple seeds:")
        for instance_id, names in multi_seed_instances:
            print(f"  {instance_id}: {names}")


if __name__ == "__main__":
    main()
