"""
build_kg_prompts.py

Pre-computes KG-only completion prompts for a set of TestGenEval instances:
no code_src/test_src (the whole file), only what pycodekg's
TestContextExtractor + LLMSerializer surface for the one seed function the
instance's patch touches -- its own source, structural metadata (module,
class), callers/callees/siblings, and any existing tests already linked to
it in the KG.

This answers "does KG-only, surgical context work as well as (or better/worse
than) dumping the whole file" -- a deliberately different question from
"does KG context help on TOP OF the whole file" (which would need the KG
prompt to also include code_src/test_src, additively).

Run from repo-kg-construction's own environment (not testgeneval's --
pycodekg isn't installed there by design, to keep the two projects'
dependency stacks decoupled). Writes one JSON file mapping instance id ->
{first, last, extra} prompt strings, meant to be merged into the dataset on
the testgeneval side before running inference.

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
    "complete the next test given structural context about the function "
    "under test (no full source file is provided -- work from the "
    "function's own source and its callers/callees/related tests)."
)

PROMPT_TEMPLATE = """Function under test: {function_name}
Module: {module}
Class: {class_name}

Source:
```python
{source_code}
```

Declared exceptions: {exceptions}

{sections}
Your job is to write the Python code for the next test for this function. Ideally your next
test should improve coverage of the function's behavior, including error cases and boundary
conditions.

Only output the next unit test, preserve indentation and formatting. Do not output anything
else. Format like this:

```python
Next unit test Python code
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


def _build_prompt(serialized: dict) -> str:
    seed = serialized["seed"]
    context = serialized["context"]

    sections = "\n".join(filter(None, [
        _snippet_section("Callers", context.get("callers", [])),
        _snippet_section("Callees", context.get("callees", [])),
        _snippet_section("Sibling methods", context.get("sibling_methods", [])),
        _snippet_section(
            "Existing tests already covering related code", context.get("existing_tests", [])
        ),
    ]))

    return PROMPT_TEMPLATE.format(
        function_name=seed.get("function_name", ""),
        module=seed.get("module", ""),
        class_name=seed.get("class_name", "") or "(none -- top-level function)",
        source_code=seed.get("source_code", ""),
        exceptions=", ".join(seed.get("exceptions", [])) or "(none declared)",
        sections=sections,
    )


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

    for row in rows:
        repo_slug = row["repo"].replace("/", "_")
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

            prompt_text = _build_prompt(serialized)
            # Same prompt text for all 3 completion settings -- the KG
            # context doesn't depend on which test-file fragment the plain
            # arm would have seen, since we deliberately don't include
            # test_src/code_src at all in this arm.
            prompts[row["id"]] = {
                "first": prompt_text,
                "last": prompt_text,
                "extra": prompt_text,
            }
            print(f"  OK: {row['id']} (seed: {serialized['seed'].get('function_name')})")
        except Exception as e:
            failures.append((row["id"], f"{type(e).__name__}: {e}"))
            print(f"  FAILED: {row['id']}: {type(e).__name__}: {e}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\n{len(prompts)}/{len(rows)} prompts built -> {args.output}")
    if failures:
        print(f"{len(failures)} failures:")
        for instance_id, err in failures:
            print(f"  {instance_id}: {err}")


if __name__ == "__main__":
    main()
