# Pipeline walkthrough

How a `(repo, base_commit, patch, code_file, test_file)` instance becomes
LLM-ready JSON context, traced through a single running example.

This doc is meant to be read alongside the code, not instead of it -- each
section names the exact file/function so you can jump straight to source.
Docstrings in the code explain *why* a given check or guard exists (often
citing the specific bug that motivated it); this doc explains how the
pieces fit together as one pipeline.

## The example

A hypothetical patch to `widgets/gadget.py` in some repo:

```python
# widgets/gadget.py, as it exists at base_commit (pre-patch)
class Gadget:
    def build(self):
        return self.helper()

    def helper(self):
        return 42
```

```diff
--- a/widgets/gadget.py
+++ b/widgets/gadget.py
@@ -2,3 +2,3 @@
     def build(self):
-        return self.helper()
+        return self.helper() + 1
```

The instance dict every stage of the pipeline consumes:

```python
instance = {
    "repo": "acme/widgets",
    "base_commit": "abc1234...",
    "patch": "<the diff above>",
    "code_file": "widgets/gadget.py",
    "test_file": "tests/test_gadget.py",
}
```

## Stage 1 -- Build the KG

**Code:** `kg/repo_manager.py` (`RepoManager`), `kg/builder.py`
(`RepoKGBuilder`, `RepoASTParser`, `_parse_file`)

This stage doesn't know or care about `instance` at all -- it builds a
structural graph of the **entire repo** at `base_commit`, once, and caches
it. Every instance that happens to share a `(repo, base_commit)` reuses the
same build.

1. `RepoManager.ensure_clone("acme/widgets")` -- clones the repo as a bare
   mirror (no working tree) into `repo_cache/`, once ever, reused forever.
2. `RepoManager.extract_at_commit(repo, "abc1234...", dest)` -- runs
   `git archive` against the bare clone (read-only, no checkout, safe for
   concurrent commits of the same repo) and untars the result into a temp
   directory. Now real `.py` files exist on disk.
3. `RepoASTParser.parse_repo` walks every `.py` file:
   - **Pass 1 (parallel):** `_parse_file` runs once per file, in its own
     worker process. Each file independently produces nodes (`file`,
     `class`, `method`, `function`, `import`) and *unresolved* edges
     (`calls`, `inherits`, ...) -- a bare name string like `"helper"`, plus
     resolution hints (`class_hint`, `local_type_hint`, `import_resolved`).
     A single file has no visibility into any other file, so nothing gets
     resolved here.
   - **Pass 2 (sequential):** `_aggregate_and_index` merges every file's
     result and builds repo-wide name -> ID lookup tables. `_resolve_edges`
     walks every unresolved edge and matches it against those tables using
     the hints from pass 1, tagging each resolution `qualified` / `exact` /
     `ambiguous`, or dropping it if nothing trustworthy matches (this repo
     deliberately prefers a missing edge over a wrong one -- see #61, #65).
   - `_add_call_context` annotates every function/method node with
     `caller_count` / `direct_callers`, derived from the now-resolved
     `calls` edges.
4. `RepoKGBuilder.save` writes the result to
   `kg_output/kg_acme_widgets_abc1234.json`.

**What exists after this stage**, for our example: a `class` node for
`Gadget`, `method` nodes for `build` and `helper`, a `contains` edge
`Gadget -> build`, `Gadget -> helper`, and a resolved `calls` edge
`build -> helper` (confidence `qualified`, since `self.helper()` gave a
strong `class_hint`).

## Stage 2 -- Parse the patch

**Code:** `extraction/patch.py` (`PatchParser`)

Given just the diff text and `code_file`, figure out *which* function(s) or
class(es) actually changed -- by parsing the **real pre-patch AST** and
mapping the diff's changed line numbers against real `lineno`/`end_lineno`
ranges, not by scanning the diff text itself (an earlier, text-scanning
approach caused four separate bugs: #14, #43, #63, #71).

1. `RepoManager.read_file_at_commit` fetches `widgets/gadget.py`'s real
   text at `base_commit` (a single-file `git show`, cheaper than a full
   tree extract for just one file).
2. `PatchParser.extract_changed_functions_with_scope(patch, code_file, pre_patch_source)`:
   - `_function_ranges(pre_patch_source)` -> real line ranges: `Gadget`
     spans lines 1-6, `build` spans 2-3, `helper` spans 5-6.
   - `_changed_pre_patch_lines` walks the diff: the `-` line is old line 3
     (a real removal, trustworthy). The `+` line has no pre-patch line
     number, so its insertion *point* (line 3, and line 2 as a weaker
     neighbor) is recorded as an anchor.
   - Line 3 resolves via `_innermost_range` to `build` (the smallest range
     containing it) -- not `Gadget`, even though `Gadget`'s range also
     technically contains line 3.
   - The post-patch source is independently reconstructed and re-checked
     too (this is what catches a wholly new function/class that has no
     home in the pre-patch source at all -- #84), and a redundancy filter
     drops a class-level match when a real method match already explains
     the same hunk (#85's follow-up finding).
3. Result: `{("build", "Gadget")}` -- one changed method, with its
   enclosing class name attached (needed because a bare name like `build`
   could collide with a same-named method on a different class -- #63).

## Stage 3 -- Extract a subgraph

**Code:** `extraction/context.py` (`TestContextExtractor`)

Turns the changed-entity set from stage 2 into real KG node IDs (seeds),
then does BFS outward over the graph built in stage 1.

1. `find_function_by_name("build")` -> matches, filtered to
   `widgets/gadget.py`, narrowed by the `"Gadget"` class hint if
   ambiguous. If nothing matches as a function/method, `find_class_by_name`
   is tried instead (#85 -- a changed *class-body* attribute has no
   function/method node to find at all).
2. `seed_ids = [<build's node id>]`.
3. `_bfs(seed_ids, depth=2, edge_filter={contains, calls, accesses,
   inherits, tests, uses})` walks outward, both directions. From `build`:
   one hop out via `calls` reaches `helper` (a callee); one hop in via
   `calls` would reach anything that calls `build` (none here); via
   `contains` reaches the enclosing `Gadget` class.
4. `test_nodes` are populated only from `tests` edges whose *target* is
   literally the seed (`build`) -- not any test reachable somewhere in the
   depth-2 subgraph, which would misattribute unrelated tests (confirmed
   on a real instance where 372 BFS-reachable test nodes were found, but
   only a handful actually tested the seed).
5. Returns a `TestContext`: `seeds=[build]`, `context_nodes=[helper,
   Gadget]`, `edges=[build->helper (calls), Gadget->build (contains),
   Gadget->helper (contains)]`, `test_nodes=[...]`.

**Special case -- a wholly new file:** if `code_file` doesn't exist at
`base_commit` at all (patch creates it), there's no KG node for it and no
possible BFS context (nothing at `base_commit` could reference a file that
doesn't exist yet). `_extract_for_new_file` builds seed node(s) directly
from the reconstructed post-patch source via AST -- same shape as a real
node, tagged `newly_created_file: True` -- with empty context/edges. This
is the *correct* answer here, not a degraded one (#93).

## Stage 4 -- Validate

**Code:** `extraction/validator.py` (`TestContextValidator`)

Runs a battery of structural/semantic checks on the `TestContext` before
it's ever shown to a model:

- no orphaned nodes (except `newly_created_file` seeds, which are
  genuinely, correctly edge-less -- #93)
- no broken edges (endpoints missing from the node set)
- seeds have real connectivity -- checked in **both** directions (a seed
  with only incoming edges, i.e. real callers, still has usable context;
  checking outgoing-only wrongly rejected leaf-like functions -- #91)
- no duplicate edges, no ambiguous same-named seeds left unresolved
- (soft) some test coverage exists, context isn't suspiciously thin

`pipeline.extract_and_validate(instance, strict=True)` raises if any
must-have check fails -- this exists because a real validator check
(`_check_seed_types`) once caught a genuine bug but nothing actually read
its result (`verbose=False` silently discarded the report), so a broken
context reached the LLM undetected for a while (#54).

## Stage 5 -- Serialize for the LLM

**Code:** `llm/llm_serializer.py` (`LLMSerializer`)

Turns the validated `TestContext` into hierarchical JSON:
```json
{
  "seed": { "name": "build", "class": "Gadget", "signature": "...",
            "source_code": "def build(self):\n    return self.helper()",
            "callers": [...], "callees": [...] },
  "context": { "related_functions": [...], "related_classes": [...] },
  "instructions": { "...": "..." }
}
```
This is the final artifact this repo produces. A sibling repo,
`kg-test-generation`, is what actually calls `serialize_context()` and
sends this to an LLM (via Groq) to generate tests, comparing this
KG-augmented context against a plain-text baseline (just the seed's raw
source, no graph traversal at all).

## End to end, one call

```python
from kg_construction.pipeline import extract_and_validate, serialize_context

context, report = extract_and_validate(instance, depth=2, strict=True)
llm_input = serialize_context(context)
```

`extract_and_validate` runs stages 1 (load-or-build), 3, and 4 in order
(stage 2 happens inside stage 3). `serialize_context` is stage 5, called
separately since the caller may want to inspect/save the raw
`TestContext` first.
