# `llm/llm_serializer.py` deep dive

A complete walkthrough of `LLMSerializer`: how a flat `TestContext`
subgraph (nodes + edges) becomes the actual hierarchical JSON handed to an
LLM for test generation. This is the last stage of the pipeline -- what
this module produces is the real, final artifact everything before it
exists to build.

## Why hierarchical, not flat

The module docstring states the design intent directly: minimize token
usage, let the model prioritize relevant information, and mirror how a
developer actually reasons about a code change -- "here's what changed,
here's what surrounds it, here's what I need you to do." The output has
exactly three top-level sections: `seed`, `context`, `instructions`.

## `_filepath_to_module`

A small but consequential helper: converts a repo-relative path
(`"requests/sessions.py"`) to a dotted module path (`"requests.sessions"`)
-- the same derivation `kg/builder.py` uses internally to resolve
qualified names (see its `qualified_to_ids` index), but that result was
never carried through to the LLM-facing output before this existed. The
comment documents a real, concrete bug this fixed (issue #6): without a
real module path, the model had to GUESS an import path, and it
fabricated placeholder imports like `from your_module import X` --
obviously never going to actually work when the generated test runs.

## `LLMSerializer.serialize` -- the orchestrator

Takes the flat `TestContext`-shaped dict (as produced by
`TestContext.save()`/`context.__dict__`-equivalent), builds a
`node_by_id` lookup across ALL nodes (seeds + context_nodes + test_nodes
together), then calls three section-builders in sequence and assembles
their results into the final `{seed, context, instructions}` dict.

## `_build_seed_section`

Builds the "here's what changed" section. Notably: `# For now, assume
single seed (most common case)` -- only `seeds[0]` is ever used here, a
real, acknowledged simplification (not every instance has exactly one
seed; a genuine multi-function patch, or a wholly-new-file instance with
several classes, can produce more than one). This is precisely why
`seeds[0]`'s ORDERING mattered so much elsewhere in the pipeline --
`context.py`'s design (never adding the test file to `seed_ids`, #54) and
`extraction/validator.py`'s ambiguous-seed-name check (#63/#93) both exist
specifically because this serializer trusts `seeds[0]` unconditionally,
with no fallback or disambiguation logic of its own.

Fields returned: `function_name`, `type`, `module` (via
`_filepath_to_module`), `filepath`, `class_name` (empty string if the seed
is a module-level function, not a method -- documented as needed because
a method isn't importable by name on its own; `from module import
some_method` doesn't exist as a concept, the model needs the CLASS to
import and instantiate instead), `signature`, `docstring`, `exceptions`,
`source_code`, `decorators`, `type_hints`.

## `_build_context_section` -- the most involved part of this module

Builds five sub-lists from the subgraph's edges, each with its own real
scoping bug history behind why it's restricted the way it is.

### `seed_class_ids`

Computed first, since several of the lists below need it: the class (if
any) that DIRECTLY contains a seed, found via a `contains` edge from a
class node to the seed itself. This exists to distinguish "a sibling
method of the seed's OWN class" from any other class->method `contains`
edge that might appear elsewhere in the BFS subgraph (e.g. via some
unrelated class reached through inheritance or instantiation a few hops
away).

### `callers` / `callees`

Straightforward: any `calls` edge whose target is a seed -> caller; any
`calls` edge whose source is a seed -> callee. Each rendered via
`_node_to_snippet` (name, type, module, class_name, signature, docstring,
source_code -- a smaller shape than the full seed section, since these
are supporting context, not the thing being tested).

### `related` (inheritance/instantiation)

This one has real, documented history. `inherits`/`uses`/`instantiates`
edges are only included when the edge's SOURCE is the seed itself, or the
seed's own class (`src_id in seed_ids or src_id in seed_class_ids`) --
NOT any such edge anywhere in the subgraph. Without this restriction, a
caller or callee that itself happens to instantiate some class gets that
fact wrongly attributed to "related" as if it were about the seed. The
comment documents the exact real case that surfaced this
(`kg-test-generation#49`'s investigation): `Session` and `Request` both
instantiate `PreparedRequest` somewhere in their own bodies, which
produced THREE near-identical "instantiation: PreparedRequest" entries
for a seed (`prepare_body`) that doesn't itself instantiate
`PreparedRequest` at all. Confirmed at the time that the underlying KG
data/edges were entirely correct -- this was a pure serialization-scoping
bug, not a bad graph or real duplication.

The seed's own CLASS is deliberately included as a valid source too (not
just the seed function/method itself), since a method typically inherits
or instantiates via its class, not usually in its own body -- e.g.
`PreparedRequest.__init__` instantiating `CaseInsensitiveDict` is relevant
context for every method on `PreparedRequest`, not just `__init__`
specifically. Each entry's `source` field (`"seed"` vs. `"seed_class"`)
is what lets the model tell "the seed itself does this" apart from "the
seed's class does this somewhere else" -- these are genuinely different
kinds of context, and collapsing them would lose real information.

### `sibling_methods`

Other methods on the seed's OWN class -- e.g. `__init__`, or a setup
method like `prepare()` whose side effects the seed's body depends on but
doesn't itself set up. This is real context a flat, single-function
extraction (the baseline, no-KG arm) can never provide at all, since a
flat extraction has no notion of "what else does this class define" --
documented as a real gap found in `kg-test-generation#50`. Excludes the
seed's own containment edge (`tgt_id not in seed_ids`) so a seed never
lists itself as its own sibling.

### `existing_tests`

Just `name` + `source_code` for every node in `test_nodes` -- the
filtering/scoping work for which tests actually belong here already
happened upstream, in `context.py`'s step 9 (restricting `tests` edges to
ones whose target is literally a seed, not any test reachable within the
BFS depth).

### `patterns`

Delegated to `_extract_patterns` (below).

## `_build_instructions_section`

The one section with NO dependency on the actual subgraph content at
all -- a fixed, hand-authored set of `coverage_targets` (boundary
conditions, happy path, error cases, edge cases) and `conventions`
(naming pattern, assertion style, mocking guidance, test isolation),
plus a fixed `task` string. This is deliberately generic guidance, not
derived from the seed's own actual behavior -- the seed-specific
information (what to actually test) lives entirely in the `seed` and
`context` sections above; this section only tells the model HOW to write
tests, not WHAT this particular function does.

## `_node_to_snippet`

The shared, smaller node-rendering shape used by `callers`/`callees`/
`sibling_methods`: `name`, `type`, `module`, `class_name`, `signature`,
`docstring`, `source_code`. Notably smaller than the full seed section
(no `exceptions`, `decorators`, `type_hints`) -- these are supporting
context nodes, not the thing under test, so less detail is needed.

## `_extract_patterns`

Scans every seed AND context node's metadata for three optional keys:
`branch_count` (rendered as a `"Branches: N"` string), `type_hints`
(merged into one dict across all nodes), and `exceptions` (extended into
a flat list). None of these three keys are ever actually present in real
node metadata -- `_build_func_metadata` (`ast/helpers.py`) populates
`branches` (not `branch_count`) and `raises`/`catches` (not `exceptions`),
and never populates `type_hints` as a per-node key at all. So
`patterns.control_flow`, `patterns.type_hints`, and
`patterns.error_handling` are effectively always empty in every real
serialized output.

Deliberately left unfixed here, distinct from `_build_seed_section`'s own
`exceptions` field (fixed -- see below): `control_flow`'s bare branch
count is weak signal already visible in the seed's own `source_code`;
`error_handling` becomes redundant with the seed section's `exceptions`
field once that's populated; and `type_hints` has no underlying data
anywhere in the pipeline to wire up at all (not a lookup-key mismatch --
the source data genuinely doesn't exist yet). Fixing these would need new
extraction logic, not just a key-name correction.

**`_build_seed_section`'s own `exceptions` field WAS fixed** (this key had
the identical mismatch: reading `metadata.get("exceptions", [])`, which
never matched anything real). Now reads `metadata.get("raises", [])` --
the actual key `_build_func_metadata` populates, and the one genuinely
useful signal here (what a test generator needs to know to write
`pytest.raises(...)` assertions). `catches` (what a function handles
internally) is a separate, less directly test-relevant signal and
deliberately not folded in here.
