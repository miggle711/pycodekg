# `extraction/context.py` deep dive

A complete walkthrough of `TestContextExtractor`: how a changed-entity set
(from `PatchParser`) becomes a real KG subgraph -- seed nodes plus
BFS-expanded surrounding context -- ready for validation and serialization.

## `TestContext` -- the output shape

A plain dataclass: `seeds` (nodes representing the changed
functions/classes), `context_nodes` (BFS-expanded neighbors),
`edges` (within the subgraph), `test_nodes` (existing tests found via
`tests` edges), plus `repo`/`base_commit` for reference. `save`/`load`
round-trip it to/from JSON for debugging; `summary()` gives a quick
human-readable digest (seed labels, edge-type counts).

## `TestContextExtractor.extract` -- the main path

Takes one dataset instance (`{repo, base_commit, patch, code_file,
test_file}`) and an already-loaded `KGQueryEngine`. Default `edge_filter`
is `{contains, calls, accesses, inherits, tests, uses}` -- `depends_on`
(imports) is excluded by default as noise, added back separately via
`include_seed_imports` if wanted.

### Step 1 -- detect the new-file special case first

Before anything else: `is_newly_created_file(patch, code_file)`. If the
patch itself creates this file, none of the rest of this method applies
at all -- there's no KG node for a file that doesn't exist at
`base_commit`, so control diverges entirely to `_extract_for_new_file`
(covered separately below). This check has to come first, since every
subsequent step in the normal path assumes the file genuinely exists at
`base_commit` and can be fetched.

### Step 2 -- resolve changed entities via PatchParser

Fetches the file's real pre-patch text (`RepoManager.read_file_at_commit`)
and calls `PatchParser.extract_changed_functions_with_scope` -- this is
where the `patch.py` module (see its own deep dive) actually runs, one
call, right here.

### Step 3 -- verify the file node exists in the KG

`find_file_by_path` -- mostly a sanity check. If the file genuinely isn't
in the built KG at all (skipped by `MAX_FILE_LINES`, a syntax error, etc.),
this raises immediately with a clear message rather than silently
continuing with an empty seed set.

### Step 4 -- turn each changed entity into real seed node IDs

This is the most heavily hedged part of the whole method -- every hedge
here maps to a real, previously-shipped bug:

1. `find_function_by_name(name)` -- bare-name lookup (a function/method
   name alone isn't unique across a whole repo).
2. Filter to only matches whose `filepath` equals `code_file` -- a
   same-named function in an unrelated file must never become a seed.
3. If STILL ambiguous (>1 match in this file) AND the patch told us which
   class it belongs to (`enclosing_class` from step 2), narrow to that
   class. This is #63's fix -- without it, EVERY same-named match becomes
   a seed, and `LLMSerializer._build_seed_section` trusts `seeds[0]`
   unconditionally, so the model could be shown an arbitrary one of them.
   If narrowing doesn't land on exactly one match, matches are left
   unfiltered -- `TestContextValidator._check_no_ambiguous_seed_names` is
   the deliberate backstop that refuses to silently proceed with an
   unresolved collision, rather than this method guessing.
4. If NO function/method match was found at all, try `find_class_by_name`
   instead (#85) -- a changed entity can be a CLASS itself (e.g. a
   class-body attribute assignment outside any method), and
   `find_function_by_name` can structurally never match this, since it
   only searches function/method/test_function node types.

### Step 5 -- fallback: no changed entity resolved at all

Use the whole file as the seed. A genuine last resort, not a preferred
path -- everything above is designed to avoid ever reaching this.

### Step 6 -- the test file is deliberately never a seed

This is a documented, real historical bug (#54), worth understanding in
full: `context.seeds`'s actual order comes from BFS VISITED-NODE order,
not from the order `seed_ids` was constructed in. When the test file used
to be unconditionally appended to `seed_ids`, this made it
non-deterministic which node landed at `seeds[0]` on any given run -- and
`LLMSerializer._build_seed_section` trusts `seeds[0]` blindly. When the
test file won that race, the LLM-augmented arm's entire "seed" was the
test file itself (empty signature/source_code) instead of the real target
function, on roughly a third of instances/runs. Fixed by never adding the
test file to `seed_ids` in the first place -- it's still fully reachable:
`tests` edges point FROM the test function TO the seed (see `builder.py`),
so BFS from the seed alone, with incoming-direction traversal already
enabled, reaches the test function without it ever needing to be a seed.

### Step 7 -- BFS

`_bfs(seed_ids, depth, edge_filter)` -- the actual graph-traversal step,
covered below.

### Step 8 -- optionally add seed-only import edges

`_add_seed_imports`, gated by `include_seed_imports` (default `True`).
Kept deliberately separate from the main BFS edge filter so import edges
don't bloat the whole subgraph (~10-15% size increase if included) while
still being available for import-usage validation.

### Step 9 -- extract existing tests, tightly scoped

Only `tests` edges whose TARGET is literally one of the seed IDs -- not
any `tests` edge found anywhere in the depth-N BFS subgraph. This
restriction matters a lot in practice: confirmed empirically that
`handle_401`'s subgraph contained 372 BFS-reachable `tests`-edge nodes,
but only a handful of them actually tested `handle_401` -- most were tests
of other, unrelated functions/methods nearby. Without this restriction, a
test of some unrelated function 1-2 hops away gets misattributed as an
existing test FOR the seed (the same class of scoping bug as
`kg-test-generation#49`'s `related`-list issue). This restriction became
necessary specifically because calls-based `tests`-edge derivation
(`builder.py`'s `_derive_tests_edges_from_calls`, added for #57's audit)
made `tests` edges far more common than the old naming-convention-only
heuristic ever produced.

### Step 10 -- split seeds from context, return

Straightforward: partition `subgraph_nodes` by whether each ID is in
`seed_node_ids`.

## `_extract_for_new_file` -- the wholly-new-file path (#93)

Triggered by step 1's `is_newly_created_file` check. No KG lookup or BFS
is possible here at all: the file doesn't exist at `base_commit`, so it
has no node in the base_commit-built KG, AND nothing at `base_commit`
could reference it either -- there is genuinely no context to traverse,
not merely a lookup that failed to find something.

1. `reconstruct_post_patch_source("", patch, code_file)` -- rebuilds the
   file's full post-patch text starting from an empty pre-patch source
   (this works correctly because `reconstruct_post_patch_source` treats
   an empty starting source as "every line in the diff is an addition",
   with no special-casing needed).
2. `ast.parse`s the reconstructed source directly and walks its top-level
   statements, building seed node dicts by hand -- mirroring EXACTLY what
   `_parse_file` in `builder.py` does for a real file (same
   `_build_func_metadata` call, same ID-construction scheme via
   `_make_id`), so these synthetic nodes are structurally identical to
   real ones and downstream code (serialization, validation) treats them
   identically.
3. Every synthetic node's metadata is tagged `newly_created_file: True`.
   This marker is what `TestContextValidator` checks (in
   `_check_seed_connectivity`, `_check_no_orphaned_nodes`, and
   `_check_no_ambiguous_seed_names`) to know that zero edges -- and
   potentially multiple same-named seeds across different classes, e.g.
   several classes each defining their own `__init__` -- is the CORRECT,
   expected shape here, not a sign of a bug. (Discovered the hard way: a
   real pytest instance, `src/_pytest/mark/expression.py`, has several
   classes each with their own `__init__`/`__str__` -- every one of them
   genuinely IS a changed entity, since the whole file is new, so this
   isn't the same "unresolved collision" #63 was about at all.)
4. Returns a `TestContext` with real seeds but empty
   `context_nodes`/`edges`/`test_nodes` -- there is genuinely nothing
   else to report.

## `_add_seed_imports`

For each seed, looks at its own OUTGOING `depends_on`/`module_depends_on`
edges only (not any import reachable via BFS), adds the target node and
edge if not already present. Deliberately narrow -- this exists so
downstream validation can check "does this seed actually use what it
imports" without bloating the subgraph with the whole module's import
graph.

## `_bfs`

A thin wrapper around `GraphTraversal.bfs` (a separate, generic utility).
Filters `seed_ids` down to ones that actually exist in the KG first
(defensive -- in case a seed ID somehow doesn't resolve), then delegates
the real traversal: BOTH `outgoing` and `incoming` edges are walked, up to
`depth` hops, restricted to `edge_filter`'s relations. Walking both
directions is what makes callers AND callees both count as real context,
not just one direction -- this is also exactly what makes the #91 fix (a
seed with only incoming edges still has real context) meaningful: the BFS
was already finding that context correctly; only the VALIDATOR's
connectivity check was looking in the wrong direction.
