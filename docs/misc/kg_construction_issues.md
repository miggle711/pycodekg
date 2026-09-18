# KG Construction: Issues Found in a Full Pass

A line-by-line review of the entire KG construction and extraction pipeline (`kg/builder.py`, `ast/helpers.py`, `kg/traversal.py`, `kg/query.py`, `kg/repo_manager.py`, `extraction/patch.py`), looking specifically for correctness bugs, silent data loss, and gaps in the graph the LLM-facing pipeline depends on. Findings are grouped by severity.

**Test coverage note:** `tests/` currently has no test file for `kg/builder.py` or `ast/helpers.py` — the two files with the most findings below. `test_traversal.py`, `test_patch_parser.py`, `test_validators.py`, and `test_data_flows.py` exist and pass (40 tests), but the call-resolution logic, edge emission, and most AST helpers are untested. Several of the bugs below (in particular #1 and #2) would be caught immediately by a unit test with two same-named files or a multi-level inheritance fixture.

---

## Correctness bugs (produce a wrong answer, not just a missing one)

### 1. `module_depends_on` edges can attach to the wrong file when filenames collide

`kg/builder.py`, `_resolve_edges`, lines 643–670.

```python
file_label_to_id: Dict[str, str] = {
    n['label']: n['id'] for n in all_nodes
    if n['type'] in ('file', 'test_file')
}
```

`file_label_to_id` is keyed by **bare filename** (`n['label']`, e.g. `"utils.py"`), not the full relative path. Any repo with two files sharing a name in different packages (`pkg_a/utils.py` and `pkg_b/utils.py` — a very common layout) will silently collide: the dict keeps whichever file was inserted last, and every `module_depends_on` edge meant for the other `utils.py` gets attached to the wrong node. This is not a missing edge — it's a wrong edge, with no error or warning surfaced anywhere.

**Fix:** key `file_label_to_id` by `metadata['path']` (already present on every file node) instead of `label`, and match candidates by resolving the import's dotted path against real file paths rather than just the last path segment.

### 2. `overrides` resolution only checks one level of inheritance

`kg/builder.py`, `_resolve_edges`, lines 615–631.

```python
if class_label_to_ids.get(base_simple):
    for target_id in label_to_ids.get(method_name, []):
        node = nodes_by_id.get(target_id, {})
        if node.get('metadata', {}).get('class') == base_simple:
```

This checks whether the *candidate target method's own class* equals the *immediate* base class name. For `class C(B)`, `class B(A)`, if `A` defines `foo` and neither `B` nor `C` redefine it except `C.foo`, the override relationship to `A.foo` is invisible — the check only ever matches a direct parent, not a grandparent. Diamond and multi-level inheritance chains are silently under-linked; this degrades quietly rather than erroring, so it will not show up in the existing structural validators (no orphan, no cycle, no broken edge — just a missing `overrides` edge).

**Fix:** requires walking the resolved `inherits` chain transitively (once base classes are resolved) rather than comparing name strings at emission time. This is a second-pass problem: `overrides` resolution currently happens in the same pass as `inherits` resolution, before the full inheritance chain is known.

### 3. `_get_instantiated_classes` misfires in both directions on the uppercase heuristic

`ast/helpers.py`, lines 716–739, feeding `uses` edges (builder.py:330-332) and `external_deps` (builder.py:849-854).

```python
if isinstance(child.func, ast.Name) and child.func.id[0].isupper():
```

- **False positive:** any uppercase-named callable that isn't a class — an all-caps constant-style factory function, a class method used as a bound callable assigned to an uppercase alias, etc. — gets recorded as an instantiated class.
- **False negative:** the extremely common `requests`-style pattern of lowercase factory functions returning instances (`session = requests.session()`, `conn = pool.connection()`) is invisible to this heuristic entirely, so `uses` edges under-report real object dependencies.

This is a fundamental limitation of name-based heuristics without type inference, so it may not be fully fixable, but the false-positive half is cheap to reduce: cross-check the candidate name against `class_label_to_ids` during pass 2 and drop `uses` edges that never resolve to an actual class node in the repo (currently unresolved `uses` edges to non-existent classes are simply dropped in `_resolve_edges` — line 601-613 — so this is *partially* self-correcting for names from other files, but still produces false signal for same-named coincidences within the repo).

### 4. Property/classmethod/staticmethod access is invisible to call-edge extraction

`ast/helpers.py`, `_extract_callee_name`, lines 83–99, and its only caller, `_emit_call_edges` in `builder.py`, which only walks `ast.Call` nodes.

A `@property`-decorated method is invoked as `obj.attr`, never `obj.attr()` — it never appears as an `ast.Call` node at all, so no `calls` edge is ever emitted for property access, no matter how central the property is to program behavior. This is a structural blind spot: the KG has zero signal for an entire category of attribute access that is extremely common in real Python (e.g. `requests.Session.headers`, any `@property`-based API). It's not wrong, but it's a complete gap the docstrings don't currently disclose.

**Fix (partial):** detect `ast.Attribute` reads (already half-collected via `_get_attribute_accesses`, though that's `self.*`-scoped only) against known `@property`-decorated methods in the same or related classes, and emit a distinct `accesses` edge type. Full fix requires knowing which attribute names are properties across the whole repo, which needs the same two-pass approach already used for `calls`.

---

## Silent data loss / wasted work

### 5. `reads`, `writes`, and `returns` edges are fully computed, then unconditionally discarded

`kg/builder.py`, `_emit_func_edges` (lines 255–303) builds real edges:

```python
for attr in reads:
    edges.append(asdict(KGEdge(source=func_id, target=attr, relation='reads', ...)))
for attr in writes:
    edges.append(asdict(KGEdge(source=func_id, target=attr, relation='writes', ...)))
for ret_type in _get_return_types(func_node):
    edges.append(asdict(KGEdge(source=func_id, target=ret_type, relation='returns', ...)))
```

And `_resolve_edges` (line 633-634) throws every one of them away:

```python
elif meta.get('unresolved'):
    continue
```

Because attribute/type names aren't graph node IDs, these can never resolve. This means every worker process, for every function, in every file, computes `_get_attribute_accesses` and `_get_return_types` a second time (they're also called directly in `_build_func_metadata` for the `side_effects` and `data_flows` metadata fields — see builder.py:857, 860) purely to build edges that are deterministically deleted 100% of the time. It's dead CPU work at the scale of the whole repo, and it clutters the unresolved-edge list that pass 2 has to iterate over.

**Fix:** delete the `reads`/`writes`/`returns` edge emission in `_emit_func_edges` entirely. The same information already lives on the node as metadata (`side_effects`, `data_flows.returns`) via `_build_func_metadata`, which is the form validators and the LLM serializer actually consume. This is a pure win — less work, no behavior change, one less category of edge for `_resolve_edges` to skip over.

### 6. `_collect_local_types` ignores type-annotated parameters, the cheapest possible signal

`ast/helpers.py`, lines 117–150. Only tracks `x = SomeClass(...)` assignments within the function body. It does not look at `_get_signature`'s already-extracted parameter annotations (`ast/helpers.py:185-240`), even though that data is computed in the same pass and is a direct, unambiguous type hint for the parameter's local name.

```python
def process(self, request: Request) -> None:
    request.prepare()   # 'request' could resolve via param annotation, but doesn't
```

Concretely: any call through a type-annotated parameter falls through to bare-name resolution in `_resolve_call` (`label_to_ids.get(callee_name, [])`) and gets marked `ambiguous` whenever another class in the repo happens to share a method name — even though the annotation fully disambiguates it.

**Fix:** seed `_collect_local_types`'s returned dict with `{arg.arg: bare_type_name}` for every annotated parameter before walking assignments, using the same `_safe_unparse`-then-strip-generics logic already used in `_get_annotation_type_names`. Cheap, no new AST walk needed, directly reduces spurious `ambiguous` confidence.

---

## Known limitations already surfaced by comments (not new, but worth restating explicitly)

- `_extract_callee_name` (ast/helpers.py:83-99) has an explicit `TODO` acknowledging it doesn't handle dynamic calls, lambdas, or calls-via-variable-indirection. Subscript calls (`handlers['key']()`) and chained calls (`foo()()`) return `None` and are silently dropped — no edge, no warning.
- `_check_cycles` in `kg/validator.py` explicitly excludes `depends_on`/`imports` from cycle detection because those are "resolved at runtime" — an implicit acknowledgment that the validator can't reason about import-time semantics, only static structure.
- Call edges already carry `exact`/`ambiguous`/`unresolved` confidence tags specifically because name-based resolution is known to be lossy — this is the right instinct, but nothing downstream (validators, the LLM serializer) currently *filters or surfaces* that confidence to the consumer. `KGQueryEngine.find_callers`/`find_callees` explicitly say "confidence... is not filtered" (query.py:142-143) — a caller has no way to ask for exact-only edges without re-implementing the filter themselves.

---

## Future work: real type inference (research direction, not a fix)

Issue #3's false-negative half (lowercase factory functions like `session = requests.session()` being invisible to `_get_instantiated_classes`) can't be closed by heuristic patching — it needs actual type inference, which is a different scale of work from everything else in this document. Worth scoping explicitly so it isn't confused with a quick follow-up.

What it would take:

- **A real type environment per scope**, replacing `_collect_local_types`'s single flat dict: types need to merge at branch joins (`x = A() if cond else B()` → `x: A | B`), narrow correctly across reassignment in loops (a fixpoint, not one linear pass), and handle closures over outer scope instead of skipping nested functions entirely.
- **Cross-function return-type inference.** To know `factory()` returns a `Session`, you need to inspect `factory`'s body — which may itself depend on resolving other calls' return types. This is recursive by nature: a call graph with return-type propagation and a fixpoint (or depth cap) for cycles/recursion, not a single AST walk.
- **Cross-module propagation of inferred types**, building on the pass-2 cross-file `calls` resolution that already exists, but propagating a *return type* along an edge is new plumbing, not an extension of it.
- **An explicit "unknown/external" outcome.** Most real instantiation happens through stdlib or third-party types with no AST in the repo (`bool(x)`, `some_orm.Session()`). A real implementation needs a stub/typeshed-style lookup or an honest "type unknown" result — not silent guessing, not silent dropping.

Realistic scope: this is a new subsystem, likely multi-week, and arguably duplicates what existing tools (`pyright`, `pyre`) already do well. The more promising angle may be **shelling out to an existing type checker's inferred types** rather than reimplementing inference from scratch — worth a research spike before committing to either approach. Tracked as a standalone research issue, not a bugfix.

---

## Smaller / lower-severity observations

- **`RepoManager.ensure_clone`** (`kg/repo_manager.py:39-54`) never refreshes an existing bare clone. If a repo was cloned once and the KG is later rebuilt at a commit that's landed on the remote *after* the clone was cached, `git archive` will fail with an opaque `CalledProcessError` rather than a clear "commit not found, try fetching" message. There's no `git fetch` path at all — the cache is effectively immutable once created.
- **`PatchParser._extract_defs_from_hunk`** (`extraction/patch.py:80-106`) uses a regex match against `def`/`class` lines within changed hunks, but does not account for decorated definitions where the `@decorator` line changed but the `def` line itself is unchanged context outside the hunk window — a patch that only adds/modifies a decorator on an otherwise-untouched function would not identify that function as changed. This is a real gap for any repo that uses decorator-heavy patterns (Flask routes, pytest fixtures, dataclasses).
- **`MAX_FILE_LINES = 5000`** (`kg/builder.py:86`) skips any file over 5000 lines with no metadata recorded about the skip — the file simply doesn't appear in the KG at all, and nothing downstream (validators included) can distinguish "this file doesn't exist" from "this file was skipped as too large." A single line in `kg['metadata']` recording skipped files would make this diagnosable.
- **`KGQueryEngine.visualize`** (`kg/query.py:322-340`) does its own separate BFS implementation rather than reusing `GraphTraversal.bfs` from `kg/traversal.py` — two BFS implementations with slightly different signatures (`visualize` takes no `edge_filter`, walks outgoing-only) that will drift out of sync over time. Low priority, but worth consolidating into one traversal primitive since `traversal.py`'s docstring already claims to be the shared utility "used by subgraph extraction (Phase 3), test execution (Phase 6), and metrics (Phase 7)" — `visualize` should be on that list too.

---

## Suggested priority order

1. **#1 (module_depends_on filename collision)** — genuine correctness bug, cheap fix, no design questions.
2. **#5 (dead reads/writes/returns edges)** — pure win, removes wasted computation with zero behavior change.
3. **#6 (parameter-annotation type hints)** — cheap, directly reduces false `ambiguous` confidence on call edges.
4. **#3's false-positive half (uses edge cross-check against class_label_to_ids)** — cheap, same pattern already used elsewhere in `_resolve_edges`.
5. **#2 (multi-level overrides)** and **#4 (property access)** — both require a design decision (transitive inheritance walk; new edge type for attribute access) rather than a local fix, so worth scoping as separate follow-ups rather than quick patches.
6. Everything under "smaller observations" — worth a pass once the above are settled, none are urgent on their own.
