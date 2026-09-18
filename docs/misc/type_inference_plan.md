# Plan: pyright-backed type inference for factory-function `uses` edges

Follow-up to #19 (research spike, see issue comment for the feasibility findings). This
plan scopes the first implementation slice — not the full multi-week subsystem the issue
describes, just the concrete mechanism validated by the spike.

## Problem recap

`_get_instantiated_classes` / `_get_instantiated_classes_in_class` (`ast/helpers.py`) only
detect `uses` edges from calls whose callee name starts with an uppercase letter
(`SomeClass()`). Factory functions returning an instance under a lowercase name
(`self.session = requests.session()`) are invisible to this heuristic — no edge is
emitted at all, not even an unresolved one.

The spike (issue #19 comment) confirmed pyright correctly infers these return types,
handles branch-join merging, and resolves external/stdlib types, at a workable per-repo
cost (~1.25s for an 18-file package).

## Scope of this slice

**In scope:**
- Detect lowercase-callee assignment sites (`x = some_call(...)`) that today produce no
  `uses` edge candidate at all.
- Run pyright once per repo build, resolve the RHS call's inferred return type at each
  detected site.
- Emit a `uses` edge when the inferred type resolves to a class already in the KG's
  `class_label_to_ids` index (same resolution path the existing string-heuristic edges
  use — no new resolution mechanism).
- Graceful degradation: if pyright is not installed, times out, or fails for any reason,
  skip this enrichment step entirely and fall back to today's heuristic-only behavior.
  A KG build must never fail because of this step.
- Tests using a real local pyright invocation (no network) against synthetic fixtures.

**Explicitly out of scope for this slice** (tracked as follow-ups, not blockers):
- Cross-module return-type propagation beyond what pyright already resolves in one pass
  (pyright itself handles this internally; we are not building our own propagation).
- Recursive/fixpoint handling for mutually recursive factories — rely on whatever pyright
  reports for these; no special-casing.
- Any change to the existing `calls` edge resolution — this only adds `uses` edges.
- Making type inference the default for every KG build without an opt-in (see Decision 2).

## Design decisions

### 1. Where pyright runs in the pipeline

Add it as a distinct step between Pass 1 and Pass 2 in `RepoASTParser.parse_repo`:

```
Pass 1 (parallel, per-file)          — unchanged
  -> also records "candidate factory-call sites":
     (file rel_path, lineno, col_offset, enclosing func_id) for any
     assignment `x = call(...)` where the call's resolved callee name
     (from _extract_callee_name) does NOT start with an uppercase letter.

Pass 1.5 (new, sequential, once per repo)
  -> if any candidate sites were recorded:
       run `pyright --outputjson <repo_dir>` once
       parse "reveal-type"-equivalent info... (see Decision 3 on mechanism)
       for each candidate site, look up the inferred type name by
       (file, line, col)
       if the type name resolves to a class defined in this repo
       (class_label_to_ids will have it after Pass 2's aggregation —
       so this step only *records* inferred type names against call
       sites; matching against class_label_to_ids still happens in
       existing Pass 2 resolution, reusing that path)

Pass 2 (sequential)                   — unchanged mechanism,
  -> _resolve_edges gains one more edge shape to resolve:
     unresolved 'uses' edges whose target is a pyright-inferred type
     name, resolved via the *same* class_label_to_ids lookup already
     used for the uppercase-heuristic 'uses' edges.
```

This keeps the existing two-pass architecture's shape intact (matches the module
docstring's existing framing of "Pass 1 (parallel) / Pass 2 (sequential)" and the pattern
established by `_add_call_context` running after resolution) rather than introducing a
parallel pipeline.

### 2. Opt-in vs. default-on

**Decision: opt-in, default off.** Add `builder.build(repo, commit, infer_types=False)`.

Reasoning: this adds a new external dependency (`pyright`), a new subprocess invocation,
and a new failure mode to every KG build if made default. Existing callers (`pipeline.py`,
`kg-run`) should not silently start requiring `pyright` to be installed, or silently start
paying an extra ~1-2s per repo, without asking for it. Once this slice has run against a
handful of real repos and the enrichment is trusted, flipping the default is a one-line
change — cheap to revisit, expensive to walk back if it breaks someone's existing
pyright-less environment on day one.

### 3. How pyright's output gets correlated to call sites

pyright's `--outputjson` mode over a whole project reports diagnostics, not a queryable
"type at this location" API by default. Two ways to get per-site inferred types:

- **`reveal_type()` injection (what the spike used):** for each candidate site, this
  would require rewriting the source file to insert `reveal_type(x)` right after the
  assignment, which is invasive (temp-file rewriting per candidate) and doesn't scale to
  "many sites in the same file" cleanly.
- **pyright's hover/type-query LSP mode**, or the `--stats`/JSON per-symbol output pyright
  supports for editors: use pyright's language-server protocol (`pyright-langserver`) and
  issue `textDocument/hover` requests at each candidate's (line, col) directly — no source
  rewriting needed, one long-lived pyright process can answer many queries.

**Decision: use the LSP hover approach**, not `reveal_type` injection. It avoids mutating
extracted source files (which are meant to be read-only per `RepoManager`'s design) and
scales to many call sites per file without N temp-file rewrites. This is more
implementation work up front (spin up `pyright-langserver`, speak minimal LSP over stdio,
issue hover requests, parse the hover markdown for a type name) but is the only approach
that doesn't fight the existing "extract once via git archive, don't mutate" design.

This is the main risk area of the slice — LSP stdio plumbing is fiddlier than a single
subprocess call. If this proves too heavy, the fallback is `reveal_type` injection against
a temporary copy of just the files that have candidate sites (not the full tree), accepting
the extra temp-file management.

### 4. Dependency management

Add `pyright` as a new optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
types = ["pyright"]
```

`infer_types=True` without `pyright` installed raises a clear `ImportError` at call time
(matching the existing pattern removed from `llm/groq.py` before it was deleted — catch
the import, raise with an actionable message), not a silent no-op.

## Implementation checklist (for the actual PR)

1. `ast/helpers.py`: new pure function `_get_factory_call_sites(func_node) -> List[Tuple[int, int]]`
   — returns (lineno, col_offset) for `x = call(...)` assignments where callee is lowercase.
   Pure AST-in/data-out, consistent with the rest of that module.
2. `kg/builder.py` `_parse_file`: call the new helper per function/method, attach recorded
   sites to the parse result dict (new key, e.g. `factory_call_sites`).
3. New module `kg/type_inference.py`: wraps pyright LSP invocation — `resolve_types(repo_dir, sites) -> Dict[(file, line, col), str]`.
   Handles: pyright not installed (raise clear error only if `infer_types=True` requested
   it), timeout, LSP protocol errors (log + return empty dict, don't crash the build).
4. `RepoASTParser.parse_repo`: wire in the new pass 1.5 step, only when `infer_types=True`.
5. `_resolve_edges`: extend the existing `uses`-edge resolution branch to also check
   pyright-inferred type names via the same `class_label_to_ids` lookup.
6. `RepoKGBuilder.build`: thread `infer_types: bool = False` through to `parse_repo`.
7. Tests: synthetic fixture repo with a factory-function case (real pyright invocation,
   no network — matches the spike), plus a pyright-not-installed fallback test.
8. `pyproject.toml`: add the `types` extra.
9. Update `CLAUDE.md`/`README.md` edge-type tables if `uses` edge metadata gains a new
   `source: 'pyright'` vs `source: 'heuristic'` provenance tag (recommended — makes it
   possible to distinguish confidence/origin later without re-deriving it).

## Open question to confirm before coding

Should resolved-via-pyright `uses` edges carry a distinguishing metadata field (e.g.
`metadata: {confidence: 'exact', source: 'pyright'}`) so they're distinguishable from the
existing uppercase-heuristic edges downstream? Recommended: yes, cheap to add now, and
consistent with how `calls` edges already carry a `confidence` provenance-like field.
