# `extraction/patch.py` deep dive

A complete walkthrough of `PatchParser`: how a unified diff gets turned
into a set of genuinely-changed function/class names, using real AST line
ranges rather than the diff text itself.

## Why this module exists in this form

An earlier version inferred function/class boundaries directly from the
diff's hunk text (its `@@` header and the small, variable context window
around each change). That approach caused four distinct real bugs over
time (#14, #43, #63, #71), all from the same root cause: a hunk's context
window can miss a real `def`/`class` line entirely, or sweep an unrelated
one into view, and there's no way to tell the two apart from the diff text
alone. The fix (#75): parse the REAL pre-patch source with `ast`, get real
`lineno`/`end_lineno` ranges for every function/class, and map the diff's
changed line numbers against those ranges instead of guessing from
context. This eliminates the whole bug class, since a real AST range is
never ambiguous about what it contains.

## `_FunctionRange` and `_function_ranges`

`_FunctionRange` is a `NamedTuple`: `name`, `is_class`, `enclosing_class`,
`enclosing_function`, `start_line`, `end_line`. `start_line` deliberately
includes any decorator lines (`child.decorator_list[0].lineno`, taken as
the earlier of the two) -- a changed `@property` decorator is just as much
"this function changed" as a change to the `def` line itself, even though
`ast`'s own `.lineno` only ever points at the `def`/`class` line proper.

`_function_ranges(source)` walks the real AST (`ast.parse`) recursively --
not just top-level statements, but every nested function/method too --
producing a flat list of every range in the file. Recursion tracks
`enclosing_class`/`enclosing_function` as it descends: entering a
`ClassDef` sets `enclosing_class` for its children (and clears
`enclosing_function`); entering a `FunctionDef` sets `enclosing_function`
(and clears `enclosing_class`) for anything nested inside it. A name can
legitimately appear more than once in the returned list (two different
classes each with their own `__init__`, or the same name at different
nesting levels) -- this is fine by construction, since every caller
matches by LINE NUMBER against a range, never by name alone.

## `_innermost_range`

Given a line number, returns the smallest range that contains it (by
`end_line - start_line`), or `None` if nothing contains it (a module-level
statement, outside any function/class). This is what makes "a change
inside a method" attribute to the method and not its enclosing class --
both ranges technically contain the same line, but the method's range is
strictly smaller.

## `_changed_pre_patch_lines`

Walks the raw diff text line by line, tracking `old_lineno` (the pre-patch
file's own line counter, advanced by context/removed lines, not by
additions, matching how a unified diff's `@@ -a,b +c,d @@` header actually
counts). Returns a `(removed_lines, insertion_anchor_lines)` pair -- two
sets with genuinely different trust levels:

- **`removed_lines`**: a `-` line has its own real pre-patch line number.
  This is always a trustworthy match -- the removed content genuinely
  existed at that exact line, so whatever range contains it is correct by
  construction.
- **`insertion_anchor_lines`**: a `+` line has NO pre-patch line number of
  its own (it doesn't exist in the pre-patch file at all). Its insertion
  POINT is recorded instead -- both the line immediately before and after
  it (`old_lineno` and `old_lineno - 1`), since an addition at the very
  first or very last line of a function's body only has one real neighbor
  on the correct side. This anchor is inherently weaker than a real
  removal: confirmed on a real sympy patch (#84's follow-up) that an
  anchor can land EXACTLY on some unrelated sibling function/class's own
  `start_line` -- a false match, since the insertion sits just before that
  sibling begins, not inside it. Callers must reject a match where
  `line == match.start_line` for an anchor (never for a real removal,
  where that same condition is never actually a false match).

This anchor approach only finds a home for an addition when the insertion
point falls INSIDE some pre-patch range at all. A wholly new top-level
function/class inserted into the blank-line gap between two existing ones
has no such home -- neither neighbor's range contains that blank line.
That case needs the post-patch side (below).

## `reconstruct_post_patch_source`

Replays a diff's hunks against `pre_patch_source` directly, in memory, to
produce the file's real post-patch text -- without ever shelling out to
`git apply`. Tracks a 0-indexed cursor (`pre_idx`) into the pre-patch
source's lines:

- On a `@@` hunk header: copies everything between the end of the last
  hunk (or file start) and this hunk's start, unchanged, then jumps the
  cursor to the hunk's start line.
- `-` line: cursor advances (present in pre, absent from post) but nothing
  is appended to the output.
- `+` line: appended to the output directly (its own text, minus the `+`
  marker) -- present in post only, cursor doesn't move.
- ` ` (context) line: appended from the pre-patch source at the cursor,
  cursor advances.
- A genuinely blank diff line with no leading marker at all (some diffs
  omit the leading space on an empty context line) is treated the same as
  context.

Returns `None` if the patch has no hunks for `code_file` at all (nothing
to reconstruct) -- this is later distinguished by callers, e.g.
`_extract_for_new_file` in `context.py` raises a real error if this
happens for a file the patch itself claims to create.

This function exists specifically to resolve a changed line that has no
home anywhere in the pre-patch source's own ranges (#84): a wholly new
function/class doesn't exist yet pre-patch, so there's nothing there to
attribute it to. Parsing the file it WILL become and re-checking against
real ranges there finds it directly -- reusing the exact same
range-matching logic as the pre-patch path, rather than a second, bespoke
diff-scanning implementation.

It also transparently handles an empty starting source (`""`) correctly:
a wholly new FILE's hunk starts at line 0, so every line in it is an
addition -- this is what lets `is_newly_created_file`/`context.py`'s
new-file path reuse this same function with no special-casing (#93).

## `_changed_post_patch_lines`

The mirror image of `_changed_pre_patch_lines`, tracking the new-file
side's line numbers (`new_lineno`, from the `+c,d` half of the `@@`
header) instead of the old-file side's.

- A `+` line has its own real post-patch line number -- trustworthy,
  used directly.
- A `-` line has no post-patch line number of its own -- its removal
  POINT (`new_lineno` at the moment it's encountered) is used as a weaker
  anchor.

Deliberately asymmetric with the pre-patch side: a removal's anchor here
does NOT also check `new_lineno - 1`. Checking the line before would pull
in whatever unrelated content happens to precede the removal point --
confirmed directly that a removed module-level comment right after some
function's closing line would otherwise misattribute to that function,
reintroducing the exact shape of bug #71 already fixed once.

## `is_newly_created_file`

Scans the diff's own header text for git's `new file mode` marker. The
tricky part is WHERE to track `current_file` from: `new file mode`
appears right after each file's `diff --git a/x b/x` header line, BEFORE
that file's own `+++` line -- so this function tracks `current_file` from
`diff --git`, not `+++` (unlike every other function in this module, which
only needs `+++` since they only care about hunk content, which always
comes after `+++`). Getting this wrong would mean the check never sees
`new file mode` in time.

This exists so a caller (`RepoManager.read_file_at_commit` via
`context.py`) can check BEFORE trying to fetch pre-patch source at all --
a newly-created file genuinely doesn't exist at `base_commit`, so fetching
it would raise, even though that's not a data error (#93; confirmed on
real scikit-learn/astropy commits that create a brand-new file via their
patch).

## `PatchParser.extract_changed_functions` / `extract_changed_functions_with_scope`

The public entry point. `extract_changed_functions` is a thin wrapper
returning bare names only; `extract_changed_functions_with_scope` is the
real implementation, returning `(name, enclosing_class_or_None)` pairs --
the class hint exists because a bare name alone can collide across
classes (#63: a real encode/httpx patch to `AsyncClient.aclose` also
matched the unrelated `BoundAsyncStream.aclose` in the same file).

Three phases, always run together, results unioned:

**Phase 1 -- resolve against pre-patch ranges.** Every real removed line
resolves via `_innermost_range` directly. Every insertion-anchor line also
resolves this way, but rejected if the match's own `start_line` equals the
anchor line (the sibling-collision case from #84, described above).
Anchor matches that resolve to a CLASS (not a method) are separately
tracked in `anchor_only_class_matches`, since these are provisional --
explained in phase 3.

**Phase 2 -- resolve against the reconstructed post-patch source,
independently.** This is not merely a fallback for when phase 1 finds
nothing. The key insight from #84: "phase 1 matched something" is not a
reliable signal that it matched the RIGHT thing -- a wholly new function's
own changed lines resolve to nothing in phase 1, while OTHER changed lines
in the same hunk can still resolve to a real but wrong neighboring
function. Checking both sides and taking the union catches this without
needing to first detect that phase 1 "failed" globally. For an ordinary
modification (no new function involved), phase 2 just re-derives the same
`(name, class)` pair phase 1 already found -- the union adds nothing
extra in that case.

**Phase 3 -- drop redundant anchor-only class matches (#85's discovery).**
An insertion anchored to the blank line right after some method's closing
line has no smaller range to match (the method's own range already
ended), so it resolves to the ENCLOSING CLASS -- but if the same patch
already resolved a real METHOD inside that class (from phase 1's removed
lines, or phase 2's post-patch pass), the class-level match is redundant
noise from the anchor's imprecision, not a genuine class-body-level
change. This filter only strips a class match that came exclusively from
the ambiguous anchor pass AND is redundant with a trustworthy method match
found elsewhere in the same result. A genuine class-body-only change (no
method touched at all -- e.g. a bare class attribute assignment) is
unaffected and still resolves correctly, since nothing in
`class_names_with_method_match` would match it.

## Summary of the design principle running through the whole file

Every ambiguous signal (an insertion anchor, a same-named collision, a
redundant class-level match) is either resolved by a stronger, independent
signal, or deliberately left unresolved/dropped -- never guessed at. This
mirrors the same bias documented in `kg/builder.py`'s edge resolution
(#61, #65): a missing result is recoverable and visible; a wrong one
silently propagates downstream into a seed the model gets shown as if it
were the real target.
