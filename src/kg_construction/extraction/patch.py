"""
patch.py

Unified diff parsing for code change extraction.

Identifies which functions/classes in code_file were genuinely changed by a
patch, by parsing the PRE-PATCH source file's real AST and mapping each
changed line number (from the diff) against each function/class's actual
lineno/end_lineno range -- rather than inferring boundaries from whatever
happens to be visible in a hunk's small, variable context window.

Replaces an earlier hunk/line-scanning approach that inferred boundaries
from the diff text alone. That approach caused four distinct bugs over
time (kg_construction#14, #43, #63, #71), all from the same root cause: a
diff's context window can miss a real def/class line, or sweep in an
unrelated one, and there was no way to tell the two apart from the diff
text alone. Resolving by real line ranges eliminates that whole class --
see kg_construction#75 for the investigation and rationale.
"""

import ast
import re
from typing import List, NamedTuple, Optional, Set, Tuple


class _FunctionRange(NamedTuple):
    """A function/class's real line range in the pre-patch source, plus its
    enclosing scope. start_line includes any decorators (their own lines
    are part of "this function changed" just as much as the def line
    itself), even though ast.FunctionDef.lineno alone points only at the
    def line.
    """
    name: str
    is_class: bool
    enclosing_class: Optional[str]
    enclosing_function: Optional[str]
    start_line: int
    end_line: int


def _function_ranges(source: str) -> List[_FunctionRange]:
    """Collect every function/class definition's real line range in source,
    recursively (including nested functions and methods), via ast.

    Returns:
        List of _FunctionRange, in no particular order. A name can appear
        more than once (same-named methods on different classes, or a
        name reused at different nesting levels) -- callers match by line
        number, not name, so this ambiguity is resolved by construction
        rather than needing to be disambiguated after the fact.
    """
    tree = ast.parse(source)
    ranges: List[_FunctionRange] = []

    def visit(node: ast.AST, enclosing_class: Optional[str], enclosing_function: Optional[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = child.lineno
                if child.decorator_list:
                    start_line = min(start_line, child.decorator_list[0].lineno)
                ranges.append(_FunctionRange(
                    name=child.name, is_class=False,
                    enclosing_class=enclosing_class, enclosing_function=enclosing_function,
                    start_line=start_line, end_line=child.end_lineno,
                ))
                visit(child, enclosing_class=None, enclosing_function=child.name)
            elif isinstance(child, ast.ClassDef):
                start_line = child.lineno
                if child.decorator_list:
                    start_line = min(start_line, child.decorator_list[0].lineno)
                ranges.append(_FunctionRange(
                    name=child.name, is_class=True,
                    enclosing_class=enclosing_class, enclosing_function=enclosing_function,
                    start_line=start_line, end_line=child.end_lineno,
                ))
                visit(child, enclosing_class=child.name, enclosing_function=None)
            else:
                visit(child, enclosing_class, enclosing_function)

    visit(tree, enclosing_class=None, enclosing_function=None)
    return ranges


def _innermost_range(ranges: List[_FunctionRange], line: int) -> Optional[_FunctionRange]:
    """Return the smallest range containing line, or None if no range
    contains it (e.g. the line is a module-level statement, outside any
    function/class).
    """
    containing = [r for r in ranges if r.start_line <= line <= r.end_line]
    if not containing:
        return None
    return min(containing, key=lambda r: r.end_line - r.start_line)


def _changed_pre_patch_lines(patch: str, code_file: str) -> Tuple[Set[int], Set[int]]:
    """Return (removed_lines, insertion_anchor_lines): pre-patch line
    numbers, in code_file, that a changed line should be attributed
    against -- split by how reliable that attribution is.

    A '-' (removed) line has its own real pre-patch line number, used
    directly -- a removal's line genuinely IS the changed content, so a
    range containing it is always a correct match.

    A '+' (added) line has no pre-patch line number of its own (it doesn't
    exist in the pre-patch file) -- its insertion POINT is used instead:
    both the pre-patch line just before and just after it, since either
    can be the one actually inside the enclosing function's range (an
    addition at a function's very first or very last body line only has
    one real neighbor on the correct side). This anchor is inherently
    weaker than a real removal's line, though: if the insertion point
    happens to land exactly on some OTHER function/class's own def line,
    that's a false match (the insertion sits just before that function
    starts, not inside it) -- kg_construction#84's follow-up finding,
    confirmed directly on a real sympy patch inserting a new method
    immediately before an existing one. Callers must reject a match
    against an insertion-anchor line when line == match.start_line;
    a real removal's line is never subject to this exclusion.

    This anchor-based approach only finds a home for an addition when the
    insertion point falls INSIDE some pre-patch function/class's range at
    all -- a wholly new top-level function/class inserted in the
    blank-line gap between two others has no such home (kg_construction#84);
    that case is handled separately, against the reconstructed POST-patch
    source, by _changed_post_patch_lines/_reconstruct_post_patch_source
    below.
    """
    removed_lines: Set[int] = set()
    insertion_anchor_lines: Set[int] = set()
    current_file = None
    old_lineno = None

    for line in patch.split('\n'):
        if line.startswith('+++'):
            match = re.match(r'^\+\+\+ b/(.+)$', line)
            current_file = match.group(1) if match else None
            continue

        if current_file != code_file:
            continue

        if line.startswith('@@'):
            match = re.match(r'^@@ -(\d+)', line)
            old_lineno = int(match.group(1)) if match else None
            continue

        if old_lineno is None:
            continue

        if line.startswith('-') and not line.startswith('---'):
            removed_lines.add(old_lineno)
            old_lineno += 1
        elif line.startswith('+') and not line.startswith('+++'):
            insertion_anchor_lines.add(old_lineno)
            if old_lineno > 1:
                insertion_anchor_lines.add(old_lineno - 1)
        elif line.startswith(' '):
            old_lineno += 1

    return removed_lines, insertion_anchor_lines


def reconstruct_post_patch_source(pre_patch_source: str, patch: str, code_file: str) -> Optional[str]:
    """Reconstruct code_file's post-patch text by applying patch's hunks
    for that file directly to pre_patch_source, in memory.

    Used to resolve a changed line that has no home in the pre-patch
    source's own function/class ranges (kg_construction#84) -- a wholly
    new top-level function/class doesn't exist yet in pre_patch_source, so
    there's nothing there to attribute it to; parsing the file it WILL
    become and re-checking against real ranges there finds it directly,
    reusing the same range-matching logic as the pre-patch path rather
    than a second, bespoke diff-scanning path.

    Returns:
        The reconstructed post-patch text, or None if patch has no hunks
        for code_file at all (nothing to reconstruct).
    """
    pre_lines = pre_patch_source.split('\n')
    out_lines: List[str] = []
    pre_idx = 0  # 0-indexed cursor into pre_lines
    current_file = None
    saw_any_hunk = False

    # A trailing '' is a real artifact of splitting a newline-terminated
    # patch string, not a genuine blank context line. Left in, it looks
    # exactly like one and fails validation (kg_construction#128).
    lines = patch.split('\n')
    if patch.endswith('\n'):
        lines = lines[:-1]
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith('+++'):
            match = re.match(r'^\+\+\+ b/(.+)$', line)
            current_file = match.group(1) if match else None
            i += 1
            continue

        if current_file != code_file:
            i += 1
            continue

        if line.startswith('@@'):
            match = re.match(r'^@@ -(\d+)', line)
            if not match:
                i += 1
                continue
            saw_any_hunk = True
            hunk_start = int(match.group(1)) - 1  # 0-indexed
            # -1 is the real "new file" convention (@@ -0,0 @@), not a
            # mismatch. Anything beyond pre_lines' real length, or behind
            # pre_idx (an out-of-order hunk), means pre_patch_source
            # doesn't match what this patch assumes (kg_construction#128).
            if hunk_start > len(pre_lines) or (hunk_start != -1 and hunk_start < pre_idx):
                return None
            # Copy everything between the end of the last hunk (or file
            # start) and the start of this one, unchanged.
            out_lines.extend(pre_lines[pre_idx:hunk_start])
            pre_idx = hunk_start
            i += 1
            continue

        if not saw_any_hunk:
            i += 1
            continue

        if line.startswith('-') and not line.startswith('---'):
            # Must match pre_patch_source at this position (kg_construction#128).
            if pre_idx >= len(pre_lines) or pre_lines[pre_idx] != line[1:]:
                return None
            pre_idx += 1  # removed: present in pre, absent from post
        elif line.startswith('+') and not line.startswith('+++'):
            out_lines.append(line[1:])  # added: present in post only
        elif line.startswith(' '):
            if pre_idx >= len(pre_lines) or pre_lines[pre_idx] != line[1:]:
                return None
            out_lines.append(pre_lines[pre_idx])
            pre_idx += 1
        # A blank line with no marker at all (some diffs omit the leading
        # space on a genuinely empty context line) behaves like context.
        elif line == '':
            if pre_idx >= len(pre_lines) or pre_lines[pre_idx] != '':
                return None
            out_lines.append(pre_lines[pre_idx])
            pre_idx += 1

        i += 1

    if not saw_any_hunk:
        return None

    out_lines.extend(pre_lines[pre_idx:])
    return '\n'.join(out_lines)


def _changed_post_patch_lines(patch: str, code_file: str) -> Set[int]:
    """Return the set of POST-patch line numbers, in code_file, that a
    changed line should be attributed against -- the mirror image of
    _changed_pre_patch_lines, tracking the new-file side's line numbers
    instead of the old-file side's.

    A '+' (added) line has its own real post-patch line number, used
    directly. A '-' (removed) line has no post-patch line number of its
    own -- its removal POINT (new_lineno at the moment it's encountered)
    is used instead. Unlike the pre-patch side's insertion-point anchor
    (which also checks the line before, since an insertion's only real
    neighbor can be on either side), a removal's anchor does NOT also
    check new_lineno - 1: that would incorrectly pull in whatever
    unrelated content precedes the removal point (confirmed directly: a
    removed module-level comment right after a function's closing line
    would otherwise misattribute to that function, reintroducing the
    exact shape of bug kg_construction#71 fixed).
    """
    changed_lines: Set[int] = set()
    current_file = None
    new_lineno = None

    for line in patch.split('\n'):
        if line.startswith('+++'):
            match = re.match(r'^\+\+\+ b/(.+)$', line)
            current_file = match.group(1) if match else None
            continue

        if current_file != code_file:
            continue

        if line.startswith('@@'):
            match = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)', line)
            new_lineno = int(match.group(1)) if match else None
            continue

        if new_lineno is None:
            continue

        if line.startswith('+') and not line.startswith('+++'):
            changed_lines.add(new_lineno)
            new_lineno += 1
        elif line.startswith('-') and not line.startswith('---'):
            changed_lines.add(new_lineno)
        elif line.startswith(' '):
            new_lineno += 1

    return changed_lines


def is_newly_created_file(patch: str, code_file: str) -> bool:
    """True if patch's own diff header shows code_file as wholly new (git's
    'new file mode' marker), rather than an existing file being modified.

    Callers resolving pre-patch source (RepoManager.read_file_at_commit)
    should check this BEFORE fetching -- a newly-created file genuinely
    doesn't exist at base_commit, so fetching would raise even though this
    isn't a data error (kg_construction#93; confirmed on real scikit-learn/
    astropy commits that create a brand-new file via their patch). Treat
    pre-patch source as "" instead; _reconstruct_post_patch_source already
    handles an empty pre-patch source correctly (an empty file's hunk starts
    at line 0, so every line is an addition), so the existing #84 resolution
    path finds the new file's functions/classes without any special-casing
    beyond this check.
    """
    current_file = None
    for line in patch.split('\n'):
        # 'diff --git a/x b/x' starts each file's header block; 'new file
        # mode' (when present) always follows it immediately, before that
        # file's own '+++' line -- so current_file must be set here, not
        # from '+++', for this check to see it in time.
        match = re.match(r'^diff --git a/.+ b/(.+)$', line)
        if match:
            current_file = match.group(1)
            continue
        if current_file == code_file and line.startswith('new file mode'):
            return True
    return False


class PatchParser:
    """Parse unified diffs to extract changed function/class names, using
    the pre-patch source's real AST rather than the diff text alone.
    """

    @staticmethod
    def extract_changed_functions(
        patch: str, code_file: str, pre_patch_source: str
    ) -> Set[str]:
        """Extract function and class names genuinely changed in code_file.

        Args:
            patch: Unified diff string (multi-file).
            code_file: Relative path to the file to extract changes from
                      (e.g. 'requests/sessions.py').
            pre_patch_source: code_file's actual text before the patch was
                              applied -- required to resolve real function/
                              class line ranges via ast (kg_construction#75).

        Returns:
            Set of function/class name strings genuinely changed in
            code_file. Bare names only -- callers needing to disambiguate a
            name that matches more than one class' method should use
            extract_changed_functions_with_scope instead.

        Raises:
            SyntaxError: If pre_patch_source doesn't parse. Raised directly
                         rather than silently falling back to a guess --
                         a wrong guess here is exactly the bug class this
                         AST-based approach replaces.
        """
        return {
            name for name, _class_name in PatchParser.extract_changed_functions_with_scope(
                patch, code_file, pre_patch_source
            )
        }

    @staticmethod
    def extract_changed_functions_with_scope(
        patch: str, code_file: str, pre_patch_source: str
    ) -> Set[Tuple[str, Optional[str]]]:
        """Same as extract_changed_functions, but also reports the
        enclosing class name for each changed method, when there is one.

        Each changed line's real pre-patch line number is looked up
        against every function/class's real ast-derived range, and
        attributed to the innermost (smallest) containing range -- so a
        change inside a method is attributed to that method, not its
        enclosing class, and a change inside a nested closure is
        attributed to the closure, not its enclosing function
        (kg_construction#74).

        Changed lines are ALSO independently resolved against the
        reconstructed POST-patch source's own ranges, and the two result
        sets are unioned. This isn't just a fallback for when the
        pre-patch pass finds nothing: a wholly new top-level function/class
        (kg_construction#84) has no home in the pre-patch source at all, so
        its OWN changed lines resolve to nothing there, while OTHER changed
        lines in the same hunk can still resolve to a real (but wrong)
        neighboring function -- meaning "the pre-patch pass matched
        something" is not a reliable signal that it matched the right
        thing. Checking both sides and taking the union catches this
        without needing to first detect that the pre-patch pass "failed"
        globally. For an ordinary modification (no new function), the
        post-patch side re-derives the same (name, class) pair the
        pre-patch side already found, adding nothing extra.

        Returns:
            Set of (name, enclosing_class_or_None) tuples. A changed
            nested function (not a class method) also has
            enclosing_class=None -- the enclosing SCOPE information is
            still real, just not a class; only kg_construction#63's
            same-class-method-name ambiguity needs the class hint, and a
            nested function can't collide with a class method that way.
        """
        pre_ranges = _function_ranges(pre_patch_source)
        removed_lines, insertion_anchor_lines = _changed_pre_patch_lines(patch, code_file)

        changed: Set[Tuple[str, Optional[str]]] = set()
        # Class-level matches found ONLY via the insertion-anchor pass are
        # provisional: an anchor landing on a blank line right after some
        # method's closing line has no smaller range to match (the method's
        # own range already ended), so it matches the ENCLOSING CLASS --
        # but if the same hunk's real removed/changed line already
        # resolved to a method inside that class, the insertion is
        # genuinely "right after that method", not a real class-body-level
        # change, and the class-level match is redundant noise
        # (kg_construction#85's discovery: harmless while nothing looked up
        # class nodes, but a real method match in the same patch already
        # fully explains it). A class-level match from a REAL removed line
        # or the post-patch pass is trustworthy and always kept.
        anchor_only_class_matches: Set[Tuple[str, Optional[str]]] = set()

        for line in removed_lines:
            match = _innermost_range(pre_ranges, line)
            if match is None:
                continue  # module-level change
            changed.add((match.name, match.enclosing_class))

        for line in insertion_anchor_lines:
            match = _innermost_range(pre_ranges, line)
            if match is None or match.start_line == line:
                # No home at all, or the anchor merely landed on a
                # DIFFERENT sibling def/class's own start line -- not a
                # real match, since the insertion sits just before that
                # sibling begins, not inside it (kg_construction#84's
                # follow-up finding). Left for the post-patch pass below.
                continue
            pair = (match.name, match.enclosing_class)
            if match.is_class and pair not in changed:
                anchor_only_class_matches.add(pair)
            changed.add(pair)

        post_patch_source = reconstruct_post_patch_source(pre_patch_source, patch, code_file)
        if post_patch_source is not None:
            try:
                post_ranges = _function_ranges(post_patch_source)
            except SyntaxError:
                post_ranges = []
            post_changed_lines = _changed_post_patch_lines(patch, code_file)
            for line in post_changed_lines:
                match = _innermost_range(post_ranges, line)
                if match is None:
                    continue  # module-level change either way
                pair = (match.name, match.enclosing_class)
                anchor_only_class_matches.discard(pair)
                changed.add(pair)

        class_names_with_method_match = {
            enclosing_class for _, enclosing_class in changed if enclosing_class is not None
        }
        changed -= {
            pair for pair in anchor_only_class_matches
            if pair[0] in class_names_with_method_match
        }

        return changed
