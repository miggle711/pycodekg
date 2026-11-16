"""The no-KG audit uses real target-range and serializer selection rules."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze_module_constants import analyze_row, distribution, source_pair  # noqa: E402


def _row(before, after, *, new_file=False):
    import difflib

    patch = "".join(difflib.unified_diff(before.splitlines(keepends=True),
                                        after.splitlines(keepends=True),
                                        fromfile="a/mod.py", tofile="b/mod.py"))
    if new_file:
        patch = "diff --git a/mod.py b/mod.py\nnew file mode 100644\n" + patch
    return {"id": "example", "repo": "test/repo", "base_commit": "base",
            "code_file": "mod.py", "patch": patch, "code_src": after}


def test_post_patch_dataset_is_reversed_then_reapplied():
    before = "LIMIT = 10\ndef f():\n    return LIMIT\n"
    after = before.replace("10", "20").replace("return LIMIT", "return LIMIT + 1")
    assert source_pair(_row(before, after)) == (before, after)


def test_new_file_has_empty_base():
    after = "LIMIT = 10\ndef f():\n    return LIMIT\n"
    assert source_pair(_row("", after, new_file=True)) == ("", after)


def test_mismatched_patch_is_reported():
    row = _row("LIMIT = 1\n", "LIMIT = 2\n")
    row["code_src"] = "LIMIT = 3\n"
    with pytest.raises(ValueError, match="reverse patch"):
        source_pair(row)


def test_exact_context_can_relocate_shifted_hunk_coordinates():
    before = "# unrelated\n# prelude\nLIMIT = 1\ndef f():\n    return LIMIT\n"
    after = before.replace("return LIMIT", "return LIMIT + 1")
    row = _row(before, after)
    # Remove two lines before the hunk from its stated coordinates only.
    row["patch"] = row["patch"].replace("@@ -2,4 +2,4 @@", "@@ -1,4 +1,4 @@")
    diagnostics = {}
    assert source_pair(row, diagnostics) == (before, after)
    assert diagnostics["reverse_offset_relocation"]


def test_changed_removed_and_added_referenced_constants_are_reported():
    before = "CHANGE = 10\nREMOVE = 3\nUNUSED = 5\ndef f():\n    return CHANGE + REMOVE\n"
    after = "CHANGE = 20\nADD = 2\nUNUSED = 6\ndef f():\n    return CHANGE + REMOVE + ADD\n"
    # Isolate constant selection from target selection in this unit test.
    result = analyze_row(_row(before, after), lambda *_: [(4, 5)])
    assert result["referenced_constants"] == ["ADD", "CHANGE"]
    assert {(c["name"], c["kind"]) for c in result["stale_constants"]} == {
        ("ADD", "added"), ("CHANGE", "changed"), ("REMOVE", "removed")}
    assert result["added_characters"] > 0
    assert result["truncated_values"] == []


def test_constant_only_patch_has_no_function_target():
    result = analyze_row(_row("LIMIT = 1\n", "LIMIT = 2\n"), lambda *_: None)
    assert result["target_ranges"] == []
    assert result["referenced_constants"] == []
    assert result["stale_constants"] == []
    assert result["added_characters"] == 0


def test_no_kg_growth_uses_the_same_truncation_and_renderer():
    source = f"LONG = {'x' * 500!r}\ndef f():\n    return LONG\n"
    result = analyze_row(_row(source, source.replace("return LONG", "return LONG + ''")),
                         lambda *_: [(2, 3)])
    assert result["truncated_values"] == ["LONG"]
    assert result["added_characters"] == result["base_kg_added_characters"]
    assert result["added_characters"] < 300


def test_nearest_rank_p90_and_median():
    assert distribution(list(range(1, 11))) == {"n": 10, "median": 5.5, "p90": 9, "max": 10}


def test_post_patch_rendering_has_no_stale_or_missing_lines_even_when_the_base_differs():
    before = "CHANGE = 10\nREMOVE = 3\ndef f():\n    return CHANGE + REMOVE\n"
    after = "CHANGE = 20\nADD = 2\ndef f():\n    return CHANGE + REMOVE + ADD\n"
    result = analyze_row(_row(before, after), lambda *_: [(3, 4)])
    # The base-versus-post comparison still records what a KG-only lookup shows.
    assert {c["kind"] for c in result["stale_constants"]} == {"added", "changed", "removed"}
    # The prompt as built now shows exactly the post-patch assignments.
    assert result["post_patch_rendering"] == {
        "unavailable": None, "has_section": True, "stale_or_wrong": [], "missing": []}
