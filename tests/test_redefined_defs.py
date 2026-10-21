"""Regression tests for redefined and conditionally-defined functions/classes.

Two related defects, fixed together because fixing the second alone makes the
first worse:

1. Definitions nested under compound statements (`if`/`try`/`with`/`for`/
   `while`/`match`) were never emitted as KG nodes at all, only direct
   children of a module/class/function body were visited. A patch to e.g.
   `def u(s)` under `if sys.version_info[0] == 2:` had nothing to seed from.

2. Two definitions with the same qualified name in one file (a property
   getter and its `@x.setter`, `if/else` alternative defs, `@overload`)
   shared one node ID, so `_aggregate_and_index` silently kept only the
   first. Seed refresh then keyed post-patch source by (label, class), last
   definition winning, so a patch to the getter put the unchanged setter's
   source in the prompt.

Fix: every def gets a unique qualname within its file (`C.x`, then `C.x#2`;
children use the parent's unique qualname, e.g. `C#2.m`), stored in node
metadata['qualname']; the first occurrence keeps its pre-fix ID.
"""

import ast
import difflib
from pathlib import Path

from kg_construction.ast.helpers import _make_id, assign_qualnames
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.extraction.patch import PatchParser
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.kg.builder import SCHEMA_VERSION, RepoASTParser
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.repo_manager import RepoManager

REPO = "test/repo"


def _build(tmp_path: Path, source: str, name: str = "mod.py"):
    (tmp_path / name).write_text(source)
    return RepoASTParser(max_workers=1).parse_repo(REPO, tmp_path)


def _nodes(kg, label):
    return [n for n in kg["nodes"] if n["label"] == label]


def _qualnames(source: str):
    return [q for _node, q in assign_qualnames(ast.parse(source))]


PROPERTY_SOURCE = (
    "class C:\n"
    "    def __init__(self):\n"
    "        self._x = 0\n"
    "\n"
    "    @property\n"
    "    def x(self):\n"
    "        return self._x\n"
    "\n"
    "    @x.setter\n"
    "    def x(self, v):\n"
    "        self._x = v\n"
)


class TestAssignQualnames:
    def test_plain_nesting(self):
        src = "class A:\n    def m(self):\n        def inner(): pass\n"
        assert _qualnames(src) == ["A", "A.m", "A.m.inner"]

    def test_second_definition_gets_ordinal_suffix(self):
        assert _qualnames(PROPERTY_SOURCE) == ["C", "C.__init__", "C.x", "C.x#2"]

    def test_third_definition_gets_third_ordinal(self):
        src = "def f(): pass\ndef f(): pass\ndef f(): pass\n"
        assert _qualnames(src) == ["f", "f#2", "f#3"]

    def test_children_use_parents_unique_qualname(self):
        src = (
            "if PY2:\n"
            "    class C:\n"
            "        def m(self): pass\n"
            "else:\n"
            "    class C:\n"
            "        def m(self): pass\n"
        )
        assert _qualnames(src) == ["C", "C.m", "C#2", "C#2.m"]

    def test_descends_through_compound_statements(self):
        src = (
            "import sys\n"
            "if sys.version_info[0] == 2:\n"
            "    def u(s): pass\n"
            "else:\n"
            "    def u(s): pass\n"
            "try:\n"
            "    import x\n"
            "except ImportError:\n"
            "    def fallback(): pass\n"
            "with open('f') as fh:\n"
            "    def in_with(): pass\n"
            "for _ in range(1):\n"
            "    def in_for(): pass\n"
            "while False:\n"
            "    def in_while(): pass\n"
        )
        assert _qualnames(src) == ["u", "u#2", "fallback", "in_with", "in_for", "in_while"]

    def test_nested_def_is_reported_once_under_its_real_parent(self):
        src = "def outer():\n    if True:\n        def inner(): pass\n"
        assert _qualnames(src) == ["outer", "outer.inner"]


class TestBuilderEmitsRedefinedDefs:
    def test_property_getter_and_setter_are_distinct_nodes(self, tmp_path):
        kg = _build(tmp_path, PROPERTY_SOURCE)
        xs = _nodes(kg, "x")
        assert len(xs) == 2
        assert len({n["id"] for n in xs}) == 2
        assert sorted(n["metadata"]["qualname"] for n in xs) == ["C.x", "C.x#2"]
        assert sorted(n["metadata"]["lineno"] for n in xs) == [6, 10]

    def test_first_occurrence_keeps_its_pre_fix_id(self, tmp_path):
        kg = _build(tmp_path, PROPERTY_SOURCE)
        first = next(n for n in _nodes(kg, "x") if n["metadata"]["qualname"] == "C.x")
        assert first["id"] == _make_id(f"func_{REPO}_mod.py_C.x")
        init = _nodes(kg, "__init__")[0]
        assert init["id"] == _make_id(f"func_{REPO}_mod.py_C.__init__")

    def test_both_property_defs_are_contained_by_the_class(self, tmp_path):
        kg = _build(tmp_path, PROPERTY_SOURCE)
        class_id = _nodes(kg, "C")[0]["id"]
        x_ids = {n["id"] for n in _nodes(kg, "x")}
        contained = {e["target"] for e in kg["edges"]
                     if e["relation"] == "contains" and e["source"] == class_id}
        assert x_ids <= contained

    def test_if_else_alternative_defs_both_emitted(self, tmp_path):
        kg = _build(tmp_path, (
            "import sys\n"
            "if sys.version_info[0] == 2:\n"
            "    def u(s):\n"
            "        return unicode(s)\n"
            "else:\n"
            "    def u(s):\n"
            "        return str(s)\n"
        ))
        us = _nodes(kg, "u")
        assert len(us) == 2
        assert len({n["id"] for n in us}) == 2
        file_id = _nodes(kg, "mod.py")[0]["id"]
        for n in us:
            assert any(e["relation"] == "contains" and e["source"] == file_id
                       and e["target"] == n["id"] for e in kg["edges"])

    def test_try_except_fallback_def_emitted(self, tmp_path):
        kg = _build(tmp_path, (
            "try:\n"
            "    from fast import encode\n"
            "except ImportError:\n"
            "    def encode(s):\n"
            "        return s\n"
        ))
        assert len(_nodes(kg, "encode")) == 1
        assert _nodes(kg, "encode")[0]["type"] == "function"

    def test_method_defined_under_if_in_class_body_is_a_method(self, tmp_path):
        kg = _build(tmp_path, (
            "class K:\n"
            "    if True:\n"
            "        def m(self):\n"
            "            return 1\n"
        ))
        m = _nodes(kg, "m")
        assert len(m) == 1
        assert m[0]["type"] == "method"
        assert m[0]["metadata"]["class"] == "K"
        class_id = _nodes(kg, "K")[0]["id"]
        assert any(e["relation"] == "contains" and e["source"] == class_id
                   and e["target"] == m[0]["id"] for e in kg["edges"])

    def test_def_under_if_inside_function_is_emitted_once_with_real_parent(self, tmp_path):
        kg = _build(tmp_path, (
            "def outer(flag):\n"
            "    if flag:\n"
            "        def inner():\n"
            "            return 1\n"
            "    return None\n"
        ))
        inner = _nodes(kg, "inner")
        assert len(inner) == 1
        assert inner[0]["metadata"]["parent_function"] == "outer"
        outer_id = _nodes(kg, "outer")[0]["id"]
        assert any(e["relation"] == "contains" and e["source"] == outer_id
                   and e["target"] == inner[0]["id"] for e in kg["edges"])

    def test_redefined_class_children_do_not_collide(self, tmp_path):
        kg = _build(tmp_path, (
            "if PY2:\n"
            "    class C:\n"
            "        def m(self): return 1\n"
            "else:\n"
            "    class C:\n"
            "        def m(self): return 2\n"
        ))
        ms = _nodes(kg, "m")
        assert len(ms) == 2
        assert len({n["id"] for n in ms}) == 2
        assert sorted(n["metadata"]["qualname"] for n in ms) == ["C#2.m", "C.m"]
        assert len(_nodes(kg, "C")) == 2

    def test_build_is_deterministic(self, tmp_path):
        a = _build(tmp_path, PROPERTY_SOURCE)
        b = _build(tmp_path, PROPERTY_SOURCE)
        assert [n["id"] for n in a["nodes"]] == [n["id"] for n in b["nodes"]]

    def test_schema_version_bumped_for_new_nodes(self):
        # An older cached KG is silently missing conditionally-defined nodes
        # and has collapsed redefinitions; it must be rebuilt, not served.
        assert SCHEMA_VERSION >= 4


class TestPatchParserReportsWhichDuplicateChanged:
    def _patch(self, old: str, new: str) -> str:
        return "".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), "a/mod.py", "b/mod.py"))

    def test_getter_change_reports_getter_def_line_only(self):
        new = PROPERTY_SOURCE.replace("return self._x", "return self._x + 1")
        defs = PatchParser.extract_changed_defs(
            self._patch(PROPERTY_SOURCE, new), "mod.py", PROPERTY_SOURCE)
        assert defs[("x", "C")] == {6}

    def test_setter_change_reports_setter_def_line_only(self):
        new = PROPERTY_SOURCE.replace("self._x = v", "self._x = v * 2")
        defs = PatchParser.extract_changed_defs(
            self._patch(PROPERTY_SOURCE, new), "mod.py", PROPERTY_SOURCE)
        assert defs[("x", "C")] == {10}

    def test_scope_api_still_returns_plain_pairs(self):
        new = PROPERTY_SOURCE.replace("return self._x", "return self._x + 1")
        pairs = PatchParser.extract_changed_functions_with_scope(
            self._patch(PROPERTY_SOURCE, new), "mod.py", PROPERTY_SOURCE)
        assert pairs == {("x", "C")}


class _LocalFileRepoManager(RepoManager):
    def __init__(self, repo_dir: Path):
        self._repo_dir = repo_dir

    def read_file_at_commit(self, repo: str, commit: str, path: str) -> str:
        return (self._repo_dir / path).read_text()


def _extract(tmp_path: Path, old: str, new: str):
    kg = _build(tmp_path, old)
    patch = "".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), "a/mod.py", "b/mod.py"))
    extractor = TestContextExtractor(
        KGQueryEngine(kg), repo_manager=_LocalFileRepoManager(tmp_path))
    return extractor.extract({
        "repo": REPO, "base_commit": "deadbeef", "patch": patch,
        "code_file": "mod.py", "test_file": "test_mod.py",
    }, depth=1)


class TestSeedSelectionAndRefreshWithDuplicates:
    def test_getter_patch_seeds_only_the_getter_with_post_patch_source(self, tmp_path):
        new = PROPERTY_SOURCE.replace("return self._x", "return self._x + 1")
        ctx = _extract(tmp_path, PROPERTY_SOURCE, new)
        xs = [s for s in ctx.seeds if s["label"] == "x"]
        assert len(xs) == 1
        assert xs[0]["metadata"]["qualname"] == "C.x"
        assert "return self._x + 1" in xs[0]["metadata"]["source_code"]
        assert "self._x = v" not in xs[0]["metadata"]["source_code"]
        assert ctx.stale_seed_labels == []

    def test_setter_patch_seeds_only_the_setter_with_post_patch_source(self, tmp_path):
        new = PROPERTY_SOURCE.replace("self._x = v", "self._x = v * 2")
        ctx = _extract(tmp_path, PROPERTY_SOURCE, new)
        xs = [s for s in ctx.seeds if s["label"] == "x"]
        assert len(xs) == 1
        assert xs[0]["metadata"]["qualname"] == "C.x#2"
        assert "self._x = v * 2" in xs[0]["metadata"]["source_code"]
        assert ctx.stale_seed_labels == []

    def test_duplicate_seed_names_do_not_trip_the_ambiguity_check(self, tmp_path):
        new = PROPERTY_SOURCE.replace("return self._x", "return self._x + 1")
        ctx = _extract(tmp_path, PROPERTY_SOURCE, new)
        _valid, report = TestContextValidator(ctx).validate()
        assert "Ambiguous seed names" not in report

    def test_changed_conditional_def_is_a_seed(self, tmp_path):
        old = (
            "import sys\n"
            "if sys.version_info[0] == 2:\n"
            "    def u(s):\n"
            "        return unicode(s)\n"
            "else:\n"
            "    def u(s):\n"
            "        return str(s)\n"
        )
        new = old.replace("return str(s)", "return str(s).strip()")
        ctx = _extract(tmp_path, old, new)
        us = [s for s in ctx.seeds if s["label"] == "u"]
        assert len(us) == 1
        assert us[0]["metadata"]["qualname"] == "u#2"
        assert "str(s).strip()" in us[0]["metadata"]["source_code"]

    def test_closure_inside_method_is_refreshed_not_stale(self, tmp_path):
        old = (
            "class W:\n"
            "    def build(self, items):\n"
            "        def key(e):\n"
            "            return e[0]\n"
            "        return sorted(items, key=key)\n"
        )
        new = old.replace("return e[0]", "return e[1]")
        ctx = _extract(tmp_path, old, new)
        keys = [s for s in ctx.seeds if s["label"] == "key"]
        assert len(keys) == 1
        assert "return e[1]" in keys[0]["metadata"]["source_code"]
        assert ctx.stale_seed_labels == []
