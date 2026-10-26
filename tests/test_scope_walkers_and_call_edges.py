"""Tests for how nested functions (closures) and call sites are analysed.

The rule these tests pin down:

* Facts about a function's own control flow stay in its own scope: returns,
  mutations, side effects. They are never merged in from a closure.
* Dependencies are transitive. An outer function that defines a closure which
  calls `normalize` keeps a calls edge to `normalize`, tagged
  `via_closure: True` so a consumer can tell it from a direct call.
"""

import ast
import textwrap
from pathlib import Path

import pytest

from kg_construction.ast.helpers import (
    _extract_data_flows,
    _get_attribute_accesses,
    _get_class_attributes,
)
from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.query import KGQueryEngine


def _build(tmp_path: Path, files: dict):
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)


def _edges(kg, relation):
    """{(source label, target label): edge metadata} for one relation."""
    nodes = {n["id"]: n for n in kg["nodes"]}
    return {
        (nodes[e["source"]]["label"], nodes[e["target"]]["label"]): e.get("metadata", {})
        for e in kg["edges"] if e["relation"] == relation
    }


def _func(source: str):
    """The first function or method defined in `source`."""
    tree = ast.parse(textwrap.dedent(source))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))


CLOSURE_SOURCE = """
    def normalize(x):
        return x

    def create_index(entries):
        def keyfunc(entry):
            return normalize(entry)
        return sorted(entries, key=keyfunc)
"""


class TestClosureCallsAreKeptAndTagged:
    def test_outer_function_keeps_the_closures_call_tagged(self, tmp_path):
        calls = _edges(_build(tmp_path, {"m.py": CLOSURE_SOURCE}), "calls")
        assert calls[("keyfunc", "normalize")].get("via_closure") is None
        assert calls[("create_index", "normalize")]["via_closure"] is True

    def test_a_direct_call_wins_over_the_closures_call(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            def normalize(x):
                return x

            def create_index(entries):
                normalize(entries)
                def keyfunc(entry):
                    return normalize(entry)
                return sorted(entries, key=keyfunc)
        """})
        assert _edges(kg, "calls")[("create_index", "normalize")].get("via_closure") is None

    def test_caller_counts_and_queries_separate_direct_from_closure_callers(self, tmp_path):
        kg = _build(tmp_path, {"m.py": CLOSURE_SOURCE})
        normalize = next(n for n in kg["nodes"] if n["label"] == "normalize")
        meta = normalize["metadata"]
        assert meta["caller_count"] == 1
        assert [c["label"] for c in meta["direct_callers"]] == ["keyfunc"]
        assert [c["label"] for c in meta["closure_callers"]] == ["create_index"]

        engine = KGQueryEngine(kg)
        assert {n["label"] for n in engine.find_callers(normalize["id"])} == {"keyfunc", "create_index"}
        assert {n["label"] for n in engine.find_callers(normalize["id"], include_closure=False)} == {"keyfunc"}

    def test_test_whose_only_call_is_in_a_local_callback_is_still_linked(self, tmp_path):
        kg = _build(tmp_path, {
            "m.py": "def target():\n    return 1\n",
            "tests/test_m.py": (
                "from m import target\n"
                "def test_callback():\n"
                "    def callback():\n"
                "        return target()\n"
                "    callback()\n"
            ),
        })
        tests = _edges(kg, "tests")
        assert tests[("test_callback", "target")]["via_closure"] is True


class TestCallsOnDifferentReceiversAreNotCollapsed:
    @pytest.mark.parametrize("untyped_first", [True, False])
    def test_typed_call_resolves_whatever_the_statement_order(self, tmp_path, untyped_first):
        calls = ["x.save()", "m = Model()\n    m.save()"]
        body = "\n    ".join(calls if untyped_first else reversed(calls))
        kg = _build(tmp_path, {"m.py": (
            "class Model:\n    def save(self): pass\n"
            "class Other:\n    def save(self): pass\n"
            f"def f(x):\n    {body}\n"
        )})
        meta = _edges(kg, "calls")[("f", "save")]
        assert meta["confidence"] == "qualified"


class TestNestedScopeLeakage:
    def test_closure_returns_are_not_the_outer_functions_but_its_parameter_use_counts(self):
        flows = _extract_data_flows(_func("""
            def outer(items):
                def inner():
                    if True:
                        return 5
                    return items
                return 1
        """))
        assert flows["returns"] == ["1"]
        assert "items" in flows["parameter_usage"]


class TestWriteFormsAreRecorded:
    def test_side_effects_cover_every_assignment_form_and_skip_nested_scopes(self):
        writes = _get_attribute_accesses(_func("""
            def f(self, items, p):
                self.n += 1
                self.a, self.b = 1, 2
                [self.c, *self.d] = items
                for self.i in items:
                    pass
                with open(p) as self.fh:
                    pass
                self.items.append(3)
                def callback():
                    self.hidden = 1
        """))[1]
        assert sorted(writes) == ["a", "b", "c", "d", "fh", "i", "n"]

    def test_class_attributes_cover_the_same_forms_and_skip_nested_scopes(self):
        cls = ast.parse(textwrap.dedent("""
            class K:
                def __init__(self, items):
                    self.n = 0
                    self.a, self.b = items
                    self.total += 1
                    def later():
                        self.hidden = 1
        """)).body[0]
        assert _get_class_attributes(cls) == ["n", "a", "b", "total"]
