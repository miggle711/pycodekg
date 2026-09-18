"""Regression test: operator syntax (a | b, a == b, etc) resolves to
the dunder method Python actually dispatches to (kg_construction#122).
`a | b` never appears as an ast.Call, so calls-edge extraction has no
signal for it at all without this.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


class TestOperatorDispatchEdges:
    def test_locally_constructed_operand_resolves(self, tmp_path):
        """The issue's own real example: Django's Q() & Q(), a fresh
        constructor combined via an operator.
        """
        (tmp_path / "query_utils.py").write_text(
            "class Q:\n"
            "    def _combine(self, other, conn):\n"
            "        return Q()\n"
            "\n"
            "    def __and__(self, other):\n"
            "        return self._combine(other, 'AND')\n"
            "\n"
            "    def __or__(self, other):\n"
            "        return self._combine(other, 'OR')\n"
        )
        (tmp_path / "test_query_utils.py").write_text(
            "from query_utils import Q\n"
            "\n"
            "def test_combine():\n"
            "    q1 = Q()\n"
            "    q2 = Q()\n"
            "    combined = q1 & q2\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_combine"
        )
        and_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "__and__"
        )
        calls_edges = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == test_id and e["target"] == and_id
        ]
        assert len(calls_edges) == 1

    def test_operator_dispatch_derives_a_tests_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Q:\n"
            "    def __or__(self, other):\n"
            "        return self\n"
            "\n"
            "def test_or():\n"
            "    q1 = Q()\n"
            "    q2 = Q()\n"
            "    q1 | q2\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_or"
        )
        or_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "__or__"
        )
        tests_edges = [
            e for e in kg["edges"]
            if e["relation"] == "tests" and e["source"] == test_id and e["target"] == or_id
        ]
        assert len(tests_edges) == 1
        assert tests_edges[0]["metadata"]["derived_from"] == "calls"

    def test_self_receiver_resolves(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Vector:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "    def __add__(self, other):\n"
            "        return Vector(self.x + other.x)\n"
            "\n"
            "    def combine_with(self, other):\n"
            "        return self + other\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        combine_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "combine_with"
        )
        add_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "__add__"
        )
        calls_edges = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == combine_id and e["target"] == add_id
        ]
        assert len(calls_edges) == 1

    def test_equality_operator_resolves(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Value:\n"
            "    def __eq__(self, other):\n"
            "        return True\n"
            "\n"
            "def test_eq():\n"
            "    v1 = Value()\n"
            "    v2 = Value()\n"
            "    assert v1 == v2\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_eq"
        )
        eq_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "__eq__"
        )
        calls_edges = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == test_id and e["target"] == eq_id
        ]
        assert len(calls_edges) == 1

    def test_unresolvable_receiver_creates_no_edge(self, tmp_path):
        """A bare, unannotated parameter has no inferable type -- drop
        rather than guess, same principle as every other resolution
        path in this codebase.
        """
        (tmp_path / "mod.py").write_text(
            "def combine(a, b):\n"
            "    return a | b\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        calls_edges = [e for e in kg["edges"] if e["relation"] == "calls"]
        assert calls_edges == []

    def test_unresolvable_site_does_not_block_a_later_resolvable_one(self, tmp_path):
        """An earlier operator site for a dunder that can't be resolved
        must not consume the dedup slot a later, resolvable site for the
        same dunder needs (kg_construction#137 review). Order must not
        matter.
        """
        source = (
            "class Thing:\n"
            "    def __or__(self, other):\n"
            "        return self\n"
            "\n"
            "def combine(unresolvable, y: Thing):\n"
            "    a = unresolvable | y\n"
            "    b = y | y\n"
        )
        (tmp_path / "mod.py").write_text(source)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        calls_edges = [e for e in kg["edges"] if e["relation"] == "calls"]
        assert len(calls_edges) == 1

    def test_unresolvable_site_does_not_block_a_resolvable_one_reverse_order(self, tmp_path):
        """Same as above with the two sites swapped, confirms the fix
        isn't order dependent in either direction.
        """
        source = (
            "class Thing:\n"
            "    def __or__(self, other):\n"
            "        return self\n"
            "\n"
            "def combine(unresolvable, y: Thing):\n"
            "    b = y | y\n"
            "    a = unresolvable | y\n"
        )
        (tmp_path / "mod.py").write_text(source)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        calls_edges = [e for e in kg["edges"] if e["relation"] == "calls"]
        assert len(calls_edges) == 1

    def test_chained_comparison_is_skipped(self, tmp_path):
        """a < b < c is a real, if less common, shape (single Compare
        node with 2 ops) -- explicitly out of scope for this first cut,
        confirmed it doesn't crash and doesn't emit a wrong edge.
        """
        (tmp_path / "mod.py").write_text(
            "class Value:\n"
            "    def __lt__(self, other):\n"
            "        return True\n"
            "\n"
            "def test_chained():\n"
            "    a = Value()\n"
            "    b = Value()\n"
            "    c = Value()\n"
            "    assert a < b < c\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_chained"
        )
        lt_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "__lt__"
        )
        # 3 real Value() constructor calls do create edges (kg_construction#120,
        # unrelated to this test) -- only __lt__ specifically must be absent.
        calls_to_lt = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == test_id and e["target"] == lt_id
        ]
        assert calls_to_lt == []
