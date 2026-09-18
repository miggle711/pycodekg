"""Regression test: a bare constructor call (SomeClass(...)) resolves to
the class's own node, not just calls to functions/methods
(kg_construction#120). Previously label_to_ids (functions/methods only)
was the sole bare-name index, so a test that only constructs an
instance and never calls a named method on it (e.g. only uses builtins
like hash()/repr() on the result) had no calls/tests edge to the class
at all.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


class TestConstructorCallEdges:
    def test_bare_same_file_constructor_call_creates_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Value:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "    def __hash__(self):\n"
            "        return hash(self.x)\n"
            "\n"
            "def test_it():\n"
            "    v1 = Value(1)\n"
            "    v2 = Value(2)\n"
            "    assert hash(v1) != hash(v2)\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_it"
        )
        class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Value"
        )

        calls_edges = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == test_id and e["target"] == class_id
        ]
        assert len(calls_edges) == 1
        assert calls_edges[0]["metadata"]["confidence"] == "exact"

    def test_constructor_call_derives_a_tests_edge(self, tmp_path):
        """The existing calls -> tests derivation should pick this up
        for free, no separate wiring needed.
        """
        (tmp_path / "mod.py").write_text(
            "class Value:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "\n"
            "def test_it():\n"
            "    v = Value(1)\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_it"
        )
        class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Value"
        )

        tests_edges = [
            e for e in kg["edges"]
            if e["relation"] == "tests" and e["source"] == test_id and e["target"] == class_id
        ]
        assert len(tests_edges) == 1
        assert tests_edges[0]["metadata"]["derived_from"] == "calls"

    def test_imported_constructor_call_creates_edge(self, tmp_path):
        """The qualified/import_resolved path already indexed classes
        (qualified_to_ids), confirms it still works alongside the bare
        name fix.
        """
        (tmp_path / "models.py").write_text(
            "class ModelChoiceIteratorValue:\n"
            "    def __init__(self, value, instance):\n"
            "        self.value = value\n"
            "\n"
            "    def __hash__(self):\n"
            "        return hash(self.value)\n"
        )
        (tmp_path / "test_models.py").write_text(
            "from models import ModelChoiceIteratorValue\n"
            "\n"
            "def test_choice_value_hash():\n"
            "    value_1 = ModelChoiceIteratorValue(1, None)\n"
            "    value_2 = ModelChoiceIteratorValue(2, None)\n"
            "    assert hash(value_1) != hash(value_2)\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_choice_value_hash"
        )
        class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "ModelChoiceIteratorValue"
        )

        calls_edges = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == test_id and e["target"] == class_id
        ]
        assert len(calls_edges) == 1

    def test_name_colliding_with_a_function_prefers_same_file(self, tmp_path):
        """A class and a bare function share a name in different files --
        the caller's own file wins, same as the existing bare-call
        same-file preference for plain function/function collisions.
        """
        (tmp_path / "a.py").write_text(
            "class Widget:\n"
            "    def __init__(self):\n"
            "        pass\n"
        )
        (tmp_path / "b.py").write_text(
            "def Widget():\n"
            "    return {}\n"
            "\n"
            "def test_it():\n"
            "    w = Widget()\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        test_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "test_function" and n["label"] == "test_it"
        )
        calls_edges = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == test_id
        ]
        assert len(calls_edges) == 1
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}
        target = nodes_by_id[calls_edges[0]["target"]]
        assert target["type"] == "function"
        assert target["metadata"]["filepath"] == "b.py"
        assert calls_edges[0]["metadata"]["confidence"] == "exact"

    def test_class_with_no_matching_name_still_drops(self, tmp_path):
        """Sanity check: a truly external/unknown bare call still
        resolves to nothing, the fix only widens what's checked, it
        doesn't change the drop-when-unmatched behavior.
        """
        (tmp_path / "mod.py").write_text(
            "def test_it():\n"
            "    x = SomeExternalThing()\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        calls_edges = [e for e in kg["edges"] if e["relation"] == "calls"]
        assert calls_edges == []
