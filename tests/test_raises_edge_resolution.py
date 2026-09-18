"""Regression test: raise/except targets that name a real in-repo
exception class get a real 'raises' edge to that class's own node
(kg_construction#112), instead of only existing as an unresolved raw
string in metadata.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


class TestRaisesEdgeResolution:
    def test_raise_of_in_repo_class_creates_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class ValidationError(Exception):\n"
            "    pass\n"
            "\n"
            "def process(x):\n"
            "    if x < 0:\n"
            "        raise ValidationError('bad input')\n"
            "    return x\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        func_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "process"
        )
        class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "ValidationError"
        )

        raises_edges = [
            e for e in kg["edges"]
            if e["relation"] == "raises" and e["source"] == func_id
        ]
        assert len(raises_edges) == 1
        assert raises_edges[0]["target"] == class_id
        assert raises_edges[0]["metadata"]["confidence"] == "exact"

    def test_except_of_in_repo_class_creates_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class ValidationError(Exception):\n"
            "    pass\n"
            "\n"
            "def consume():\n"
            "    try:\n"
            "        pass\n"
            "    except ValidationError:\n"
            "        return None\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        func_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "consume"
        )
        class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "ValidationError"
        )

        raises_edges = [
            e for e in kg["edges"]
            if e["relation"] == "raises" and e["source"] == func_id
        ]
        assert len(raises_edges) == 1
        assert raises_edges[0]["target"] == class_id

    def test_tuple_except_handler_resolves_every_element(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class ErrorA(Exception):\n"
            "    pass\n"
            "\n"
            "class ErrorB(Exception):\n"
            "    pass\n"
            "\n"
            "def consume():\n"
            "    try:\n"
            "        pass\n"
            "    except (ErrorA, ErrorB):\n"
            "        return None\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        func_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "consume"
        )
        targets = {
            e["target"] for e in kg["edges"]
            if e["relation"] == "raises" and e["source"] == func_id
        }
        class_ids = {
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] in ("ErrorA", "ErrorB")
        }
        assert targets == class_ids

    def test_builtin_exception_creates_no_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def f():\n"
            "    raise ValueError('not in this repo')\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        raises_edges = [e for e in kg["edges"] if e["relation"] == "raises"]
        assert raises_edges == []

    def test_ambiguous_name_links_to_every_candidate(self, tmp_path):
        (tmp_path / "a.py").write_text(
            "class CustomError(Exception):\n"
            "    pass\n"
        )
        (tmp_path / "b.py").write_text(
            "class CustomError(Exception):\n"
            "    pass\n"
            "\n"
            "def f():\n"
            "    raise CustomError('ambiguous')\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        raises_edges = [e for e in kg["edges"] if e["relation"] == "raises"]
        assert len(raises_edges) == 2
        assert all(e["metadata"]["confidence"] == "ambiguous" for e in raises_edges)

    def test_attribute_form_raise_creates_no_edge(self, tmp_path):
        """raise self.exc_class(...) and raise module.Error(...) aren't
        bare class names -- left unresolved rather than guessed at.
        """
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    def f(self):\n"
            "        raise self.exc_class('x')\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        raises_edges = [e for e in kg["edges"] if e["relation"] == "raises"]
        assert raises_edges == []
