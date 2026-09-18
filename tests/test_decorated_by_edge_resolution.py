"""Regression test: a bare decorator name that resolves to a real
in-repo function or class gets a real 'decorated_by' edge to that
definition's own node (kg_construction#113), instead of only existing
as an unresolved raw string in metadata.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


class TestDecoratedByEdgeResolution:
    def test_bare_decorator_of_in_repo_function_creates_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def my_decorator(func):\n"
            "    return func\n"
            "\n"
            "@my_decorator\n"
            "def process(x):\n"
            "    return x\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        process_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "process"
        )
        decorator_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "my_decorator"
        )

        dec_edges = [
            e for e in kg["edges"]
            if e["relation"] == "decorated_by" and e["source"] == process_id
        ]
        assert len(dec_edges) == 1
        assert dec_edges[0]["target"] == decorator_id
        assert dec_edges[0]["metadata"]["confidence"] == "exact"

    def test_bare_decorator_call_form_resolves(self, tmp_path):
        """@my_decorator(arg) -- a Call, not a bare Name, still resolves
        via its func target.
        """
        (tmp_path / "mod.py").write_text(
            "def my_decorator(arg):\n"
            "    def wrap(func):\n"
            "        return func\n"
            "    return wrap\n"
            "\n"
            "@my_decorator('x')\n"
            "def process():\n"
            "    pass\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        process_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "process"
        )
        dec_edges = [
            e for e in kg["edges"]
            if e["relation"] == "decorated_by" and e["source"] == process_id
        ]
        assert len(dec_edges) == 1

    def test_stdlib_decorator_creates_no_edge(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Foo:\n"
            "    @staticmethod\n"
            "    def bar():\n"
            "        pass\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        dec_edges = [e for e in kg["edges"] if e["relation"] == "decorated_by"]
        assert dec_edges == []

    def test_dotted_decorator_creates_no_edge(self, tmp_path):
        """@pytest.mark.parametrize(...) is a real, common shape --
        module-qualified, not a bare repo-local name. Left unresolved
        rather than guessed at (kg_construction#113's own scope note).
        """
        (tmp_path / "mod.py").write_text(
            "import pytest\n"
            "\n"
            "@pytest.mark.parametrize('x', [1, 2, 3])\n"
            "def test_f(x):\n"
            "    pass\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        dec_edges = [e for e in kg["edges"] if e["relation"] == "decorated_by"]
        assert dec_edges == []

    def test_decorator_resolving_to_a_class_creates_edge(self, tmp_path):
        """A class used as a decorator (its __call__ wraps the function)
        is a real, if less common, Python pattern.
        """
        (tmp_path / "mod.py").write_text(
            "class Registered:\n"
            "    def __call__(self, func):\n"
            "        return func\n"
            "\n"
            "registered = Registered()\n"
            "\n"
            "@Registered\n"
            "def process():\n"
            "    pass\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        process_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "process"
        )
        class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Registered"
        )
        dec_edges = [
            e for e in kg["edges"]
            if e["relation"] == "decorated_by" and e["source"] == process_id
        ]
        assert len(dec_edges) == 1
        assert dec_edges[0]["target"] == class_id

    def test_name_colliding_across_class_and_function_prefers_same_file(self, tmp_path):
        """A class and a function share the same bare name in different
        files -- must not silently pick the class and call it exact
        just because class_label_to_ids was checked first
        (kg_construction#135 review). The decorated function's own
        module scope resolves the real Python semantics here: same
        file as the decorated function wins, same as bare calls do.
        """
        (tmp_path / "a.py").write_text(
            "class registered:\n"
            "    def __call__(self, func):\n"
            "        return func\n"
        )
        (tmp_path / "b.py").write_text(
            "def registered(func):\n"
            "    return func\n"
            "\n"
            "@registered\n"
            "def process():\n"
            "    pass\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        process_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "process"
        )
        func_registered_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "registered"
        )
        dec_edges = [
            e for e in kg["edges"]
            if e["relation"] == "decorated_by" and e["source"] == process_id
        ]
        assert len(dec_edges) == 1
        assert dec_edges[0]["target"] == func_registered_id
        assert dec_edges[0]["metadata"]["confidence"] == "exact"

    def test_name_colliding_in_neither_same_file_stays_ambiguous(self, tmp_path):
        """Same collision, but now neither candidate is in the decorated
        function's own file -- no same-file signal to prefer one, so
        both stay linked as ambiguous.
        """
        (tmp_path / "a.py").write_text(
            "class registered:\n"
            "    def __call__(self, func):\n"
            "        return func\n"
        )
        (tmp_path / "b.py").write_text(
            "def registered(func):\n"
            "    return func\n"
        )
        (tmp_path / "c.py").write_text(
            "@registered\n"
            "def process():\n"
            "    pass\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        process_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "function" and n["label"] == "process"
        )
        dec_edges = [
            e for e in kg["edges"]
            if e["relation"] == "decorated_by" and e["source"] == process_id
        ]
        assert len(dec_edges) == 2
        assert all(e["metadata"]["confidence"] == "ambiguous" for e in dec_edges)

        targets = {
            n["type"] for n in kg["nodes"]
            if n["id"] in {e["target"] for e in dec_edges}
        }
        assert targets == {"class", "function"}
