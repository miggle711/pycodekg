"""Imports that are not direct children of the module are collected too.

Before, only the module's direct children were looked at, so an import under
`try`/`except ImportError`, under `if TYPE_CHECKING:`, or inside a function
(a lazy import) produced no import node, no `imports`/`depends_on` edge, and
no `import_resolved` hint for the calls that use it.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser


def _build(tmp_path: Path, files: dict):
    for rel, source in files.items():
        (tmp_path / rel).write_text(textwrap.dedent(source))
    return RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)


def _labels(kg, relation):
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["source"]]["label"], nodes[e["target"]]["label"])
        for e in kg["edges"] if e["relation"] == relation
    )


def _run_targets(kg):
    """(caller label, defining file) for each calls edge to a function named run."""
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["source"]]["label"], nodes[e["target"]]["metadata"]["filepath"])
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["target"]]["label"] == "run"
    )


class TestImportsOutsideTheModuleBody:
    def test_imports_under_try_and_type_checking_are_collected_once(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            from typing import TYPE_CHECKING
            try:
                import chardet
            except ImportError:
                import charset_normalizer
            if TYPE_CHECKING:
                from models import Model

            def lazy():
                import chardet
                return chardet
        """})
        assert _labels(kg, "imports") == [
            ("m.py", "chardet"), ("m.py", "charset_normalizer"),
            ("m.py", "models.Model"), ("m.py", "typing.TYPE_CHECKING"),
        ]

    def test_lazy_import_is_visible_inside_its_function_and_closures_only(self, tmp_path):
        kg = _build(tmp_path, {
            "helper.py": "def run(): ...\n",
            "m.py": """
                def f():
                    import helper
                    def inner():
                        return helper.run()
                    return helper.run()

                def g():
                    return helper.run()
            """,
        })
        # f and its closure see the import: a depends_on edge and a call
        # resolved through it. g has no such import, so it gets neither.
        assert _labels(kg, "depends_on") == [("f", "helper"), ("inner", "helper")]
        nodes = {n["id"]: n["label"] for n in kg["nodes"]}
        confidence = {nodes[e["source"]]: e["metadata"]["confidence"]
                      for e in kg["edges"] if e["relation"] == "calls"}
        assert confidence["f"] == "qualified"
        assert confidence["inner"] == "qualified"
        assert confidence["g"] != "qualified"

    @pytest.mark.parametrize("header", [
        "import a as client\nif TYPE_CHECKING:\n    import b as client\n",
        "if TYPE_CHECKING:\n    import b as client\nimport a as client\n",
    ], ids=["runtime-first", "type-checking-first"])
    def test_type_checking_import_does_not_replace_a_runtime_binding(self, tmp_path, header):
        kg = _build(tmp_path, {
            "a.py": "def run(): ...\n",
            "b.py": "def run(): ...\n",
            "m.py": "from typing import TYPE_CHECKING\n" + header + "\ndef f():\n    return client.run()\n",
        })
        # The call is to a.run, and b is still recorded as a dependency.
        assert _run_targets(kg) == [("f", "a.py")]
        assert ("m.py", "b") in _labels(kg, "imports")

    @pytest.mark.parametrize("body, expected", [
        pytest.param(
            "import a as client\n    first = client.run()\n    import b as client\n    return first\n",
            [("f", "a.py")], id="call-before-the-rebinding"),
        pytest.param(
            "import a as client\n    first = client.run()\n    import b as client\n"
            "    second = client.run()\n    return first, second\n",
            [("f", "a.py"), ("f", "b.py")], id="calls-on-both-sides"),
    ])
    def test_a_call_goes_through_the_import_that_has_run_by_then(self, tmp_path, body, expected):
        kg = _build(tmp_path, {
            "a.py": "def run(): ...\n",
            "b.py": "def run(): ...\n",
            "m.py": "def f():\n    " + body,
        })
        assert _run_targets(kg) == expected

    def test_a_closures_call_goes_through_the_closures_own_import(self, tmp_path):
        kg = _build(tmp_path, {
            "a.py": "def run(): ...\n",
            "b.py": "def run(): ...\n",
            "m.py": """
                def outer():
                    import a as client
                    def inner():
                        import b as client
                        return client.run()
                    return inner
            """,
        })
        # inner imports b itself, so the call is to b.run, and so is outer's
        # copy of it (made through the closure). Neither is a call to a.run.
        assert _run_targets(kg) == [("inner", "b.py"), ("outer", "b.py")]
