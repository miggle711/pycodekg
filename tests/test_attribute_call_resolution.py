"""An attribute call `x.name()` is resolved through its receiver, never
through an import of `name` itself.

`v.cross(w)` used to be linked to an imported function `cross` as well as to
the method, because the callee name was looked up in the imports whenever the
receiver was not an import (#195). And since `x` is not a module, the call
can only reach a method: a module-level function or class of the same name
is not a candidate.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser

VEC = """
    class Vector:
        def cross(self, other): ...
    def cross(a, b): ...
"""


def _call_targets(tmp_path: Path, files, caller: str):
    """(owning class or "<module>", label, confidence) for each calls edge
    made by `caller`."""
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["target"]]["metadata"].get("class") or "<module>",
         nodes[e["target"]]["label"], e["metadata"]["confidence"])
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["source"]]["label"] == caller
    )


@pytest.mark.parametrize("files, caller, expected", [
    # The issue's repro: v is not the imported cross, so only the method.
    ({"vec.py": VEC, "use.py": """
        from vec import cross
        def f(v, w):
            return v.cross(w)
    """}, "f", [("Vector", "cross", "name_only")]),
    # psf/requests: `import json` (external) used to hide r.json() entirely.
    ({"models.py": """
        import json
        class Response:
            def json(self): ...
        def h(r):
            return r.json()
    """}, "h", [("Response", "json", "name_only")]),
    # x is not a module, so a module-level function is not a candidate.
    ({"m.py": """
        def helper(): ...
        def g(x):
            return x.helper()
    """}, "g", []),
])
def test_attribute_call_ignores_import_of_its_name(tmp_path, files, caller, expected):
    assert _call_targets(tmp_path, files, caller) == expected


@pytest.mark.parametrize("files, caller, expected", [
    # A bare call still goes through the import.
    ({"vec.py": VEC, "use.py": """
        from vec import cross
        def f(a, b):
            return cross(a, b)
    """}, "f", [("<module>", "cross", "qualified")]),
    # A call through an imported module still reaches its function.
    ({"vec.py": VEC, "use.py": """
        import vec
        def f(a, b):
            return vec.cross(a, b)
    """}, "f", [("<module>", "cross", "qualified")]),
    # A dotted receiver that starts with an imported name may be a module
    # path, so a function is still a candidate.
    ({"pkg/__init__.py": "", "pkg/mod.py": "def run(): ...\n", "use.py": """
        import pkg
        def f():
            return pkg.mod.run()
    """}, "f", [("<module>", "run", "name_only")]),
])
def test_calls_through_imports_unchanged(tmp_path, files, caller, expected):
    assert _call_targets(tmp_path, files, caller) == expected
