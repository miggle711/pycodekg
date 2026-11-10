"""An attribute call is not resolved through an import of its method name.

`from vec import cross` makes `cross(a, b)` a call of the imported function,
but `v.cross(w)` is a method call whatever `v` is, and was linked to the
imported function as well.
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


def _cross_targets(tmp_path: Path, use: str):
    """Confidence of f's calls to the module level function named cross."""
    (tmp_path / "vec.py").write_text(textwrap.dedent(VEC))
    (tmp_path / "use.py").write_text(textwrap.dedent(use))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        e["metadata"]["confidence"]
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["source"]]["label"] == "f"
        and nodes[e["target"]]["label"] == "cross" and nodes[e["target"]]["type"] == "function"
    )


@pytest.mark.parametrize("use, expected", [
    # A method call on an untyped receiver is not the imported function.
    ("from vec import cross\ndef f(v, w):\n    return v.cross(w)\n", []),
    # A bare call is the imported function.
    ("from vec import cross\ndef f(a, b):\n    return cross(a, b)\n",
     ["qualified"]),
    # A call on an imported module is still resolved through that module.
    ("import vec\ndef f(a, b):\n    return vec.cross(a, b)\n",
     ["qualified"]),
], ids=["attribute_call", "bare_call", "module_receiver"])
def test_attribute_call_is_not_resolved_through_an_imported_function(tmp_path, use, expected):
    assert _cross_targets(tmp_path, use) == expected


def test_attribute_call_on_a_dotted_module_path_still_reaches_the_function(tmp_path):
    # `import pkg.util` then `pkg.util.helper()`: the receiver is the module
    # path, which is not looked up as an import, so this edge comes from the
    # unique name match. Dropping plain functions from attribute calls would
    # lose it (requests' `requests.utils.rewind_body(prep)` is one).
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "util.py").write_text("def helper(): ...\n")
    (tmp_path / "use.py").write_text("import pkg.util\ndef f():\n    return pkg.util.helper()\n")
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    assert [(nodes[e["target"]]["type"], e["metadata"]["confidence"]) for e in kg["edges"]
            if e["relation"] == "calls" and nodes[e["source"]]["label"] == "f"] == [("function", "name_only")]
