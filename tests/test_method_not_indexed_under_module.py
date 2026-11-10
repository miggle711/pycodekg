"""A method is indexed under module.Class.name only, not module.name.

A bare call through `from vec import cross`, or a call on the module
(`vec.cross()`), can only reach the module-level function cross, not the
method Vector.cross defined in the same module (#203).
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
    """(owning class or "<module>", confidence) of f's calls to cross."""
    (tmp_path / "vec.py").write_text(textwrap.dedent(VEC))
    (tmp_path / "use.py").write_text(textwrap.dedent(use))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["target"]]["metadata"].get("class") or "<module>", e["metadata"]["confidence"])
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["source"]]["label"] == "f"
        and nodes[e["target"]]["label"] == "cross"
    )


@pytest.mark.parametrize("use", [
    "from vec import cross\ndef f(a, b):\n    return cross(a, b)\n",
    "import vec\ndef f(a, b):\n    return vec.cross(a, b)\n",
], ids=["bare_call", "module_receiver"])
def test_a_module_level_name_does_not_reach_a_method(tmp_path, use):
    assert _cross_targets(tmp_path, use) == [("<module>", "qualified")]


def test_a_method_is_still_found_through_its_class(tmp_path):
    # `from vec import Vector` then Vector.cross(v, w): the import hint is
    # vec.Vector.cross, which still finds the method.
    use = "from vec import Vector\ndef f(v, w):\n    return Vector.cross(v, w)\n"
    assert _cross_targets(tmp_path, use) == [("Vector", "qualified")]
