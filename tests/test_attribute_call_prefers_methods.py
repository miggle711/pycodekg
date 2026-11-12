"""An unresolved attribute call is a call on an object, not on a plain function (#202)."""

from textwrap import dedent

import pytest

from kg_construction.kg.builder import RepoASTParser


def _calls_from(tmp_path, files, caller="f"):
    for name, source in files.items():
        (tmp_path / name).write_text(dedent(source))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return [(nodes[e["target"]]["type"], nodes[e["target"]]["metadata"].get("class"),
             e["metadata"]["confidence"])
            for e in kg["edges"]
            if e["relation"] == "calls" and nodes[e["source"]]["label"] == caller]


VECTOR = """
    class Vector:
        def cross(self, other): ...
"""


def test_attribute_call_goes_to_the_method_not_the_same_named_function(tmp_path):
    calls = _calls_from(tmp_path, {
        "vec.py": VECTOR,
        "util.py": "def cross(a, b): ...\n",
        "use.py": "def f(v, w):\n    return v.cross(w)\n",
    })
    assert calls == [("method", "Vector", "name_only")]


@pytest.mark.parametrize("use_file, use", [
    # A bare call resolves in the caller's own module, so the function there wins.
    ("util.py", "def cross(a, b): ...\n\ndef f(v, w):\n    return cross(v, w)\n"),
    # Through an imported module the import is the hint, so the function wins.
    ("use.py", "import util\n\ndef f(v, w):\n    return util.cross(v, w)\n"),
])
def test_calls_that_are_not_on_an_object_still_reach_the_function(tmp_path, use_file, use):
    files = {"vec.py": VECTOR, "util.py": "def cross(a, b): ...\n"}
    files[use_file] = use
    calls = _calls_from(tmp_path, files)
    assert [kind for kind, _, _ in calls] == ["function"]


def test_two_methods_of_the_name_stay_dropped(tmp_path):
    calls = _calls_from(tmp_path, {
        "vec.py": VECTOR,
        "mat.py": "class Matrix:\n    def cross(self, other): ...\n",
        "util.py": "def cross(a, b): ...\n",
        "use.py": "def f(v, w):\n    return v.cross(w)\n",
    })
    assert calls == []
