"""A call is resolved with the types of the function that makes it.

An outer function also gets the calls made inside its closures (tagged
`via_closure`). Those were resolved with the outer function's types, so a
closure that rebinds a name got an edge for the wrong class, and a type
assigned inside the closure never reached the outer function's copy.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser

CLASSES = """
    class Model:
        def save(self): ...
    class Other:
        def save(self): ...
"""
MODEL_ONLY = """
    class Model:
        def save(self): ...
"""
IMPORTS_M = "\n    import models as m\n"


def _outer_save_edges(tmp_path: Path, source: str):
    """(class of the target, confidence, via_closure) for outer's calls to save."""
    (tmp_path / "m.py").write_text(source)
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    owner = {e["target"]: nodes[e["source"]]["label"]
             for e in kg["edges"] if e["relation"] == "contains" and nodes[e["source"]]["type"] == "class"}
    return sorted(
        (owner[e["target"]], e["metadata"]["confidence"], bool(e["metadata"].get("via_closure")))
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["source"]]["label"] == "outer"
        and nodes[e["target"]]["label"] == "save"
    )


@pytest.mark.parametrize("classes, body, expected", [
    # The closure's parameter is an Other, whatever outer's m is.
    (CLASSES, """
        def outer():
            m = Model()
            def cb(m: Other):
                m.save()
    """, [("Other", "qualified", True)]),
    # A type assigned inside the closure reaches outer's copy of the edge.
    (MODEL_ONLY, """
        def outer():
            def cb():
                n = Model()
                n.save()
    """, [("Model", "qualified", True)]),
    # An untyped parameter hides outer's m: no type, so a name match only.
    (MODEL_ONLY, """
        def outer():
            m = Model()
            def cb(m):
                m.save()
    """, [("Model", "name_only", True)]),
    # A name the closure does not bind is outer's variable, so its type applies.
    (MODEL_ONLY, """
        def outer():
            m = Model()
            def cb():
                m.save()
    """, [("Model", "qualified", True)]),
    # The same text on both sides is two different calls.
    (CLASSES, """
        def outer():
            m = Model()
            m.save()
            def cb(m: Other):
                m.save()
    """, [("Model", "qualified", False), ("Other", "qualified", True)]),
    # A comprehension's loop variable is its own scope, not a binding of the closure.
    (MODEL_ONLY, """
        def outer():
            m = Model()
            def cb(items):
                vals = [m for m in items]
                m.save()
    """, [("Model", "qualified", True)]),
    # A name bound without a known type does not fall through to a module import.
    (IMPORTS_M + MODEL_ONLY, """
        def outer():
            def cb(m):
                m.save()
    """, [("Model", "name_only", True)]),
    # A nested def binds its name in the enclosing function.
    (IMPORTS_M + MODEL_ONLY, """
        def outer():
            def m(): ...
            def cb():
                m.save()
    """, [("Model", "name_only", True)]),
    # `global m` names the module level variable, not outer's.
    (MODEL_ONLY, """
        def outer():
            m = Model()
            def cb():
                global m
                m.save()
    """, [("Model", "name_only", True)]),
    # `global m` then `m = Model()` in the closure: the assignment says what m holds there.
    (MODEL_ONLY, """
        def outer():
            def cb():
                global m
                m = Model()
                m.save()
    """, [("Model", "qualified", True)]),
    # `case ... as m` binds m, with no known type.
    (MODEL_ONLY, """
        def outer():
            m = Model()
            def cb(x):
                match x:
                    case Other() as m:
                        m.save()
    """, [("Model", "name_only", True)]),
], ids=["typed_parameter_rebinds", "type_assigned_in_closure", "untyped_parameter_hides_outer",
        "free_variable_keeps_outer_type", "same_call_text_two_types", "comprehension_variable",
        "parameter_hides_import", "nested_def_hides_import", "global_declaration", "global_then_assigned",
        "match_capture"])
def test_closure_call_uses_the_closures_own_types(tmp_path, classes, body, expected):
    assert _outer_save_edges(tmp_path, textwrap.dedent(classes) + textwrap.dedent(body)) == expected
