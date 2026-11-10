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


def _save_edges(tmp_path: Path, source: str, caller: str = "outer"):
    """(class of the target or "<module>" for a plain function, confidence,
    via_closure) for the caller's calls to save."""
    (tmp_path / "m.py").write_text(source)
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    owner = {e["target"]: nodes[e["source"]]["label"]
             for e in kg["edges"] if e["relation"] == "contains" and nodes[e["source"]]["type"] == "class"}
    return sorted(
        (owner.get(e["target"], "<module>"), e["metadata"]["confidence"], bool(e["metadata"].get("via_closure")))
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["source"]]["label"] == caller
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
    assert _save_edges(tmp_path, textwrap.dedent(classes) + textwrap.dedent(body)) == expected


MODELS_PY = "def save(): ...\n"


@pytest.mark.parametrize("body, caller, expected", [
    # The closure's own node sees the function around it, not only itself:
    # m is outer's local, not the module, so models.save is not a candidate
    # and only the method matches, by name (#195).
    ("""
        import models as m
        def outer():
            def m(): ...
            def cb():
                m.save()
    """, "cb", [("Model", "name_only", False)]),
    ("""
        import models as m
        def outer(m):
            def cb():
                m.save()
    """, "cb", [("Model", "name_only", False)]),
    ("""
        import models as m
        def outer():
            m = Model()
            def cb():
                m.save()
    """, "cb", [("Model", "qualified", False)]),
    # An import in the closure hides outer's variable, in the closure and in outer's copy.
    ("""
        def outer():
            m = Model()
            def cb():
                import models as m
                m.save()
    """, "cb", [("<module>", "qualified", False)]),
    ("""
        def outer():
            m = Model()
            def cb():
                import models as m
                m.save()
    """, "outer", [("<module>", "qualified", True)]),
    # A comprehension variable or lambda parameter is its own name.
    ("""
        def outer(items):
            m = Model()
            return [m.save() for m in items]
    """, "outer", [("Model", "name_only", False)]),
    ("""
        def outer():
            m = Model()
            return lambda m: m.save()
    """, "outer", [("Model", "name_only", False)]),
    # The first iterable is evaluated in the enclosing scope, so it is still outer's m.
    ("""
        def outer():
            m = Model()
            return [x for x in m.save()]
    """, "outer", [("Model", "qualified", False)]),
], ids=["closure_node_nested_def_hides_import", "closure_node_param_hides_import",
        "closure_node_sees_outer_type", "closure_import_hides_outer_variable",
        "outer_copy_of_closure_import", "comprehension_receiver", "lambda_receiver",
        "first_iterable_is_enclosing_scope"])
def test_a_name_resolves_in_the_scope_that_binds_it(tmp_path, body, caller, expected):
    if "import models" in body:
        (tmp_path / "models.py").write_text(MODELS_PY)
    source = textwrap.dedent(MODEL_ONLY) + textwrap.dedent(body)
    assert _save_edges(tmp_path, source, caller) == expected
