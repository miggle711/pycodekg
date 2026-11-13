"""Calls through an import that has alternatives.

`try: import a as c` / `except ImportError: import b as c` binds c to a
whenever a is installed, so a call through c goes to a, not to the fallback
that comes last in the file. An `if`/`else` (or `elif`) that imports the same
name in more than one branch can bind it to either, so a call after it links
to every branch. A name that is both imported and assigned in one function
(`except ImportError: m = None`) still goes through its import.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser

MODULES = {"a.py": "def run(): ...\n", "b.py": "def run(): ...\n", "d.py": "def run(): ...\n"}


def _run_targets(tmp_path: Path, source: str):
    """(caller, file defining the run() it calls) for each calls edge."""
    for name, text in {**MODULES, "m.py": textwrap.dedent(source)}.items():
        (tmp_path / name).write_text(text)
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["source"]]["label"], nodes[e["target"]]["metadata"]["filepath"])
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["target"]]["label"] == "run"
    )


@pytest.mark.parametrize("source, expected", [
    ("""
        try:
            import a as c
        except ImportError:
            import b as c
        def f():
            c.run()
    """, [("f", "a.py")]),
    ("""
        try:
            from a import run
        except ImportError:
            from b import run
        def f():
            run()
    """, [("f", "a.py")]),
    ("""
        def f():
            try:
                import a as c
            except ImportError:
                import b as c
            c.run()
    """, [("f", "a.py")]),
    # A handler that imports a name the try body does not bind still counts.
    ("""
        try:
            import a as c
        except ImportError:
            import b as other
        def f():
            other.run()
    """, [("f", "b.py")]),
], ids=["module_import", "from_import", "inside_function", "handler_only_name"])
def test_try_body_import_wins_over_its_fallback(tmp_path, source, expected):
    assert _run_targets(tmp_path, source) == expected


@pytest.mark.parametrize("source, expected", [
    ("""
        import sys
        if sys.platform == "win32":
            import a as c
        else:
            import b as c
        def f():
            c.run()
    """, [("f", "a.py"), ("f", "b.py")]),
    ("""
        import sys
        if sys.platform == "win32":
            from a import run
        elif sys.platform == "darwin":
            from b import run
        else:
            from d import run
        def f():
            run()
    """, [("f", "a.py"), ("f", "b.py"), ("f", "d.py")]),
    ("""
        import sys
        def f():
            if sys.platform == "win32":
                import a as c
            else:
                import b as c
            c.run()
    """, [("f", "a.py"), ("f", "b.py")]),
    # Inside a branch, only that branch's import has run.
    ("""
        import sys
        def f():
            if sys.platform == "win32":
                import a as c
                c.run()
            else:
                import b as c
    """, [("f", "a.py")]),
], ids=["module_if_else", "module_elif_chain", "inside_function", "call_inside_branch"])
def test_a_call_after_if_else_links_every_branch(tmp_path, source, expected):
    assert _run_targets(tmp_path, source) == expected


def _build(tmp_path: Path, source: str):
    for name, text in {**MODULES, "m.py": textwrap.dedent(source)}.items():
        (tmp_path / name).write_text(text)
    return RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)


def test_depends_on_covers_every_branch(tmp_path):
    kg = _build(tmp_path, """
        import sys
        if sys.platform == "win32":
            import a as c
        else:
            import b as c
        def f():
            c.run()
    """)
    nodes = {n["id"]: n["label"] for n in kg["nodes"]}
    assert sorted(nodes[e["target"]] for e in kg["edges"]
                  if e["relation"] == "depends_on" and nodes[e["source"]] == "f") == ["a", "b"]


@pytest.mark.parametrize("source", [
    """
        def f():
            try:
                import a as m
            except ImportError:
                m = None
            m.run()
    """,
    """
        def f(m=None):
            if m is None:
                import a as m
            m.run()
    """,
], ids=["try_import_or_none", "parameter_default_then_import"])
def test_a_name_both_imported_and_assigned_keeps_its_import(tmp_path, source):
    assert _run_targets(tmp_path, source) == [("f", "a.py")]


@pytest.mark.parametrize("source, expected", [
    # An if with no else keeps the binding from before it.
    ("""
        import sys
        import a as c
        if sys.platform == "win32":
            import b as c
        def f():
            c.run()
    """, [("f", "a.py"), ("f", "b.py")]),
    ("""
        import sys
        def f():
            import a as c
            if sys.platform == "win32":
                import b as c
            c.run()
    """, [("f", "a.py"), ("f", "b.py")]),
    # Nested if/else: every branch of both.
    ("""
        import sys
        def f():
            if sys.platform == "win32":
                if sys.maxsize > 2:
                    import a as c
                else:
                    import b as c
            else:
                import d as c
            c.run()
    """, [("f", "a.py"), ("f", "b.py"), ("f", "d.py")]),
    # In the else branch, before its own import, the if branch's import has
    # not run.
    ("""
        import sys
        def f():
            if sys.platform == "win32":
                import a as c
            else:
                c.run()
                import b as c
    """, []),
    # A branch that returns does not reach the code after the if.
    ("""
        import sys
        def f():
            import a as c
            if sys.platform == "win32":
                import b as c
                return
            c.run()
    """, [("f", "a.py")]),
], ids=["module_if_without_else", "function_if_without_else", "nested_if_else",
        "else_before_its_import", "returning_branch"])
def test_bindings_follow_every_path_to_the_call(tmp_path, source, expected):
    assert _run_targets(tmp_path, source) == expected


def test_a_comprehension_variable_does_not_hide_an_import(tmp_path):
    # A comprehension's loop variable is its own scope in Python 3, so the
    # function's m is still the import after it.
    assert _run_targets(tmp_path, """
        def f(items):
            import a as m
            names = [m for m in items]
            m.run()
    """) == [("f", "a.py")]


@pytest.mark.parametrize("source", [
    # The parameter is what m.run() uses; the import comes later.
    """
        import b as m
        def f(m=None):
            m.run()
            import a as m
    """,
    # m is None by the time of the call.
    """
        def f():
            import a as m
            m = None
            m.run()
    """,
], ids=["parameter_before_later_import", "assigned_after_import"])
def test_a_local_binding_at_the_call_is_not_an_import(tmp_path, source):
    assert _run_targets(tmp_path, source) == []
