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


def test_depends_on_covers_every_branch(tmp_path):
    _run_targets(tmp_path, """
        import sys
        if sys.platform == "win32":
            import a as c
        else:
            import b as c
        def f():
            c.run()
    """)
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
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
