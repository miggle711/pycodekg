"""Code outside the tests is never linked to a definition in a test file.

Production code does not import from a test file, so when it uses a name that
nothing outside the tests defines (a builtin like `int`, or a library class
like `argparse.Action`), the one same-named definition in a test file is not
what it means, however unique (#210). Edges from test files are unchanged.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser


def _edges(tmp_path: Path, files, relation):
    """(source label, target label, target file, confidence) for `relation`."""
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["source"]]["label"], nodes[e["target"]]["label"],
         nodes[e["target"]]["metadata"]["filepath"], e["metadata"]["confidence"])
        for e in kg["edges"] if e["relation"] == relation
    )


@pytest.mark.parametrize("files, relation", [
    # pytest: the builtin int() linked to a test-local int.
    ({"app.py": "def parse(s):\n    return int(s)\n",
      "tests/test_repr.py": "def int(x): ...\n"}, "calls"),
    # pytest: argparse.Action subclasses linked to a test class Action.
    ({"app.py": "import argparse\nclass HelpAction(argparse.Action): ...\n",
      "tests/test_cache.py": "class Action: ...\n"}, "inherits"),
    # Django: collections.Counter linked to a test model Counter.
    ({"app.py": "from collections import Counter\nclass Apps:\n    def populate(self):\n        return Counter()\n",
      "tests/models.py": "class Counter: ...\n"}, "uses"),
    ({"app.py": "from logging import Handler\ndef f():\n    raise Handler()\n",
      "tests/test_signals.py": "class Handler(Exception): ...\n"}, "raises"),
], ids=["builtin_call", "library_base", "library_class_used", "library_raise"])
def test_production_code_is_not_linked_into_test_files(tmp_path, files, relation):
    assert [e for e in _edges(tmp_path, files, relation) if e[0] not in ("Handler", "Action", "Counter", "int")] == []


def test_a_production_match_is_exact_despite_a_test_namesake(tmp_path):
    # One production Model and one test Model: from production code the name
    # is unique, so the edge is exact rather than ambiguous over both.
    edges = _edges(tmp_path, {
        "models.py": "class Model: ...\n",
        "app.py": "from models import Model\nclass Article(Model): ...\n",
        "tests/models.py": "class Model: ...\n",
    }, "inherits")
    assert edges == [("Article", "Model", "models.py", "exact")]


def test_test_code_still_reaches_test_helpers(tmp_path):
    edges = _edges(tmp_path, {
        "tests/helpers.py": "def make_user(): ...\n",
        "tests/test_users.py": "from helpers import make_user\ndef test_create():\n    make_user()\n",
    }, "calls")
    assert ("test_create", "make_user", "tests/helpers.py") in [e[:3] for e in edges]
