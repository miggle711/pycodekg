"""inherits, uses, raises and decorated_by follow where a name was imported.

They used to match by simple name only, so a name imported from the standard
library or another package linked to a same-named definition in the repo:
Django's `from functools import cached_property` to
django.utils.functional.cached_property, sympy's `from ast import Tuple` to
sympy's Tuple (#211). A name imported from the repo, relatively or not, still
resolves.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser


def _targets(tmp_path: Path, files, relation, source):
    """(target label, target file) for `relation` edges from `source`."""
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["target"]]["label"], nodes[e["target"]]["metadata"]["filepath"])
        for e in kg["edges"]
        if e["relation"] == relation and nodes[e["source"]]["label"] == source
    )


@pytest.mark.parametrize("files, relation, source", [
    ({"pkg/functional.py": "def cached_property(f): ...\n",
      "pkg/apps.py": "from functools import cached_property\nclass A:\n    @cached_property\n    def label(self): ...\n"},
     "decorated_by", "label"),
    ({"pkg/models.py": "class Counter: ...\n",
      "pkg/registry.py": "from collections import Counter\nclass Apps:\n    def populate(self):\n        return Counter()\n"},
     "uses", "Apps"),
    ({"pkg/actions.py": "class Action: ...\n",
      "pkg/help.py": "import argparse\nclass HelpAction(argparse.Action): ...\n"},
     "inherits", "HelpAction"),
    ({"pkg/errors.py": "class DomainError(Exception): ...\n",
      "pkg/ring.py": "from flint import DomainError\ndef divide():\n    raise DomainError()\n"},
     "raises", "divide"),
], ids=["stdlib_decorator", "stdlib_class", "stdlib_base", "third_party_exception"])
def test_a_name_imported_from_outside_is_not_linked(tmp_path, files, relation, source):
    assert _targets(tmp_path, files, relation, source) == []


@pytest.mark.parametrize("files", [
    {"pkg/__init__.py": "", "pkg/models.py": "class Model: ...\n",
     "pkg/app.py": "from pkg.models import Model\nclass Article(Model): ...\n"},
    {"pkg/__init__.py": "", "pkg/models.py": "class Model: ...\n",
     "pkg/app.py": "from .models import Model\nclass Article(Model): ...\n"},
    {"pkg/__init__.py": "", "pkg/models.py": "class Model: ...\n",
     "pkg/app.py": "from pkg import models\nclass Article(models.Model): ...\n"},
], ids=["absolute_import", "relative_import", "module_attribute"])
def test_a_name_imported_from_the_repo_still_resolves(tmp_path, files):
    assert _targets(tmp_path, files, "inherits", "Article") == [("Model", "pkg/models.py")]
