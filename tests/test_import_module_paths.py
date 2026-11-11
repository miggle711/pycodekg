"""Calls through an import are resolved through the module it names.

An absolute import whose top level name is not one of the repo's packages is
outside the repo, even when a nested module happens to share the name
(`from pprint import pformat` is not src/_pytest/_io/pprint.py, #212). A
relative import keeps its dots and is resolved from the importing file. A call
on a dotted module path (`pkg.utils.f()`) goes through the import of its first
name (#206).
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser


def _calls(tmp_path: Path, files, caller):
    """(target file, target qualname, confidence) for `caller`'s calls edges."""
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["target"]]["metadata"]["filepath"], nodes[e["target"]]["metadata"].get("qualname"),
         e["metadata"]["confidence"])
        for e in kg["edges"]
        if e["relation"] == "calls" and nodes[e["source"]]["label"] == caller
    )


PKG = {"pkg/__init__.py": "", "pkg/io/__init__.py": ""}


@pytest.mark.parametrize("files", [
    {**PKG, "pkg/io/pprint.py": "def pformat(x): ...\n",
     "pkg/recwarn.py": "from pprint import pformat\ndef f(x):\n    return pformat(x)\n"},
    {**PKG, "pkg/io/html.py": "def escape(s): ...\n",
     "pkg/text.py": "from html import escape\ndef f(s):\n    return escape(s)\n"},
    {**PKG, "pkg/io/path.py": "def join(*a): ...\n",
     "pkg/files.py": "import os\ndef f(a, b):\n    return os.path.join(a, b)\n"},
], ids=["stdlib_name_shared_by_a_nested_module", "stdlib_function", "dotted_stdlib_path"])
def test_an_import_from_outside_the_repo_is_not_linked(tmp_path, files):
    assert _calls(tmp_path, files, "f") == []


TWO_PACKAGES = {"a/__init__.py": "", "b/__init__.py": "",
                "a/exceptions.py": "def boom(): ...\n", "b/exceptions.py": "def boom(): ...\n"}


@pytest.mark.parametrize("use", [
    "from .exceptions import boom\ndef f():\n    return boom()\n",
    "from . import exceptions\ndef f():\n    return exceptions.boom()\n",
    "from a.exceptions import boom\ndef f():\n    return boom()\n",
    "import a\ndef f():\n    return a.exceptions.boom()\n",
], ids=["relative_from_import", "relative_module_receiver", "absolute_from_import", "dotted_module_path"])
def test_an_import_reaches_the_module_it_names(tmp_path, use):
    # boom is defined in two packages, so only the import says which one.
    files = {**TWO_PACKAGES, "a/models.py": use}
    assert _calls(tmp_path, files, "f") == [("a/exceptions.py", "boom", "qualified")]


def test_a_src_layout_dotted_path_reaches_its_function(tmp_path):
    # psf/requests: requests.utils.rewind_body(prep) with `import requests`.
    files = {"src/requests/__init__.py": "", "src/requests/utils.py": "def rewind_body(p): ...\n",
             "src/other/utils.py": "def rewind_body(p): ...\n",
             "src/requests/models.py": "import requests\ndef f(prep):\n    return requests.utils.rewind_body(prep)\n"}
    assert _calls(tmp_path, files, "f") == [("src/requests/utils.py", "rewind_body", "qualified")]


def test_an_object_reached_through_a_module_stays_a_name_match(tmp_path):
    # pkg.conn is an object, not a module: its method is matched by name and
    # does not become more certain because pkg was imported.
    files = {"pkg/__init__.py": "class Conn:\n    def cursor(self): ...\nconn = Conn()\n",
             "use.py": "import pkg\ndef f():\n    return pkg.conn.cursor()\n"}
    assert _calls(tmp_path, files, "f") == [("pkg/__init__.py", "Conn.cursor", "name_only")]


def test_every_definition_of_an_overloaded_name_is_reached(tmp_path):
    # sympy's symbols: @overload stubs, then the real def. All are the name
    # the import refers to, so none is dropped.
    files = {"pkg/__init__.py": "",
             "pkg/core.py": ("from typing import overload\n"
                             "@overload\ndef symbols(n: str) -> int: ...\n"
                             "@overload\ndef symbols(n: int) -> str: ...\n"
                             "def symbols(n):\n    return n\n"),
             "pkg/use.py": "from pkg.core import symbols\ndef f():\n    return symbols('x')\n"}
    targets = _calls(tmp_path, files, "f")
    assert len(targets) == 3 and {t[2] for t in targets} == {"qualified"}



def test_a_class_re_exported_by_the_imported_module_is_reached(tmp_path):
    # sympy: dfm.py does `from ._dfm import DFM`, and domainmatrix.py does
    # `from .dfm import DFM` then `DFM._supports_domain(...)`.
    files = {"pkg/__init__.py": "",
             "pkg/_impl.py": "class DFM:\n    @classmethod\n    def supports(cls, d): ...\n",
             "pkg/dfm.py": "from ._impl import DFM\n",
             "pkg/user.py": "from .dfm import DFM\ndef f(d):\n    return DFM.supports(d)\n"}
    assert _calls(tmp_path, files, "f") == [("pkg/_impl.py", "DFM.supports", "qualified")]
