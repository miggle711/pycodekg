"""Framework test utilities are production files, unlike a project's tests (#168)."""

import pytest

from kg_construction.kg.builder import RepoASTParser, _parse_file


@pytest.mark.parametrize("relative_path, expected_type", [
    ("django/test/testcases.py", "file"),
    ("django/test/utils.py", "file"),
    ("sympy/testing/runtests.py", "file"),
    ("tests/helpers.py", "test_file"),
    ("pkg/tests/helpers.py", "test_file"),
    ("test/helpers.py", "test_file"),
    ("testing/helpers.py", "test_file"),
    ("pkg/test_widget.py", "test_file"),
    ("pkg/widget_test.py", "test_file"),
    ("conftest.py", "test_file"),
    ("pkg/conftest.py", "test_file"),
    ("django/test/test_utils.py", "test_file"),
])
def test_file_node_classification(tmp_path, relative_path, expected_type):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def helper(value):\n    assert value\n")

    raw = _parse_file(("test/repo", relative_path, str(path)))
    file_node = next(n for n in raw["nodes"] if n["type"] in {"file", "test_file"})
    assert file_node["type"] == expected_type
    helper = next(n for n in raw["nodes"] if n["label"] == "helper")
    # Classification also governs extraction of test assertion patterns.
    assert bool(helper["metadata"]["assert_patterns"]) == (expected_type == "test_file")


def test_framework_definitions_remain_visible_to_production_code(tmp_path):
    files = {
        "django/test/testcases.py": "class TestCase: pass\n",
        "django/test/utils.py": "def setup_environment(): pass\n",
        "tests/helpers.py": "def setup_fixture(): pass\n",
        "app.py": (
            "from django.test.testcases import TestCase\n"
            "from django.test.utils import setup_environment\n"
            "from tests.helpers import setup_fixture\n"
            "class ApplicationCase(TestCase): pass\n"
            "def configure():\n"
            "    setup_environment()\n"
            "    setup_fixture()\n"
        ),
    }
    for relative_path, source in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    graph = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in graph["nodes"]}
    file_types = {n["metadata"]["path"]: n["type"] for n in graph["nodes"]
                  if n["type"] in {"file", "test_file"}}
    assert file_types == {"django/test/testcases.py": "file", "django/test/utils.py": "file",
                          "tests/helpers.py": "test_file", "app.py": "file"}
    links = {(nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"])
             for e in graph["edges"] if e["relation"] in {"calls", "inherits"}}
    assert ("ApplicationCase", "inherits", "TestCase") in links
    assert ("configure", "calls", "setup_environment") in links
    assert ("configure", "calls", "setup_fixture") not in links
