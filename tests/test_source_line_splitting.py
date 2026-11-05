"""Source must split into the lines `ast` numbers: after \\n only (#157).

str.splitlines also splits on form feeds and other separators, so every
function after one in a file was sliced from the wrong lines.
"""

import pytest

from kg_construction.ast.helpers import split_source_lines
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.kg.builder import RepoASTParser, _parse_file
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.kg.query import KGQueryEngine


class _FileRepoManager(RepoManager):
    def __init__(self, repo_dir):
        super().__init__()
        self._repo_dir = repo_dir

    def read_file_at_commit(self, repo, commit, path):
        return (self._repo_dir / path).read_text()


@pytest.mark.parametrize("source, expected", [
    ("", []),
    ("a", ["a"]),
    ("a\n", ["a\n"]),
    ("a\nb", ["a\n", "b"]),
    ("a\n\nb\n", ["a\n", "\n", "b\n"]),
    ("a\x0c\nb\n", ["a\x0c\n", "b\n"]),
])
def test_split_source_lines(source, expected):
    assert split_source_lines(source) == expected


@pytest.mark.parametrize("separator", ["\x0c", "\x0b", "\x1c", "\x1d", "\x1e", "\x85", " ", " "])
def test_only_newlines_split(separator):
    assert split_source_lines(f"x = '{separator}'\ny = 1\n") == [f"x = '{separator}'\n", "y = 1\n"]


def _function(result, label):
    return next(n for n in result["nodes"] if n["label"] == label)


def test_function_after_a_form_feed_has_its_own_source(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("def a():\n    return 1\n\x0c\ndef b():\n    return 2\n")

    result = _parse_file(("test/repo", "mod.py", str(path)))

    metadata = _function(result, "b")["metadata"]
    assert metadata["source_code"] == "def b():\n    return 2\n"
    assert metadata["signature"] == "def b():"


def test_function_after_a_separator_in_a_string_has_its_own_source(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("S = 'a b'\n\n\ndef b():\n    return 2\n", encoding="utf-8")

    result = _parse_file(("test/repo", "mod.py", str(path)))

    assert _function(result, "b")["metadata"]["source_code"] == "def b():\n    return 2\n"


def _extract(repo_dir, patch, code_file):
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", repo_dir)
    extractor = TestContextExtractor(KGQueryEngine(kg), repo_manager=_FileRepoManager(repo_dir))
    return extractor.extract({
        "repo": "test/repo", "base_commit": "deadbeef", "patch": patch,
        "code_file": code_file, "test_file": "test_mod.py",
    }, depth=2)


def _seed_source(context, label):
    return next(n for n in context.seeds if n["label"] == label)["metadata"]["source_code"]


def test_post_patch_seed_source_after_a_form_feed(tmp_path):
    (tmp_path / "mod.py").write_text("def a():\n    return 1\n\x0c\ndef b():\n    return 2\n")
    patch = (
        "--- a/mod.py\n+++ b/mod.py\n@@ -3,3 +3,3 @@\n"
        " \x0c\n def b():\n-    return 2\n+    return 3\n"
    )

    context = _extract(tmp_path, patch, "mod.py")

    assert _seed_source(context, "b") == "def b():\n    return 3\n"


def test_new_file_seed_source_after_a_form_feed(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    patch = (
        "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n"
        "@@ -0,0 +1,5 @@\n+def a():\n+    return 1\n+\x0c\n+def b():\n+    return 2\n"
    )

    context = _extract(tmp_path, patch, "new.py")

    assert _seed_source(context, "b") == "def b():\n    return 2\n"
