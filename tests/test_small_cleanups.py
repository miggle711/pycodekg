"""Regression tests for a bundle of small, independent correctness fixes.

Each class covers one finding from the correctness audit:

* TestNoDuplicateTestNodes: a test that exercises two seeds was listed twice.
* TestCodeFileLookupIsExact: the extractor and the CLI picked the first
  substring match for a file path, so a vendored or build copy could win.
* TestBuildCommandValidatesPath: building from a path that does not exist
  (for example a GitHub name given without a commit) saved an empty KG.
* TestSavedContextKeepsStaleSeedLabels: save and load dropped the field.
* TestCommitMatchingAcceptsShortSha: a KG saved under a full SHA was not
  found when loaded with a short SHA, and the reverse.
* TestNewFileSeedIds: seed IDs for methods of a newly created file did not
  follow the builder's qualname scheme.
* TestPytestDoesNotCollectDomainClasses: classes named Test* that are not
  tests.
"""

import difflib
import json
from pathlib import Path
from unittest.mock import patch

from kg_construction.ast.helpers import _make_id
from kg_construction.cli import _cmd_build, _cmd_query, build_parser
from kg_construction.extraction.context import TestContext, TestContextExtractor
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.kg.builder import RepoASTParser, RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.repo_manager import RepoManager

REPO = "test/repo"


class _FakeRepoManager(RepoManager):
    def __init__(self, source: str):
        self._source = source

    def read_file_at_commit(self, repo: str, commit: str, path: str) -> str:
        return self._source


def _build(tmp_path: Path, files: dict):
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return RepoASTParser(max_workers=1).parse_repo(REPO, tmp_path)


def _patch(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))


def _extract(kg, code_file: str, source: str, patch_text: str, test_file: str = "t.py"):
    extractor = TestContextExtractor(KGQueryEngine(kg), repo_manager=_FakeRepoManager(source))
    return extractor.extract({
        "repo": REPO, "base_commit": "deadbeef", "patch": patch_text,
        "code_file": code_file, "test_file": test_file,
    }, depth=1)


class TestNoDuplicateTestNodes:
    def test_test_covering_two_seeds_is_listed_once(self, tmp_path):
        src = "def a(): return 1\ndef b(): return 2\n"
        kg = _build(tmp_path, {
            "m.py": src,
            "tests/test_m.py": "from m import a, b\ndef test_it():\n    a()\n    b()\n",
        })
        new = src.replace("return 1", "return 11").replace("return 2", "return 22")
        ctx = _extract(kg, "m.py", src, _patch("m.py", src, new), "tests/test_m.py")
        assert sorted(s["label"] for s in ctx.seeds) == ["a", "b"]
        assert [t["label"] for t in ctx.test_nodes] == ["test_it"]

    def test_distinct_tests_are_all_kept_in_stable_order(self, tmp_path):
        src = "def a(): return 1\n"
        kg = _build(tmp_path, {
            "m.py": src,
            "tests/test_m.py": (
                "from m import a\n"
                "def test_one():\n    a()\n"
                "def test_two():\n    a()\n"
            ),
        })
        new = src.replace("return 1", "return 11")
        ctx = _extract(kg, "m.py", src, _patch("m.py", src, new), "tests/test_m.py")
        labels = [t["label"] for t in ctx.test_nodes]
        assert sorted(labels) == ["test_one", "test_two"]
        assert len(labels) == len(set(labels))


class TestCodeFileLookupIsExact:
    def test_fallback_seed_is_the_real_file_not_a_build_copy(self, tmp_path):
        # The change is to a module level line, so no function resolves and
        # the extractor falls back to seeding the code file node itself.
        src = "LIMIT = 1\n\n\ndef a():\n    return 1\n"
        kg = _build(tmp_path, {"build/lib/pkg/mod.py": src, "pkg/mod.py": src})
        new = src.replace("LIMIT = 1", "LIMIT = 2")
        ctx = _extract(kg, "pkg/mod.py", src, _patch("pkg/mod.py", src, new))
        assert [s["metadata"]["path"] for s in ctx.seeds] == ["pkg/mod.py"]

    def test_cli_file_query_prefers_exact_path_over_substring(self, tmp_path, capsys):
        kg = _build(tmp_path / "repo", {
            "build/lib/pkg/mod.py": "def a(): pass\n",
            "pkg/mod.py": "def a(): pass\n",
        })
        kg_file = tmp_path / "kg.json"
        kg_file.write_text(json.dumps(kg))
        args = build_parser().parse_args(["query", str(kg_file), "--file", "pkg/mod.py", "--json"])
        assert _cmd_query(args) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["file"]["metadata"]["path"] == "pkg/mod.py"

    def test_cli_file_query_still_accepts_a_partial_path(self, tmp_path, capsys):
        kg = _build(tmp_path / "repo", {"pkg/mod.py": "def a(): pass\n"})
        kg_file = tmp_path / "kg.json"
        kg_file.write_text(json.dumps(kg))
        args = build_parser().parse_args(["query", str(kg_file), "--file", "mod.py", "--json"])
        assert _cmd_query(args) == 0
        assert json.loads(capsys.readouterr().out)["file"]["metadata"]["path"] == "pkg/mod.py"


class TestBuildCommandValidatesPath:
    def _run(self, argv):
        args = build_parser().parse_args(argv)
        with patch("kg_construction.cli.RepoKGBuilder") as builder_cls:
            instance = builder_cls.return_value
            instance.build_from_dir.return_value = {"nodes": [], "edges": [], "metadata": {}}
            code = _cmd_build(args)
        return code, instance

    def test_missing_path_fails_and_saves_nothing(self, tmp_path, capsys):
        code, instance = self._run(["build", str(tmp_path / "nope")])
        assert code != 0
        assert "does not exist" in capsys.readouterr().err
        instance.build_from_dir.assert_not_called()
        instance.save_local.assert_not_called()

    def test_owner_name_without_commit_hints_at_the_commit_option(self, capsys):
        code, instance = self._run(["build", "psf/requests"])
        assert code != 0
        err = capsys.readouterr().err
        assert "commit" in err
        instance.save_local.assert_not_called()

    def test_file_instead_of_directory_fails(self, tmp_path, capsys):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        code, instance = self._run(["build", str(f)])
        assert code != 0
        assert "not a directory" in capsys.readouterr().err
        instance.save_local.assert_not_called()

    def test_directory_with_no_python_files_fails_and_saves_nothing(self, tmp_path, capsys):
        d = tmp_path / "empty"
        d.mkdir()
        args = build_parser().parse_args(["build", str(d)])
        with patch("kg_construction.cli.RepoKGBuilder") as builder_cls:
            instance = builder_cls.return_value
            instance.build_from_dir.return_value = {
                "nodes": [], "edges": [], "metadata": {"file_count": 0}}
            code = _cmd_build(args)
        assert code != 0
        assert "no Python files" in capsys.readouterr().err
        instance.save_local.assert_not_called()

    def test_real_directory_still_builds_and_saves(self, tmp_path):
        d = tmp_path / "ok"
        d.mkdir()
        (d / "a.py").write_text("def f(): pass\n")
        args = build_parser().parse_args(["build", str(d), "--name", "ok-repo"])
        with patch("kg_construction.cli.RepoKGBuilder") as builder_cls:
            instance = builder_cls.return_value
            instance.build_from_dir.return_value = {
                "nodes": [{}], "edges": [], "metadata": {"file_count": 1}}
            code = _cmd_build(args)
        assert code == 0
        instance.save_local.assert_called_once()


class TestSavedContextKeepsStaleSeedLabels:
    def test_round_trip(self, tmp_path):
        ctx = TestContext(
            seeds=[], context_nodes=[], edges=[], test_nodes=[],
            repo="r", base_commit="c" * 8, stale_seed_labels=["x", "y"],
        )
        path = str(tmp_path / "ctx.json")
        ctx.save(path)
        assert TestContext.load(path).stale_seed_labels == ["x", "y"]

    def test_older_file_without_the_field_still_loads(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "repo": "r", "base_commit": "c" * 8, "seeds": [], "context_nodes": [],
            "edges": [], "test_nodes": [],
        }))
        assert TestContext.load(str(path)).stale_seed_labels == []


class TestCommitMatchingAcceptsShortSha:
    def _builder(self, tmp_path):
        return RepoKGBuilder(output_dir=tmp_path / "out", cache_dir=tmp_path / "rc")

    def _kg(self, commit):
        from kg_construction.kg.builder import SCHEMA_VERSION
        return {"nodes": [], "edges": [],
                "metadata": {"base_commit": commit, "schema_version": SCHEMA_VERSION}}

    def test_full_saved_short_loaded(self, tmp_path):
        b = self._builder(tmp_path)
        b.save("o/r", self._kg("a" * 40))
        assert b.load("o/r", "a" * 8) is not None

    def test_short_saved_full_loaded(self, tmp_path):
        b = self._builder(tmp_path)
        b.save("o/r", self._kg("a" * 8))
        assert b.load("o/r", "a" * 40) is not None

    def test_different_commit_with_same_first_eight_is_rejected(self, tmp_path):
        b = self._builder(tmp_path)
        b.save("o/r", self._kg("a" * 8 + "1" * 32))
        assert b.load("o/r", "a" * 8 + "2" * 32) is None

    def test_too_short_a_prefix_is_rejected(self, tmp_path):
        b = self._builder(tmp_path)
        b.save("o/r", self._kg("a" * 40))
        assert b.load("o/r", "a") is None


class TestNewFileSeedIds:
    def test_method_seed_id_matches_builder_qualname_scheme(self):
        patch_text = (
            "diff --git a/new.py b/new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+class K:\n"
            "+    def m(self):\n"
            "+        return 1\n"
        )
        kg = {"nodes": [], "edges": [], "metadata": {}}
        ctx = TestContextExtractor(KGQueryEngine(kg), repo_manager=_FakeRepoManager("")).extract({
            "repo": REPO, "base_commit": "deadbeef", "patch": patch_text,
            "code_file": "new.py", "test_file": "t.py",
        })
        method = next(s for s in ctx.seeds if s["label"] == "m")
        assert method["id"] == _make_id(f"func_{REPO}_new.py_K.m")


class TestPytestDoesNotCollectDomainClasses:
    def test_marked_as_not_tests(self):
        assert TestContext.__test__ is False
        assert TestContextExtractor.__test__ is False
        assert TestContextValidator.__test__ is False
