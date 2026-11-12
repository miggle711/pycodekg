"""Tests for the scriptable CLI (kg_construction#83).

Covers the argparse wiring, build --commit vs. local-directory dispatch,
and query subcommand output -- not a full end-to-end subprocess test,
since build --commit needs a real git clone; that path is already covered
by RepoKGBuilder.build's own tests.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from kg_construction.cli import build_parser, _cmd_build, _cmd_query
from kg_construction.kg.builder import RepoASTParser


def _write_repo(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(
        "class Widget:\n"
        "    def build(self):\n"
        "        return self.helper()\n"
        "\n"
        "    def helper(self):\n"
        "        return 42\n"
        "\n"
        "\n"
        "def standalone():\n"
        "    return Widget().build()\n"
    )
    (tmp_path / "test_mod.py").write_text(
        "def test_helper():\n"
        "    assert Widget().helper() == 42\n"
    )
    return tmp_path


class TestBuildParser:
    def test_build_subcommand_with_commit(self):
        parser = build_parser()
        args = parser.parse_args(["build", "psf/requests", "--commit", "abc123"])
        assert args.command == "build"
        assert args.repo == "psf/requests"
        assert args.commit == "abc123"

    def test_build_subcommand_local_dir(self):
        parser = build_parser()
        args = parser.parse_args(["build", ".", "--name", "my-repo"])
        assert args.repo == "."
        assert args.commit is None
        assert args.name == "my-repo"

    def test_query_subcommand_flags(self):
        parser = build_parser()
        args = parser.parse_args(["query", "kg.json", "--callers", "send", "--json"])
        assert args.kg_file == "kg.json"
        assert args.callers == "send"
        assert args.json is True

    def test_query_subcommand_min_confidence_flag(self):
        parser = build_parser()
        args = parser.parse_args(["query", "kg.json", "--callers", "send", "--min-confidence", "exact"])
        assert args.min_confidence == "exact"
        assert parser.parse_args(["query", "kg.json", "--callers", "send"]).min_confidence is None

    def test_query_subcommand_class_flag_uses_cls_dest(self):
        parser = build_parser()
        args = parser.parse_args(["query", "kg.json", "--class", "Session"])
        assert args.cls == "Session"

    def test_no_command_is_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestCmdBuild:
    def test_commit_given_uses_clone_by_sha_path(self):
        parser = build_parser()
        args = parser.parse_args(["build", "psf/requests", "--commit", "abc123"])

        with patch("kg_construction.cli.RepoKGBuilder") as MockBuilder:
            instance = MockBuilder.return_value
            instance.build.return_value = {"nodes": [], "edges": [], "metadata": {}}
            _cmd_build(args)

            instance.build.assert_called_once_with("psf/requests", "abc123")
            instance.save.assert_called_once()
            instance.build_from_dir.assert_not_called()

    def test_no_commit_uses_local_dir_path(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["build", str(repo_dir), "--name", "my-repo"])

        with patch("kg_construction.cli.RepoKGBuilder") as MockBuilder:
            instance = MockBuilder.return_value
            instance.build_from_dir.return_value = {"nodes": [], "edges": [], "metadata": {}}
            _cmd_build(args)

            instance.build_from_dir.assert_called_once_with("my-repo", repo_dir)
            instance.save_local.assert_called_once()
            instance.build.assert_not_called()


class TestCmdQuery:
    def _kg_path(self, tmp_path) -> Path:
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)
        kg_path = tmp_path / "kg.json"
        kg_path.write_text(json.dumps(kg))
        return kg_path

    def test_callers_prints_human_readable_by_default(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--callers", "helper"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        assert "build" in out
        assert "mod.py" in out

    def test_callers_json_output_is_valid_json(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--callers", "helper", "--json"])

        _cmd_query(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert {n["label"] for n in parsed} == {"build", "test_helper"}

    def test_callees_lists_functions_called_by_the_target(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--callees", "build"])

        _cmd_query(args)
        out = capsys.readouterr().out
        assert "helper" in out

    def test_file_lists_classes_and_functions(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--file", "mod.py"])

        _cmd_query(args)
        out = capsys.readouterr().out
        assert "Widget" in out
        assert "standalone" in out

    def test_unknown_function_name_returns_error_exit_code(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--callers", "does_not_exist"])

        result = _cmd_query(args)
        assert result == 1

    def test_no_query_flag_given_returns_error_exit_code(self, tmp_path):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path)])

        result = _cmd_query(args)
        assert result == 1

    def test_tests_lists_test_functions_calling_the_target(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--tests", "helper"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        assert "test_helper" in out

    def test_tests_reports_no_tests_found(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--tests", "standalone"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        assert "No existing tests found" in out

    def test_class_lists_methods_of_the_class(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--class", "Widget"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        assert "build" in out
        assert "helper" in out

    def test_class_unknown_name_returns_error_exit_code(self, tmp_path):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--class", "DoesNotExist"])

        result = _cmd_query(args)
        assert result == 1

    def test_list_files_lists_all_files(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--list-files"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        assert "mod.py" in out
        assert "test_mod.py" in out

    def test_list_functions_lists_all_functions_and_methods(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--list-functions"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        for name in ("build", "helper", "standalone", "test_helper"):
            assert name in out

    def test_list_classes_lists_all_classes(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--list-classes"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        assert result == 0
        assert "Widget" in out

    def test_export_produces_subgraph_json_for_named_functions(self, tmp_path, capsys):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--export", "build,helper"])

        result = _cmd_query(args)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert result == 0
        assert "nodes" in parsed and "edges" in parsed
        labels = {n["label"] for n in parsed["nodes"]}
        assert "build" in labels
        assert "helper" in labels

    def test_export_unknown_function_returns_error_exit_code(self, tmp_path):
        kg_path = self._kg_path(tmp_path)
        parser = build_parser()
        args = parser.parse_args(["query", str(kg_path), "--export", "does_not_exist"])

        result = _cmd_query(args)
        assert result == 1
