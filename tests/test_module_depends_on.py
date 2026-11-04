"""Regression test for module_depends_on edge resolution across same-named files.

Covers the bug where file_label_to_id was keyed by bare filename, causing
identically-named files in different packages (pkg_a/utils.py vs
pkg_b/utils.py) to collide and misattribute module_depends_on edges to
whichever file happened to be indexed last.

TestModuleDependsOnResolution covers the other half: an import has to find its
file under a src/ layout, as a package or submodule, and through relative
imports, without linking a stdlib name to an unrelated deep file.
"""

from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser


def _write_repo(tmp_path: Path) -> Path:
    """Create a small repo with two same-named 'utils.py' files in different packages.

    pkg_a/main.py imports pkg_a.utils (should depend on pkg_a/utils.py).
    pkg_b/main.py imports pkg_b.utils (should depend on pkg_b/utils.py).
    """
    (tmp_path / "pkg_a").mkdir()
    (tmp_path / "pkg_b").mkdir()

    (tmp_path / "pkg_a" / "__init__.py").write_text("")
    (tmp_path / "pkg_b" / "__init__.py").write_text("")

    (tmp_path / "pkg_a" / "utils.py").write_text(
        "def helper_a():\n    return 'a'\n"
    )
    (tmp_path / "pkg_b" / "utils.py").write_text(
        "def helper_b():\n    return 'b'\n"
    )

    (tmp_path / "pkg_a" / "main.py").write_text(
        "from pkg_a.utils import helper_a\n\n"
        "def run():\n    return helper_a()\n"
    )
    (tmp_path / "pkg_b" / "main.py").write_text(
        "from pkg_b.utils import helper_b\n\n"
        "def run():\n    return helper_b()\n"
    )

    return tmp_path


class TestModuleDependsOnPathCollision:
    """module_depends_on edges must resolve to the correct same-named file."""

    def test_same_named_files_resolve_independently(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        nodes_by_id = {n["id"]: n for n in kg["nodes"]}

        def file_id_for(rel_path: str) -> str:
            matches = [
                n["id"] for n in kg["nodes"]
                if n["type"] == "file" and n["metadata"].get("path") == rel_path
            ]
            assert len(matches) == 1, f"expected exactly one node for {rel_path}"
            return matches[0]

        main_a_id = file_id_for("pkg_a/main.py")
        main_b_id = file_id_for("pkg_b/main.py")
        utils_a_id = file_id_for("pkg_a/utils.py")
        utils_b_id = file_id_for("pkg_b/utils.py")

        module_depends_on = [
            e for e in kg["edges"] if e["relation"] == "module_depends_on"
        ]

        # pkg_a/main.py must depend on pkg_a/utils.py, not pkg_b/utils.py
        assert any(
            e["source"] == main_a_id and e["target"] == utils_a_id
            for e in module_depends_on
        ), "pkg_a/main.py should depend on pkg_a/utils.py"
        assert not any(
            e["source"] == main_a_id and e["target"] == utils_b_id
            for e in module_depends_on
        ), "pkg_a/main.py must not depend on pkg_b/utils.py"

        # pkg_b/main.py must depend on pkg_b/utils.py, not pkg_a/utils.py
        assert any(
            e["source"] == main_b_id and e["target"] == utils_b_id
            for e in module_depends_on
        ), "pkg_b/main.py should depend on pkg_b/utils.py"
        assert not any(
            e["source"] == main_b_id and e["target"] == utils_a_id
            for e in module_depends_on
        ), "pkg_b/main.py must not depend on pkg_a/utils.py"


def _module_edges(tmp_path: Path, files: dict) -> set:
    """{(importing path, imported path)} of the module_depends_on edges."""
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    paths = {n["id"]: n["metadata"]["path"] for n in kg["nodes"] if n["type"] in ("file", "test_file")}
    return {(paths[e["source"]], paths[e["target"]])
            for e in kg["edges"] if e["relation"] == "module_depends_on"}


class TestModuleDependsOnResolution:
    """Imports resolve to files under src/ layouts, as packages and submodules,
    and through relative imports."""

    @pytest.mark.parametrize("files, expected", [
        pytest.param(
            {"src/pkg/__init__.py": "", "src/pkg/b.py": "", "src/pkg/a.py": "from pkg.b import f\n"},
            {("src/pkg/a.py", "src/pkg/b.py")}, id="src-layout"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/b.py": "", "pkg/a.py": "from pkg import b\n"},
            {("pkg/a.py", "pkg/__init__.py"), ("pkg/a.py", "pkg/b.py")}, id="submodule"),
        pytest.param(
            {"pkg/__init__.py": "", "main.py": "import pkg\n"},
            {("main.py", "pkg/__init__.py")}, id="package"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/b.py": "", "pkg/a.py": "from .b import f\n"},
            {("pkg/a.py", "pkg/b.py")}, id="relative-module"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/b.py": "", "pkg/a.py": "from . import b\n"},
            {("pkg/a.py", "pkg/__init__.py"), ("pkg/a.py", "pkg/b.py")}, id="relative-submodule"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/x.py": "", "pkg/sub/__init__.py": "",
             "pkg/sub/a.py": "from ..x import y\n"},
            {("pkg/sub/a.py", "pkg/x.py")}, id="relative-two-levels"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/feature.py": "", "pkg/feature/__init__.py": "",
             "main.py": "import pkg.feature\n"},
            {("main.py", "pkg/feature/__init__.py")}, id="package-wins-over-module"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/feature.py": "", "pkg/feature/__init__.py": "",
             "pkg/a.py": "from . import feature\n"},
            {("pkg/a.py", "pkg/__init__.py"), ("pkg/a.py", "pkg/feature/__init__.py")},
            id="package-wins-over-module-relative"),
    ])
    def test_import_links_to_the_files_it_depends_on(self, tmp_path, files, expected):
        assert _module_edges(tmp_path, files) == expected

    @pytest.mark.parametrize("files", [
        pytest.param(
            {"pkg/__init__.py": "", "pkg/utils/__init__.py": "", "pkg/utils/json.py": "",
             "pkg/a.py": "import json\n"}, id="top-level-name-inside-a-package"),
        pytest.param(
            {"tools/json.py": "", "main.py": "import json\n"}, id="stdlib-name-under-a-plain-directory"),
        pytest.param(
            {"src/pkg/__init__.py": "", "src/pkg/b.py": "", "lib/pkg/__init__.py": "",
             "lib/pkg/b.py": "", "main.py": "from pkg.b import x\n"}, id="two-source-roots"),
        pytest.param(
            {"src/pkg/__init__.py": "", "src/pkg/b.py": "", "lib/pkg/__init__.py": "",
             "lib/pkg/b.py": "", "b.py": "", "main.py": "import pkg.b\n"},
            id="ambiguous-roots-do-not-fall-back-to-a-root-file"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/a.py": "from ...x import y\n"}, id="relative-beyond-the-repo"),
        pytest.param(
            {"src/pkg/__init__.py": "", "src/outside.py": "", "src/pkg/a.py": "from ..outside import x\n"},
            id="relative-above-the-top-level-package"),
        pytest.param(
            {"pkg/__init__.py": "", "pkg/utils/missing_external.py": "",
             "main.py": "import missing_external\n"}, id="plain-directory-inside-a-package"),
    ])
    def test_no_edge_when_the_target_is_not_the_repos_module(self, tmp_path, files):
        assert _module_edges(tmp_path, files) == set()
