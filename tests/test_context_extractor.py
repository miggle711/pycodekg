"""Regression test: TestContextExtractor.extract() must not crash.

KGQueryEngine.kg_extraction._bfs previously called self.engine.edges, an
attribute KGQueryEngine has never had (edges live at self.engine.kg['edges']).
Extract() has no test coverage anywhere, so this AttributeError was only
discovered when kg-test-generation actually tried to call it against a real
repo -- there had never been a synthetic end-to-end test exercising this path.

TestContextExtractor now fetches each instance's pre-patch source via
RepoManager (kg_construction#75: PatchParser resolves changed functions by
real ast line ranges, not the diff text alone). These tests use a stub
RepoManager that reads directly from the synthetic tmp_path repo instead of
a real git clone -- there's no real git history here, just plain files.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.extraction.context import TestContextExtractor


class _LocalFileRepoManager(RepoManager):
    """Reads a file's CURRENT content directly from repo_dir, ignoring
    repo/commit entirely -- these tests have no real git history, just a
    synthetic repo written straight to tmp_path.
    """

    def __init__(self, repo_dir: Path):
        self._repo_dir = repo_dir

    def read_file_at_commit(self, repo: str, commit: str, path: str) -> str:
        return (self._repo_dir / path).read_text()


def _write_repo(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(
        "class Widget:\n"
        "    def build(self):\n"
        "        return self.helper()\n"
        "\n"
        "    def helper(self):\n"
        "        return 42\n"
    )
    return tmp_path


class TestContextExtractorEndToEnd:
    def test_extract_does_not_crash_and_returns_seed(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     def build(self):\n"
            "-        return self.helper()\n"
            "+        return self.helper() + 1\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        assert len(context.seeds) >= 1
        seed_labels = {s["label"] for s in context.seeds}
        assert "build" in seed_labels

        # The BFS must have actually traversed edges (helper is one hop away).
        context_labels = {n["label"] for n in context.context_nodes}
        assert "helper" in context_labels


class TestSeedNeverIncludesTestFile:
    """kg_construction#54: extract() used to unconditionally add the test
    file's own node to seed_ids alongside the real target function.
    context.seeds' order comes from BFS visited-node order, not seed_ids'
    construction order, so it was non-deterministic which one landed at
    seeds[0] -- and LLMSerializer._build_seed_section trusts seeds[0]
    blindly. When the test file won that race, the LLM-augmented arm's
    entire seed section was the test file (empty signature/source_code)
    instead of the real function, discovered via kg-test-generation#49's
    investigation into prepare_body_2015's repeated collection failures.
    """

    def _write_repo_with_test_file(self, tmp_path: Path) -> Path:
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    def build(self):\n"
            "        return self.helper()\n"
            "\n"
            "    def helper(self):\n"
            "        return 42\n"
        )
        (tmp_path / "test_mod.py").write_text(
            "from mod import Widget\n"
            "\n"
            "def test_build():\n"
            "    assert Widget().build() == 42\n"
        )
        return tmp_path

    def test_seeds_never_contains_a_test_file_node(self, tmp_path):
        repo_dir = self._write_repo_with_test_file(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     def build(self):\n"
            "-        return self.helper()\n"
            "+        return self.helper() + 1\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        # Run several times -- the bug was non-deterministic (BFS visited-
        # node order), so a single run passing wouldn't rule it out.
        for _ in range(10):
            context = extractor.extract(instance, depth=2)

            seed_types = {s.get("type") for s in context.seeds}
            assert "test_file" not in seed_types

            assert context.seeds, "the real function must still be a seed"
            assert context.seeds[0]["label"] == "build"
            assert context.seeds[0].get("type") != "test_file"

    def test_test_function_still_reachable_via_tests_edge(self, tmp_path):
        """Removing the test file from seed_ids must not break test_nodes
        -- 'tests' edges are resolved by function-naming convention
        (test_<name> -> <name>), not via any relationship to the test
        file's own node, so BFS from the seed alone should still reach it.
        """
        repo_dir = self._write_repo_with_test_file(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     def build(self):\n"
            "-        return self.helper()\n"
            "+        return self.helper() + 1\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        test_labels = {t["label"] for t in context.test_nodes}
        assert "test_build" in test_labels


class TestCallsBasedTestDetection:
    """kg_construction#57's follow-up investigation: the naming-convention
    'tests' edge (test_<name> -> <name>, exact match) assumes a test's name
    mechanically derives from the function it tests -- checked directly
    against psf/requests' real test suite and found true for only 1 of 159
    test functions; the rest use descriptive names with no derivable
    relationship to the function under test, so existing_tests/test_nodes
    was empty for 21 of 22 real benchmark instances even when a real,
    directly-relevant test existed. A test that actually CALLS the target
    function is a naming-independent signal already computable from the
    ordinary 'calls' edge resolution -- these tests cover deriving 'tests'
    edges from that instead of/in addition to the naming heuristic.
    """

    def _write_repo(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    def build(self):\n"
            "        return self.helper()\n"
            "\n"
            "    def helper(self):\n"
            "        return 42\n"
            "\n"
            "    def unrelated(self):\n"
            "        return 0\n"
        )
        (tmp_path / "test_mod.py").write_text(
            "from mod import Widget\n"
            "\n"
            "def test_descriptive_name_with_no_relation_to_build():\n"
            "    assert Widget().build() == 42\n"
            "\n"
            "def test_covers_unrelated_function():\n"
            "    assert Widget().unrelated() == 0\n"
        )
        return tmp_path

    def _instance(self):
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     def build(self):\n"
            "-        return self.helper()\n"
            "+        return self.helper() + 1\n"
        )
        return {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

    def test_descriptively_named_test_is_found_via_calls_not_naming(self, tmp_path):
        repo_dir = self._write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))
        context = extractor.extract(self._instance(), depth=2)

        test_labels = {t["label"] for t in context.test_nodes}
        assert "test_descriptive_name_with_no_relation_to_build" in test_labels

    def test_unrelated_functions_test_is_not_attributed_to_the_seed(self, tmp_path):
        """A test that calls some OTHER function reachable in the subgraph
        (here, a sibling method 'unrelated') must not be misattributed as
        an existing test FOR the seed -- the same scoping bug class as
        kg-test-generation#49's 'related' list, here for test_nodes.
        """
        repo_dir = self._write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))
        context = extractor.extract(self._instance(), depth=2)

        test_labels = {t["label"] for t in context.test_nodes}
        assert "test_covers_unrelated_function" not in test_labels


class TestAmbiguousSeedNameDisambiguation:
    """kg_construction#63: a changed method name can match more than one
    class' same-named method in the same file -- found via a real
    encode/httpx patch to AsyncClient.aclose, which also matched the
    unrelated BoundAsyncStream.aclose in the same file. extract() resolves
    this via PatchParser's real ast line ranges (kg_construction#75) --
    two same-named methods on different classes have different ranges, so
    the ambiguity can't arise in the first place, unlike the old
    hunk-scanning approach's class-hint-based workaround.
    """

    def _write_repo_with_name_collision(self, tmp_path: Path) -> Path:
        (tmp_path / "mod.py").write_text(
            "class Alpha:\n"
            "    def aclose(self):\n"
            "        return 1\n"
            "\n"
            "class Beta:\n"
            "    def aclose(self):\n"
            "        return 2\n"
        )
        return tmp_path

    def test_class_hint_resolves_the_collision(self, tmp_path):
        repo_dir = self._write_repo_with_name_collision(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,4 +1,5 @@\n"
            " class Alpha:\n"
            "     def aclose(self):\n"
            "-        return 1\n"
            "+        return 10\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        assert len(context.seeds) == 1
        assert context.seeds[0]["label"] == "aclose"
        assert context.seeds[0]["metadata"].get("class") == "Alpha"

    def test_both_classes_changed_reports_both_seeds_correctly_scoped(self, tmp_path):
        """When a patch genuinely touches both same-named methods, both
        must be reported as seeds, each correctly scoped to its own class
        -- resolved unambiguously by real line range, no ambiguity to
        leave for the validator at all (unlike the old approach, where
        this exact shape needed TestContextValidator's
        _check_no_ambiguous_seed_names as a deliberate backstop).
        """
        repo_dir = self._write_repo_with_name_collision(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,7 +1,9 @@\n"
            " class Alpha:\n"
            "     def aclose(self):\n"
            "-        return 1\n"
            "+        return 10\n"
            "\n"
            " class Beta:\n"
            "     def aclose(self):\n"
            "-        return 2\n"
            "+        return 20\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        seeds_by_class = {
            (s["label"], s["metadata"].get("class")) for s in context.seeds
        }
        assert ("aclose", "Alpha") in seeds_by_class
        assert ("aclose", "Beta") in seeds_by_class


class TestExtractForNewFile:
    """kg_construction#93: a file the patch itself creates has no node in
    the base_commit-built KG. extract() must build seed nodes directly from
    the reconstructed post-patch source instead of raising or looking the
    file up in the KG.
    """

    def test_new_file_function_becomes_a_seed_with_no_context(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "diff --git a/new_mod.py b/new_mod.py\n"
            "new file mode 100644\n"
            "index 0000000..1234567\n"
            "--- /dev/null\n"
            "+++ b/new_mod.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def brand_new():\n"
            "+    return 1\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "new_mod.py",
            "test_file": "test_new_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        seed_labels = {s["label"] for s in context.seeds}
        assert "brand_new" in seed_labels
        assert context.context_nodes == []
        assert context.edges == []
        seed = next(s for s in context.seeds if s["label"] == "brand_new")
        assert seed["metadata"]["newly_created_file"] is True
        assert "return 1" in seed["metadata"]["source_code"]

    def test_new_file_class_and_method_become_seeds(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "diff --git a/new_mod.py b/new_mod.py\n"
            "new file mode 100644\n"
            "index 0000000..1234567\n"
            "--- /dev/null\n"
            "+++ b/new_mod.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+class NewImputer:\n"
            "+    def transform(self, x):\n"
            "+        return x\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "new_mod.py",
            "test_file": "test_new_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        seeds_by_label_type = {(s["label"], s["type"]) for s in context.seeds}
        assert ("NewImputer", "class") in seeds_by_label_type
        assert ("transform", "method") in seeds_by_label_type


class TestClassLevelSeedLookup:
    """kg_construction#85: a patch changing a class-body attribute (outside
    any method) has its changed entity correctly named as the class itself
    by PatchParser, but seed lookup only ever searched
    function/method/test_function node types -- silently falling through to
    the file-level fallback instead of finding the real class node.
    """

    def test_class_body_attribute_change_seeds_the_class_itself(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    template_name = 'widget.html'\n"
            "\n"
            "    def build(self):\n"
            "        return 1\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(tmp_path))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,2 +1,3 @@\n"
            " class Widget:\n"
            "+    template_name = 'widget.html'\n"
            "\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        seed_labels_types = {(s["label"], s["type"]) for s in context.seeds}
        assert ("Widget", "class") in seed_labels_types
        # The old fallback behavior (file-as-seed) must not happen now that
        # the class itself resolves correctly.
        assert "mod.py" not in {s["label"] for s in context.seeds}


class TestSeedSourceIsPostPatch:
    """kg_construction#124: the KG is built once from base_commit, with no
    patch applied, so every stored node's source_code -- including a
    seed's -- is PRE-patch content. Generated tests are evaluated against
    the real, current (POST-patch) repository state either way, so a
    seed's source_code must reflect the patch's fix, not the KG's stored
    (stale) value. Context nodes are unaffected since the patch never
    touches them.
    """

    def _write_pre_patch_repo(self, tmp_path: Path) -> Path:
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    def build(self):\n"
            "        return self.helper() + OLD_VALUE\n"
            "\n"
            "    def helper(self):\n"
            "        return 42\n"
        )
        return tmp_path

    def test_seed_source_reflects_the_patch_not_the_kg(self, tmp_path):
        # KG is built from the PRE-patch file -- same as a real KG built
        # once from base_commit, before the patch is ever applied.
        repo_dir = self._write_pre_patch_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        # _LocalFileRepoManager also reads the same pre-patch file, same
        # as RepoManager.read_file_at_commit would for a real base_commit.
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n"
            " class Widget:\n"
            "     def build(self):\n"
            "-        return self.helper() + OLD_VALUE\n"
            "+        return self.helper() + NEW_VALUE\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        seed = next(s for s in context.seeds if s["label"] == "build")
        assert "NEW_VALUE" in seed["metadata"]["source_code"]
        assert "OLD_VALUE" not in seed["metadata"]["source_code"]

        # helper is a context node (a callee), never touched by the
        # patch -- its source must stay exactly what the KG has, since
        # pre-patch and post-patch content are identical for it.
        helper = next(n for n in context.context_nodes if n["label"] == "helper")
        assert "return 42" in helper["metadata"]["source_code"]

    def test_kg_own_cached_node_is_not_mutated(self, tmp_path):
        # The fix must return a COPY of the seed node, never mutate the
        # KG's own cached node in place -- that node object is shared
        # across every extract() call that happens to touch it.
        repo_dir = self._write_pre_patch_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n"
            " class Widget:\n"
            "     def build(self):\n"
            "-        return self.helper() + OLD_VALUE\n"
            "+        return self.helper() + NEW_VALUE\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        extractor.extract(instance, depth=2)

        raw_build_node = next(
            n for n in kg["nodes"] if n["label"] == "build" and n["type"] == "method"
        )
        assert "OLD_VALUE" in raw_build_node["metadata"]["source_code"]
        assert "NEW_VALUE" not in raw_build_node["metadata"]["source_code"]

    def test_stale_seed_labels_is_empty_on_a_successful_refresh(self, tmp_path):
        repo_dir = self._write_pre_patch_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n"
            " class Widget:\n"
            "     def build(self):\n"
            "-        return self.helper() + OLD_VALUE\n"
            "+        return self.helper() + NEW_VALUE\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)
        assert context.stale_seed_labels == []

    def test_stale_seed_labels_reports_a_seed_that_could_not_be_refreshed(self, tmp_path):
        # A patch string with hunks for a DIFFERENT file than code_file --
        # reconstruct_post_patch_source has nothing to apply, so the seed
        # can't be found in a post-patch AST and must fall back to stale
        # (pre-patch) source. This should be visible on the TestContext,
        # not just silent.
        repo_dir = self._write_pre_patch_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))

        patch = (
            "--- a/other.py\n"
            "+++ b/other.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)
        # No changed functions found for mod.py in this patch, so extract()
        # falls back to the file itself as the seed (see "If no changed
        # functions found, use the code_file itself as seed" in extract()).
        # A file-type seed has no matching (label, class) key in
        # post_patch_ranges, so it must be reported as stale.
        assert context.stale_seed_labels == ["mod.py"]


class TestNestedSeedSourceIsPostPatch:
    """A closure or a class defined inside a function body is a real,
    resolvable seed too (builder.py's _emit_function recurses the same
    way for kg_construction#74's nested-scope case). _PostPatchVisitor
    must recurse into function bodies too, or a patch changing such a
    seed silently keeps stale pre-patch source -- caught in review on
    kg_construction#124's PR (#127).
    """

    def test_nested_closure_seed_reflects_the_patch(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def outer():\n"
            "    def inner():\n"
            "        return OLD_VALUE\n"
            "    return inner\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(tmp_path))

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,4 +1,4 @@\n"
            " def outer():\n"
            "     def inner():\n"
            "-        return OLD_VALUE\n"
            "+        return NEW_VALUE\n"
            "     return inner\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        seed = next(s for s in context.seeds if s["label"] == "inner")
        assert "NEW_VALUE" in seed["metadata"]["source_code"]
        assert "OLD_VALUE" not in seed["metadata"]["source_code"]
        assert context.stale_seed_labels == []
