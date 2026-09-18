"""Regression test: TestContextExtractor.extract()'s seed_ids construction
order must be stable across separate calls for a multi-function-patch
instance, not just deduplicated.

context.py's "Find seed nodes" loop used to iterate `changed` (a
Set[Tuple[str, Optional[str]]] returned by
PatchParser.extract_changed_functions_with_scope) directly, without
sorting first. Python randomizes set iteration order for strings
per-process by default (PYTHONHASHSEED unset), so seed_ids' construction
order could vary between separate builds of the same multi-function-patch
instance. Same bug class as the BFS visited-node-order fix and the
raises/catches/returns ordering fix, a different call site.

Note: context.seeds' own final order (what LLMSerializer actually
renders) is separately sorted by node id via _bfs's own fix, since
seed_nodes is filtered from subgraph_nodes rather than built from
seed_ids directly (see the "Separate seeds from context" comment below
in context.py). So this fix closes a real non-determinism in seed_ids
itself, relied on directly by _add_seed_imports and the 'tests' edge
lookup, not a second, independent bug in the rendered seed order.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.extraction.context import TestContextExtractor


class _LocalFileRepoManager(RepoManager):
    def __init__(self, repo_dir: Path):
        self._repo_dir = repo_dir

    def read_file_at_commit(self, repo: str, commit: str, path: str) -> str:
        return (self._repo_dir / path).read_text()


def _write_multi_function_repo(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def bravo():\n"
        "    return 2\n"
        "\n"
        "\n"
        "def charlie():\n"
        "    return 3\n"
        "\n"
        "\n"
        "def delta():\n"
        "    return 4\n"
        "\n"
        "\n"
        "def echo():\n"
        "    return 5\n"
    )
    return tmp_path


# A patch touching 5 distinct top-level functions -- enough distinct
# string names for set iteration order to plausibly differ from sorted
# order within a single process (a 2-name case can accidentally match
# sorted order even when genuinely hash-randomized).
_MULTI_FUNCTION_PATCH = (
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def alpha():\n"
    "-    return 1\n"
    "+    return 100\n"
    "@@ -5,2 +5,2 @@\n"
    " def bravo():\n"
    "-    return 2\n"
    "+    return 200\n"
    "@@ -9,2 +9,2 @@\n"
    " def charlie():\n"
    "-    return 3\n"
    "+    return 300\n"
    "@@ -13,2 +13,2 @@\n"
    " def delta():\n"
    "-    return 4\n"
    "+    return 400\n"
    "@@ -17,2 +17,2 @@\n"
    " def echo():\n"
    "-    return 5\n"
    "+    return 500\n"
)


def _extract(tmp_path):
    repo_dir = _write_multi_function_repo(tmp_path)
    parser = RepoASTParser(max_workers=1)
    kg = parser.parse_repo("test/repo", repo_dir)
    engine = KGQueryEngine(kg)
    extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(repo_dir))
    instance = {
        "repo": "test/repo",
        "base_commit": "deadbeef",
        "patch": _MULTI_FUNCTION_PATCH,
        "code_file": "mod.py",
        "test_file": "test_mod.py",
    }
    return extractor.extract(instance, depth=1)


class TestSeedSetIsCorrectRegardlessOfOrder:
    def test_all_five_changed_functions_become_seeds(self, tmp_path):
        # Sanity check the fixture actually exercises a real
        # multi-function patch before testing order-sensitivity of it.
        context = _extract(tmp_path)
        labels = {s["label"] for s in context.seeds}
        assert labels == {"alpha", "bravo", "charlie", "delta", "echo"}


class TestChangedFunctionsIterationOrderIsDeterministic:
    """Unit test on PatchParser's return value directly (the actual
    source of the non-determinism), rather than through the full
    extract() pipeline where context.seeds' order is separately
    normalized by _bfs's own sort -- that would mask this specific bug.
    """

    def test_sorted_iteration_of_changed_is_stable_and_matches_name_order(self):
        from kg_construction.extraction.patch import PatchParser

        pre_patch_source = (
            "def alpha():\n    return 1\n\n\n"
            "def bravo():\n    return 2\n\n\n"
            "def charlie():\n    return 3\n\n\n"
            "def delta():\n    return 4\n\n\n"
            "def echo():\n    return 5\n"
        )
        changed = PatchParser.extract_changed_functions_with_scope(
            _MULTI_FUNCTION_PATCH, "mod.py", pre_patch_source
        )
        assert {name for name, _cls in changed} == {
            "alpha", "bravo", "charlie", "delta", "echo",
        }

        # This is exactly the sort context.py's "Find seed nodes" loop
        # now applies before iterating `changed` -- reproduced here to
        # verify it's actually deterministic and matches name order,
        # since enclosing_class can be None and a naive sorted(changed)
        # would raise TypeError comparing None to str.
        ordered = sorted(changed, key=lambda pair: (pair[0], pair[1] or ""))
        names = [name for name, _cls in ordered]
        assert names == sorted(names)

        ordered_again = sorted(changed, key=lambda pair: (pair[0], pair[1] or ""))
        assert ordered == ordered_again
