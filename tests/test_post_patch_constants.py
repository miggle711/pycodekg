"""A seed's own-file module constants come from the post-patch focal file
(miggle711/pycodekg#114, decided for the rerun on 13 Nov).

The KG is built at the base commit, so its file node holds pre-patch
constants. The seed's source is already refreshed to post-patch; its
constants now follow the same rule, through an overlay on the extracted
context. A constant the patch adds, changes or removes is shown as the
post-patch file has it, the cached KG is not changed, and a focal file that
can't be reconstructed or parsed gives no constants section (reported),
never the pre-patch values presented as current.
"""

import sys
from pathlib import Path

from kg_construction.extraction.context import TestContextExtractor
from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.llm.llm_serializer import LLMSerializer

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_kg_prompts import _build_context  # noqa: E402

BASE = (
    "LIMIT = 10\n"
    "OLD = 1\n"
    "GONE = 5\n"
    "\n"
    "def f(x):\n"
    "    return x < LIMIT + OLD + GONE\n"
)
# Changes LIMIT, keeps OLD, removes GONE and adds NEW, which f now uses.
PATCH = (
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,6 +1,6 @@\n"
    "-LIMIT = 10\n"
    "+LIMIT = 20\n"
    " OLD = 1\n"
    "-GONE = 5\n"
    "+NEW = 3\n"
    " \n"
    " def f(x):\n"
    "-    return x < LIMIT + OLD + GONE\n"
    "+    return x < LIMIT + OLD + NEW\n"
)


class _LocalFileRepoManager(RepoManager):
    def __init__(self, repo_dir: Path):
        self._repo_dir = repo_dir

    def read_file_at_commit(self, repo: str, commit: str, path: str) -> str:
        return (self._repo_dir / path).read_text()


def _extract(tmp_path, patch):
    (tmp_path / "mod.py").write_text(BASE)
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    engine = KGQueryEngine(kg)
    extractor = TestContextExtractor(engine, repo_manager=_LocalFileRepoManager(tmp_path))
    instance = {"repo": "test/repo", "base_commit": "deadbeef", "patch": patch,
                "code_file": "mod.py", "test_file": "test_mod.py"}
    return engine, extractor.extract(instance, depth=2)


def _serialize(engine, context):
    """The same context dict build_kg_prompts.py passes to the serializer."""
    return LLMSerializer().serialize({
        "seeds": context.seeds, "context_nodes": context.context_nodes,
        "edges": context.edges, "test_nodes": context.test_nodes,
        "file_nodes": engine.find_file_by_path("mod.py", exact=True),
        "post_patch_constants": context.post_patch_constants,
        "constants_unavailable": context.constants_unavailable,
    })


def _file_node_constants(engine):
    return engine.find_file_by_path("mod.py", exact=True)[0]["metadata"]["constants"]


def test_added_changed_and_removed_constants_follow_the_post_patch_file(tmp_path):
    engine, context = _extract(tmp_path, PATCH)
    assert context.constants_unavailable == {}
    assert context.post_patch_constants == {"mod.py": {"LIMIT": "20", "OLD": "1", "NEW": "3"}}
    serialized = _serialize(engine, context)
    assert serialized["context"]["module_constants"] == [
        {"module": "mod", "filepath": "mod.py", "source_code": "LIMIT = 20\nNEW = 3\nOLD = 1"}]
    prompt = _build_context(serialized)
    assert "LIMIT = 20" in prompt and "NEW = 3" in prompt and "OLD = 1" in prompt
    assert "LIMIT = 10" not in prompt and "GONE" not in prompt


def test_cached_kg_keeps_its_pre_patch_constants(tmp_path):
    engine, context = _extract(tmp_path, PATCH)
    _serialize(engine, context)
    assert _file_node_constants(engine) == {"LIMIT": "10", "OLD": "1", "GONE": "5"}


def test_without_the_overlay_the_pre_patch_values_would_show(tmp_path):
    """What the draft did before this change: the stale case this fixes."""
    engine, context = _extract(tmp_path, PATCH)
    serialized = LLMSerializer().serialize({
        "seeds": context.seeds, "context_nodes": [], "edges": [], "test_nodes": [],
        "file_nodes": engine.find_file_by_path("mod.py", exact=True)})
    assert serialized["context"]["module_constants"][0]["source_code"] == "LIMIT = 10\nOLD = 1"


def test_unreconstructable_focal_file_gives_no_constants_and_says_why(tmp_path):
    bad_patch = PATCH.replace("-LIMIT = 10\n", "-LIMIT = 99\n")   # not what the file has
    engine, context = _extract(tmp_path, bad_patch)
    assert context.post_patch_constants == {}
    assert "could not be reconstructed" in context.constants_unavailable["mod.py"]
    serialized = _serialize(engine, context)
    assert serialized["context"]["module_constants"] == []
    assert serialized["context"]["module_constants_unavailable"] == [
        {"filepath": "mod.py", "reason": context.constants_unavailable["mod.py"]}]
    prompt = _build_context(serialized)
    assert "Module constants" not in prompt and "LIMIT = 10" not in prompt


def test_overlay_without_the_seeds_file_reports_it_instead_of_using_the_kg():
    seed = {"id": "s", "type": "function", "label": "f",
            "metadata": {"filepath": "mod.py", "source_code": "def f():\n    return LIMIT\n"}}
    file_node = {"id": "file", "type": "file", "metadata": {"path": "mod.py", "constants": {"LIMIT": "10"}}}
    serialized = LLMSerializer().serialize({"seeds": [seed], "context_nodes": [], "edges": [],
                                            "test_nodes": [], "file_nodes": [file_node],
                                            "post_patch_constants": {"other.py": {}}})
    assert serialized["context"]["module_constants"] == []
    assert serialized["context"]["module_constants_unavailable"] == [
        {"filepath": "mod.py", "reason": "no post-patch constants for this file"}]


def test_post_patch_file_constants_reports_parse_failures():
    from kg_construction.extraction.context import post_patch_file_constants
    assert post_patch_file_constants("LIMIT = 1\nX = (\n")[0] is None
    assert "does not parse" in post_patch_file_constants("LIMIT = 1\nX = (\n")[1]
    assert post_patch_file_constants(None) == (
        None, "post-patch focal file could not be reconstructed from the patch")
    assert post_patch_file_constants("LIMIT = 1\nlower = 2\n") == ({"LIMIT": "1"}, None)
