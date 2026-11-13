"""build_kg_prompts.py --without-module-constants renders the same prompts
minus the module constants section, from the same KGs and code, so the
section's effect (e.g. on the token budget) can be measured on its own."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "build_kg_prompts_flag", Path(__file__).parent.parent / "scripts" / "build_kg_prompts.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def _run(tmp_path, monkeypatch, *extra):
    rows = [{"id": "example__repo-1-0", "repo": "example/repo", "base_commit": "12345678",
             "patch": "patch", "code_file": "module.py", "test_file": "tests/test_module.py"}]
    datasets = ModuleType("datasets")
    datasets.load_from_disk = lambda path: {"test": rows}
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    (tmp_path / "kg_example_repo_12345678.json").write_text(json.dumps({"nodes": [], "edges": []}))

    class Extractor:
        def __init__(self, engine):
            pass

        def extract(self, instance, depth):
            return SimpleNamespace(
                seeds=[{"id": "f", "type": "function", "label": "f", "metadata": {
                    "filepath": "module.py", "source_code": "def f(x):\n    return x < LIMIT\n"}}],
                context_nodes=[], edges=[], test_nodes=[], retrieval_time_s=0,
                localization_outcome="exact", node_count=1, edge_count=0,
                cross_file_node_proportion=0, relationship_type_counts={}, stale_seed_labels=[],
                post_patch_constants={"module.py": {"LIMIT": "10"}}, constants_unavailable={})

    monkeypatch.setattr(builder, "TestContextExtractor", Extractor)
    output = tmp_path / f"prompts{len(extra)}.json"
    monkeypatch.setattr(sys, "argv", ["build_kg_prompts.py", "--kg-dir", str(tmp_path),
                                      "--dataset-path", "fixture", "--output", str(output), *extra])
    builder.main()
    return json.loads(output.read_text())["example__repo-1-0"]


def test_flag_removes_only_the_constants_section(tmp_path, monkeypatch):
    with_section = _run(tmp_path, monkeypatch)
    without = _run(tmp_path, monkeypatch, "--without-module-constants")
    assert "Module constants:" in with_section["context"] and "LIMIT = 10" in with_section["context"]
    assert "Module constants:" not in without["context"] and "LIMIT = 10" not in without["context"]
    section = builder._constants_section([{"module": "module", "source_code": "LIMIT = 10"}])
    assert with_section["context"].replace(section, "").replace("\n\n", "\n") == \
        without["context"].replace("\n\n", "\n")
    assert with_section["target_functions"] == without["target_functions"]
