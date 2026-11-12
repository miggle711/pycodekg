"""A high BFS depth must not silently promise a deeper generated prompt."""

import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "depth_warning_prompt_builder", Path(__file__).resolve().parents[1] / "scripts/build_kg_prompts.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


@pytest.mark.parametrize("depth", [None, 1, 2, 3, 4], ids=["default", "one", "two", "three", "four"])
def test_cli_warns_once_for_the_run_and_preserves_requested_depth(depth, tmp_path, monkeypatch, caplog):
    import datasets

    rows = [{"id": f"example__repo-1-{n}", "repo": "example/repo", "base_commit": "12345678",
             "patch": "patch", "code_file": "module.py", "test_file": "tests/test_module.py"}
            for n in range(2)]
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: {"test": rows})
    (tmp_path / "kg_example_repo_12345678.json").write_text(json.dumps({"nodes": [], "edges": []}))
    seen_depths = []

    class Extractor:
        def __init__(self, engine):
            pass

        def extract(self, instance, depth):
            seen_depths.append(depth)
            return SimpleNamespace(
                seeds=[{"id": "f", "type": "function", "label": "f", "metadata": {
                    "filepath": "module.py", "source_code": "def f(): return 1"}}],
                context_nodes=[], edges=[], test_nodes=[], retrieval_time_s=0,
                localization_outcome="exact", node_count=1, edge_count=0,
                cross_file_node_proportion=0, relationship_type_counts={}, stale_seed_labels=[])

    monkeypatch.setattr(builder, "TestContextExtractor", Extractor)
    output = tmp_path / "prompts.json"
    args = ["build_kg_prompts.py", "--kg-dir", str(tmp_path), "--dataset-path", "fixture",
            "--output", str(output)]
    if depth is not None:
        args += ["--depth", str(depth)]
    monkeypatch.setattr(sys, "argv", args)
    with caplog.at_level(logging.WARNING):
        builder.main()
    warnings = [r for r in caplog.records if "depth" in r.getMessage()]
    assert len(warnings) == int(depth is not None and depth > 2)
    if warnings:
        assert "does not add prompt content" in warnings[0].getMessage()
    assert seen_depths == [2 if depth is None else depth] * 2
    prompts = json.loads(output.read_text())
    assert set(prompts) == {row["id"] for row in rows}
    for prompt in prompts.values():
        assert prompt["target_functions"] == ["f"]
        assert "def f(): return 1" in prompt.get("context", prompt.get("prompt", ""))
