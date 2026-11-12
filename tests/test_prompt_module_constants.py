"""Only referenced seed-file constants reach the optional prompt section."""

import sys
from pathlib import Path

import pytest

from kg_construction.kg.builder import _parse_file
from kg_construction.llm.llm_serializer import LLMSerializer

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_kg_prompts import _build_context  # noqa: E402


def _serialize(tmp_path, source, *, file_in_context=True):
    path = tmp_path / "mod.py"
    path.write_text(source)
    raw = _parse_file(("test/repo", "mod.py", str(path)))
    seed = next(n for n in raw["nodes"] if n["label"] == "f")
    file_node = next(n for n in raw["nodes"] if n["type"] == "file")
    context = {"seeds": [seed], "edges": [], "test_nodes": [],
               "context_nodes": [file_node] if file_in_context else [],
               "file_nodes": [] if file_in_context else [file_node]}
    return LLMSerializer().serialize(context)


@pytest.mark.parametrize("file_in_context", [True, False], ids=["bfs_file", "depth_zero_file"])
def test_only_referenced_constants_are_rendered(tmp_path, file_in_context):
    result = _serialize(tmp_path, "LIMIT = 10\nUNUSED = 20\ndef f(x):\n    return x < LIMIT\n",
                        file_in_context=file_in_context)
    assert result["context"]["module_constants"] == [
        {"module": "mod", "filepath": "mod.py", "source_code": "LIMIT = 10"}]
    prompt = _build_context(result)
    assert "Module constants" in prompt
    assert "LIMIT = 10" in prompt
    assert "UNUSED" not in prompt


@pytest.mark.parametrize("body", [
    '    return "LIMIT"  # LIMIT\n',
    '    return obj.LIMIT\n',
    '    LIMIT = 2\n    return LIMIT\n',
    '    return [LIMIT for LIMIT in range(3)]\n',
])
def test_non_global_references_do_not_add_constants(tmp_path, body):
    result = _serialize(tmp_path, "LIMIT = 10\ndef f(obj=None):\n" + body)
    assert result["context"]["module_constants"] == []
    assert "Module constants" not in _build_context(result)


def test_section_is_absent_without_module_constants(tmp_path):
    result = _serialize(tmp_path, "def f():\n    return 1\n")
    assert result["context"]["module_constants"] == []
    assert "Module constants" not in _build_context(result)


def test_long_value_is_explicitly_truncated(tmp_path):
    from kg_construction.llm.llm_serializer import MAX_CONSTANT_VALUE_CHARS

    result = _serialize(tmp_path, f"LONG = {'a' * 500!r}\ndef f():\n    return LONG\n")
    code = result["context"]["module_constants"][0]["source_code"]
    assert "# value truncated" in code
    assert "a" * (MAX_CONSTANT_VALUE_CHARS + 1) not in code
    assert "# value truncated" in _build_context(result)


def test_constants_are_from_own_file_and_deduplicated_across_seeds():
    seed = {"id": "f", "label": "f", "type": "function", "metadata": {
        "filepath": "own.py", "source_code": "def f(): return LIMIT"}}
    other_seed = dict(seed, id="g", label="g")
    files = [{"id": path, "type": "file", "metadata": {
        "path": path, "constants": {"LIMIT": value}}}
        for path, value in [("own.py", "10"), ("other.py", "999")]]
    result = LLMSerializer().serialize({"seeds": [seed, other_seed], "context_nodes": files})
    assert result["context"]["module_constants"] == [
        {"module": "own", "filepath": "own.py", "source_code": "LIMIT = 10"}]


def test_annotated_seed_keeps_source_and_docstring_without_dead_type_hints(tmp_path):
    source = 'def f(x: int) -> int:\n    """Return the input."""\n    return x\n'
    result = _serialize(tmp_path, source)
    assert "type_hints" not in result["seed"][0]
    assert "type_hints" not in result["context"]["patterns"]
    prompt = _build_context(result)
    assert "Docstring: Return the input." in prompt
    assert "def f(x: int) -> int:" in prompt


def test_global_reference_inside_closure_is_included(tmp_path):
    result = _serialize(tmp_path, "LIMIT = 10\ndef f():\n    def inner():\n        return LIMIT\n    return inner\n")
    assert result["context"]["module_constants"][0]["source_code"] == "LIMIT = 10"
