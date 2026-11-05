"""A file that starts with a UTF-8 BOM must parse like any other (#161)."""

from kg_construction.kg.builder import _parse_file


def _functions(result):
    return {n["label"]: n for n in result["nodes"] if n["type"] == "function"}


def test_file_with_bom_produces_nodes(tmp_path):
    path = tmp_path / "mod.py"
    path.write_bytes(b"\xef\xbb\xbfdef a():\n    return 1\n")

    result = _parse_file(("test/repo", "mod.py", str(path)))

    assert result is not None
    assert any(n["type"] == "file" for n in result["nodes"])
    assert _functions(result)["a"]["metadata"]["source_code"] == "def a():\n    return 1\n"


def test_file_without_bom_is_unchanged(tmp_path):
    path = tmp_path / "mod.py"
    path.write_bytes(b"def a():\n    return 1\n")

    result = _parse_file(("test/repo", "mod.py", str(path)))

    assert _functions(result)["a"]["metadata"]["source_code"] == "def a():\n    return 1\n"
