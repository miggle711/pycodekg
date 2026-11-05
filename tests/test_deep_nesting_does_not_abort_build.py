"""One very deeply nested file must not abort a whole build."""

from kg_construction.kg.builder import RepoASTParser


def test_a_too_deep_file_is_skipped_and_the_rest_is_built(tmp_path):
    terms = " + ".join(["x"] * 1500)
    (tmp_path / "deep.py").write_text(f"def deep(x):\n    return {terms}\n")
    (tmp_path / "fine.py").write_text("def fine():\n    return 1\n")

    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)

    labels = {n["label"] for n in kg["nodes"]}
    assert "fine" in labels
    assert "deep" not in labels
