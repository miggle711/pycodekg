import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_kg_prompts import _build_context, _related_section, _repo_slug  # noqa: E402


def _serialized(seed_overrides=None, related=None):
    seed = {
        "function_name": "foo",
        "module": "m",
        "class_name": "",
        "docstring": "",
        "source_code": "def foo(): ...",
        "exceptions": [],
    }
    if seed_overrides:
        seed.update(seed_overrides)
    return {
        "seed": [seed],
        "context": {
            "callers": [],
            "callees": [],
            "sibling_methods": [],
            "related": related or [],
        },
    }


class TestSeedBlockDocstring:
    def test_docstring_is_rendered(self):
        prompt = _build_context(_serialized({"docstring": "Does the thing."}))
        assert "Docstring: Does the thing." in prompt

    def test_missing_docstring_renders_as_none(self):
        prompt = _build_context(_serialized({"docstring": ""}))
        assert "Docstring: (none)" in prompt


class TestContextHasNoTaskWording:
    """The task wording and output format come from testgeneval's shared
    template, the same for both arms (fyp-paper#5, decision B), so the KG
    context must not carry its own."""

    def test_context_holds_the_seed_and_no_instructions(self):
        context = _build_context(_serialized())
        assert "Function under test: foo" in context
        assert "def foo(): ..." in context
        for wording in ("Your job", "Unit test Python code", "test_", "coverage"):
            assert wording not in context


class TestRelatedSection:
    def test_no_related_omits_section_entirely(self):
        prompt = _build_context(_serialized(related=[]))
        assert "Related classes" not in prompt

    def test_parent_class_includes_source_code(self):
        related = [{
            "type": "parent_class", "name": "Base", "module": "m",
            "source_code": "class Base: ...", "source": "seed",
        }]
        result = _related_section(related)
        assert "Base (m), parent class of the seed" in result
        assert "class Base: ..." in result

    def test_instantiation_includes_its_outline(self):
        related = [{
            "type": "instantiation", "name": "Helper", "module": "m",
            "source_code": "class Helper:\n    def __init__(self, x): ...", "source": "seed_class",
        }]
        result = _related_section(related)
        assert "Helper (m), instantiated by the seed's class:\n```python\nclass Helper:" in result

    def test_class_without_source_has_no_empty_code_block(self):
        related = [{
            "type": "parent_class", "name": "Base", "module": "m",
            "source_code": "", "source": "seed",
        }]
        result = _related_section(related)
        assert "Base (m), parent class of the seed" in result
        assert "```python" not in result

    def test_source_seed_vs_seed_class_is_distinguished(self):
        related = [
            {"type": "instantiation", "name": "A", "module": "m", "source": "seed"},
            {"type": "instantiation", "name": "B", "module": "m", "source": "seed_class"},
        ]
        result = _related_section(related)
        assert "A (m), instantiated by the seed\n" in result
        assert "B (m), instantiated by the seed's class" in result


class TestRepoSlug:
    """Must match RepoKGBuilder._cache_path's sanitization exactly
    (kg/builder.py) -- a mismatch here means this script can never find
    the KG file a real build actually produced (kg_construction#141).
    """

    def test_slashes_become_underscores(self):
        assert _repo_slug("psf/requests") == "psf_requests"

    def test_hyphens_become_underscores(self):
        assert _repo_slug("scikit-learn/scikit-learn") == "scikit_learn_scikit_learn"

    def test_dots_become_underscores(self):
        assert _repo_slug("foo.bar/baz.qux") == "foo_bar_baz_qux"
