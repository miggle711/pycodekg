import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_kg_prompts import _build_prompt, _related_section, _repo_slug  # noqa: E402


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
        prompt = _build_prompt(_serialized({"docstring": "Does the thing."}))
        assert "Docstring: Does the thing." in prompt

    def test_missing_docstring_renders_as_none(self):
        prompt = _build_prompt(_serialized({"docstring": ""}))
        assert "Docstring: (none)" in prompt


class TestRelatedSection:
    def test_no_related_omits_section_entirely(self):
        prompt = _build_prompt(_serialized(related=[]))
        assert "Related classes" not in prompt

    def test_parent_class_includes_source_code(self):
        related = [{
            "type": "parent_class", "name": "Base", "module": "m",
            "source_code": "class Base: ...", "source": "seed",
        }]
        result = _related_section(related)
        assert "Base (m), parent class of the seed" in result
        assert "class Base: ..." in result

    def test_instantiation_has_no_source_code_block(self):
        related = [{
            "type": "instantiation", "name": "Helper", "module": "m",
            "source": "seed_class",
        }]
        result = _related_section(related)
        assert "Helper (m), instantiated by the seed's class" in result
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
