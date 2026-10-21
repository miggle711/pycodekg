"""Regression tests for inheritance and call resolution (pass 2 of the builder).

Three defects fixed together because they share code.

First, inherits and overrides. A generic base (`Base[int]`) never matched a
class. Also, `class Foo(a.Foo)` matched itself (a self loop) and demoted the
real parent to 'ambiguous', which the serializer then drops.

Second, self.method() calls. They were resolved by class name, so same named
classes in different files were conflated (both labelled 'qualified'). A
method inherited from a base class was only found by accident, when its name
happened to be unique.

Third, external calls. `json.loads(s)` was linked to a repo method named
`loads` whenever that was the only one, and labelled 'exact'. Any unique name
match on an untyped receiver (`d.loads()`) was also 'exact'.
"""

import textwrap
from pathlib import Path

from kg_construction.kg.builder import SCHEMA_VERSION, RepoASTParser


def _build(tmp_path: Path, files: dict):
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)


def _name(node):
    cls = node["metadata"].get("class")
    return f"{cls}.{node['label']}" if cls else node["label"]


def _edges(kg, relation):
    """Sorted list of (source, target, confidence), each node shown as
    'Class.name@file'."""
    nodes = {n["id"]: n for n in kg["nodes"]}
    out = []
    for e in kg["edges"]:
        if e["relation"] != relation:
            continue
        s, t = nodes[e["source"]], nodes[e["target"]]
        out.append((
            f"{_name(s)}@{s['metadata'].get('filepath')}",
            f"{_name(t)}@{t['metadata'].get('filepath')}",
            e.get("metadata", {}).get("confidence"),
        ))
    return sorted(out)


class TestInheritsAndOverrides:
    def test_generic_base_resolves(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            from typing import Generic, TypeVar
            T = TypeVar('T')
            class Base(Generic[T]):
                def run(self): pass
            class Child(Base[int]):
                def run(self): pass
        """})
        assert ("Child@m.py", "Base@m.py", "exact") in _edges(kg, "inherits")
        assert _edges(kg, "overrides") == [("Child.run@m.py", "Base.run@m.py", None)]

    def test_dotted_generic_base_resolves(self, tmp_path):
        kg = _build(tmp_path, {
            "a.py": "class Base:\n    def run(self): pass\n",
            "b.py": """
                import a
                class Child(a.Base[int]):
                    def run(self): pass
            """,
        })
        assert _edges(kg, "inherits") == [("Child@b.py", "Base@a.py", "exact")]
        assert _edges(kg, "overrides") == [("Child.run@b.py", "Base.run@a.py", None)]

    def test_class_with_same_name_as_its_parent_has_no_self_loop(self, tmp_path):
        kg = _build(tmp_path, {
            "a.py": "class Foo:\n    def bar(self): pass\n",
            "b.py": """
                import a
                class Foo(a.Foo):
                    def bar(self): pass
            """,
        })
        # One edge, to the real parent, and trusted (exact) so the
        # serializer keeps it.
        assert _edges(kg, "inherits") == [("Foo@b.py", "Foo@a.py", "exact")]
        assert _edges(kg, "overrides") == [("Foo.bar@b.py", "Foo.bar@a.py", None)]

    def test_ordinary_single_inheritance_unchanged(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            class A:
                def f(self): pass
            class B(A):
                def f(self): pass
            class C(B):
                pass
        """})
        assert _edges(kg, "inherits") == [("B@m.py", "A@m.py", "exact"), ("C@m.py", "B@m.py", "exact")]
        assert _edges(kg, "overrides") == [("B.f@m.py", "A.f@m.py", None)]


class TestSelfCallResolution:
    def test_self_call_links_only_to_its_own_class_not_a_same_named_one(self, tmp_path):
        kg = _build(tmp_path, {
            "a.py": "class Foo:\n    def go(self): pass\n    def run(self): self.go()\n",
            "b.py": "class Foo:\n    def go(self): pass\n",
        })
        assert _edges(kg, "calls") == [("Foo.run@a.py", "Foo.go@a.py", "qualified")]

    def test_inherited_method_found_even_when_name_is_shared(self, tmp_path):
        kg = _build(tmp_path, {
            "base.py": "class Base:\n    def helper(self): pass\n",
            "other.py": "class Other:\n    def helper(self): pass\n",
            "child.py": """
                from base import Base
                class Child(Base):
                    def run(self):
                        self.helper()
            """,
        })
        assert _edges(kg, "calls") == [("Child.run@child.py", "Base.helper@base.py", "qualified")]

    def test_method_inherited_through_several_levels(self, tmp_path):
        kg = _build(tmp_path, {
            "base.py": "class Base:\n    def helper(self): pass\n",
            "other.py": "class Other:\n    def helper(self): pass\n",
            "mid.py": "from base import Base\nclass Mid(Base):\n    pass\n",
            "leaf.py": "from mid import Mid\nclass Leaf(Mid):\n    def run(self):\n        self.helper()\n",
        })
        assert _edges(kg, "calls") == [("Leaf.run@leaf.py", "Base.helper@base.py", "qualified")]

    def test_own_method_wins_over_inherited(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            class Base:
                def helper(self): pass
            class Child(Base):
                def helper(self): pass
                def run(self):
                    self.helper()
        """})
        assert _edges(kg, "calls") == [("Child.run@m.py", "Child.helper@m.py", "qualified")]

    def test_self_call_to_unknown_method_falls_back_to_name_only(self, tmp_path):
        # zap is defined by neither X nor any parent (it could come from an
        # unparsed mixin), so a unique repo wide name match is only a guess.
        kg = _build(tmp_path, {
            "y.py": "class Y:\n    def zap(self): pass\n",
            "x.py": "class X:\n    def run(self):\n        self.zap()\n",
        })
        assert _edges(kg, "calls") == [("X.run@x.py", "Y.zap@y.py", "name_only")]


class TestExternalAndUntypedCalls:
    def test_call_through_external_import_is_not_linked_to_repo_name(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            import json
            class Codec:
                def loads(self, s): pass
            def f(s):
                return json.loads(s)
        """})
        assert _edges(kg, "calls") == []

    def test_bare_call_to_externally_imported_name_is_not_linked(self, tmp_path):
        kg = _build(tmp_path, {
            "util.py": "def get(url):\n    pass\n",
            "m.py": "from requests import get\ndef f():\n    return get('x')\n",
        })
        assert _edges(kg, "calls") == []

    def test_relative_import_of_repo_module_is_still_resolved(self, tmp_path):
        kg = _build(tmp_path, {
            "pkg/__init__.py": "",
            "pkg/utils.py": "def helper():\n    pass\n",
            "pkg/a.py": "from .utils import helper\ndef f():\n    helper()\n",
        })
        # The import path loses its package ('utils.helper'), so the precise
        # lookup misses and the bare name fallback links it. It must not be
        # treated as external just because the precise lookup missed.
        assert _edges(kg, "calls") == [("f@pkg/a.py", "helper@pkg/utils.py", "exact")]

    def test_untyped_receiver_unique_name_is_name_only_not_exact(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            class Codec:
                def loads(self, s): pass
            def g(d):
                return d.loads()
        """})
        assert _edges(kg, "calls") == [("g@m.py", "Codec.loads@m.py", "name_only")]

    def test_bare_unique_call_is_still_exact(self, tmp_path):
        kg = _build(tmp_path, {
            "helpers.py": "def helper():\n    pass\n",
            "m.py": "def f():\n    return helper()\n",
        })
        assert _edges(kg, "calls") == [("f@m.py", "helper@helpers.py", "exact")]

    def test_typed_local_receiver_is_still_qualified(self, tmp_path):
        kg = _build(tmp_path, {"m.py": """
            class Model:
                def save(self): pass
            class Other:
                def save(self): pass
            def f():
                m = Model()
                m.save()
        """})
        assert ("f@m.py", "Model.save@m.py", "qualified") in _edges(kg, "calls")


def test_schema_version_bumped():
    # An older cached KG has wrong or missing inherits edges and the old
    # 'exact' label on untyped receiver calls.
    assert SCHEMA_VERSION >= 5
