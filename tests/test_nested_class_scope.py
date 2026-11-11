"""self, cls and super() in a nested class's methods mean that class.

A function that defines a class still gets the calls made in that class's
methods, tagged via_closure, as it does for closures: they are dependencies
of the function. But inside those methods `self` and `cls` are the nested
class (and super() walks its bases), not the class of the function that
defines it (#177).
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser

SOURCE = """
    def util(): ...
    class OtherBase:
        def setup(self): ...
    class Base:
        def setup(self): ...
    class K(OtherBase):
        def helper(self): ...
        def build(self): ...
        def run(self):
            class Inner(Base):
                def helper(self): ...
                @classmethod
                def build(cls): ...
                def go(self):
                    self.helper()
                    util()
                    super().setup()
                @classmethod
                def make(cls):
                    return cls.build()
"""


@pytest.fixture
def calls(tmp_path: Path):
    (tmp_path / "m.py").write_text(textwrap.dedent(SOURCE))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}

    def of(caller):
        return sorted(
            (nodes[e["target"]]["metadata"].get("qualname"), e["metadata"]["confidence"],
             bool(e["metadata"].get("via_closure")))
            for e in kg["edges"]
            if e["relation"] == "calls" and nodes[e["source"]]["metadata"].get("qualname") == caller
        )
    return of


@pytest.mark.parametrize("target", ["K.run.Inner.helper", "Base.setup", "K.run.Inner.build"],
                         ids=["self", "super", "cls"])
def test_the_outer_function_resolves_the_nested_class_receiver(calls, target):
    assert (target, "qualified", True) in calls("K.run")


@pytest.mark.parametrize("wrong", ["K.helper", "OtherBase.setup", "K.build"],
                         ids=["self", "super", "cls"])
def test_not_against_the_outer_class(calls, wrong):
    assert wrong not in [c[0] for c in calls("K.run")]


def test_the_outer_function_keeps_the_nested_class_calls(calls):
    # A bare call in a nested class's method is still a dependency of the
    # function that defines the class.
    assert ("util", "exact", True) in calls("K.run")


def test_the_nested_methods_resolve_the_same_way(calls):
    go = calls("K.run.Inner.go")
    assert ("K.run.Inner.helper", "qualified", False) in go
    assert ("Base.setup", "qualified", False) in go
