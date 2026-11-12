"""Class nodes carry an outline of the class as source_code (#142).

Before this, class nodes had no source_code, so every parent class, class
callee and related class in a kg_only prompt was an empty code block.
"""

import ast

from kg_construction.ast.helpers import _get_class_outline, split_source_lines
from kg_construction.kg.builder import RepoASTParser

SOURCE = '''\
import dataclasses


@dataclasses.dataclass
class Point(Base, metaclass=Meta):
    """A point."""

    x: int = 0
    LIMITS = {
        "a": 1,
    }

    class Meta:
        ordering = ["x"]

        def nested(self):
            return 1

    def __init__(self, x):
        self.x = x
        self.ready = True

    @property
    def doubled(self):
        return self.x * 2

    def move(
        self,
        dx,
    ):
        """Move by dx."""
        self.x += dx

    def short(self): return 1
'''

EXPECTED = '''\
@dataclasses.dataclass
class Point(Base, metaclass=Meta):
    """A point."""
    x: int = 0
    LIMITS = {
        "a": 1,
    }
    class Meta:
        ordering = ["x"]
        def nested(self):
            ...
    def __init__(self, x):
        self.x = x
        self.ready = True
    @property
    def doubled(self):
        ...
    def move(
        self,
        dx,
    ):
        ...
    def short(self): return 1'''


def _outline(source, **kwargs):
    lines = split_source_lines(source)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    return _get_class_outline(node, lines, **kwargs)


def test_outline_keeps_signatures_attributes_and_init():
    header, outline = _outline(SOURCE)
    assert header == "@dataclasses.dataclass\nclass Point(Base, metaclass=Meta):"
    assert outline == EXPECTED


def test_one_line_class_is_kept_as_is():
    assert _outline("class A: pass\n") == ("class A: pass", "class A: pass")


def test_signature_that_closes_on_the_body_line_is_kept_whole():
    # The typed @overload form: the body shares the signature's last line.
    # Cutting before the body dropped that line and left the "(" open.
    source = (
        "class A:\n"
        "    @overload\n"
        "    def f(\n"
        "        self,\n"
        "        data: int,\n"
        "    ) -> int: ...\n"
        "    def g(self):\n"
        "        return 1\n"
    )
    _, outline = _outline(source)
    assert outline == (
        "class A:\n"
        "    @overload\n"
        "    def f(\n"
        "        self,\n"
        "        data: int,\n"
        "    ) -> int: ...\n"
        "    def g(self):\n"
        "        ..."
    )
    ast.parse(outline)


def test_body_on_the_signature_line_is_kept_to_its_last_line():
    source = (
        "class A:\n"
        "    def f(\n"
        "        self,\n"
        "    ): return dict(\n"
        "        a=1,\n"
        "    )\n"
    )
    _, outline = _outline(source)
    assert outline == source.rstrip()
    ast.parse(outline)


def test_decorator_of_a_nested_function_is_not_part_of_the_header():
    source = (
        "class A:\n"
        "    def f(self):\n"
        "        @decorator\n"
        "        def inner():\n"
        "            pass\n"
        "        return inner\n"
    )
    _, outline = _outline(source)
    assert outline == "class A:\n    def f(self):\n        ..."
    ast.parse(outline)


def test_decorator_of_the_first_member_is_not_repeated():
    # The class header used to run up to the first member's `def` line, so
    # that member's decorator appeared in the header and again in the body.
    source = (
        "class A:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return 1\n"
        "    class Meta:\n"
        "        @staticmethod\n"
        "        def m():\n"
        "            return 1\n"
    )
    header, outline = _outline(source)
    assert header == "class A:"
    assert outline == (
        "class A:\n"
        "    @property\n"
        "    def x(self):\n"
        "        ...\n"
        "    class Meta:\n"
        "        @staticmethod\n"
        "        def m():\n"
        "            ..."
    )


def test_class_whose_header_closes_on_the_body_line_is_kept_as_is():
    source = "class A(\n    Base,\n): pass\n"
    assert _outline(source) == (source.rstrip(), source.rstrip())


def test_long_outline_is_cut_with_a_note():
    source = "class Big:\n" + "".join(f"    a{i} = {i}\n" for i in range(10))
    _, outline = _outline(source, max_lines=3)
    assert outline == "class Big:\n    a0 = 0\n    a1 = 1\n    a2 = 2\n    # ... 7 more lines not shown"


def test_class_nodes_in_the_kg_carry_the_outline(tmp_path):
    (tmp_path / "m.py").write_text(SOURCE)
    kg = RepoASTParser(max_workers=1).parse_repo("t/r", tmp_path)
    point = next(n for n in kg["nodes"] if n["type"] == "class" and n["label"] == "Point")
    assert point["metadata"]["source_code"] == EXPECTED
    assert point["metadata"]["signature"].endswith("class Point(Base, metaclass=Meta):")
