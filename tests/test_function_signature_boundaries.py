"""Function headers use the same statement boundaries as class outlines."""

import ast

import pytest

from kg_construction.ast.helpers import _get_signature_and_source_text, split_source_lines


def _signature(source, name="f"):
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    lines = split_source_lines(source)
    signature, full_source = _get_signature_and_source_text(node, lines)
    # Header corrections must not change the separately stored function body.
    assert full_source == "".join(lines[node.lineno - 1:node.end_lineno])
    return signature


def test_wrapped_header_keeps_its_inline_body_line():
    source = "def f(\n    value: int,\n) -> int: ...\n"
    signature = _signature(source)
    assert signature == source.rstrip()
    ast.parse(signature)


@pytest.mark.parametrize("nested", [
    "    @decorate\n    def inner():\n        pass\n",
    "    @decorate\n    class Inner:\n        pass\n",
], ids=["nested_function", "nested_class"])
def test_first_body_statement_owns_its_decorators(nested):
    assert _signature("def f():\n" + nested) == "def f():"


def test_inline_body_continuation_keeps_balanced_brackets():
    source = "def f(\n    value,\n): return dict(\n    value=value,\n)\n"
    signature = _signature(source)
    assert signature == source.rstrip()
    ast.parse(signature)


def test_signature_keeps_its_own_decorators():
    source = "@first\n@second(flag=True)\nasync def f(value: int = 1) -> int:\n    return value\n"
    signature = _signature(source)
    assert signature == "@first\n@second(flag=True)\nasync def f(value: int = 1) -> int:"
    parsed = ast.parse(signature + "\n    ...\n").body[0]
    assert [ast.dump(d) for d in parsed.decorator_list] == [
        ast.dump(d) for d in ast.parse(source).body[0].decorator_list]


def test_existing_single_line_signature_is_unchanged():
    assert _signature("def f(): return 1\n") == "def f(): return 1"
