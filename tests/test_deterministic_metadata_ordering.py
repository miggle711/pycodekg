"""Regression test: metadata lists built by deduplicating through a set
(raises/catches in _get_exceptions, returns/mutates_attributes in
_extract_data_flows) must be sorted before being returned, not just
list(set(...)) -- Python randomizes set iteration order for strings
per-process by default (PYTHONHASHSEED unset), so two separate builds
of the same function could otherwise report these fields in a
different order. raises/catches feeds the rendered prompt directly via
SEED_BLOCK_TEMPLATE's "Declared exceptions" field. Same bug class as
#145's BFS visited-node-order fix, different call sites.
"""

import ast

from kg_construction.ast.helpers import _get_exceptions, _extract_data_flows


def _parse_function(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return tree.body[0]


class TestExceptionOrderingIsDeterministic:
    def test_raises_are_sorted(self):
        node = _parse_function(
            "def f(x):\n"
            "    if x == 1:\n"
            "        raise ValueError('a')\n"
            "    if x == 2:\n"
            "        raise KeyError('b')\n"
            "    if x == 3:\n"
            "        raise TypeError('c')\n"
        )
        result = _get_exceptions(node)
        assert result["raises"] == sorted(result["raises"])

    def test_catches_are_sorted(self):
        node = _parse_function(
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError:\n"
            "        pass\n"
            "    except KeyError:\n"
            "        pass\n"
            "    except TypeError:\n"
            "        pass\n"
        )
        result = _get_exceptions(node)
        assert result["catches"] == sorted(result["catches"])

    def test_raises_order_is_stable_across_repeated_calls(self):
        # Same AST, called twice -- must produce the identical order
        # both times, not just an order that happens to be sorted once.
        node = _parse_function(
            "def f(x):\n"
            "    if x == 1:\n"
            "        raise ValueError('a')\n"
            "    if x == 2:\n"
            "        raise KeyError('b')\n"
            "    if x == 3:\n"
            "        raise TypeError('c')\n"
            "    if x == 4:\n"
            "        raise IndexError('d')\n"
            "    if x == 5:\n"
            "        raise AttributeError('e')\n"
        )
        first = _get_exceptions(node)
        second = _get_exceptions(node)
        assert first["raises"] == second["raises"]
        assert first["catches"] == second["catches"]


class TestDataFlowOrderingIsDeterministic:
    def test_returns_are_sorted(self):
        node = _parse_function(
            "def f(x):\n"
            "    if x == 1:\n"
            "        return 'z'\n"
            "    if x == 2:\n"
            "        return 'a'\n"
            "    if x == 3:\n"
            "        return 'm'\n"
        )
        flows = _extract_data_flows(node)
        assert flows["returns"] == sorted(flows["returns"])

    def test_mutates_attributes_values_are_sorted(self):
        node = _parse_function(
            "def f(self, x):\n"
            "    if x == 1:\n"
            "        self.value = 'z'\n"
            "    if x == 2:\n"
            "        self.value = 'a'\n"
            "    if x == 3:\n"
            "        self.value = 'm'\n"
        )
        flows = _extract_data_flows(node)
        assert flows["mutates_attributes"]["value"] == sorted(
            flows["mutates_attributes"]["value"]
        )
