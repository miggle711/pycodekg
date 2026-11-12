"""Call edges can be filtered by confidence (#12).

The builder labels every resolved call `qualified`, `exact` or `name_only`,
from most to least certain. `min_confidence` lets a consumer leave out the
weaker ones without reading raw edge metadata.
"""

import pytest

from kg_construction.kg.builder import _CALL_CONFIDENCE_RANK
from kg_construction.kg.query import CALL_CONFIDENCE_ORDER, KGQueryEngine
from kg_construction.llm.llm_serializer import LLMSerializer


def _node(node_id, ntype="function", filepath="app.py"):
    return {"id": node_id, "label": node_id, "type": ntype, "metadata": {"filepath": filepath}}


def _call(source, target, confidence):
    metadata = {} if confidence is None else {"confidence": confidence}
    return {"source": source, "target": target, "relation": "calls", "metadata": metadata}


# seed is called by four functions, one per label plus one with no label.
CALLERS = {"by_type": "qualified", "by_name": "exact", "guess": "name_only", "old": None}
NODES = [_node("seed")] + [_node(name) for name in CALLERS]
EDGES = [_call(name, "seed", label) for name, label in CALLERS.items()]
CALLEE_EDGES = [_call("seed", name, label) for name, label in CALLERS.items()]


@pytest.mark.parametrize("minimum, expected", [
    (None, {"by_type", "by_name", "guess", "old"}),
    ("name_only", {"by_type", "by_name", "guess"}),
    ("exact", {"by_type", "by_name"}),
    ("qualified", {"by_type"}),
], ids=["no_filter", "name_only", "exact", "qualified"])
def test_callers_callees_and_tests_respect_min_confidence(minimum, expected):
    callers = KGQueryEngine({"nodes": NODES, "edges": EDGES})
    assert {n["id"] for n in callers.find_callers("seed", min_confidence=minimum)} == expected

    callees = KGQueryEngine({"nodes": NODES, "edges": CALLEE_EDGES})
    assert {n["id"] for n in callees.find_callees("seed", min_confidence=minimum)} == expected

    tests = [dict(n, type="test_function") if n["id"] != "seed" else n for n in NODES]
    covering = KGQueryEngine({"nodes": tests, "edges": EDGES})
    assert {n["id"] for n in covering.find_test_functions_for("seed", min_confidence=minimum)} == expected


def test_an_unknown_confidence_is_an_error():
    with pytest.raises(ValueError, match="unknown confidence"):
        KGQueryEngine({"nodes": NODES, "edges": EDGES}).find_callers("seed", min_confidence="certain")


def test_the_order_matches_the_builders():
    assert sorted(CALL_CONFIDENCE_ORDER, key=_CALL_CONFIDENCE_RANK.get) == list(CALL_CONFIDENCE_ORDER)


@pytest.mark.parametrize("minimum, expected", [
    (None, ["by_name", "guess"]),
    ("exact", ["by_name"]),
], ids=["default_keeps_every_call", "exact_drops_guesses"])
def test_the_serializer_leaves_out_weak_callers_and_callees(minimum, expected):
    seed = _node("seed")
    others = [_node("by_name"), _node("guess")]
    edges = [_call("by_name", "seed", "exact"), _call("guess", "seed", "name_only"),
             _call("seed", "by_name", "exact"), _call("seed", "guess", "name_only")]
    result = LLMSerializer(min_call_confidence=minimum).serialize(
        {"seeds": [seed], "context_nodes": others, "edges": edges, "test_nodes": []})

    assert sorted(c["name"] for c in result["context"]["callers"]) == expected
    assert sorted(c["name"] for c in result["context"]["callees"]) == expected
