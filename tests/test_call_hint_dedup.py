"""Dedup pass-1 calls by the hints consumed in pass 2 (#178)."""

from copy import deepcopy
from textwrap import dedent

import pytest

from kg_construction.kg.builder import RepoASTParser, _parse_file


def _raw(tmp_path, source):
    path = tmp_path / "mod.py"
    path.write_text(dedent(source))
    return _parse_file(("test/repo", "mod.py", str(path)))


def _calls(graph, caller="f", target="save", *, raw=False):
    nodes = {n["id"]: n for n in graph["nodes"]}
    return [e for e in graph["edges"]
            if e["relation"] == "calls" and nodes[e["source"]]["label"] == caller
            and (e["target"] if raw else nodes[e["target"]]["label"]) == target]


def _resolve(tmp_path, raw):
    parser = RepoASTParser(max_workers=1)
    # Feed exactly these pass-1 edges through the real pass-2 pipeline.
    parser._run_parallel_parse = lambda file_args: [raw]
    return parser.parse_repo("test/repo", tmp_path)


def test_untyped_receivers_share_one_raw_edge_and_preserve_final_graph(tmp_path):
    raw = _raw(tmp_path, """
        class Model:
            def save(self): ...

        def f(a, b, c):
            a.save()
            b.save()
            c.save()
    """)
    calls = _calls(raw, raw=True)
    assert len(calls) == 1
    assert calls[0]["metadata"]["unresolved"] is True

    # Recreate the old per-receiver edges for the issue's exact repro.
    old_raw = deepcopy(raw)
    for receiver in ("b", "c"):
        edge = deepcopy(calls[0])
        edge["metadata"]["receiver"] = receiver
        old_raw["edges"].append(edge)
    before, after = _resolve(tmp_path, old_raw), _resolve(tmp_path, raw)
    assert before["nodes"] == after["nodes"]
    assert before["edges"] == after["edges"]
    final_calls = _calls(after)
    assert len(final_calls) == 1
    assert final_calls[0]["metadata"] == {
        "confidence": "name_only", "import_resolved": None,
    }


def test_distinct_local_type_hints_remain_separate(tmp_path):
    raw = _raw(tmp_path, """
        class Model:
            def save(self): ...
        class Other:
            def save(self): ...
        def f(a: Model, b: Other, c: Model):
            a.save()
            b.save()
            c.save()
    """)
    calls = _calls(raw, raw=True)
    assert len(calls) == 2
    assert {e["metadata"]["local_type_hint"] for e in calls} == {"Model", "Other"}
    final_calls = _calls(_resolve(tmp_path, raw))
    assert len(final_calls) == 2
    assert {e["metadata"]["confidence"] for e in final_calls} == {"qualified"}


@pytest.mark.parametrize("body, hints", [
    ("a.save(); save()", {None, "a"}),
    ("import first as a; import second as b; a.save(); b.save()",
     {"first.save", "second.save"}),
    ("a.save(); super().save()", {False, True}),
])
def test_other_resolution_hints_remain_separate(tmp_path, body, hints):
    raw = _raw(tmp_path, f"def f(a, b):\n    {body}\n")
    calls = _calls(raw, raw=True)
    assert len(calls) == 2
    field = ("import_resolved" if "import " in body else
             "is_super_call" if "super()" in body else "receiver")
    assert {e["metadata"][field] for e in calls} == hints


def test_direct_call_wins_over_equivalent_closure_receiver(tmp_path):
    raw = _raw(tmp_path, """
        class Model:
            def save(self): ...
        def f(a, b):
            def cb():
                b.save()
            a.save()
    """)
    calls = _calls(raw, raw=True)
    assert len(calls) == 1
    assert not calls[0]["metadata"].get("via_closure")
    assert not _calls(_resolve(tmp_path, raw))[0]["metadata"].get("via_closure")


def test_nested_super_contexts_remain_separate(tmp_path):
    raw = _raw(tmp_path, """
        def f():
            class First:
                def save(self):
                    super().save()
            class Second:
                def save(self):
                    super().save()
    """)
    calls = _calls(raw, raw=True)
    assert len(calls) == 2
    assert len({e["metadata"]["super_class_id"] for e in calls}) == 2


def test_skipped_receiver_still_suppresses_operator_dispatch(tmp_path):
    raw = _raw(tmp_path, """
        class Model:
            def __or__(self, other): ...
        def f(a: Model, b: Model):
            a.__or__(b)
            b.__or__(a)
            b | a
    """)
    assert len(_calls(raw, target="__or__", raw=True)) == 1
