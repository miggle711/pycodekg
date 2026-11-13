"""scripts/kg_edge_audit.py: a correct KG passes every rule, and each rule
catches the defect it is written for (#239)."""

import copy
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import kg_edge_audit as audit  # noqa: E402
from kg_construction.kg.builder import RepoKGBuilder  # noqa: E402

FILES = {
    "src/pkg/__init__.py": "from .models import Thing\n",
    "src/pkg/util.py": "def helper():\n    return 1\n",
    "src/pkg/models.py": """
        from .util import helper as _helper


        class Base:
            def run(self):
                return 1


        class Thing(Base):
            @property
            def size(self):
                return 1

            def run(self):
                self.go()
                return _helper() + self.size

            def go(self):
                pass


        class Other:
            def go(self):
                pass
        """,
    "tests/test_models.py": """
        import pytest
        from pkg import Thing
        from pkg.util import helper


        @pytest.mark.parametrize("x", [helper()])
        def test_thing(x):
            assert Thing().run() == x
        """,
}


@pytest.fixture(scope="module")
def kg(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip())
    return RepoKGBuilder(max_workers=1).build_from_dir("fixture__repo", root)


def _ids(kg):
    return {(n["metadata"].get("qualname") or n["metadata"].get("path")): n["id"] for n in kg["nodes"]}


def _broken(kg):
    found, counted, notes = audit.audit(kg)
    return {r: len(v) for r, v in found.items() if v}, counted, notes


def test_fixture_is_clean_and_exercises_each_rule(kg):
    broken, counted, notes = _broken(kg)
    assert broken == {}
    for rule in ("call_name", "qualified_module", "self_call_class", "accesses_property",
                 "inherits_base", "overrides_parent"):
        assert counted[rule] > 0, rule
    assert notes["call_name: called by its import alias"] >= 1        # _helper()
    assert notes["call_name: called in a decorator (e.g. parametrize arguments)"] >= 1
    assert notes["qualified_module: reached through a re-export"] >= 1  # pkg.Thing


def _with(kg, edit):
    k = copy.deepcopy(kg)
    edit(k, _ids(k))
    return _broken(k)[0]


def _edge(k, rel, src, tgt, **meta):
    k["edges"].append({"source": src, "target": tgt, "relation": rel, "metadata": meta})


def test_dangling_edge(kg):
    assert _with(kg, lambda k, i: _edge(k, "calls", i["Thing.run"], "no_such_node")) == {"dangling": 1}


def test_duplicate_edge(kg):
    def edit(k, i):
        k["edges"].append(copy.deepcopy(next(e for e in k["edges"] if e["relation"] == "inherits")))
    assert _with(kg, edit) == {"duplicate_edge": 1}


def test_endpoint_types(kg):
    assert _with(kg, lambda k, i: _edge(k, "inherits", i["Thing"], i["Thing.go"])) == {"endpoint_types": 1}


def test_contains_parent(kg):
    def edit(k, i):
        k["edges"] = [e for e in k["edges"] if not (e["relation"] == "contains" and e["target"] == i["Other"])]
    assert _with(kg, edit) == {"contains_parent": 1}


def test_call_name(kg):
    # Thing.go's body calls nothing, so an edge to `run` has no call site.
    assert _with(kg, lambda k, i: _edge(k, "calls", i["Thing.go"], i["Base.run"],
                                        confidence="name_only")) == {"call_name": 1}


def test_self_call_on_unrelated_class(kg):
    assert _with(kg, lambda k, i: _edge(k, "calls", i["Thing.run"], i["Other.go"],
                                        confidence="name_only")) == {"self_call_class": 1}


def test_qualified_into_the_wrong_file(kg):
    def edit(k, i):
        e = next(e for e in k["edges"] if e["relation"] == "calls" and e["source"] == i["Thing.run"]
                 and e["target"] == i["helper"])
        e["metadata"]["import_resolved"] = ".models.helper"
    assert _with(kg, edit) == {"qualified_module": 1}


def test_accesses_non_property(kg):
    assert _with(kg, lambda k, i: _edge(k, "accesses", i["Thing.run"], i["Thing.go"])) == {"accesses_property": 1}


def test_inherits_unnamed_base(kg):
    assert _with(kg, lambda k, i: _edge(k, "inherits", i["Thing"], i["Other"])) == {"inherits_base": 1}


def test_overrides_non_ancestor(kg):
    assert _with(kg, lambda k, i: _edge(k, "overrides", i["Thing.go"], i["Other.go"])) == {"overrides_parent": 1}


@pytest.mark.parametrize("dotted, from_path, expected", [
    ("a.b", "src/pkg/x.py", ["a", "b"]),
    (".util.helper", "src/pkg/models.py", ["src", "pkg", "util", "helper"]),
    (".util", "src/pkg/__init__.py", ["src", "pkg", "util"]),
    ("..core.f", "src/pkg/sub/x.py", ["src", "pkg", "core", "f"]),
])
def test_resolve_relative(dotted, from_path, expected):
    assert audit._resolve_relative(dotted, from_path) == expected


def test_sample_is_seeded_and_blind(kg, tmp_path):
    import json as _json
    import kg_edge_audit_sample as sample
    p = tmp_path / "fixture__repo.json"
    p.write_text(_json.dumps(kg))
    outs = []
    for run in ("a", "b"):
        stem = str(tmp_path / run)
        sample.main(["--kg-json", str(p), "--per-cell", "2", "--flagged-per-rule", "2",
                     "--seed", "7", "--out", stem])
        outs.append((Path(stem + ".md").read_text(), Path(stem + ".key.csv").read_text()))
    assert outs[0] == outs[1]
    sheet, key = outs[0]
    assert "## 1." in sheet and "**To**" in sheet
    # the labelling sheet does not lead the label
    for word in ("name_only", "qualified", "exact", "flagged", "random"):
        assert word not in sheet
    assert key.splitlines()[0].startswith("item,repo,relation,confidence")
    assert Path(str(tmp_path / "a") + ".csv").read_text().splitlines()[0] == "item,repo,label,reason"
