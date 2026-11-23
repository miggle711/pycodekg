"""scripts/rq1_draw_rebuilt.py: RQ1's replacement edges from the rebuilt KGs (testgeneval#70)."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import rq1_draw_rebuilt as dr  # noqa: E402

REPO, COMMIT = "django/django", "98cfb90182a8baa806fc4e09e294b6cfc5d09eff"


def node(i, typ, label, path, line):
    return {"id": i, "type": typ, "label": label, "metadata": {"filepath": path, "lineno": line}}


def edge(rel, s, t, conf):
    return {"relation": rel, "source": s, "target": t, "metadata": {"confidence": conf}}


# Non-test sources a, b; test sources t0..t5 (in tests/); targets x, y.
NODES = [node("a", "class", "A", "django/a.py", 1), node("b", "class", "B", "django/b.py", 1),
         node("x", "class", "X", "django/x.py", 1), node("y", "class", "Y", "django/y.py", 1)]
NODES += [node(f"t{i}", "class", f"T{i}", f"tests/t{i}.py", 1) for i in range(6)]
EDGES = [edge("inherits", "a", "x", "ambiguous"), edge("uses", "b", "y", "ambiguous"),   # non-test ambiguous: 2
         edge("calls", "a", "y", "exact")]                                              # confident: not wanted
EDGES += [edge("inherits", f"t{i}", "x", "ambiguous") for i in range(4)]                # test ambiguous inherits: 4
EDGES += [edge("uses", f"t{i}", "y", "ambiguous") for i in range(4, 6)]                 # test ambiguous uses: 2
KG = {"metadata": {"repo": REPO, "base_commit": COMMIT}, "nodes": NODES, "edges": EDGES}

COLS = ["sample_id", "stratum", "repo", "relation", "confidence", "source_type", "source_name",
        "source_file", "source_line", "target_type", "target_name", "target_file", "target_line", "commit"]


def sample_row(sid, stratum, rel, s, t):
    return {"sample_id": sid, "stratum": stratum, "repo": REPO, "relation": rel, "confidence": "ambiguous",
            "source_type": "class", "source_name": s, "source_file": f"tests/{s.lower()}.py" if s[0] == "T"
            else f"django/{s.lower()}.py", "source_line": "1", "target_type": "class", "target_name": t,
            "target_file": f"django/{t.lower()}.py", "target_line": "1", "commit": COMMIT}


def setup(tmp, sample_rows):
    kg_dir = tmp / "kg"
    kg_dir.mkdir()
    (kg_dir / f"kg_django_django_{COMMIT[:8]}.json").write_text(json.dumps(KG))
    s = tmp / "s.csv"
    with open(s, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(sample_rows)
    return ["--kg-dir", str(kg_dir), "--sample", str(s)]


OLD = [sample_row("RQ1-001", "resolved_confident", "calls", "A", "Y"),
       sample_row("RQ1-002", "test_resolved_ambiguous", "inherits", "T0", "X")]   # still there: never redrawn


def run(tmp, *extra, rows=OLD):
    return dr.main(setup(tmp, rows) + ["--seed", "7", "--first-id", "111", "--out", str(tmp / "o.csv"), *extra])


def test_census_takes_every_edge_of_the_stratum(tmp_path):
    rows = run(tmp_path, "--census", "resolved_ambiguous")
    assert sorted((r["source_name"], r["target_name"]) for r in rows) == [("A", "X"), ("B", "Y")]
    assert [r["sample_id"] for r in rows] == ["RQ1-111", "RQ1-112"]
    assert {r["annotator"] for r in rows} == {"both"}


def test_draw_allocates_by_cell_and_skips_sampled_edges(tmp_path):
    rows = run(tmp_path, "--draw", "test_resolved_ambiguous=3")
    assert len(rows) == 3
    assert ("T0", "X") not in {(r["source_name"], r["target_name"]) for r in rows}
    # eligible: inherits 3 (T1..T3), uses 2 -> 3 * 3/5 = 1.8, 1.2 -> 2 inherits, 1 uses
    assert sorted(r["relation"] for r in rows) == ["inherits", "inherits", "uses"]
    assert all(r["stratum"] == "test_resolved_ambiguous" for r in rows)


def test_draw_is_reproducible(tmp_path):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    a = run(tmp_path / "1", "--draw", "test_resolved_ambiguous=3")
    b = run(tmp_path / "2", "--draw", "test_resolved_ambiguous=3")
    assert [r["source_name"] for r in a] == [r["source_name"] for r in b]


def test_written_csv_is_readable_by_the_recheck(tmp_path):
    import rq1_recheck_labels as rc
    run(tmp_path, "--census", "resolved_ambiguous", "--draw", "test_resolved_ambiguous=2")
    rows = rc.read_sample(str(tmp_path / "o.csv"))
    assert len(rows) == 4 and rows[0]["commit"] == COMMIT


def test_more_than_the_population_is_refused(tmp_path):
    try:
        run(tmp_path, "--draw", "resolved_ambiguous=3")
    except ValueError as e:
        assert "only 2" in str(e)
    else:
        raise AssertionError("a draw larger than the stratum was accepted")


def test_new_id_clashing_with_the_sample_is_refused(tmp_path):
    clash = OLD + [sample_row("RQ1-111", "syntactic", "contains", "A", "X")]
    try:
        run(tmp_path, "--census", "resolved_ambiguous", rows=clash)
    except ValueError as e:
        assert "RQ1-111" in str(e)
    else:
        raise AssertionError("a clashing id was accepted")


def test_equal_chance_can_reach_a_cell_the_allocation_rounds_to_zero(tmp_path):
    # allocation of 1 draw over inherits 3 / uses 2 always takes inherits; equal chance must reach uses
    seen = set()
    for seed in range(40):
        d = tmp_path / str(seed)
        d.mkdir()
        rows = dr.main(setup(d, OLD) + ["--seed", str(seed), "--first-id", "111", "--out", str(d / "o.csv"),
                                        "--draw", "test_resolved_ambiguous=1", "--equal-chance"])
        assert len(rows) == 1 and ("T0", "X") != (rows[0]["source_name"], rows[0]["target_name"])
        seen.add(rows[0]["relation"])
    assert seen == {"inherits", "uses"}


def test_census_is_the_same_with_equal_chance(tmp_path):
    rows = run(tmp_path, "--census", "resolved_ambiguous", "--equal-chance")
    assert sorted((r["source_name"], r["target_name"]) for r in rows) == [("A", "X"), ("B", "Y")]
