"""scripts/rq1_recheck_labels.py: each sampled edge gets the status the
rebuilt KG implies (testgeneval#70)."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import rq1_recheck_labels as rc  # noqa: E402

REPO, COMMIT = "django/django", "98cfb90182a8baa806fc4e09e294b6cfc5d09eff"


def node(i, typ, label, path, line=None):
    md = {"filepath": path}
    if line is not None:
        md["lineno"] = line
    return {"id": i, "type": typ, "label": label, "metadata": md}


def edge(rel, s, t, conf=None):
    return {"relation": rel, "source": s, "target": t, "metadata": {"confidence": conf} if conf else {}}


REBUILT = {
    "metadata": {"repo": REPO, "base_commit": COMMIT, "schema_version": 17},
    "nodes": [
        node("a", "class", "Field", "django/db/models/fields.py", 10),
        node("b", "class", "Model", "django/db/models/base.py", 5),
        node("c", "method", "save", "django/db/models/base.py", 40),
        node("d", "function", "helper", "django/utils/x.py", 3),
        node("e", "method", "clean", "django/forms/a.py", 7),     # two `clean`s in one file
        node("f", "method", "clean", "django/forms/a.py", 20),
        node("g", "file", "conftest.py", "tests/conftest.py"),     # was test_file
        node("h", "class", "TestModel", "tests/models.py", 1),
    ],
    "edges": [
        edge("inherits", "a", "b", "exact"),       # kept (was exact)
        edge("calls", "c", "d", "exact"),          # restratified (was ambiguous)
        edge("contains", "b", "c"),                # kept, syntactic
        edge("inherits", "a", "h", "ambiguous"),   # source a is not test code
        edge("calls", "e", "d", "exact"),
    ],
}

SAMPLE_COLS = ["sample_id", "stratum", "repo", "relation", "confidence", "source_type", "source_name",
               "source_file", "source_line", "target_type", "target_name", "target_file", "target_line", "commit"]


def row(sid, stratum, rel, conf, s, t):
    return {"sample_id": sid, "stratum": stratum, "repo": REPO, "relation": rel, "confidence": conf,
            "source_type": s[0], "source_name": s[1], "source_file": s[2], "source_line": s[3],
            "target_type": t[0], "target_name": t[1], "target_file": t[2], "target_line": t[3],
            "commit": COMMIT}


FIELD = ("class", "Field", "django/db/models/fields.py", "10")
MODEL = ("class", "Model", "django/db/models/base.py", "5")
SAVE = ("method", "save", "django/db/models/base.py", "40")
HELPER = ("function", "helper", "django/utils/x.py", "3")
TESTMODEL = ("class", "TestModel", "tests/models.py", "1")

SAMPLE = [
    row("RQ1-001", "resolved_confident", "inherits", "exact", FIELD, MODEL),          # kept
    row("RQ1-002", "resolved_ambiguous", "calls", "ambiguous", SAVE, HELPER),         # now exact
    row("RQ1-003", "syntactic", "contains", "", MODEL, SAVE),                         # kept
    row("RQ1-004", "resolved_ambiguous", "inherits", "ambiguous", FIELD,              # gone: Model in a
        ("class", "Model", "tests/other.py", "2")),                                  # test file no longer exists
    row("RQ1-005", "resolved_confident", "uses", "exact", FIELD, MODEL),              # gone (no uses edge)
    row("RQ1-006", "resolved_confident", "calls", "exact",                            # unclear: two cleans,
        ("method", "clean", "django/forms/a.py", ""), HELPER),                        # no line to pick
    row("RQ1-007", "resolved_confident", "calls", "exact",
        ("method", "clean", "django/forms/a.py", "7"), HELPER),                       # line picks e: kept
    row("RQ1-008", "test_syntactic", "contains", "",
        ("test_file", "conftest.py", "tests/conftest.py", ""), MODEL),                # type changed; gone
]


def write(tmp, name, rows, cols):
    path = tmp / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def run(tmp, labels=None):
    kg_dir = tmp / "kg"
    kg_dir.mkdir()
    (kg_dir / f"kg_django_django_{COMMIT[:8]}.json").write_text(json.dumps(REBUILT))
    args = ["--kg-dir", str(kg_dir), "--sample", write(tmp, "s.csv", SAMPLE, SAMPLE_COLS),
            "--out", str(tmp / "out.csv")]
    if labels:
        args += ["--labels", labels]
    return {r["sample_id"]: r for r in rc.main(args)}


def test_each_status(tmp_path):
    rep = run(tmp_path)
    assert rep["RQ1-001"]["status"] == "kept"
    assert rep["RQ1-002"]["status"] == "kept_restratified"
    assert rep["RQ1-002"]["stratum_now"] == "resolved_confident"
    assert rep["RQ1-003"]["status"] == "kept"
    assert rep["RQ1-004"]["status"] == "endpoint_missing"
    assert rep["RQ1-005"]["status"] == "gone"
    assert rep["RQ1-006"]["status"] == "endpoint_unclear"
    assert rep["RQ1-007"]["status"] == "kept"


def test_gone_edge_lists_where_the_source_points_now(tmp_path):
    rep = run(tmp_path)
    assert rep["RQ1-005"]["now_targets"] == ""            # FIELD has no `uses` edges at all
    assert rep["RQ1-004"]["note"] == "target missing"


def test_node_whose_type_changed_is_still_found_and_reported(tmp_path):
    rep = run(tmp_path)
    assert rep["RQ1-008"]["note"] == "source type now file"
    assert rep["RQ1-008"]["status"] == "gone"           # file conftest.py does not contain Model


def test_no_rebuilt_kg_for_the_commit(tmp_path):
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir()
    rep = rc.main(["--kg-dir", str(kg_dir), "--sample", write(tmp_path, "s.csv", SAMPLE[:1], SAMPLE_COLS),
                   "--out", str(tmp_path / "o.csv")])
    assert rep[0]["status"] == "no_kg"


def test_labels_from_the_sheet(tmp_path):
    cols = ["Sample", "Mun label", "Joe label", "Agreed label (both)"]
    labels = write(tmp_path, "labels.csv", [
        {"Sample": "RQ1-001", "Mun label": "correct"},
        {"Sample": "RQ1-002", "Mun label": "incorrect", "Joe label": "correct", "Agreed label (both)": "incorrect"},
        {"Sample": "RQ1-003", "Mun label": "correct", "Joe label": "incorrect"},
        {"Sample": "RQ1-005", "Joe label": "incorrect", "Mun label": "incorrect"},
    ], cols)
    rep = run(tmp_path, labels)
    assert [rep[i]["label"] for i in ("RQ1-001", "RQ1-002", "RQ1-003", "RQ1-005", "RQ1-006")] == \
        ["correct", "incorrect", "disagree", "incorrect", ""]


def test_strata_use_the_samplers_rules(tmp_path):
    kg = rc.KG(REBUILT)
    assert kg.strata() == {"resolved_confident": 3, "syntactic": 1, "resolved_ambiguous": 1}


def test_csv_written_with_one_row_per_sampled_edge(tmp_path):
    run(tmp_path)
    with open(tmp_path / "out.csv", newline="") as f:
        assert [r["sample_id"] for r in csv.DictReader(f)] == [r["sample_id"] for r in SAMPLE]
