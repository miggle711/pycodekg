"""KG build timing for RQ4 (pycodekg#40).

build() records how long extraction took and whether the bare clone was
already cached, next to the parse and resolve times parse_repo records, and
scripts/aggregate_build_times.py reports cold and warm extraction separately.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

from kg_construction.kg.builder import RepoKGBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeRepoManager:
    """Stands in for RepoManager: no network, and a clone appears in the
    cache the first time a repo is extracted."""

    def __init__(self):
        self.cloned = set()

    def is_cached(self, repo):
        return repo in self.cloned

    def extract_at_commit(self, repo, commit, dest):
        self.cloned.add(repo)
        dest.mkdir(parents=True)
        (dest / "mod.py").write_text("def f():\n    return 1\n")


def test_build_records_extraction_time_and_whether_the_clone_was_cached(tmp_path):
    builder = RepoKGBuilder(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache", max_workers=1)
    builder.repo_manager = _FakeRepoManager()

    cold = builder.build("acme/widgets", "a" * 40)["metadata"]
    warm = builder.build("acme/widgets", "b" * 40)["metadata"]

    assert cold["extraction_cached"] is False
    assert warm["extraction_cached"] is True
    for meta in (cold, warm):
        assert isinstance(meta["build_time_extraction_s"], float)
        assert meta["build_time_extraction_s"] >= 0
        assert {"build_time_parsing_s", "build_time_resolution_s"} <= meta.keys()


def test_build_from_dir_has_no_extraction_timing(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def f():\n    return 1\n")
    builder = RepoKGBuilder(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache", max_workers=1)
    meta = builder.build_from_dir("local", tmp_path / "src")["metadata"]
    assert "build_time_extraction_s" not in meta


def _kg(name, tmp, **extra):
    meta = {"repo": "acme/widgets", "base_commit": name, "file_count": 1, "total_lines_of_code": 1000,
            "node_count": 10, "edge_count": 20, "build_time_parsing_s": 1.0,
            "build_time_resolution_s": 0.5, **extra}
    (tmp / f"kg_{name}.json").write_text(json.dumps({"nodes": [], "edges": [], "metadata": meta}))


def test_aggregate_reports_cold_and_warm_extraction_separately(tmp_path):
    kgs = tmp_path / "kgs"
    kgs.mkdir()
    _kg("cold", kgs, build_time_extraction_s=4.0, extraction_cached=False)
    _kg("warm1", kgs, build_time_extraction_s=0.1, extraction_cached=True)
    _kg("warm2", kgs, build_time_extraction_s=0.3, extraction_cached=True)
    _kg("old", kgs)  # built before extraction was timed
    out = tmp_path / "times.csv"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "aggregate_build_times.py"),
         "--kg-dir", str(kgs), "--out", str(out)],
        capture_output=True, text=True, check=True,
    )

    rows = {r["base_commit"]: r for r in csv.DictReader(out.open())}
    assert rows["cold"]["build_time_extraction_s"] == "4.0"
    assert rows["cold"]["extraction_cached"] == "False"
    assert rows["old"]["build_time_extraction_s"] == ""
    assert rows["old"]["extraction_cached"] == ""
    # total stays parsing + resolution, comparable with older KGs
    assert {r["build_time_total_s"] for r in rows.values()} == {"1.5"}
    assert "extraction, cold (cloned): n=1  mean 4.000s" in result.stdout
    assert "extraction, warm (cached): n=2  mean 0.200s" in result.stdout
    assert "1 KG(s) have no extraction timing" in result.stdout
