# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Show how a change alters the KGs built from real repositories.

Builds a KG from the same source tree with the code at two git refs and
compares them: per relation edge counts, how many edges were removed and
added, a labelled sample of each, and the nodes whose type changed (for
example `test_file` to `file`). Unit tests cover synthetic cases; this
catches what a change does to real code (false links going away, edges being
retargeted, new edges appearing), so run it for any PR that changes KG
contents and paste the table into the PR description.

Usage:
    python scripts/kg_diff.py
    python scripts/kg_diff.py --before origin/main --after HEAD --repos requests pytest django

Only the `src` directory of each ref is exported (git archive), so the
working tree is never touched and uncommitted changes are not included:
commit first, or pass a ref that names what you want to measure.

The repositories come from the bare clones in --cache-dir (default
repo_cache/, names like psf__requests). A short name such as `requests`
matches the clone whose directory ends in `__requests`. Their HEAD is
exported once and used for both builds, so the only difference between the
two KGs is the code under test. Django takes a few minutes per build. The
commit of each repository is printed on a `repos:` line under the refs, so a
table can be compared with one from another cache (two caches at different
commits give different numbers).

Edges are compared by (relation, source id, target id), and nodes by id. Node
IDs are deterministic hashes of the qualified name, so a difference is a real
change and not ordering noise. A node that keeps its id but changes type is
invisible to the edge comparison, so it is reported in its own table.

If the KGs differ but SCHEMA_VERSION is the same at both refs, a reminder is
printed: cached KGs from before the change would otherwise be loaded as if
they were current.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = ["requests", "pytest", "django"]


def _run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def _clone_commit(clone: Path) -> str:
    """The short commit a bare clone's HEAD points at: the source tree both
    builds use, so a table can be compared with one from another cache."""
    out = _run(["git", "--git-dir", str(clone), "rev-parse", "--short=10", "HEAD"],
               capture_output=True, text=True)
    return out.stdout.strip()


def _resolve(ref: str) -> str:
    out = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", ref],
               capture_output=True, text=True)
    return out.stdout.strip()


def _export(git_args: list, ref: str, dest: Path, subdir: str = ""):
    """Extract `ref` (optionally only `subdir`) into dest. `git_args` selects
    the repository: ["-C", worktree] or ["--git-dir", bare_clone]."""
    dest.mkdir(parents=True)
    cmd = ["git", *git_args, "archive", ref] + ([subdir] if subdir else [])
    archive = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    _run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
    if archive.wait() != 0:
        raise RuntimeError(f"git archive failed for {ref} ({git_args})")


def _find_clone(cache_dir: Path, short: str) -> Path:
    matches = sorted(cache_dir.glob(f"*__{short}"))
    if not matches:
        raise SystemExit(f"no clone matching *__{short} in {cache_dir}")
    return matches[0]


def _schema_version(src: Path):
    m = re.search(r"^SCHEMA_VERSION\s*=\s*(\d+)", (src / "kg_construction/kg/builder.py").read_text(), re.M)
    return int(m.group(1)) if m else None


def dump_kg(name: str, source_dir: str, out: str):
    """Build one KG and write its nodes and edges as JSON. Runs in a child
    process whose PYTHONPATH points at the ref under test."""
    from kg_construction.kg.builder import RepoKGBuilder

    kg = RepoKGBuilder().build_from_dir(name, Path(source_dir))
    nodes = {n["id"]: [n["type"], n["label"], (n.get("metadata") or {}).get("path", "")] for n in kg["nodes"]}
    edges = [[e["relation"], e["source"], e["target"], (e.get("metadata") or {}).get("confidence", "")]
             for e in kg["edges"]]
    Path(out).write_text(json.dumps({"nodes": nodes, "edges": edges}))


def _build(src: Path, name: str, source_dir: Path, out: Path):
    env = {**os.environ, "PYTHONPATH": str(src)}
    _run([sys.executable, str(Path(__file__).resolve()), "--dump", name, str(source_dir), str(out)], env=env)
    return json.loads(out.read_text())


def compare(name: str, before: dict, after: dict, sample: int):
    """Return (edge table rows, node type rows, detail lines, differs) for one repository."""
    eb = {(r, s, t) for r, s, t, _ in before["edges"]}
    ea = {(r, s, t) for r, s, t, _ in after["edges"]}
    removed, added = eb - ea, ea - eb
    cb, ca = Counter(r for r, _, _ in eb), Counter(r for r, _, _ in ea)
    rm, ad = Counter(r for r, _, _ in removed), Counter(r for r, _, _ in added)

    rows = []
    for rel in sorted(set(cb) | set(ca)):
        if cb[rel] != ca[rel] or rm[rel] or ad[rel]:
            rows.append((name if not rows else "", rel, cb[rel], ca[rel], rm[rel], ad[rel]))

    conf_b = {(r, s, t): c for r, s, t, c in before["edges"]}
    conf_a = {(r, s, t): c for r, s, t, c in after["edges"]}
    moved = Counter((conf_b[k], conf_a[k]) for k in conf_b.keys() & conf_a.keys() if conf_b[k] != conf_a[k])

    nodes = {**before["nodes"], **after["nodes"]}
    label = lambda i: f"{nodes[i][1]} ({nodes[i][0]})" if i in nodes else i
    changed_type = sorted((before["nodes"][i][0], after["nodes"][i][0], i)
                          for i in before["nodes"].keys() & after["nodes"].keys()
                          if before["nodes"][i][0] != after["nodes"][i][0])
    type_rows = [(name if k == 0 else "", f"{old} -> {new}", n)
                 for k, ((old, new), n) in enumerate(sorted(Counter((o, n) for o, n, _ in changed_type).items()))]

    detail = [f"{name}: nodes {len(before['nodes'])} -> {len(after['nodes'])}, "
              f"edges {len(before['edges'])} -> {len(after['edges'])}, "
              f"confidence changed on {sum(moved.values())} kept edges"
              + (": " + ", ".join(f"{old} -> {new} x{n}" for (old, new), n in sorted(moved.items())) if moved else "")]
    for title, es in (("removed", removed), ("added", added)):
        for rel, s, t in sorted(es)[:sample]:
            detail.append(f"  {title}: {rel}: {label(s)} -> {label(t)}")
    kept_moved = sorted(k for k in conf_b.keys() & conf_a.keys() if conf_b[k] != conf_a[k])
    for rel, src, tgt in kept_moved[:sample]:
        detail.append(f"  confidence {conf_b[(rel, src, tgt)]} -> {conf_a[(rel, src, tgt)]}: {rel}: {label(src)} -> {label(tgt)}")
    for old, new, i in changed_type[:sample]:
        detail.append(f"  type changed: {old} -> {new}: {after['nodes'][i][2] or after['nodes'][i][1]}")
    differs = bool(removed or added or moved or changed_type or set(before["nodes"]) != set(after["nodes"]))
    return rows, type_rows, detail, differs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", default="origin/main", help="git ref for the baseline (default origin/main)")
    parser.add_argument("--after", default="HEAD", help="git ref under test (default HEAD)")
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS, help="short repo names in --cache-dir")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "repo_cache")
    parser.add_argument("--sample", type=int, default=10, help="removed and added edges to list per repo")
    parser.add_argument("--dump", nargs=3, metavar=("NAME", "SOURCE_DIR", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.dump:
        dump_kg(*args.dump)
        return

    refs = {"before": args.before, "after": args.after}
    print(f"before: {args.before} ({_resolve(args.before)})   after: {args.after} ({_resolve(args.after)})")
    clones = {short: _find_clone(args.cache_dir, short) for short in args.repos}
    print("repos: " + ", ".join(f"{short} @ {_clone_commit(clone)}" for short, clone in clones.items()))
    with tempfile.TemporaryDirectory(prefix="kg_diff_") as tmp:
        tmp = Path(tmp)
        srcs = {}
        for side, ref in refs.items():
            _export(["-C", str(REPO_ROOT)], ref, tmp / side, "src")
            srcs[side] = tmp / side / "src"

        rows, type_rows, details, any_diff = [], [], [], False
        for short in args.repos:
            clone = clones[short]
            source = tmp / f"source_{short}"
            _export(["--git-dir", str(clone)], "HEAD", source)
            print(f"building {short} ...", flush=True)
            kgs = {side: _build(srcs[side], short, source, tmp / f"{side}_{short}.json") for side in refs}
            r, t, d, differs = compare(short, kgs["before"], kgs["after"], args.sample)
            rows += r or [(short, "(no edge differences)", "", "", "", "")]
            type_rows += t
            details += d
            any_diff |= differs

        print("\n| Repo | relation | before | after | removed | added |\n|---|---|---|---|---|---|")
        for row in rows:
            print("| " + " | ".join(str(c) for c in row) + " |")
        if type_rows:
            print("\n| Repo | node type change | nodes |\n|---|---|---|")
            for row in type_rows:
                print("| " + " | ".join(str(c) for c in row) + " |")
        print("\n" + "\n".join(details))

        versions = {side: _schema_version(srcs[side]) for side in refs}
        if any_diff and versions["before"] == versions["after"]:
            print(f"\nNote: the KGs differ but SCHEMA_VERSION is {versions['before']} at both refs. "
                  "If this change alters KG contents, bump it (or confirm a bump is already queued).")


if __name__ == "__main__":
    main()
