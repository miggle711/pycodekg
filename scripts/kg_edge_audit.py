"""Structural checks over a whole KG, for the cumulative correctness audit (#239).

Each rule is a property a correct KG has whatever the resolution logic did,
so a count above zero is either a defect or a case the rule did not foresee.
The script reports counts and examples; it decides nothing about precision.
That needs labelled samples (scripts/kg_edge_audit_sample.py).

Rules (H = must hold, the script exits 1 if any is broken; C = check, every
case is reported and examined, but a count above zero is not by itself a bug):

  H dangling          both ends of every edge are nodes
  H duplicate_edge    no two edges share (relation, source, target)
  H duplicate_node    no two nodes share an id
  H endpoint_types    each relation joins the node types listed in ENDPOINTS
  H contains_parent   every class and function node has exactly one container
  C call_name         a calls edge's target name is called in the caller's
                      source (`name(...)` or `x.name(...)`); a class target
                      matches its name or the name it was imported as;
                      a dunder target needs operator syntax or a `super()`
                      call; `__init__` may come from constructing the class
  C qualified_module  a calls edge with import_resolved lands in the file the
                      dotted path names (relative paths resolved from the
                      caller's package), or in the file that path re-exports
                      the name from, followed through up to 4 imports.
                      Broken cases say where the path led: 'external' (no
                      file in the repo, or several) or 'elsewhere' (another
                      file, which does not import the name)
  C self_call_class   when every call site of a name in a method is
                      `self.name(...)`, its calls edge targets lands on the
                      method's class, a parent, or a subclass (an override
                      the edge points to). Other classes are counted
  C self_loop         a calls edge from a function to itself has a call site
                      that can mean itself: a bare name, self., cls., the
                      own class, or super(). `Other.f(self)` with an Other
                      that is not the own class does not
  C accesses_property an accesses edge's target is decorated as a property
  C inherits_base     an inherits edge's target is named in the source's bases
  C overrides_parent  an overrides edge joins two methods of the same name,
                      the target's class an ancestor of the source's

Usage:

    PYTHONPATH=src .venv/bin/python scripts/kg_edge_audit.py --repos requests pytest
    PYTHONPATH=src .venv/bin/python scripts/kg_edge_audit.py --kg-json some_kg.json

Repositories come from the bare clones in --cache-dir (repo_cache/), built
at their HEAD with the code in src/; the header prints the clone commits and
this checkout's commit. --kg-json checks a KG saved by an earlier build
instead (for example one of the experiment's cached KGs).
"""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CALLABLE = {"function", "method", "test_function"}
DEFINED = CALLABLE | {"class"}
FILES = {"file", "test_file"}
ENDPOINTS = {
    "contains": (FILES | DEFINED, DEFINED),
    "imports": (FILES, {"import"}),
    "depends_on": (CALLABLE, {"import"}),
    "module_depends_on": (FILES, FILES),
    "calls": (CALLABLE, DEFINED),
    "tests": ({"test_function"}, DEFINED),
    "accesses": (CALLABLE, {"method", "test_function"}),
    "inherits": ({"class"}, {"class"}),
    "overrides": ({"method", "test_function"}, {"method", "test_function"}),
    "decorated_by": (DEFINED, DEFINED),
    "raises": (CALLABLE, {"class"}),
    "uses": (DEFINED, {"class"}),
}
HARD = ("dangling", "duplicate_edge", "duplicate_node", "endpoint_types", "contains_parent")
CHECKS = ("call_name", "qualified_module", "self_call_class", "self_loop", "accesses_property",
          "inherits_base", "overrides_parent")
OPERATOR_NODES = (ast.BinOp, ast.UnaryOp, ast.Compare, ast.AugAssign, ast.Subscript,
                  ast.BoolOp, ast.For, ast.With, ast.AsyncWith, ast.AsyncFor)
PROPERTY_DECORATORS = ("property", "cached_property")
EXAMPLES = 3


def _parse(source):
    """Parse a function's own source. A method is indented, and dedent does
    nothing when a multi-line string in it starts at column 0, so fall back
    to parsing it as the body of an `if` block."""
    try:
        return ast.parse(textwrap.dedent(source or ""))
    except (SyntaxError, ValueError):
        pass
    try:
        return ast.parse("if 1:\n" + (source or ""))
    except (SyntaxError, ValueError):
        return None


class CallSites:
    """The names a function's own source calls, with the receiver of each."""

    def __init__(self, tree, decorators=()):
        self.names = defaultdict(set)   # called name -> receivers ('' for a bare call)
        self.decorator_names = set()     # names called only inside a decorator
        self.operators = False
        self.super_call = False
        if tree is not None:
            self._walk(tree, self.names)
        for d in decorators:
            try:
                dtree = ast.parse(str(d).lstrip("@"), mode="eval")
            except (SyntaxError, ValueError):
                continue
            found = defaultdict(set)
            self._walk(dtree, found)
            self.decorator_names |= set(found) - set(self.names)

    def _walk(self, tree, names):
        for node in ast.walk(tree):
            if isinstance(node, OPERATOR_NODES):
                self.operators = True
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name):
                names[f.id].add("")
                if f.id == "super":
                    self.super_call = True
            elif isinstance(f, ast.Attribute):
                v = f.value
                receiver = v.id if isinstance(v, ast.Name) else type(v).__name__
                names[f.attr].add(receiver)
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
                    self.super_call = True


def _import_aliases(nodes, edges):
    """For each file node id, the local names bound to an imported name."""
    aliases = defaultdict(lambda: defaultdict(set))   # file id -> original -> local names
    for e in edges:
        if e["relation"] != "imports":
            continue
        for st in (e.get("metadata") or {}).get("statements") or []:
            if isinstance(st, dict) and st.get("name"):
                aliases[e["source"]][st["name"]].add(st.get("asname") or st["name"])
    return aliases


def _module_of(filepath):
    p = filepath[:-3] if filepath.endswith(".py") else filepath
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _resolve_relative(dotted, from_path):
    """Absolute module parts for a dotted path that may start with dots,
    relative to the package of the file at from_path."""
    level = len(dotted) - len(dotted.lstrip("."))
    rest = [p for p in dotted.lstrip(".").split(".") if p]
    if level == 0:
        return rest
    base = _module_of(from_path)
    if not from_path.endswith("__init__.py"):
        base = base[:-1]
    if level > 1:
        base = base[:-(level - 1)]
    return base + rest


class ModuleMap:
    """Which file a dotted path names, and what each file imports, so a
    qualified edge can be followed through re-exports."""

    def __init__(self, nodes, edges):
        self.path = {n["id"]: (n.get("metadata") or {}).get("path", "") for n in nodes if n["type"] in FILES}
        self.imports = defaultdict(dict)   # file id -> local name -> dotted origin
        self.stars = defaultdict(list)     # file id -> modules it star-imports
        for e in edges:
            if e["relation"] != "imports" or e["source"] not in self.path:
                continue
            for st in (e.get("metadata") or {}).get("statements") or []:
                if not isinstance(st, dict) or not st.get("name"):
                    continue
                dots = "." * int(st.get("level") or 0)
                if st["name"] == "*":
                    self.stars[e["source"]].append(dots + (st.get("module") or ""))
                    continue
                if st.get("module") is None and not dots:
                    continue   # `import a.b`: binds a module, not a name
                origin = dots + ".".join(x for x in (st.get("module"), st["name"]) if x)
                self.imports[e["source"]][st.get("asname") or st["name"]] = origin

    def file_for(self, parts):
        """(file id, names after the module) for the longest prefix of parts
        that is one file's module, matched as a suffix (src/ and the like
        are ignored). None when no file, or more than one, matches."""
        for cut in range(len(parts), 0, -1):
            head = parts[:cut]
            hits = [f for f, pth in self.path.items() if _module_of(pth)[-cut:] == head]
            if len(hits) == 1:
                return hits[0], parts[cut:]
            if len(hits) > 1:
                return None
        return None

    def lands(self, dotted, from_path, target_file, hops=4):
        """Where following dotted from from_path ends, compared with
        target_file: 'direct' (the file it names), 're_export' (reached
        through imports, star imports included), 'attribute_path' (the path
        goes on through an attribute this check can not follow, such as
        an instance or a class's members), 'external' (no file in the repo,
        or several, match) or 'elsewhere' (another file, which neither
        defines nor imports the name)."""
        return self._lands(_resolve_relative(dotted, from_path), target_file, hops, 0)

    def _lands(self, parts, target_file, hops, hop):
        hit = self.file_for(parts)
        if hit is None:
            return "external"
        fid, names = hit
        if fid == target_file:
            return "direct" if hop == 0 else "re_export"
        if hop >= hops or not names:
            return "elsewhere"
        if names[0] in self.imports.get(fid, {}):
            nxt = _resolve_relative(self.imports[fid][names[0]], self.path[fid]) + names[1:]
            return self._lands(nxt, target_file, hops, hop + 1)
        for star in self.stars.get(fid, ()):
            got = self._lands(_resolve_relative(star, self.path[fid]) + names, target_file, hops, hop + 1)
            if got in ("direct", "re_export"):
                return "re_export"
        return "attribute_path" if len(names) > 1 else "elsewhere"


def audit(kg):
    nodes, edges = kg["nodes"], kg["edges"]
    by_id = {}
    found = {k: [] for k in HARD + CHECKS}
    counted = Counter()
    notes = Counter()

    ids = Counter(n["id"] for n in nodes)
    for nid, c in ids.items():
        if c > 1:
            found["duplicate_node"].append(nid)
    for n in nodes:
        by_id[n["id"]] = n

    def label(nid):
        n = by_id.get(nid)
        if n is None:
            return nid
        m = n.get("metadata") or {}
        return f"{n['type']} {m.get('qualname') or n['label']} ({m.get('filepath') or m.get('path') or ''}:{m.get('lineno', '')})"

    keys = Counter((e["relation"], e["source"], e["target"]) for e in edges)
    for k, c in keys.items():
        if c > 1:
            found["duplicate_edge"].append(k)

    live = []
    for e in edges:
        s, t = by_id.get(e["source"]), by_id.get(e["target"])
        if s is None or t is None:
            found["dangling"].append((e["relation"], e["source"], e["target"]))
            continue
        allowed = ENDPOINTS.get(e["relation"])
        if allowed is None or s["type"] not in allowed[0] or t["type"] not in allowed[1]:
            found["endpoint_types"].append((e["relation"], s["type"], t["type"], label(e["source"]), label(e["target"])))
            continue   # reported once, here; the content checks assume the types
        live.append(e)

    parents = Counter(e["target"] for e in live if e["relation"] == "contains")
    for n in nodes:
        if n["type"] in DEFINED and parents[n["id"]] != 1:
            found["contains_parent"].append((parents[n["id"]], label(n["id"])))

    # Class hierarchy from inherits edges.
    bases = defaultdict(set)
    subs = defaultdict(set)
    for e in live:
        if e["relation"] == "inherits":
            bases[e["source"]].add(e["target"])
            subs[e["target"]].add(e["source"])

    def closure(start, graph):
        seen, todo = set(), [start]
        while todo:
            c = todo.pop()
            for nxt in graph.get(c, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    method_class = {}
    container = {}
    for e in live:
        if e["relation"] == "contains":
            container[e["target"]] = e["source"]
            if by_id[e["source"]]["type"] == "class":
                method_class[e["target"]] = e["source"]

    def file_of(nid):
        seen = set()
        while nid in container and nid not in seen:
            seen.add(nid)
            nid = container[nid]
        return nid if by_id.get(nid, {}).get("type") in FILES else None

    aliases = _import_aliases(nodes, live)
    modules = ModuleMap(nodes, live)
    sites = {}

    def sites_of(nid):
        if nid not in sites:
            m = by_id[nid].get("metadata") or {}
            tree = _parse(m.get("source_code"))
            if tree is None:
                notes["caller source does not parse (its call_name cases count as broken)"] += 1
            sites[nid] = CallSites(tree, m.get("decorators") or ())
        return sites[nid]

    for e in live:
        rel, s, t = e["relation"], by_id[e["source"]], by_id[e["target"]]
        meta = e.get("metadata") or {}
        tname = t["label"]
        if rel == "calls" and s["type"] in CALLABLE:
            counted["call_name"] += 1
            cs = sites_of(e["source"])
            ok = tname in cs.names
            if not ok:
                f = file_of(e["source"])
                ok = any(a in cs.names for a in aliases.get(f, {}).get(tname, ()))
                if ok:
                    notes["call_name: called by its import alias"] += 1
            if not ok and (tname in cs.decorator_names or any(
                    a in cs.decorator_names for a in aliases.get(file_of(e["source"]), {}).get(tname, ()))):
                ok = True
                notes["call_name: called in a decorator (e.g. parametrize arguments)"] += 1
            if not ok and tname == "__init__":
                cls = method_class.get(e["target"])
                ok = cls is not None and (by_id[cls]["label"] in cs.names or cs.super_call)
                if ok:
                    notes["call_name: __init__ via construction or super()"] += 1
            if not ok and tname.startswith("__") and tname.endswith("__") and (cs.operators or cs.super_call):
                ok = True
                notes["call_name: dunder with operator syntax or super()"] += 1
            if not ok:
                found["call_name"].append((tname, label(e["source"]), label(e["target"]), meta.get("confidence")))

            q = meta.get("import_resolved")
            if q:
                counted["qualified_module"] += 1
                where = modules.lands(q, modules.path.get(file_of(e["source"]), ""), file_of(e["target"]))
                if where == "re_export":
                    notes["qualified_module: reached through a re-export"] += 1
                elif where == "attribute_path":
                    notes["qualified_module: not checkable, the path continues through an attribute"] += 1
                elif where != "direct":
                    found["qualified_module"].append((where, q, label(e["source"]), label(e["target"])))

            if e["source"] == e["target"]:
                counted["self_loop"] += 1
                own = by_id[method_class[e["source"]]]["label"] if e["source"] in method_class else None
                recv = cs.names.get(tname, set()) | ({""} if tname in cs.decorator_names else set())
                if not recv & {"", "self", "cls", "Call", own}:
                    found["self_loop"].append((tname, sorted(recv), label(e["source"])))

            cls = method_class.get(e["source"])
            if s["type"] == "method" and cls and t["type"] == "method" and tname in cs.names \
                    and cs.names[tname] == {"self"}:
                counted["self_call_class"] += 1
                tcls = method_class.get(e["target"])
                family = {cls} | closure(cls, bases) | closure(cls, subs)
                if tcls not in family:
                    found["self_call_class"].append((tname, label(e["source"]), label(e["target"])))
                elif tcls in closure(cls, subs):
                    notes["self_call_class: lands on a subclass override"] += 1

        elif rel == "accesses":
            counted["accesses_property"] += 1
            decs = (t.get("metadata") or {}).get("decorators") or []
            if not any(str(d).split(".")[-1].split("(")[0] in PROPERTY_DECORATORS for d in decs):
                found["accesses_property"].append((tname, decs, label(e["source"]), label(e["target"])))

        elif rel == "inherits":
            counted["inherits_base"] += 1
            sb = (s.get("metadata") or {}).get("bases") or []
            names = {b.split("[")[0].split("(")[0].split(".")[-1] for b in sb}
            f = file_of(e["source"])
            local = set(names)
            for orig, al in aliases.get(f, {}).items():
                if al & names:
                    local.add(orig)
            if tname not in local:
                found["inherits_base"].append((tname, sb, label(e["source"]), label(e["target"])))

        elif rel == "overrides":
            counted["overrides_parent"] += 1
            sc, tc = method_class.get(e["source"]), method_class.get(e["target"])
            if s["label"] != tname or tc is None or sc is None or tc not in closure(sc, bases):
                found["overrides_parent"].append((label(e["source"]), label(e["target"])))

    return found, counted, notes


def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def build_from_clone(clone, workdir):
    from kg_construction.kg.builder import RepoKGBuilder
    dest = Path(workdir) / clone.name
    dest.mkdir(parents=True)
    archive = subprocess.Popen(["git", "--git-dir", str(clone), "archive", "HEAD"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise RuntimeError(f"git archive failed for {clone}")
    return RepoKGBuilder().build_from_dir(clone.name, dest)


def report(name, kg, found, counted, notes, out):
    edges = Counter(e["relation"] for e in kg["edges"])
    print(f"\n## {name}: {len(kg['nodes'])} nodes, {len(kg['edges'])} edges "
          f"(schema {(kg.get('metadata') or {}).get('schema_version')})\n", file=out)
    print("| rule | kind | checked | broken |\n|---|---|---|---|", file=out)
    for r in HARD:
        checked = len(kg["nodes"]) if r in ("duplicate_node", "contains_parent") else len(kg["edges"])
        print(f"| {r} | must hold | {checked} | {len(found[r])} |", file=out)
    for r in CHECKS:
        print(f"| {r} | check | {counted[r]} | {len(found[r])} |", file=out)
    if notes:
        print("\nAccepted by a named exception (counted in `checked`, not in `broken`):\n", file=out)
        for k, v in sorted(notes.items()):
            print(f"- {k}: {v}", file=out)
    for r in HARD + CHECKS:
        if found[r]:
            print(f"\n<details><summary>{r}: first {min(EXAMPLES, len(found[r]))} of {len(found[r])}</summary>\n", file=out)
            for ex in found[r][:EXAMPLES]:
                print(f"- `{ex}`", file=out)
            print("\n</details>", file=out)
    return edges


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", nargs="*", default=["requests", "pytest"],
                    help="short names matched against *__<name> in --cache-dir")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "repo_cache"))
    ap.add_argument("--kg-json", nargs="*", default=[], help="check saved KG files instead of building")
    ap.add_argument("--save-dir", help="also write each built KG here as <clone>.json")
    ap.add_argument("--json-out", help="write every broken case as JSON")
    args = ap.parse_args(argv)

    out = sys.stdout
    try:
        own = _git("-C", str(REPO_ROOT), "rev-parse", "--short=10", "HEAD")
        dirty = " (uncommitted changes)" if _git("-C", str(REPO_ROOT), "status", "--porcelain", "src") else ""
    except subprocess.CalledProcessError:
        own, dirty = "unknown", ""
    print(f"# KG edge audit\n\npycodekg {own}{dirty}", file=out)

    kgs = []
    for p in args.kg_json:
        kgs.append((Path(p).name, json.load(open(p))))
    if not args.kg_json:
        cache = Path(args.cache_dir)
        with tempfile.TemporaryDirectory() as tmp:
            for short in args.repos:
                matches = sorted(cache.glob(f"*__{short}"))
                if not matches:
                    raise SystemExit(f"no clone matching *__{short} in {cache}")
                clone = matches[0]
                print(f"repo {clone.name} at {_git('--git-dir', str(clone), 'rev-parse', '--short=10', 'HEAD')}", file=out)
                kg = build_from_clone(clone, tmp)
                if args.save_dir:
                    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                    json.dump(kg, open(Path(args.save_dir) / f"{clone.name}.json", "w"))
                kgs.append((clone.name, kg))

    broken_hard = 0
    dump = {}
    for name, kg in kgs:
        found, counted, notes = audit(kg)
        report(name, kg, found, counted, notes, out)
        broken_hard += sum(len(found[r]) for r in HARD)
        dump[name] = {r: [list(x) if isinstance(x, tuple) else x for x in v] for r, v in found.items()}
    if args.json_out:
        json.dump(dump, open(args.json_out, "w"), indent=1, default=str)
    return 1 if broken_hard else 0


if __name__ == "__main__":
    sys.exit(main())
