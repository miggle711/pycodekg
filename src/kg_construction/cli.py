"""
cli.py

Scriptable, non-interactive command-line interface (kg_construction#83).

    pkg-run build psf/requests --commit a0df2cbb     # clone-by-SHA (existing path)
    pkg-run build . --name my-local-repo             # local working tree, no git
    pkg-run query kg_output/kg_x.json --callers send
    pkg-run query kg_output/kg_x.json --callees send
    pkg-run query kg_output/kg_x.json --file sessions.py
    pkg-run query kg_output/kg_x.json --tests send
    pkg-run query kg_output/kg_x.json --class Session
    pkg-run query kg_output/kg_x.json --list-files
    pkg-run query kg_output/kg_x.json --list-functions
    pkg-run query kg_output/kg_x.json --list-classes
    pkg-run query kg_output/kg_x.json --export send,resolve_redirects

Every query subcommand prints a human-readable listing by default;
pass --json for the raw dict, for scripting/piping.

Running `pkg-run` with no arguments falls back to the pre-existing
interactive wizard (kg_construction.pipeline._interactive_mode) --
nothing that already depends on that entry point breaks.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from kg_construction.kg.builder import RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine


def _print_or_json(data, as_json: bool, human_fn) -> None:
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        human_fn(data)


def _cmd_build(args: argparse.Namespace) -> int:
    builder = RepoKGBuilder(max_workers=args.max_workers)

    if args.commit:
        kg = builder.build(args.repo, args.commit)
        builder.save(args.repo, kg)
    else:
        local_path = Path(args.repo)
        if not local_path.exists():
            hint = ""
            if re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
                hint = " To build a GitHub repo, pass a commit with --commit <sha>."
            print(f"error: '{args.repo}' does not exist.{hint}", file=sys.stderr)
            return 2
        if not local_path.is_dir():
            print(f"error: '{args.repo}' is not a directory.", file=sys.stderr)
            return 2

        name = args.name or args.repo
        print(f"Building local KG for '{name}' from {args.repo}...")
        kg = builder.build_from_dir(name, local_path)
        # Saving a KG with no files would leave an empty cache entry that
        # looks like a real build.
        if kg.get('metadata', {}).get('file_count') == 0:
            print(f"error: no Python files found under '{args.repo}'.", file=sys.stderr)
            return 1
        builder.save_local(name, kg)

    return 0


def _load_engine(kg_path: str) -> KGQueryEngine:
    with open(kg_path) as f:
        kg = json.load(f)
    return KGQueryEngine(kg)


def _fmt_func(node: Dict) -> str:
    meta = node.get('metadata', {})
    filepath = meta.get('filepath', meta.get('path', '?'))
    lineno = meta.get('lineno', '?')
    cls = meta.get('class')
    label = f"{cls}.{node['label']}" if cls else node['label']
    return f"{filepath}:{lineno} -- {label} ({node['type']})"


def _resolve_function(engine: KGQueryEngine, name: str) -> Optional[Dict]:
    matches = engine.find_function_by_name(name)
    if not matches:
        print(f"No function/method named '{name}' found in this KG.", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"'{name}' is ambiguous ({len(matches)} matches); using the first:", file=sys.stderr)
        for m in matches:
            print(f"  {_fmt_func(m)}", file=sys.stderr)
    return matches[0]


def _resolve_class(engine: KGQueryEngine, name: str) -> Optional[Dict]:
    matches = engine.find_class_by_name(name)
    if not matches:
        print(f"No class named '{name}' found in this KG.", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"'{name}' is ambiguous ({len(matches)} matches); using the first:", file=sys.stderr)
        for m in matches:
            print(f"  {m['metadata'].get('filepath', '?')}:{m['metadata'].get('lineno', '?')} -- {m['label']}", file=sys.stderr)
    return matches[0]


def _cmd_query(args: argparse.Namespace) -> int:
    engine = _load_engine(args.kg_file)

    if args.callers:
        node = _resolve_function(engine, args.callers)
        if node is None:
            return 1
        results = engine.find_callers(node['id'])
        _print_or_json(
            results, args.json,
            lambda data: print(f"Callers of {args.callers}:\n" + "\n".join(
                f"  {_fmt_func(n)}" for n in data
            ) if data else f"No callers found for {args.callers}.")
        )
        return 0

    if args.callees:
        node = _resolve_function(engine, args.callees)
        if node is None:
            return 1
        results = engine.find_callees(node['id'])
        _print_or_json(
            results, args.json,
            lambda data: print(f"Callees of {args.callees}:\n" + "\n".join(
                f"  {_fmt_func(n)}" for n in data
            ) if data else f"No callees found for {args.callees}.")
        )
        return 0

    if args.file:
        file_matches = engine.find_file(args.file)
        if not file_matches:
            print(f"No file matching '{args.file}' found in this KG.", file=sys.stderr)
            return 1
        if len(file_matches) > 1:
            print(f"'{args.file}' is ambiguous ({len(file_matches)} matches); using the first:", file=sys.stderr)
            for m in file_matches:
                print(f"  {m['metadata'].get('path')}", file=sys.stderr)
        contents = engine.get_file_contents(file_matches[0]['id'])
        _print_or_json(contents, args.json, _print_file_contents)
        return 0

    if args.tests:
        node = _resolve_function(engine, args.tests)
        if node is None:
            return 1
        results = engine.find_test_functions_for(node['id'])
        _print_or_json(
            results, args.json,
            lambda data: print(f"Existing tests for {args.tests}:\n" + "\n".join(
                f"  {_fmt_func(n)}" for n in data
            ) if data else f"No existing tests found for {args.tests}.")
        )
        return 0

    if args.cls:
        node = _resolve_class(engine, args.cls)
        if node is None:
            return 1
        results = engine.get_class_methods(node['id'])
        _print_or_json(
            results, args.json,
            lambda data: print(f"Methods of {args.cls}:\n" + "\n".join(
                f"  {_fmt_func(n)}" for n in data
            ) if data else f"No methods found for {args.cls}.")
        )
        return 0

    if args.list_files:
        results = engine.get_files()
        _print_or_json(
            results, args.json,
            lambda data: print("Files:\n" + "\n".join(
                f"  {n['metadata'].get('path', '?')}" for n in data
            ) if data else "No files found.")
        )
        return 0

    if args.list_functions:
        results = engine.get_functions()
        _print_or_json(
            results, args.json,
            lambda data: print("Functions:\n" + "\n".join(
                f"  {_fmt_func(n)}" for n in data
            ) if data else "No functions found.")
        )
        return 0

    if args.list_classes:
        results = engine.nodes_by_type.get('class', [])
        _print_or_json(
            results, args.json,
            lambda data: print("Classes:\n" + "\n".join(
                f"  {n['metadata'].get('filepath', '?')}:{n['metadata'].get('lineno', '?')} -- {n['label']}"
                for n in data
            ) if data else "No classes found.")
        )
        return 0

    if args.export:
        names = [n.strip() for n in args.export.split(',') if n.strip()]
        node_ids = []
        for name in names:
            node = _resolve_function(engine, name)
            if node is None:
                return 1
            node_ids.append(node['id'])
        subgraph = engine.export_subgraph(node_ids)
        _print_or_json(subgraph, args.json, lambda data: print(json.dumps(data, indent=2)))
        return 0

    print("Specify one of --callers, --callees, --file, --tests, --class, "
          "--list-files, --list-functions, --list-classes, --export.", file=sys.stderr)
    return 1


def _print_file_contents(contents: Dict) -> None:
    file_node = contents.get('file', {})
    print(f"{file_node.get('metadata', {}).get('path', '?')}")
    for cls in contents.get('classes', []):
        print(f"  class {cls['label']} (line {cls['metadata'].get('lineno', '?')})")
    for fn in contents.get('functions', []):
        print(f"  {_fmt_func(fn)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='pkg-run', description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')

    build_p = subparsers.add_parser('build', help='Build a KG for a repo or local directory')
    build_p.add_argument('repo', help="'owner/name' for a GitHub repo, or a local directory path")
    build_p.add_argument('--commit', help='Commit SHA (clone-by-SHA path). Omit for a local directory.')
    build_p.add_argument('--name', help='Label for a local build (defaults to the directory path)')
    build_p.add_argument('--max-workers', type=int, default=4)
    build_p.set_defaults(func=_cmd_build)

    query_p = subparsers.add_parser('query', help='Query an already-built KG JSON file')
    query_p.add_argument('kg_file', help='Path to a KG JSON file (from build --commit or build_from_dir)')
    query_p.add_argument('--callers', metavar='FUNC_NAME', help='List callers of a function/method')
    query_p.add_argument('--callees', metavar='FUNC_NAME', help='List callees of a function/method')
    query_p.add_argument('--file', metavar='PATH', help='List classes/functions defined in a file')
    query_p.add_argument('--tests', metavar='FUNC_NAME', help='List existing tests that call a function/method')
    query_p.add_argument('--class', dest='cls', metavar='CLASS_NAME', help='List methods of a class')
    query_p.add_argument('--list-files', action='store_true', help='List all files in the KG')
    query_p.add_argument('--list-functions', action='store_true', help='List all functions/methods in the KG')
    query_p.add_argument('--list-classes', action='store_true', help='List all classes in the KG')
    query_p.add_argument('--export', metavar='FUNC_NAME,...',
                          help='Export a subgraph (seed nodes + immediate neighbors) for comma-separated function names')
    query_p.add_argument('--json', action='store_true', help='Print raw JSON instead of a human-readable listing')
    query_p.set_defaults(func=_cmd_query)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        from kg_construction.pipeline import _interactive_mode
        _interactive_mode()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
