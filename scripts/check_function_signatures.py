"""Audit function signature metadata at fixed git revisions, without a KG.

Every regular .py file from each clone's git archive is inspected (including
docs/examples/tests, unlike the KG builder's directory filters). Bare clones
are supported. The signature must parse, adding an ellipsis body if needed,
and declare the original kind/name, decorators, parameters, return annotation
and type parameters. Indented fragments are wrapped without dedenting string
contents. Inline suites already have a body and need no appended ellipsis.

Run with the helper version to measure on PYTHONPATH:
  PYTHONPATH=src /path/to/python scripts/check_function_signatures.py \
    --output /tmp/before.json /path/to/repo_cache/psf__requests [...]
  # After the fix, add --compare-json /tmp/before.json.

JSON contains every signature and a full-source hash for before/after checks.
The console reports counts and a bounded sample, not every declaration.
"""

import argparse
import ast
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tokenize

from kg_construction.ast.helpers import _get_signature_and_source_text, split_source_lines


def declaration(node):
    return (type(node).__name__, node.name,
            [ast.dump(d) for d in node.decorator_list], ast.dump(node.args),
            ast.dump(node.returns) if node.returns is not None else None,
            [ast.dump(p) for p in getattr(node, "type_params", [])])


def parsed_signature(signature):
    """Parse a stored fragment, with a neutral wrapper for its indentation."""
    if not signature:
        raise SyntaxError("empty signature")
    first = signature.split("\n", 1)[0]
    indent = first[:len(first) - len(first.lstrip())]
    def parse(text):
        module = ast.parse("if True:\n" + text if indent else text)
        statements = module.body[0].body if indent else module.body
        if len(statements) != 1 or not isinstance(statements[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise SyntaxError("signature does not declare exactly one function")
        return statements[0]
    try:
        return parse(signature + "\n")
    except SyntaxError:
        return parse(signature + "\n" + indent + "    ...\n")


def check(repo, rev):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", rev], text=True).strip()
    data = subprocess.check_output(["git", "-C", str(repo), "archive", commit])
    result = {"repo": repo.name, "commit": commit, "functions": 0, "unparseable": 0,
              "declarations_differ": 0, "decorators_only_differ": 0,
              "files_skipped": [], "examples": [], "signatures": {}}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = sorted((m for m in archive if m.isfile() and m.name.endswith(".py")),
                         key=lambda m: m.name)
        for member in members:
            try:
                raw = archive.extractfile(member).read()
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                source = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
                tree = ast.parse(source, filename=member.name)
            except (SyntaxError, UnicodeError, ValueError) as e:
                result["files_skipped"].append({"path": member.name, "error": str(e)})
                continue
            lines = split_source_lines(source)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                result["functions"] += 1
                signature, full_source = _get_signature_and_source_text(node, lines)
                where = f"{member.name}:{node.lineno} {node.name}"
                result["signatures"][where] = [signature, hashlib.sha256(full_source.encode()).hexdigest()]
                try:
                    got = parsed_signature(signature)
                except SyntaxError as e:
                    result["unparseable"] += 1
                    if len(result["examples"]) < 8:
                        result["examples"].append({"where": where, "kind": "unparseable",
                                                   "error": str(e), "signature": signature})
                    continue
                expected, actual = declaration(node), declaration(got)
                if actual != expected:
                    result["declarations_differ"] += 1
                    if actual[:2] + actual[3:] == expected[:2] + expected[3:]:
                        result["decorators_only_differ"] += 1
                    if len(result["examples"]) < 8:
                        result["examples"].append({"where": where, "kind": "declaration differs",
                                                   "signature": signature})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="+", type=Path)
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-json", type=Path)
    parser.add_argument("--renderer", action="append", type=Path, default=[],
                        help="Also verify this renderer ignores signature-only changes.")
    args = parser.parse_args()
    results = [check(repo, args.rev) for repo in args.repos]
    if args.compare_json:
        before = {r["repo"]: r for r in json.loads(args.compare_json.read_text())["results"]}
        for result in results:
            old = before[result["repo"]]
            if result["commit"] != old["commit"] or result["signatures"].keys() != old["signatures"].keys():
                raise SystemExit(f"Input revision/functions changed for {result['repo']}")
            result["signature_changes"] = sum(v[0] != old["signatures"][k][0]
                                              for k, v in result["signatures"].items())
            result["full_source_changes"] = sum(v[1] != old["signatures"][k][1]
                                                for k, v in result["signatures"].items())
    renderers = []
    for i, path in enumerate(args.renderer):
        spec = importlib.util.spec_from_file_location(f"signature_renderer_{i}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        render = getattr(module, "_build_context", None) or module._build_prompt
        # Exercise every place LLMSerializer carries a function signature.
        snippet = {"name": "neighbour", "signature": "OLD_SIGNATURE_SENTINEL",
                   "source_code": "def neighbour(): return 1"}
        serialized = {"seed": [{"function_name": "f", "signature": "OLD_SIGNATURE_SENTINEL",
                                  "source_code": "def f(): return neighbour()"}],
                      "context": {"callers": [dict(snippet)], "callees": [dict(snippet)],
                                  "sibling_methods": [dict(snippet)], "related": []}}
        before_text = render(serialized)
        serialized["seed"][0]["signature"] = "NEW_SIGNATURE_SENTINEL"
        for category in ("callers", "callees", "sibling_methods"):
            serialized["context"][category][0]["signature"] = "NEW_SIGNATURE_SENTINEL"
        unchanged = render(serialized) == before_text
        renderers.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "signature_only_change_leaves_text_unchanged": unchanged})
        if not unchanged:
            raise SystemExit(f"Signature-only changes affect rendered text: {path}")
    args.output.write_text(json.dumps({"results": results, "renderer_checks": renderers}, indent=2) + "\n")
    print("repos: " + ", ".join(f"{r['repo']} @ {r['commit'][:10]}" for r in results))
    print("| Repo | Functions | Unparseable | Declaration differs | Decorators only | Skipped files |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['repo']} | {r['functions']} | {r['unparseable']} | "
              f"{r['declarations_differ']} | {r['decorators_only_differ']} | {len(r['files_skipped'])} |")
        if "signature_changes" in r:
            print(f"  signature changes: {r['signature_changes']}; full source changes: {r['full_source_changes']}")
    for r in results:
        print(f"\n{r['repo']} examples:")
        for example in r["examples"]:
            print(json.dumps(example))
        for failure in r["files_skipped"]:
            print("  skipped: " + json.dumps(failure))
    for renderer in renderers:
        print("renderer check: " + json.dumps(renderer))


if __name__ == "__main__":
    main()
