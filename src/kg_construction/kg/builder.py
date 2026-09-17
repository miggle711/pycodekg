"""
repo_kg_builder.py

Builds structural Knowledge Graphs (KGs) from Python repository source code.

Given a GitHub repo and a commit SHA, this module:
  1. Clones the repo as a bare git mirror (cached locally)
  2. Extracts the source tree at that commit via git archive
  3. Parses every .py file with Python's ast module in parallel
  4. Emits nodes (file, class, function, method, test_function, import) and
     edges (contains, imports, calls, accesses, inherits, tests, uses,
     overrides, depends_on, module_depends_on) into a JSON KG

Node metadata includes: signatures, type annotations, default values,
decorators, docstrings, raised/caught exceptions, branch counts,
assert patterns (for test functions), class attributes, module constants,
and __all__ exports.

Output format:
    {
        "nodes": [{"id": ..., "type": ..., "label": ..., "metadata": {...}}, ...],
        "edges": [{"source": ..., "target": ..., "relation": ..., "metadata": {...}}, ...],
        "metadata": {"repo": ..., "base_commit": ..., "file_count": ..., "total_lines_of_code": ..., "parse_mode": "source", "schema_version": ...,
                     "build_time_parsing_s": ..., "build_time_resolution_s": ...}
    }

Usage:
    builder = RepoKGBuilder()
    kg = builder.build("psf/requests", "<commit_sha>")
    builder.save("psf/requests", kg)

Module layout (post-split):
  - ast_helpers.py   pure AST-in/data-out utilities (_get_signature,
                     _build_func_metadata, etc.) Re-exported here for
                     back-compat with code that imports them from
                     repo_kg_builder directly.
  - repo_manager.py  git clone / archive extraction (RepoManager).
  - repo_kg_builder  this module — KGNode/KGEdge data types, _parse_file
                     (per-file AST → nodes+edges), RepoASTParser (parallel
                     driver + second-pass edge resolution), RepoKGBuilder
                     (top-level entry point).
"""

import ast
import json
import re
import tempfile
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Optional, Tuple, Union

# Re-exported from ast_helpers so test code and external callers can still
# `from kg_construction.kg.builder import _get_signature` etc. without knowing about
# the split. These are imported as names (not *) so linters resolve them.
from kg_construction.ast.helpers import (
    _make_id,
    _is_test_file,
    _safe_unparse,
    _extract_callee_name,
    _extract_call_receiver,
    _extract_property_accesses,
    _extract_operator_dispatches,
    _collect_local_types,
    _get_docstring,
    _get_decorators,
    _get_bare_decorator_names,
    _get_signature,
    _get_exceptions,
    _get_exception_class_names,
    _count_branches,
    _get_assert_patterns,
    _get_base_names,
    _get_class_attributes,
    _get_instantiated_classes_in_class,
    _get_attribute_accesses,
    _get_used_imports,
    _get_instantiated_classes,
    _get_factory_call_sites,
    _build_func_metadata,
    _collect_file_level_info,
)
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.kg import type_inference


# Directories to skip during repo traversal — typically non-source content
SKIP_DIRS = {'docs', 'doc', 'examples', 'example', 'vendor', '.git'}

# Auto-generated Django-style migration filenames (e.g. '0001_initial.py')
# are skipped individually rather than via SKIP_DIRS, since a directory
# literally named 'migrations' isn't reliably generated content -- Django's
# own migration FRAMEWORK code lives at django/db/migrations/ (real,
# hand-written source: autodetector.py, executor.py, serializer.py, etc.),
# which SKIP_DIRS previously excluded wholesale (kg_construction#70).
_GENERATED_MIGRATION_FILENAME = re.compile(r'^\d{4}_')

# Files over this line count are skipped to avoid pathological parse times.
# Raised from 5000: a 400-instance TestGenEval validation run found real,
# actively-maintained files this size (sphinx/domains/cpp.py at 7813 lines,
# matplotlib's lib/matplotlib/axes/_axes.py at 8451) being silently skipped
# at ~2.5% prevalence. ast.parse cost at this size is trivial (~37ms), so
# this cap is a safety net for truly pathological files, not a realistic
# bound on hand-written modules.
MAX_FILE_LINES = 20000

# Bumped whenever the KG node/edge shape changes. Stamped into every build's
# metadata and checked on load, so a cached KG from an older schema is
# rebuilt rather than silently served as if it matched the current shape.
# 2: _make_id widened from 8 to 16 hex chars (collision-risk fix); every
#    node/edge ID in an old cached KG is a different length now.
# 3: nested classes (a class defined inside a function or another class
#    body, e.g. Django's real `class __proxy__` inside `lazy()`) are now
#    emitted as real class/method nodes (kg_construction#102) -- an old
#    cached KG is silently missing these nodes entirely.
SCHEMA_VERSION = 3


@dataclass
class KGNode:
    """A node in the knowledge graph representing a code entity.

    Attributes:
        id: Deterministic 8-char MD5 hash of the entity's qualified name.
            Identical entities across issues/commits map to the same ID.
        type: One of 'file', 'test_file', 'class', 'function', 'method',
              'test_function', 'import'.
        label: Human-readable short name (e.g. filename, class name, function name).
        metadata: Entity-specific data. See _build_func_metadata and _parse_file
                  for the full set of keys per node type.
    """
    id: str
    type: str
    label: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class KGEdge:
    """A directed edge in the knowledge graph representing a relationship.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        relation: Relationship type. One of:
            - 'contains': file→class, file→function, class→method
            - 'imports':  file→import module
            - 'calls':    function/method→function/method (best-effort static analysis)
            - 'accesses': function/method→@property method (attribute reads, no call syntax)
            - 'inherits': class→parent class
        metadata: Edge-specific data, e.g. confidence ('exact'/'ambiguous') for
                  calls and inherits edges resolved in the second pass.
    """
    source: str
    target: str
    relation: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-file parsing (runs in worker processes)
# ---------------------------------------------------------------------------

def _parse_file(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Parse a single .py file and return its nodes and edges.

    This function runs inside worker processes spawned by ProcessPoolExecutor,
    so it must be a module-level function (not a method or closure) to be
    picklable. All helpers it calls must also be importable from the module.

    Args:
        args: Tuple of (repo, rel_path, abs_path).
            repo: e.g. 'psf/requests'
            rel_path: path relative to repo root, e.g. 'requests/sessions.py'
            abs_path: absolute path on disk in the temp extract directory

    Returns:
        Dict with 'nodes', 'edges', and 'factory_sites' lists, or None if the
        file should be skipped (unreadable, over line limit, or has a
        SyntaxError). 'factory_sites' is a list of (rel_path, class_id, line,
        col) tuples recording lowercase-callee assignment sites invisible to
        the uppercase-heuristic 'uses' edges — see _get_factory_call_sites
        and kg/type_inference.py for how these are optionally resolved.

    Node types emitted: file/test_file, import, class, function/method/test_function
    Edge types emitted: contains, imports, calls (unresolved — resolved in second pass)
    """
    repo, rel_path, abs_path = args
    try:
        source = Path(abs_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    source_lines = source.splitlines(keepends=True)

    if source.count('\n') > MAX_FILE_LINES:
        return None

    try:
        tree = ast.parse(source, filename=abs_path)
    except SyntaxError:
        return None

    nodes = []
    edges = []
    
    # Deduplicate call edges within this file at creation time to avoid
    # accumulating one edge per call-site for frequently called functions
    seen_call_targets: Set[Tuple[str, str]] = set()

    # Factory-call assignment sites recorded for optional pyright type
    # resolution (see kg/type_inference.py). Each site is
    # (line, col, source_class_id, source_name), where exactly one of
    # source_class_id / source_name is set:
    #   - source_class_id: the enclosing class (self.x = ... / bare x = ...
    #     inside a method) -- already resolved, matches the existing
    #     class-level-only shape of 'uses' edges emitted via
    #     _get_instantiated_classes_in_class below.
    #   - source_name: the receiver's own class name (other.x = ...), still
    #     unresolved -- _resolve_edges resolves it the same way it already
    #     resolves 'uses' edge targets, via class_label_to_ids.
    factory_sites: List[Tuple[int, int, Optional[str], Optional[str]]] = []

    file_id = _make_id(f"file_{repo}_{rel_path}")
    file_type = 'test_file' if _is_test_file(rel_path) else 'file'
    import_map, exports, constants = _collect_file_level_info(tree)

    nodes.append(asdict(KGNode(
        id=file_id, type=file_type, label=Path(rel_path).name,
        metadata={'path': rel_path, 'repo': repo, 'constants': constants, 'exports': exports}
    )))

    # Emit import nodes and edges
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_id = _make_id(f"import_{repo}_{alias.name}")
                nodes.append(asdict(KGNode(id=mod_id, type='import', label=alias.name,
                                           metadata={'repo': repo})))
                edges.append(asdict(KGEdge(source=file_id, target=mod_id, relation='imports')))
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ''
            for alias in node.names:
                full_name = f"{mod_name}.{alias.name}" if mod_name else alias.name
                imp_id = _make_id(f"import_{repo}_{full_name}")
                nodes.append(asdict(KGNode(id=imp_id, type='import', label=full_name,
                                           metadata={'repo': repo, 'module': mod_name,
                                                     'name': alias.name})))
                edges.append(asdict(KGEdge(source=file_id, target=imp_id, relation='imports')))

    def _emit_call_edges(func_id: str, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                         local_types: Dict[str, str], class_name: Optional[str] = None):
        """Emit one 'calls' edge per unique (caller, callee_name) pair.

        Edges are marked unresolved=True because callee_name is just a string
        at this point — cross-file resolution happens in RepoASTParser.parse_repo
        after all files are parsed and a global name index is built.

        Each edge carries resolution hints so pass 2 can disambiguate:
            class_hint:        enclosing class name when the call is self.method()
            local_type_hint:   inferred class for receivers like x.method() when
                               x = SomeClass() was seen earlier in the function
            import_resolved:   fully-qualified module path if the bare callee
                               name was imported (e.g. 'json' → 'json'),
                               or if the receiver was imported (e.g. 'json' for
                               json.loads → 'json.loads')
            receiver:          raw receiver expression for attribute calls
                               (kept for debugging / future heuristics)
            is_super_call:     True for super().method() -- resolved in pass 2
                               against the caller's own inherits chain
                               instead of the bare-name fallback, which used
                               to link e.g. ErrorList.copy() to an unrelated
                               class's copy() (kg_construction#61).

        Args:
            local_types: Precomputed _collect_local_types(func_node) result,
                         shared with _emit_access_edges to avoid walking the
                         same function body twice for the same data.
        """
        for call in ast.walk(func_node):
            if not isinstance(call, ast.Call):
                continue
            callee = _extract_callee_name(call)
            if not callee or (func_id, callee) in seen_call_targets:
                continue

            # super().method() parses as TWO Call nodes: the outer method
            # call (handled below via is_super_call) and the inner one
            # that constructs the super proxy itself (callee='super',
            # receiver=None) -- not a real call, so it must not fall
            # through to bare-name matching (kg_construction#61).
            if callee == 'super' and _extract_call_receiver(call) is None:
                continue

            seen_call_targets.add((func_id, callee))

            receiver = _extract_call_receiver(call)
            class_hint: Optional[str] = None
            local_type_hint: Optional[str] = None
            import_resolved: Optional[str] = None
            is_super_call = receiver == 'super()'

            if is_super_call:
                pass  # resolved in pass 2 via the caller's own inherits chain
            elif receiver == 'self' and class_name is not None:
                class_hint = class_name
            elif receiver and receiver in local_types:
                local_type_hint = local_types[receiver]
            elif receiver and receiver in import_map:
                # e.g. json.loads where 'json' was imported
                import_resolved = f"{import_map[receiver]}.{callee}"
            else:
                # Bare-name call: foo() where foo was imported
                import_resolved = import_map.get(callee)

            edges.append(asdict(KGEdge(
                source=func_id, target=callee, relation='calls',
                metadata={
                    'unresolved': True,
                    'receiver': receiver,
                    'class_hint': class_hint,
                    'local_type_hint': local_type_hint,
                    'import_resolved': import_resolved,
                    'is_super_call': is_super_call,
                }
            )))

    def _emit_access_edges(func_id: str, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                           local_types: Dict[str, str], class_name: Optional[str] = None):
        """Emit one 'accesses' edge per unique (caller, attr_name) pair.

        Covers @property reads (obj.attr, never obj.attr()) that
        _emit_call_edges has no signal for at all, since a property access
        never appears as an ast.Call node. Edges are marked unresolved=True;
        pass 2 resolves them against an index of @property-decorated
        methods (built the same way class_method_to_ids is, filtered to
        nodes whose decorators include 'property').

        Uses the same hint shape as 'calls' edges (class_hint,
        local_type_hint, import_resolved, receiver) so pass-2 resolution
        can share the same disambiguation logic.

        Args:
            local_types: Precomputed _collect_local_types(func_node) result,
                         shared with _emit_call_edges to avoid walking the
                         same function body twice for the same data.
        """
        seen_access_targets: Set[Tuple[str, str]] = set()
        for attr_name, receiver in _extract_property_accesses(func_node):
            if (func_id, attr_name) in seen_access_targets:
                continue
            seen_access_targets.add((func_id, attr_name))

            class_hint: Optional[str] = None
            local_type_hint: Optional[str] = None
            import_resolved: Optional[str] = None

            if receiver == 'self' and class_name is not None:
                class_hint = class_name
            elif receiver and receiver in local_types:
                local_type_hint = local_types[receiver]
            elif receiver and receiver in import_map:
                import_resolved = f"{import_map[receiver]}.{attr_name}"

            edges.append(asdict(KGEdge(
                source=func_id, target=attr_name, relation='accesses',
                metadata={
                    'unresolved': True,
                    'receiver': receiver,
                    'class_hint': class_hint,
                    'local_type_hint': local_type_hint,
                    'import_resolved': import_resolved,
                }
            )))

    def _emit_operator_edges(func_id: str, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                             local_types: Dict[str, str], class_name: Optional[str] = None):
        """Emit a 'calls' edge for operator dispatch sites (kg_construction#122).

        `a | b` never appears as an ast.Call, Python's own dispatch
        routes it to a.__or__(b). Emitted as relation='calls' with the
        same hint shape _emit_call_edges uses, so it resolves through
        the existing _resolve_call path unchanged, no new resolution
        logic needed.
        """
        for dunder, receiver in _extract_operator_dispatches(func_node):
            if (func_id, dunder) in seen_call_targets:
                continue

            class_hint: Optional[str] = None
            local_type_hint: Optional[str] = None

            if receiver == 'self' and class_name is not None:
                class_hint = class_name
            elif receiver in local_types:
                local_type_hint = local_types[receiver]
            else:
                continue  # receiver's type isn't inferable, drop rather than guess

            # Marked only once resolvable, so an earlier unresolvable
            # site for the same dunder doesn't consume the dedup slot
            # a later, genuinely resolvable site needs (kg_construction#137
            # review).
            seen_call_targets.add((func_id, dunder))

            edges.append(asdict(KGEdge(
                source=func_id, target=dunder, relation='calls',
                metadata={
                    'unresolved': True,
                    'receiver': receiver,
                    'class_hint': class_hint,
                    'local_type_hint': local_type_hint,
                    'import_resolved': None,
                    'is_super_call': False,
                }
            )))

    def _emit_func_edges(func_id: str, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                         class_name: Optional[str] = None):
        """Emit semantic edges for a function or method beyond call relationships.

        Complements _emit_call_edges with two additional edge types:
            depends_on:    imports actually referenced in the function body,
                           resolved directly to import node IDs at emit time
                           (no second pass needed — import nodes already exist).
            tests:         for test_* functions only, emits a self-referential
                           edge keyed on the function's own name so pass 2 can
                           strip the 'test_' prefix and link to the target.

        Note: reads/writes/returns are NOT emitted as edges — attribute and
        type names aren't graph node IDs, so pass 2 (_resolve_edges) can
        never resolve them and would just discard them. That information is
        already captured as node metadata instead (side_effects, data_flows)
        via _build_func_metadata.

        Args:
            func_id: Node ID of the function/method being processed.
            func_node: The AST function node.
            class_name: Name of the enclosing class (unused now that
                        reads/writes edges are gone; kept for signature
                        compatibility with callers).
        """
        # depends_on: imports actually used in this function body
        for qualified in _get_used_imports(func_node, import_map):
            imp_id = _make_id(f"import_{repo}_{qualified}")
            edges.append(asdict(KGEdge(source=func_id, target=imp_id, relation='depends_on')))

        # tests: test function → function under test (resolved in second pass)
        if func_node.name.startswith('test_'):
            edges.append(asdict(KGEdge(
                source=func_id, target=func_node.name, relation='tests',
                metadata={'unresolved': True}
            )))

    def _record_factory_sites(func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                              enclosing_class_id: Optional[str]):
        """Record lowercase factory-call sites with a resolvable 'uses' source.

        Each candidate site from _get_factory_call_sites falls into one of
        three cases:
            receiver is None, enclosing_class_id is set:
                self.x = call() or bare x = call() inside a method -- the
                'uses' edge source is the enclosing class, already known.
            receiver is a name, resolvable via _collect_local_types:
                other.x = call() where other's own type is known from a
                parameter annotation or a prior constructor call in this
                function -- the 'uses' edge source is that resolved class
                name (a string; RepoASTParser._resolve_edges resolves it
                to a node ID the same way it already resolves 'uses' edge
                targets, via class_label_to_ids).
            anything else (bare x = call() with no enclosing class, or a
            receiver whose type can't be determined here):
                skipped -- there's no class to attach the dependency to.
        """
        sites = _get_factory_call_sites(func_node)
        if not sites:
            return
        for line, col, receiver in sites:
            if receiver is None:
                if enclosing_class_id:
                    factory_sites.append((line, col, enclosing_class_id, None))
                continue
            # Position-aware: only consider assignments strictly before this
            # site's own line. A whole-function summary (no before_line)
            # would let a LATER reassignment of `receiver` overwrite the
            # type it actually held at this call, misattributing the
            # resulting 'uses' edge to the wrong class.
            local_types = _collect_local_types(func_node, before_line=line + 1)
            source_name = local_types.get(receiver)
            if source_name:
                factory_sites.append((line, col, None, source_name))

    def _emit_class(
        class_node: ast.ClassDef,
        container_id: str,
        qualname: str,
    ):
        """Emit a class node (and everything nested inside it) and a
        'contains' edge from container_id; shared by both a top-level
        class and a class nested inside another class/function body
        (kg_construction#102/#85: a class this deeply nested -- e.g.
        Django's real `class __proxy__(Promise):` inside its `lazy()`
        factory function -- previously had no KG node at all, so a patch
        changing it had nothing to seed from, and seed lookup silently
        fell back to every same-named method across the whole file
        instead, producing a spurious ambiguous-seed error).

        Args:
            container_id: Node ID of whatever directly contains this class
                          -- a file, another class, or a function.
            qualname: Dotted qualified name (e.g. 'Klass' for a top-level
                      class, 'lazy.__proxy__' for one nested inside a
                      function, 'Outer.Inner' for one nested inside
                      another class) -- used to build a unique node ID.
        """
        bases = _get_base_names(class_node)
        class_id = _make_id(f"class_{repo}_{rel_path}_{qualname}")
        nodes.append(asdict(KGNode(
            id=class_id, type='class', label=class_node.name,
            metadata={
                'filepath': rel_path, 'repo': repo, 'lineno': class_node.lineno,
                'bases': bases,
                'decorators': _get_decorators(class_node),
                'docstring': _get_docstring(class_node),
                'attributes': _get_class_attributes(class_node),
            }
        )))
        edges.append(asdict(KGEdge(source=container_id, target=class_id, relation='contains')))

        # Inheritance edges are emitted unresolved here; the second pass
        # in parse_repo resolves base names to actual class node IDs
        for base in bases:
            edges.append(asdict(KGEdge(
                source=class_id, target=base, relation='inherits',
                metadata={'unresolved': True}
            )))

        # uses: classes instantiated within any method of this class
        for inst_cls in _get_instantiated_classes_in_class(class_node):
            edges.append(asdict(KGEdge(source=class_id, target=inst_cls, relation='uses',
                                       metadata={'unresolved': True})))

        for child in ast.iter_child_nodes(class_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _emit_function(
                    child, container_id=class_id, qualname=f"{qualname}.{child.name}",
                    parent_class=class_node.name, enclosing_class_id=class_id,
                    bases=bases,
                )
            elif isinstance(child, ast.ClassDef):
                _emit_class(child, container_id=class_id, qualname=f"{qualname}.{child.name}")

    def _emit_function(
        func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        container_id: str,
        qualname: str,
        parent_class: Optional[str] = None,
        parent_function: Optional[str] = None,
        enclosing_class_id: Optional[str] = None,
        bases: Optional[List[str]] = None,
    ):
        """Emit a function/method node (and everything nested inside it)
        and a 'contains' edge from container_id -- shared by a top-level
        function, a class's own method, and a function/class nested
        inside a function body (closures, kg_construction#74).

        Args:
            container_id: Node ID of whatever directly contains this
                          function -- a file, a class, or another function.
            qualname: Dotted qualified name, used to build a unique node ID.
            parent_class: Set only when this is a direct method of a class
                          (not a closure nested further inside a method).
            parent_function: Set only when this is a closure nested inside
                             another function/method, not a direct method.
            enclosing_class_id: The real class node ID this function
                                belongs to (directly, as a method, or
                                transitively via an enclosing method) --
                                threaded through so a nested closure's own
                                factory-call sites still resolve against
                                the real class, not just its immediate
                                function.
            bases: The enclosing class's own base names, for 'overrides'
                  guesses -- only meaningful when parent_class is set.
        """
        func_type = 'test_function' if func_node.name.startswith('test_') else (
            'method' if parent_class is not None else 'function'
        )
        func_id = _make_id(f"func_{repo}_{rel_path}_{qualname}")
        nodes.append(asdict(KGNode(
            id=func_id, type=func_type, label=func_node.name,
            metadata=_build_func_metadata(func_node, rel_path, repo,
                                          parent_class=parent_class,
                                          parent_function=parent_function,
                                          import_map=import_map,
                                          source_lines=source_lines)
        )))
        edges.append(asdict(KGEdge(source=container_id, target=func_id, relation='contains')))

        if parent_class is not None and func_node.name != '__init__':
            for base in (bases or []):
                edges.append(asdict(KGEdge(
                    source=func_id, target=f"{base}.{func_node.name}", relation='overrides',
                    metadata={'unresolved': True}
                )))

        local_types = _collect_local_types(func_node)
        _emit_call_edges(func_id, func_node, local_types, class_name=parent_class)
        _emit_access_edges(func_id, func_node, local_types, class_name=parent_class)
        _emit_func_edges(func_id, func_node, class_name=parent_class)
        _emit_operator_edges(func_id, func_node, local_types, class_name=parent_class)

        # raises/decorated_by: unresolved here, resolved in _resolve_edges
        # against the repo's own class/function index, same pattern as
        # inherits/uses/overrides above (kg_construction#112, #113).
        exc_names = _get_exception_class_names(func_node)
        for exc_name in exc_names['raises'] + exc_names['catches']:
            edges.append(asdict(KGEdge(
                source=func_id, target=exc_name, relation='raises',
                metadata={'unresolved': True}
            )))
        for dec_name in _get_bare_decorator_names(func_node):
            edges.append(asdict(KGEdge(
                source=func_id, target=dec_name, relation='decorated_by',
                metadata={'unresolved': True}
            )))

        _record_factory_sites(func_node, enclosing_class_id=enclosing_class_id)

        for child in ast.iter_child_nodes(func_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _emit_function(
                    child, container_id=func_id, qualname=f"{qualname}.{child.name}",
                    parent_function=func_node.name, enclosing_class_id=enclosing_class_id,
                )
            elif isinstance(child, ast.ClassDef):
                _emit_class(child, container_id=func_id, qualname=f"{qualname}.{child.name}")

    # Emit class, method, and top-level function nodes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            _emit_class(node, container_id=file_id, qualname=node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _emit_function(node, container_id=file_id, qualname=node.name)

    factory_sites_out = [
        (rel_path, source_class_id, source_name, line, col)
        for line, col, source_class_id, source_name in factory_sites
    ]

    return {'nodes': nodes, 'edges': edges, 'factory_sites': factory_sites_out}


# ---------------------------------------------------------------------------
# Parallel driver and second-pass resolution
# ---------------------------------------------------------------------------

class RepoASTParser:
    """Parses all Python files in a repo directory and assembles a structural KG.

    File parsing runs in parallel via ProcessPoolExecutor. After all files
    are parsed, a second pass resolves unresolved 'calls' and 'inherits'
    edges by matching callee/base names against a global node index.

    Call resolution is best-effort:
        - 'qualified': resolved via class_hint, local_type_hint, or import_resolved
        - 'exact': only one function with that name exists in the repo
        - 'ambiguous': multiple functions share the name (e.g. common names
          like 'get' or '__init__'); all candidates are linked
        - Calls to external libraries (no match in repo) are dropped
    """

    def __init__(self, max_workers: int = 4, infer_types: bool = False):
        """
        Args:
            max_workers: Number of parallel worker processes for file parsing.
                         Set to 1 for debugging to get synchronous tracebacks.
            infer_types: If True, run an additional pyright-backed resolution
                         pass (Pass 1.5) to catch 'uses' edges from lowercase
                         factory-function calls that the uppercase heuristic
                         can't see (e.g. `session = requests.session()`). See
                         kg/type_inference.py. Off by default: requires the
                         optional `pyright` dependency and adds a per-repo
                         subprocess cost; failures degrade gracefully to
                         "no enrichment" either way, so this never breaks a
                         build even if pyright is missing or crashes.
        """
        self.max_workers = max_workers
        self.infer_types = infer_types

    def parse_repo(self, repo: str, repo_dir: Path) -> Dict:
        """Walk repo_dir, parse all .py files in parallel, and return the KG dict.

        Two-pass algorithm (plus an optional Pass 1.5, see infer_types):
            Pass 1 (parallel): Each file parsed independently, emitting nodes
                and unresolved call/inherits edges (targets as name strings).
            Pass 1.5 (sequential, optional): If infer_types=True, resolve
                factory-call sites recorded in Pass 1 via pyright and inject
                the results as additional unresolved 'uses' edges, using the
                exact same candidate-name shape the uppercase heuristic
                produces so Pass 2 needs no special-casing for them.
            Pass 2 (sequential): Aggregate nodes, build name→id indices,
                resolve edges, add call context.

        Timing (RQ4 instrumentation, docs/EXPERIMENT_PLAN.md's RQ4
        instrumentation section): 'build_time_parsing_s' covers file
        discovery plus Pass 1 (the parallel AST-parse);
        'build_time_resolution_s' covers everything after (Pass 1.5 if
        infer_types is on, plus Pass 2's aggregation/edge-resolution/
        call-context steps). Wall-clock via time.perf_counter(), not CPU
        time, so Pass 1's real parallelism is reflected (multiple worker
        processes overlapping) rather than summed as if sequential.

        Args:
            repo: Repository name (e.g. 'psf/requests').
            repo_dir: Root of extracted source tree.

        Returns:
            KG dict: {'nodes': [...], 'edges': [...], 'metadata': {...}}
        """
        parse_start = time.perf_counter()
        file_args = self._collect_files(repo, repo_dir)
        total_lines_of_code = self._count_lines(file_args)
        results = self._run_parallel_parse(file_args)
        parsing_time_s = time.perf_counter() - parse_start

        resolution_start = time.perf_counter()
        if self.infer_types:
            self._inject_inferred_uses_edges(results, repo_dir)

        all_nodes, all_edges, indices = self._aggregate_and_index(results)
        all_edges = self._resolve_edges(all_nodes, all_edges, indices)
        self._add_call_context(all_nodes, all_edges)
        resolution_time_s = time.perf_counter() - resolution_start

        return {
            'nodes': all_nodes,
            'edges': all_edges,
            'metadata': {
                'repo': repo,
                'file_count': len(file_args),
                'total_lines_of_code': total_lines_of_code,
                'parse_mode': 'source',
                'build_time_parsing_s': round(parsing_time_s, 3),
                'build_time_resolution_s': round(resolution_time_s, 3),
            }
        }

    def _inject_inferred_uses_edges(self, results: List[Dict], repo_dir: Path) -> None:
        """Resolve recorded factory-call sites via pyright and add 'uses' edges.

        Mutates each result's 'edges' list in place, appending one unresolved
        'uses' edge per successfully resolved site. Each recorded site has
        either a resolved source_class_id (self.x = ... / bare x = ... in a
        method — the enclosing class is already known) or an unresolved
        source_name (other.x = ... — the receiver's own class name, known
        from a parameter annotation or prior constructor call, but not yet
        an ID). The former produces the same edge shape
        _get_instantiated_classes_in_class already produces; the latter
        carries the name via metadata['unresolved_source'] so
        _resolve_edges can resolve it via class_label_to_ids exactly like
        it already resolves 'uses' edge targets — no new lookup mechanism,
        just the existing one applied to the other side of the edge too.

        Sites that pyright can't resolve (unions, unknown, external types
        with no repo class) are simply skipped — no edge is added, same as
        if the site had never been recorded at all.
        """
        sites_by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        site_index: Dict[Tuple[str, int, int], Tuple[Dict, Optional[str], Optional[str]]] = {}

        for result in results:
            for rel_path, source_class_id, source_name, line, col in result.get('factory_sites', []):
                sites_by_file[rel_path].append((line, col))
                site_index[(rel_path, line, col)] = (result, source_class_id, source_name)

        if not sites_by_file:
            return

        resolved = type_inference.resolve_types(repo_dir, dict(sites_by_file))
        for (rel_path, line, col), type_name in resolved.items():
            result, source_class_id, source_name = site_index[(rel_path, line, col)]
            metadata = {'unresolved': True, 'source': 'pyright'}
            if source_class_id is not None:
                edge_source = source_class_id
            else:
                # Placeholder; _resolve_edges replaces this with a real ID
                # (or drops the edge) using metadata['unresolved_source'].
                edge_source = None
                metadata['unresolved_source'] = source_name
            result['edges'].append(asdict(KGEdge(
                source=edge_source, target=type_name, relation='uses',
                metadata=metadata
            )))

    def _collect_files(self, repo: str, repo_dir: Path) -> List[Tuple[str, str, str]]:
        """Collect all Python files to parse, excluding SKIP_DIRS and
        individually-detected generated migration files (kg_construction#70).

        Returns list of (repo, rel_path, abs_path) tuples.
        """
        file_args = []
        for py_file in repo_dir.rglob('*.py'):
            rel = py_file.relative_to(repo_dir)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if 'migrations' in rel.parts and _GENERATED_MIGRATION_FILENAME.match(py_file.name):
                continue
            file_args.append((repo, str(rel), str(py_file)))
        # rglob's traversal order isn't guaranteed deterministic across
        # runs/filesystems -- sort so downstream node/edge ordering (and
        # anything that truncates by list position, e.g.
        # llm_serializer's MAX_CONTEXT_ITEMS_PER_CATEGORY caps) doesn't
        # vary between separate builds of the same repo+commit.
        file_args.sort(key=lambda args: args[1])
        return file_args

    def _count_lines(self, file_args: List[Tuple[str, str, str]]) -> int:
        """Sum real line counts across every file _collect_files returned.

        RQ4 instrumentation (docs/EXPERIMENT_PLAN.md's RQ4 instrumentation
        section): build timing is only meaningful normalized against real
        code size, not file_count alone (a 50-line file and a 5000-line
        file both count as '1' there). Read synchronously in the main
        process, not part of _run_parallel_parse's worker pool -- this is
        a plain line count, no AST work, and keeping it out of the
        parallel path avoids touching that pool's existing behavior.
        A file that fails to decode (rare, e.g. real encoding issues)
        contributes 0 rather than aborting the whole count.
        """
        total = 0
        for _repo, _rel_path, abs_path in file_args:
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total += sum(1 for _ in f)
            except OSError:
                continue
        return total

    def _run_parallel_parse(self, file_args: List[Tuple[str, str, str]]) -> List[Dict]:
        """Run _parse_file in parallel via ProcessPoolExecutor.

        Returns list of parse results (nodes + unresolved edges from each
        file), in a fixed order matching file_args -- as_completed()
        previously returned results in whichever order each worker
        process happened to finish, which varies run-to-run and made
        all_nodes/all_edges ordering (and anything downstream that
        truncates by list position) non-deterministic across separate
        builds of the same repo+commit.
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_parse_file, file_args))
        return [r for r in results if r]

    def _aggregate_and_index(self, results: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
        """Aggregate nodes from parse results and build resolution indices.

        Returns (all_nodes, all_edges, indices_dict) where indices_dict contains:
            - label_to_ids, class_label_to_ids, class_method_to_ids,
            - qualified_to_ids, nodes_by_id
        """
        all_nodes: List[Dict] = []
        seen_node_ids: Set[str] = set()
        label_to_ids: Dict[str, List[str]] = defaultdict(list)
        class_label_to_ids: Dict[str, List[str]] = defaultdict(list)
        class_method_to_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        qualified_to_ids: Dict[str, List[str]] = defaultdict(list)

        for result in results:
            for node in result['nodes']:
                if node['id'] not in seen_node_ids:
                    all_nodes.append(node)
                    seen_node_ids.add(node['id'])
                    ntype = node['type']
                    label = node['label']
                    if ntype in ('function', 'method', 'test_function'):
                        label_to_ids[label].append(node['id'])
                        parent = node['metadata'].get('class')
                        if parent:
                            class_method_to_ids[(parent, label)].append(node['id'])
                    elif ntype == 'class':
                        class_label_to_ids[label].append(node['id'])

                    fp = node.get('metadata', {}).get('filepath')
                    if fp and ntype in ('function', 'method', 'class', 'test_function'):
                        mod = fp.removesuffix('.py').replace('/', '.')
                        qualified_to_ids[f"{mod}.{label}"].append(node['id'])
                        parent = node.get('metadata', {}).get('class')
                        if parent:
                            qualified_to_ids[f"{mod}.{parent}.{label}"].append(node['id'])

        nodes_by_id: Dict[str, Dict] = {n['id']: n for n in all_nodes}
        all_edges: List[Dict] = [edge for result in results for edge in result['edges']]

        # Derived from class_method_to_ids (no extra AST walk needed): the
        # same index, filtered to @property/@cached_property-decorated
        # methods only, so 'accesses' edge resolution can't accidentally
        # link a property read to a same-named plain method. Matches on
        # the decorator's final dotted component, since ast.unparse can
        # render it as 'property', 'cached_property', or
        # 'functools.cached_property' depending on how it was imported --
        # an exact 'property' string match previously missed the
        # cached_property spellings entirely (a common real pattern for
        # expensive derived attributes).
        _PROPERTY_DECORATOR_NAMES = {'property', 'cached_property'}
        property_method_to_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for (class_label, method_label), ids in class_method_to_ids.items():
            for node_id in ids:
                decorators = nodes_by_id[node_id]['metadata'].get('decorators', [])
                if any(d.rsplit('.', 1)[-1] in _PROPERTY_DECORATOR_NAMES for d in decorators):
                    property_method_to_ids[(class_label, method_label)].append(node_id)

        indices = {
            'label_to_ids': label_to_ids,
            'class_label_to_ids': class_label_to_ids,
            'class_method_to_ids': class_method_to_ids,
            'property_method_to_ids': property_method_to_ids,
            'qualified_to_ids': qualified_to_ids,
            'nodes_by_id': nodes_by_id,
        }

        return all_nodes, all_edges, indices

    def _resolve_edges(self, all_nodes: List[Dict], all_edges: List[Dict], indices: Dict) -> List[Dict]:
        """Resolve unresolved edges using name→id indices.

        Processes 'calls', 'accesses', 'inherits', 'tests', 'uses',
        'overrides' edges.
        Drops reads/writes/returns edges (attribute strings, not node IDs).
        Returns resolved edge list.
        """
        label_to_ids = indices['label_to_ids']
        class_label_to_ids = indices['class_label_to_ids']
        class_method_to_ids = indices['class_method_to_ids']
        property_method_to_ids = indices['property_method_to_ids']
        qualified_to_ids = indices['qualified_to_ids']
        nodes_by_id = indices['nodes_by_id']

        def _resolve_access(meta: Dict, attr_name: str) -> Tuple[List[str], str]:
            """Resolve an accesses-edge target against known @property methods.

            Unlike _resolve_call, there is no bare-name fallback against
            label_to_ids: an unqualified property access with no receiver
            hint is too weak a signal to link against every same-named
            property in the repo, so it is left unresolved instead.
            """
            class_hint = meta.get('class_hint')
            if class_hint:
                hits = property_method_to_ids.get((class_hint, attr_name), [])
                if hits:
                    return hits, 'qualified'

            local_hint = meta.get('local_type_hint')
            if local_hint:
                hits = property_method_to_ids.get((local_hint, attr_name), [])
                if hits:
                    return hits, 'qualified'

            qualified = meta.get('import_resolved')
            if qualified:
                parts = qualified.rsplit('.', 2)
                if len(parts) == 3:
                    hits = property_method_to_ids.get((parts[1], parts[2]), [])
                    if hits:
                        return hits, 'qualified'

            return [], 'unresolved'

        def _resolve_call(meta: Dict, callee_name: str, caller_filepath: Optional[str] = None) -> Tuple[List[str], str]:
            """Resolve a call-edge target using the metadata hints."""
            class_hint = meta.get('class_hint')
            if class_hint:
                hits = class_method_to_ids.get((class_hint, callee_name), [])
                if hits:
                    return hits, 'qualified'

            local_hint = meta.get('local_type_hint')
            if local_hint:
                hits = class_method_to_ids.get((local_hint, callee_name), [])
                if hits:
                    return hits, 'qualified'

            qualified = meta.get('import_resolved')
            if qualified:
                hits = qualified_to_ids.get(qualified, [])
                if hits:
                    return hits, 'qualified'
                parts = qualified.rsplit('.', 2)
                if len(parts) == 3:
                    hits = class_method_to_ids.get((parts[1], parts[2]), [])
                    if hits:
                        return hits, 'qualified'

            # Bare Foo(...) is a constructor call when Foo is a class, so
            # check class names too, not just functions/methods
            # (kg_construction#120).
            hits = label_to_ids.get(callee_name, []) + class_label_to_ids.get(callee_name, [])

            # A bare call (no receiver at all, e.g. `request(...)` calling
            # a module-level function, as opposed to `x.request()`) that
            # matches more than one same-named node: prefer the one(s) in
            # the SAME FILE as the caller. A bare name in Python resolves
            # via the caller's own module/import scope, so a same-file
            # match is a real, principled preference, not a guess --
            # confirmed necessary on a real psf/requests case: api.py's
            # post() calls bare request(...), meaning api.py's OWN
            # module-level request() function, but request also exists as
            # unrelated methods on Session and RequestMethods in other
            # files (kg_construction#65).
            if len(hits) > 1 and meta.get('receiver') is None and caller_filepath:
                same_file_hits = [
                    hid for hid in hits
                    if nodes_by_id.get(hid, {}).get('metadata', {}).get('filepath') == caller_filepath
                ]
                if len(same_file_hits) == 1:
                    return same_file_hits, 'exact'

            if len(hits) != 1:
                # 0 hits: genuinely nothing in this repo has that name
                # (very likely an external library/stdlib call, e.g.
                # requests.get() or ConfigParser.get() -- neither is
                # parsed into this KG, so there's nothing to link to).
                # >1 hits (and same-file preference above didn't resolve
                # it): no real hint resolved this call to a specific
                # receiver, so linking it to EVERY same-named method in
                # the repo is not a best-effort guess, it's simply wrong --
                # confirmed on real django/django and psf/requests calls
                # that had nothing to do with the arbitrary same-named
                # method they'd have been linked to (kg_construction#65,
                # a generalization of #61's super()-specific case; 80-93%
                # of calls edges in both real repos hit this fallback).
                # Drop rather than fan out to every candidate.
                return [], 'unresolved'
            return hits, 'exact'

        resolved_edges: List[Dict] = []
        seen_edges: Set[Tuple] = set()
        # 'overrides' needs the fully-resolved 'inherits' graph to walk
        # transitive base classes, so raw edges are collected here and
        # resolved in a dedicated pass after this loop finishes.
        pending_overrides: List[Dict] = []
        # super().method() calls need the same deferred treatment: the
        # caller's OWN class's 'inherits' edges aren't necessarily
        # resolved yet at this point in the loop (kg_construction#61) --
        # collected here, resolved in _resolve_super_calls once 'inherits'
        # is fully populated, walking the real base-class chain instead of
        # ever falling through to the bare-name fallback below.
        pending_super_calls: List[Dict] = []

        for edge in all_edges:
            meta = edge.get('metadata', {})

            if edge['relation'] == 'calls' and meta.get('unresolved') and meta.get('is_super_call'):
                pending_super_calls.append(edge)
                continue

            if edge['relation'] == 'calls' and meta.get('unresolved'):
                callee_name = edge['target']
                caller_filepath = nodes_by_id.get(edge['source'], {}).get('metadata', {}).get('filepath')
                matches, confidence = _resolve_call(meta, callee_name, caller_filepath)
                if not matches:
                    continue
                for target_id in matches:
                    key = (edge['source'], target_id, 'calls')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='calls',
                            metadata={'confidence': confidence,
                                      'import_resolved': meta.get('import_resolved')}
                        )))

            elif edge['relation'] == 'accesses' and meta.get('unresolved'):
                attr_name = edge['target']
                matches, confidence = _resolve_access(meta, attr_name)
                if not matches:
                    continue
                for target_id in matches:
                    key = (edge['source'], target_id, 'accesses')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='accesses',
                            metadata={'confidence': confidence}
                        )))

            elif edge['relation'] == 'inherits' and meta.get('unresolved'):
                base_name = edge['target'].split('.')[-1]
                matches = class_label_to_ids.get(base_name, [])
                if not matches:
                    continue
                confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                for target_id in matches:
                    key = (edge['source'], target_id, 'inherits')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='inherits',
                            metadata={'confidence': confidence}
                        )))

            elif edge['relation'] == 'raises' and meta.get('unresolved'):
                # Only resolves when the raised/caught name is a real
                # in-repo class (kg_construction#112) -- builtins and
                # external exceptions have nothing to match and are
                # dropped, same "drop rather than guess" rule as calls.
                exc_name = edge['target']
                matches = class_label_to_ids.get(exc_name, [])
                if not matches:
                    continue
                confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                for target_id in matches:
                    key = (edge['source'], target_id, 'raises')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='raises',
                            metadata={'confidence': confidence}
                        )))

            elif edge['relation'] == 'decorated_by' and meta.get('unresolved'):
                # Check both indices unconditionally, not short-circuited,
                # a class and function of the same name can both exist
                # (kg_construction#135 review).
                dec_name = edge['target']
                matches = class_label_to_ids.get(dec_name, []) + label_to_ids.get(dec_name, [])
                if not matches:
                    continue
                # A bare decorator name resolves via the decorated
                # function's own module scope, same reasoning as bare
                # calls above. Prefer a single same-file match over
                # reporting every same-named candidate as ambiguous.
                if len(matches) > 1:
                    decorated_filepath = nodes_by_id.get(edge['source'], {}).get('metadata', {}).get('filepath')
                    same_file_matches = [
                        mid for mid in matches
                        if nodes_by_id.get(mid, {}).get('metadata', {}).get('filepath') == decorated_filepath
                    ]
                    if len(same_file_matches) == 1:
                        matches = same_file_matches
                confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                for target_id in matches:
                    key = (edge['source'], target_id, 'decorated_by')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='decorated_by',
                            metadata={'confidence': confidence}
                        )))

            elif edge['relation'] == 'tests' and meta.get('unresolved'):
                target_name = edge['target']
                if target_name.startswith('test_'):
                    target_name = target_name[5:]
                matches = label_to_ids.get(target_name, [])
                if not matches:
                    continue
                confidence = 'exact' if len(matches) == 1 else 'ambiguous'
                for target_id in matches:
                    key = (edge['source'], target_id, 'tests')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='tests',
                            metadata={'confidence': confidence}
                        )))

            elif edge['relation'] == 'uses' and meta.get('unresolved'):
                matches = class_label_to_ids.get(edge['target'], [])
                if not matches:
                    continue
                # The uppercase-first-letter heuristic that produced this
                # candidate name (_get_instantiated_classes) can't tell a
                # constructor call from a call to an unrelated uppercase-named
                # function/method that happens to share the name. If a
                # function/method with this exact name also exists anywhere
                # in the repo, the "instantiation" is only a guess even when
                # a single class name matches, so confidence is downgraded
                # to 'ambiguous' rather than reported as 'exact'.
                name_collides_with_callable = edge['target'] in label_to_ids
                if len(matches) > 1:
                    confidence = 'ambiguous'
                elif name_collides_with_callable:
                    confidence = 'ambiguous'
                else:
                    confidence = 'exact'
                edge_metadata = {'confidence': confidence}
                if meta.get('source') == 'pyright':
                    edge_metadata['source'] = 'pyright'

                # Some pyright-derived 'uses' edges (other.x = factory())
                # carry an unresolved source class name instead of a real
                # ID (the enclosing class isn't the source; the receiver's
                # own class is) -- resolve it the same way the target
                # above was just resolved, via class_label_to_ids, including
                # the same name-collides-with-a-real-callable downgrade (a
                # source name is exactly as capable of colliding with an
                # unrelated function/method name as a target name is).
                unresolved_source_name = meta.get('unresolved_source')
                if unresolved_source_name is not None:
                    source_matches = class_label_to_ids.get(unresolved_source_name, [])
                    if not source_matches:
                        continue
                    source_collides_with_callable = unresolved_source_name in label_to_ids
                    if len(source_matches) > 1 or source_collides_with_callable:
                        edge_metadata['confidence'] = 'ambiguous'
                    source_ids = source_matches
                else:
                    source_ids = [edge['source']]

                for source_id in source_ids:
                    for target_id in matches:
                        key = (source_id, target_id, 'uses')
                        if key not in seen_edges:
                            seen_edges.add(key)
                            resolved_edges.append(asdict(KGEdge(
                                source=source_id, target=target_id, relation='uses',
                                metadata=edge_metadata
                            )))

            elif edge['relation'] == 'overrides' and meta.get('unresolved'):
                # Deferred: resolved in _resolve_overrides after 'inherits'
                # edges are fully resolved, so the transitive base-class
                # chain is walkable (a method can override a *grandparent's*
                # definition, not just an immediate base's).
                pending_overrides.append(edge)

            elif meta.get('unresolved'):
                continue

            else:
                key = (edge['source'], edge['target'], edge['relation'])
                if key not in seen_edges:
                    seen_edges.add(key)
                    resolved_edges.append(edge)

        self._resolve_overrides(pending_overrides, resolved_edges, indices, seen_edges)

        self._resolve_super_calls(pending_super_calls, resolved_edges, indices, seen_edges)

        self._derive_tests_edges_from_calls(resolved_edges, nodes_by_id, seen_edges)

        # Add module_depends_on edges
        # Keyed by full relative path (e.g. 'pkg_a/utils.py'), not bare filename,
        # so same-named files in different packages (pkg_a/utils.py vs
        # pkg_b/utils.py) don't collide and misattribute the dependency edge.
        file_path_to_id: Dict[str, str] = {
            n['metadata'].get('path', ''): n['id'] for n in all_nodes
            if n['type'] in ('file', 'test_file')
        }
        import_to_files: Dict[str, List[str]] = defaultdict(list)
        for edge in resolved_edges:
            if edge['relation'] == 'imports':
                import_to_files[edge['target']].append(edge['source'])

        for imp_node in all_nodes:
            if imp_node['type'] != 'import':
                continue
            module = imp_node['metadata'].get('module', '')
            parts = (module or imp_node['label']).split('.')
            # Try progressively shorter path suffixes: for 'pkg.sub.mod' try
            # 'pkg/sub/mod.py', then 'sub/mod.py', then 'mod.py', matching
            # against full file paths so identically-named files in different
            # packages resolve to the correct one.
            for i in range(len(parts)):
                candidate = '/'.join(parts[i:]) + '.py'
                target_file_id = file_path_to_id.get(candidate)
                if target_file_id:
                    for src_file_id in import_to_files.get(imp_node['id'], []):
                        if src_file_id != target_file_id:
                            key = (src_file_id, target_file_id, 'module_depends_on')
                            if key not in seen_edges:
                                seen_edges.add(key)
                                resolved_edges.append(asdict(KGEdge(
                                    source=src_file_id, target=target_file_id,
                                    relation='module_depends_on'
                                )))
                    break

        return resolved_edges

    def _derive_tests_edges_from_calls(
        self,
        resolved_edges: List[Dict],
        nodes_by_id: Dict,
        seen_edges: Set[Tuple],
    ) -> None:
        """Add a 'tests' edge for every resolved 'calls' edge sourced from a
        test_function node, in addition to the naming-convention 'tests'
        edges resolved earlier in this pass (test_<name> -> <name>, exact
        string match after stripping the prefix).

        The naming heuristic assumes a test's name mechanically derives
        from the function it tests -- true for some codebases, but checked
        directly against psf/requests' real test suite and found to hold
        for only 1 of 159 test functions (the rest use descriptive names
        like test_prepared_request_hook, with no derivable relationship to
        the function under test). That made existing_tests/test_nodes
        empty for 21 of 22 real benchmark instances, not because no
        relevant test existed, but because the only detection mechanism
        couldn't see it (kg_construction#57's audit).

        A test function that actually CALLS the target function is a much
        stronger, naming-independent signal that it's a real existing test
        for that function -- and this KG already computes exactly that via
        the ordinary 'calls' edge resolution above, so this only needs to
        re-tag test_function-sourced calls edges as also being 'tests'
        edges, not build any new resolution machinery.

        Mutates resolved_edges/seen_edges in place (same pattern as the
        rest of this pass) rather than returning a new list.
        """
        for edge in list(resolved_edges):
            if edge['relation'] != 'calls':
                continue
            src_node = nodes_by_id.get(edge['source'])
            if not src_node or src_node.get('type') != 'test_function':
                continue

            key = (edge['source'], edge['target'], 'tests')
            if key in seen_edges:
                continue
            seen_edges.add(key)
            resolved_edges.append(asdict(KGEdge(
                source=edge['source'], target=edge['target'], relation='tests',
                metadata={'confidence': edge.get('metadata', {}).get('confidence', 'unknown'),
                          'derived_from': 'calls'}
            )))

    def _resolve_overrides(
        self,
        pending_overrides: List[Dict],
        resolved_edges: List[Dict],
        indices: Dict,
        seen_edges: Set[Tuple],
    ) -> None:
        """Resolve 'overrides' edges by walking the transitive inherits chain.

        A method overrides the *nearest* ancestor that defines a method of
        the same name — matching Python's actual MRO/lookup semantics. For
        `class C(B)`, `class B(A)`, if only `A` defines `foo`, `C.foo`
        overrides `A.foo` (B is skipped because it doesn't define foo). If
        `B` also defines `foo`, `C.foo` overrides `B.foo` only, not both.

        Requires 'inherits' edges to already be resolved (present in
        resolved_edges) so the base-class graph can be walked by ID rather
        than by name. Appends resolved 'overrides' edges directly onto
        resolved_edges.

        Args:
            pending_overrides: Unresolved 'overrides' edges collected during
                the main _resolve_edges loop, targets shaped 'BaseName.method'.
            resolved_edges: The in-progress resolved edge list; mutated in
                place to append newly resolved 'overrides' edges.
            indices: Name->id lookup indices from _aggregate_and_index.
            seen_edges: Shared dedup set of (source, target, relation) keys.
        """
        if not pending_overrides:
            return

        class_label_to_ids = indices['class_label_to_ids']
        class_method_to_ids = indices['class_method_to_ids']
        nodes_by_id = indices['nodes_by_id']

        # Build class_id -> [base_class_id, ...] from the resolved inherits
        # edges so the chain can be walked by ID (unambiguous), rather than
        # re-deriving base names from strings for every ancestor level.
        base_class_ids: Dict[str, List[str]] = defaultdict(list)
        for edge in resolved_edges:
            if edge['relation'] == 'inherits':
                base_class_ids[edge['source']].append(edge['target'])

        for edge in pending_overrides:
            parts = edge['target'].rsplit('.', 1)
            if len(parts) != 2:
                continue
            base_name, method_name = parts
            base_simple = base_name.split('.')[-1]

            for base_class_id in class_label_to_ids.get(base_simple, []):
                target_id = self._find_nearest_ancestor_method(
                    base_class_id, method_name, base_class_ids, class_method_to_ids, nodes_by_id
                )
                if target_id is None:
                    continue
                key = (edge['source'], target_id, 'overrides')
                if key not in seen_edges:
                    seen_edges.add(key)
                    resolved_edges.append(asdict(KGEdge(
                        source=edge['source'], target=target_id,
                        relation='overrides'
                    )))

    @staticmethod
    def _find_nearest_ancestor_method(
        base_class_id: str,
        method_name: str,
        base_class_ids: Dict[str, List[str]],
        class_method_to_ids: Dict[Tuple[str, str], List[str]],
        nodes_by_id: Dict[str, Dict],
    ) -> Optional[str]:
        """BFS up an inheritance chain for the nearest ancestor defining
        method_name, starting from base_class_id. Returns its node ID, or
        None if no ancestor defines it. Guards against cycles via `visited`.

        Shared by _resolve_overrides ('overrides') and _resolve_super_calls
        (kg_construction#61, "what does super().method() call") -- same
        question, two different relations.
        """
        visited: Set[str] = set()
        frontier = [base_class_id]
        while frontier:
            next_frontier: List[str] = []
            for class_id in frontier:
                if class_id in visited:
                    continue
                visited.add(class_id)

                class_node = nodes_by_id.get(class_id)
                if class_node:
                    hits = class_method_to_ids.get((class_node['label'], method_name), [])
                    if hits:
                        return hits[0]

                next_frontier.extend(base_class_ids.get(class_id, []))
            frontier = next_frontier
        return None

    def _resolve_super_calls(
        self,
        pending_super_calls: List[Dict],
        resolved_edges: List[Dict],
        indices: Dict,
        seen_edges: Set[Tuple],
    ) -> None:
        """Resolve super().method() calls against the caller's own
        inheritance chain (kg_construction#61), instead of the bare-name
        fallback in _resolve_call -- which used to match e.g.
        ErrorList.copy() to some unrelated class's copy() method.

        Walks the caller's class's own (already-resolved) inherits chain
        for the nearest ancestor defining the method, same as
        _resolve_overrides. If no parsed ancestor defines it (e.g. the
        real base is an unparsed builtin like list), the edge is dropped
        rather than guessed. Requires 'inherits' edges already resolved,
        same precondition as _resolve_overrides.
        """
        if not pending_super_calls:
            return

        class_label_to_ids = indices['class_label_to_ids']
        class_method_to_ids = indices['class_method_to_ids']
        nodes_by_id = indices['nodes_by_id']

        base_class_ids: Dict[str, List[str]] = defaultdict(list)
        for edge in resolved_edges:
            if edge['relation'] == 'inherits':
                base_class_ids[edge['source']].append(edge['target'])

        for edge in pending_super_calls:
            caller_node = nodes_by_id.get(edge['source'])
            if not caller_node:
                continue
            caller_class_name = caller_node.get('metadata', {}).get('class')
            caller_filepath = caller_node.get('metadata', {}).get('filepath')
            if not caller_class_name:
                continue  # super() outside a class isn't valid Python

            # class_label_to_ids can return multiple candidates for a
            # common class name across files -- narrow to the caller's own
            # file, falling back to all candidates if none match.
            candidates = class_label_to_ids.get(caller_class_name, [])
            same_file_candidates = [
                cid for cid in candidates
                if nodes_by_id.get(cid, {}).get('metadata', {}).get('filepath') == caller_filepath
            ]
            caller_class_ids = same_file_candidates or candidates

            method_name = edge['target']
            for caller_class_id in caller_class_ids:
                for base_class_id in base_class_ids.get(caller_class_id, []):
                    target_id = self._find_nearest_ancestor_method(
                        base_class_id, method_name, base_class_ids, class_method_to_ids, nodes_by_id
                    )
                    if target_id is None:
                        continue
                    key = (edge['source'], target_id, 'calls')
                    if key not in seen_edges:
                        seen_edges.add(key)
                        resolved_edges.append(asdict(KGEdge(
                            source=edge['source'], target=target_id, relation='calls',
                            metadata={'confidence': 'qualified', 'via': 'super'}
                        )))

    def _add_call_context(self, all_nodes: List[Dict], all_edges: List[Dict]) -> None:
        """Annotate functions with caller_count and direct_callers metadata."""
        callers: Dict[str, List[str]] = defaultdict(list)
        for edge in all_edges:
            if edge['relation'] == 'calls':
                callers[edge['target']].append(edge['source'])

        node_by_id = {node['id']: node for node in all_nodes}

        for node in all_nodes:
            if node['type'] not in ('function', 'method', 'test_function'):
                continue

            node_id = node['id']
            caller_ids = callers.get(node_id, [])
            caller_count = len(caller_ids)

            direct_callers = []
            for caller_id in caller_ids:
                caller_node = node_by_id.get(caller_id)
                if caller_node:
                    direct_callers.append({
                        'id': caller_id,
                        'label': caller_node['label'],
                        'type': caller_node['type'],
                    })

            node['metadata']['caller_count'] = caller_count
            node['metadata']['direct_callers'] = direct_callers


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

class RepoKGBuilder:
    """Top-level entry point for building, saving, and loading repo KGs.

    Orchestrates RepoManager (git operations) and RepoASTParser (source
    parsing) into a single build() call. Output is saved as JSON to
    kg_output/kg_{repo}_{commit}.json -- keyed on both repo and commit, so
    a KG built at one commit is never silently served for another.

    Example:
        builder = RepoKGBuilder()
        kg = builder.build('psf/requests', 'a0df2cbb...')
        builder.save('psf/requests', kg)

        # Later, same commit
        kg = builder.load('psf/requests', 'a0df2cbb...')
        engine = KGQueryEngine(kg)
    """

    def __init__(self,
                 output_dir: Path = Path('kg_output'),
                 cache_dir: Path = Path('repo_cache'),
                 max_workers: int = 4,
                 infer_types: bool = False):
        """
        Args:
            output_dir: Where KG JSON files are saved.
            cache_dir: Where bare git clones are cached.
            max_workers: Parallel workers for file parsing.
            infer_types: If True, enable the optional pyright-backed 'uses'
                         edge enrichment for factory-function call sites.
                         Requires the `pyright` package (`pip install
                         kg-construction[types]`); degrades gracefully to
                         no enrichment if pyright is missing or fails.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.repo_manager = RepoManager(cache_dir)
        self.ast_parser = RepoASTParser(max_workers=max_workers, infer_types=infer_types)

    def build(self, repo: str, commit: str) -> Dict:
        """Build a structural KG for a repo at a specific commit.

        Clones (if needed), extracts source at the commit, parses all .py
        files, resolves edges, and returns the KG dict. The source tree is
        cleaned up automatically via tempfile.TemporaryDirectory.

        Args:
            repo: GitHub repo in 'owner/name' format.
            commit: Commit SHA to build the KG from.

        Returns:
            KG dict ready for saving or querying.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'src'
            print(f"Extracting {repo}@{commit[:8]}...")
            self.repo_manager.extract_at_commit(repo, commit, dest)
            print("Parsing source...")
            kg = self.ast_parser.parse_repo(repo, dest)

        kg['metadata']['base_commit'] = commit
        kg['metadata']['schema_version'] = SCHEMA_VERSION
        return kg

    def build_from_dir(self, repo_name: str, local_path: Path) -> Dict:
        """Build a structural KG from a local directory of Python source,
        with no git dependency at all -- for a working tree, including
        uncommitted changes, rather than a specific pushed commit
        (kg_construction#83).

        RepoASTParser.parse_repo only ever needs a directory of Python
        files; it has no awareness of git, so this is a thin alternate
        entry point, not new parsing logic.

        Args:
            repo_name: A label for this build (used in node IDs and as the
                       'repo' field in metadata) -- does not need to be a
                       real GitHub 'owner/name', since there's no clone
                       involved.
            local_path: Directory containing the Python source to parse.

        Returns:
            KG dict. metadata['base_commit'] is None (no single commit this
            snapshot corresponds to) -- callers must not pass this KG to
            save()/load(), which are keyed on a real commit; use
            save_local()/load_local() instead.
        """
        kg = self.ast_parser.parse_repo(repo_name, Path(local_path))
        kg['metadata']['base_commit'] = None
        kg['metadata']['schema_version'] = SCHEMA_VERSION
        return kg

    def _local_cache_path(self, repo_name: str) -> Path:
        """On-disk path for a build_from_dir() snapshot.

        Kept separate from _cache_path (which is keyed on a real commit)
        so a local working-tree build can never collide with, or be
        silently confused for, a real commit-pinned KG -- a local build
        has no commit to key on and can go stale the moment the working
        tree changes again, so it must never be served from the same cache
        a real commit-pinned build trusts.
        """
        safe_name = repo_name.replace('/', '_').replace('-', '_').replace('.', '_')
        return self.output_dir / f"kg_local_{safe_name}.json"

    def save_local(self, repo_name: str, kg: Dict):
        """Save a build_from_dir() KG to kg_output/kg_local_{repo_name}.json.

        Args:
            repo_name: Same label passed to build_from_dir().
            kg: KG dict as returned by build_from_dir().
        """
        output_file = self._local_cache_path(repo_name)
        with open(output_file, 'w') as f:
            json.dump(kg, f, indent=2)
        print(f"Saved: {repo_name} (local) -> {output_file} "
              f"({len(kg['nodes'])} nodes, {len(kg['edges'])} edges)")

    def load_local(self, repo_name: str) -> Optional[Dict]:
        """Load a previously saved build_from_dir() KG.

        No freshness check against the working tree is possible (there's
        no commit to compare against) -- callers that need an up-to-date
        snapshot of a working tree that may have changed should call
        build_from_dir() again rather than trust a cached local build.

        Args:
            repo_name: Same label passed to build_from_dir()/save_local().

        Returns:
            KG dict, or None if no cached local build exists.
        """
        kg_file = self._local_cache_path(repo_name)
        if not kg_file.exists():
            return None
        with open(kg_file) as f:
            kg = json.load(f)
        if kg.get('metadata', {}).get('schema_version') != SCHEMA_VERSION:
            return None
        return kg

    def _cache_path(self, repo: str, commit: str) -> Path:
        """Return the on-disk cache path for a repo at a specific commit.

        Keyed on (repo, commit), not repo alone -- a KG built at one commit
        must never be silently served for a different commit of the same
        repo. Slashes, dashes, and dots in repo names are replaced with
        underscores to produce a safe filename.
        """
        safe_name = repo.replace('/', '_').replace('-', '_').replace('.', '_')
        return self.output_dir / f"kg_{safe_name}_{commit[:8]}.json"

    def save(self, repo: str, kg: Dict):
        """Serialize and save a KG to kg_output/kg_{repo}_{commit}.json.

        Args:
            repo: Repository name (used to derive the output filename).
            kg: KG dict as returned by build(). Must have metadata.base_commit
                set (build() sets this automatically).

        Raises:
            ValueError: If kg has no base_commit -- a build_from_dir() KG
                        has none, and must use save_local() instead, since
                        this method's cache is keyed on a real commit.
        """
        commit = kg['metadata']['base_commit']
        if commit is None:
            raise ValueError(
                "kg has no base_commit -- if this came from build_from_dir(), "
                "use save_local() instead."
            )
        output_file = self._cache_path(repo, commit)
        with open(output_file, 'w') as f:
            json.dump(kg, f, indent=2)
        print(f"Saved: {repo}@{commit[:8]} -> {output_file} "
              f"({len(kg['nodes'])} nodes, {len(kg['edges'])} edges)")

    def load(self, repo: str, commit: str) -> Optional[Dict]:
        """Load a previously saved KG from disk, for a specific commit.

        A cached file is only returned if it exists AND its stamped
        metadata.base_commit and metadata.schema_version both match what's
        requested/current -- guards against a pre-#45 cache file (no
        schema_version, or a filename collision) being served as if valid.

        Args:
            repo: Repository name (e.g. 'psf/requests').
            commit: Commit SHA the caller needs the KG built at.

        Returns:
            KG dict, or None if no valid cached KG exists for this
            (repo, commit) pair.
        """
        kg_file = self._cache_path(repo, commit)
        if not kg_file.exists():
            return None
        with open(kg_file) as f:
            kg = json.load(f)
        if kg.get('metadata', {}).get('base_commit') != commit:
            return None
        if kg.get('metadata', {}).get('schema_version') != SCHEMA_VERSION:
            return None
        return kg
