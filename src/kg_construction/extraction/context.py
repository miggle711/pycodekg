"""
test_context.py

Extract KG subgraphs from code changes for LLM test generation.

Given a dataset instance with {repo, base_commit, patch, code_file, test_file},
this module:
  1. Parses the patch to identify changed functions/classes
  2. Loads the pre-built KG at base_commit
  3. Finds corresponding nodes in the KG
  4. Performs BFS to extract surrounding context
  5. Returns a structured TestContext

Works generically across any dataset with the standard schema.
"""

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from kg_construction.ast.helpers import (
    _make_id,
    _build_func_metadata,
    _collect_file_level_info,
    _get_base_names,
    _get_class_attributes,
    _get_decorators,
    _get_docstring,
    _get_signature_and_source_text,
)
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.kg.traversal import GraphTraversal
from kg_construction.extraction.patch import (
    PatchParser,
    is_newly_created_file,
    reconstruct_post_patch_source,
)


@dataclass
class TestContext:
    """Structured subgraph extracted for test generation.

    Attributes:
        seeds: Nodes representing the changed functions/classes under test.
        context_nodes: BFS-expanded neighbors providing surrounding context.
        edges: Edges within the subgraph (source and target both in context).
        test_nodes: Existing test functions found via 'tests' edges.
        repo: Repository name for reference.
        base_commit: Commit SHA the KG was built from.
        stale_seed_labels: Labels of any seed whose source_code could not
            be refreshed to post-patch content (fell back to the KG's
            stored, pre-patch value). Empty in the normal case -- non-empty
            means a real run should look into why, not just that this one
            instance's seed source is stale.
    """
    seeds: List[Dict]
    context_nodes: List[Dict]
    edges: List[Dict]
    test_nodes: List[Dict]
    repo: str
    base_commit: str
    stale_seed_labels: List[str] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Save subgraph to JSON for debugging.

        Args:
            path: File path to write JSON to (e.g. 'subgraph_debug.json').
        """
        data = {
            'repo': self.repo,
            'base_commit': self.base_commit,
            'seeds': self.seeds,
            'context_nodes': self.context_nodes,
            'edges': self.edges,
            'test_nodes': self.test_nodes,
            'stats': {
                'num_seeds': len(self.seeds),
                'num_context_nodes': len(self.context_nodes),
                'num_edges': len(self.edges),
                'num_test_nodes': len(self.test_nodes),
            }
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"✓ Saved subgraph to {path}")

    @classmethod
    def load(cls, path: str) -> 'TestContext':
        """Load subgraph from JSON.

        Args:
            path: File path to read JSON from.

        Returns:
            Reconstructed TestContext instance.
        """
        data = json.loads(Path(path).read_text())
        return cls(
            seeds=data['seeds'],
            context_nodes=data['context_nodes'],
            edges=data['edges'],
            test_nodes=data['test_nodes'],
            repo=data['repo'],
            base_commit=data['base_commit'],
        )

    def summary(self) -> str:
        """Return a human-readable summary of the subgraph.

        Returns:
            Formatted string with counts and top nodes.
        """
        lines = [
            f"TestContext: {self.repo} @ {self.base_commit[:8]}",
            f"  Seeds: {len(self.seeds)}",
            f"  Context nodes: {len(self.context_nodes)}",
            f"  Edges: {len(self.edges)}",
            f"  Test nodes: {len(self.test_nodes)}",
            f"",
            "Top seeds:",
        ]
        for node in self.seeds[:5]:
            lines.append(f"  - {node['label']:40} ({node['type']})")
        if len(self.seeds) > 5:
            lines.append(f"  ... and {len(self.seeds) - 5} more")

        lines.append("")
        lines.append("Edge types:")
        edge_types = {}
        for edge in self.edges:
            rel = edge['relation']
            edge_types[rel] = edge_types.get(rel, 0) + 1
        for rel, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            lines.append(f"  {rel:20} {count:5}")

        return "\n".join(lines)




class TestContextExtractor:
    """Extract KG subgraphs from dataset instances for test generation."""

    def __init__(self, engine: KGQueryEngine, repo_manager: Optional[RepoManager] = None):
        """
        Args:
            engine: Loaded KGQueryEngine on the pre-built KG.
            repo_manager: Used to fetch each instance's pre-patch source
                         (kg_construction#75: PatchParser resolves changed
                         functions by real ast line ranges, which requires
                         the pre-patch file's actual text). Defaults to a
                         fresh RepoManager() -- injectable for tests to
                         avoid a real network clone.
        """
        self.engine = engine
        self.patch_parser = PatchParser()
        self.repo_manager = repo_manager or RepoManager()

    def extract(
        self,
        instance: Dict,
        depth: int = 2,
        edge_filter: Optional[Set[str]] = None,
        include_seed_imports: bool = True,
    ) -> TestContext:
        """Extract a KG subgraph from a dataset instance.

        Args:
            instance: Dict with keys:
                - repo: Repository name (e.g. 'psf/requests')
                - base_commit: Commit SHA the KG was built from
                - patch: Unified diff of code changes
                - code_file: Relative path to code file (e.g. 'requests/sessions.py')
                - test_file: Relative path to test file (e.g. 'tests/test_sessions.py')
            depth: BFS depth for context expansion (default 2).
            edge_filter: Set of edge relations to traverse during BFS.
                        If None, uses smart defaults: contains, calls, inherits, tests, uses.
                        Excludes depends_on (imports) as it is primarily noise for test generation.
            include_seed_imports: If True (default), add depends_on edges from seed nodes only.
                        This keeps subgraph small (~10-15% larger) while enabling import validation.

        Returns:
            TestContext with seeds, context_nodes, edges, test_nodes.
        """
        if edge_filter is None:
            # Exclude depends_on (imports) as noise; include structural edges only
            edge_filter = {'contains', 'calls', 'accesses', 'inherits', 'tests', 'uses'}

        # Extract changed function/class names from the patch, with the
        # enclosing class name too when there is one (kg_construction#63:
        # a bare name can match more than one class' same-named method in
        # the same file -- e.g. a real encode/httpx patch to
        # AsyncClient.aclose also matched the unrelated
        # BoundAsyncStream.aclose). Resolved via the pre-patch file's real
        # AST line ranges (kg_construction#75), not the diff text alone --
        # requires fetching code_file's own pre-patch content.
        # A file the patch itself creates (git's 'new file mode' header)
        # genuinely doesn't exist at base_commit -- fetching it would raise
        # even though this isn't a data error (kg_construction#93).
        # A file the patch itself creates (git's 'new file mode' header)
        # genuinely doesn't exist at base_commit -- it has no node in the
        # KG (built from the real base_commit git tree) and no BFS-reachable
        # context can exist either, since nothing at base_commit could
        # reference a file that isn't there yet. Its seed node(s) are built
        # directly from the reconstructed post-patch source instead of a KG
        # lookup (kg_construction#93).
        if is_newly_created_file(instance['patch'], instance['code_file']):
            return self._extract_for_new_file(instance)

        pre_patch_source = self.repo_manager.read_file_at_commit(
            instance['repo'], instance['base_commit'], instance['code_file']
        )
        changed = self.patch_parser.extract_changed_functions_with_scope(
            instance['patch'],
            instance['code_file'],
            pre_patch_source,
        )

        # KG nodes store pre-patch source. Seeds need post-patch source
        # since generated tests are evaluated against the real, current
        # repo state (kg_construction#124). Context nodes are unaffected
        # since the patch never touched them.
        post_patch_source = reconstruct_post_patch_source(
            pre_patch_source, instance['patch'], instance['code_file']
        )
        post_patch_ranges: Dict[Tuple[str, Optional[str]], ast.AST] = {}
        post_patch_lines: List[str] = []
        if post_patch_source is not None:
            try:
                post_tree = ast.parse(post_patch_source)
            except SyntaxError:
                post_tree = None
            if post_tree is not None:
                post_patch_lines = post_patch_source.splitlines(keepends=True)
                class_stack: List[str] = []

                class _PostPatchVisitor(ast.NodeVisitor):
                    def visit_ClassDef(self, node):
                        post_patch_ranges[(node.name, class_stack[-1] if class_stack else None)] = node
                        class_stack.append(node.name)
                        self.generic_visit(node)
                        class_stack.pop()

                    def visit_FunctionDef(self, node):
                        # Recurse so nested functions/classes are found
                        # too (matches builder.py's _emit_function).
                        post_patch_ranges[(node.name, class_stack[-1] if class_stack else None)] = node
                        self.generic_visit(node)

                    visit_AsyncFunctionDef = visit_FunctionDef

                _PostPatchVisitor().visit(post_tree)

        # Find the code file node
        code_file_results = self.engine.find_file_by_path(instance['code_file'])
        if not code_file_results:
            raise ValueError(f"Code file not found: {instance['code_file']}")
        code_file_node = code_file_results[0]

        # Find seed nodes: changed functions in code_file
        seed_ids: List[str] = []
        for name, enclosing_class in changed:
            funcs = self.engine.find_function_by_name(name)
            # Filter to only those in the code_file
            matches = [
                func for func in funcs
                if func['metadata'].get('filepath') == instance['code_file']
            ]
            # When the name alone is ambiguous (>1 match in this file) AND
            # the diff told us which class it belongs to, narrow to that
            # class -- otherwise every same-named match becomes a seed,
            # and LLMSerializer._build_seed_section's seeds[0] would show
            # the model an arbitrary one of them (same failure shape as
            # #54's test-file-as-seed defect). If the class hint doesn't
            # narrow it to exactly one match (e.g. the hint wasn't
            # available, or somehow still ambiguous), fall through with
            # all matches unfiltered -- TestContextValidator's
            # _check_no_ambiguous_seed_names is the backstop that refuses
            # to silently proceed with an unresolved collision.
            if len(matches) > 1 and enclosing_class is not None:
                narrowed = [
                    func for func in matches
                    if func['metadata'].get('class') == enclosing_class
                ]
                if len(narrowed) == 1:
                    matches = narrowed

            # A changed entity can be a CLASS itself (e.g. a class-body
            # attribute assignment outside any method) -- find_function_by_name
            # can never match this, since it only searches
            # function/method/test_function node types. Try find_class_by_name
            # when no function match was found (kg_construction#85).
            if not matches:
                classes = self.engine.find_class_by_name(name)
                matches = [
                    cls for cls in classes
                    if cls['metadata'].get('filepath') == instance['code_file']
                ]

            for func in matches:
                seed_ids.append(func['id'])

        # If no changed functions found, use the code_file itself as seed
        if not seed_ids:
            seed_ids = [code_file_node['id']]

        # The test file itself is NOT a seed -- a seed is something to
        # generate a test FOR, and the test file is existing test content,
        # already surfaced separately via test_nodes ('tests' edges,
        # extracted below). It used to be appended to seed_ids here, but
        # subgraph_nodes' order (which context.seeds is derived from) comes
        # from BFS visited-node order, not seed_ids' construction order --
        # so this was non-deterministic, and when the test file landed at
        # seeds[0], LLMSerializer._build_seed_section's `seeds[0]` picked
        # the test file (empty signature/source_code) instead of the real
        # target function, silently starving the LLM-augmented arm of the
        # seed's actual code on roughly a third of instances/runs (see
        # kg-test-generation#49's investigation, kg_construction#54).
        # Nothing needs the test file in seed_ids for BFS reachability
        # either: 'tests' edges point FROM the test function TO the seed
        # (see builder.py's edge construction), so BFS from the seed alone
        # (with incoming-direction traversal, already enabled) reaches the
        # test function without the test file needing to be a seed itself.

        # BFS to extract subgraph
        subgraph_nodes, subgraph_edges = self._bfs(
            seed_ids,
            depth=depth,
            edge_filter=edge_filter
        )

        # Optionally add import edges from seeds (for import validation)
        if include_seed_imports:
            subgraph_nodes, subgraph_edges = self._add_seed_imports(
                seed_ids, subgraph_nodes, subgraph_edges
            )

        # Find test functions via 'tests' edges -- restricted to edges
        # targeting the seed itself (edge['target'] in seed_ids), not any
        # 'tests' edge anywhere in the depth-N BFS subgraph. Without this
        # restriction, a test of some unrelated function 1-2 hops away
        # (e.g. a caller's own test, or a sibling method's test) gets
        # misattributed as an existing test FOR the seed -- the same class
        # of scoping bug as the 'related' list (kg-test-generation#49),
        # here amplified once calls-based 'tests'-edge derivation (added
        # for kg_construction#57's audit) made 'tests' edges far more
        # common than the naming-convention-only heuristic ever produced.
        # Confirmed empirically: handle_401_2017's subgraph contained 372
        # 'tests'-edge-reachable nodes, but only a handful actually call
        # handle_401 -- most were tests of other functions/methods nearby.
        seed_id_set = set(seed_ids)
        test_nodes = []
        for edge in subgraph_edges:
            if edge['relation'] == 'tests' and edge['target'] in seed_id_set:
                test_node = self.engine.nodes_by_id.get(edge['source'])
                if test_node:
                    test_nodes.append(test_node)

        # Separate seeds from context
        seed_node_ids = set(seed_ids)
        seed_nodes = []
        stale_seed_labels = []
        for n in subgraph_nodes:
            if n['id'] not in seed_node_ids:
                continue
            refreshed, was_refreshed = self._refresh_seed_source(
                n, post_patch_ranges, post_patch_lines
            )
            seed_nodes.append(refreshed)
            if not was_refreshed:
                stale_seed_labels.append(n['label'])
        context_nodes = [n for n in subgraph_nodes if n['id'] not in seed_node_ids]

        return TestContext(
            seeds=seed_nodes,
            context_nodes=context_nodes,
            edges=subgraph_edges,
            test_nodes=test_nodes,
            repo=instance['repo'],
            base_commit=instance['base_commit'],
            stale_seed_labels=stale_seed_labels,
        )

    def _refresh_seed_source(
        self,
        node: Dict,
        post_patch_ranges: Dict[Tuple[str, Optional[str]], ast.AST],
        post_patch_lines: List[str],
    ) -> Tuple[Dict, bool]:
        """Return (node, was_refreshed) -- a copy of a seed node with
        source_code/signature replaced by the post-patch version, leaving
        everything else (id, edges, docstring, decorators, etc.)
        untouched, plus whether the refresh actually happened.

        Never mutates the passed-in node in place: it's the KG's own
        cached node object, shared across every extract() call that
        touches it, so mutating it here would corrupt the KG for other
        instances/callers.

        Falls back to the node as-is (stale source_code, was_refreshed
        False) if the seed can't be located in the post-patch AST -- e.g.
        the patch had no hunks for this file, or the reconstructed source
        failed to parse. This should not normally happen for a real seed
        on a real (non-new-file) instance -- callers should treat a
        non-empty TestContext.stale_seed_labels as worth investigating,
        not routine.
        """
        key = (node['label'], node.get('metadata', {}).get('class'))
        ast_node = post_patch_ranges.get(key)
        if ast_node is None:
            return node, False

        signature, source_code = _get_signature_and_source_text(ast_node, post_patch_lines)
        if not source_code:
            return node, False

        refreshed = dict(node)
        refreshed['metadata'] = dict(node.get('metadata', {}))
        refreshed['metadata']['source_code'] = source_code
        refreshed['metadata']['signature'] = signature
        return refreshed, True

    def _extract_for_new_file(self, instance: Dict) -> TestContext:
        """Build a TestContext for a file the patch itself creates.

        No KG lookup or BFS is possible: the file doesn't exist at
        base_commit, so it has no node in the base_commit-built KG, and
        nothing there could reference it either -- there is genuinely no
        context to traverse, not a lookup failure. Seed nodes are built
        directly from the reconstructed post-patch source via AST, shaped
        identically to a real KG node (same _build_func_metadata call the
        real builder uses) so downstream serialization treats them the same
        way (kg_construction#93).
        """
        source = reconstruct_post_patch_source("", instance['patch'], instance['code_file'])
        if source is None:
            raise ValueError(
                f"'{instance['code_file']}' has a 'new file mode' header but no "
                f"hunks for it in this patch"
            )

        tree = ast.parse(source)
        import_map, _, _ = _collect_file_level_info(tree)
        source_lines = source.splitlines(keepends=True)
        repo = instance['repo']
        rel_path = instance['code_file']

        seed_nodes: List[Dict] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_id = _make_id(f"class_{repo}_{rel_path}_{node.name}")
                seed_nodes.append({
                    'id': class_id, 'type': 'class', 'label': node.name,
                    'metadata': {
                        'filepath': rel_path, 'repo': repo, 'lineno': node.lineno,
                        'bases': _get_base_names(node),
                        'decorators': _get_decorators(node),
                        'docstring': _get_docstring(node),
                        'attributes': _get_class_attributes(node),
                        'newly_created_file': True,
                    },
                })
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_id = _make_id(f"func_{repo}_{rel_path}_{node.name}_{child.name}")
                        func_metadata = _build_func_metadata(
                            child, rel_path, repo, parent_class=node.name,
                            import_map=import_map, source_lines=source_lines,
                        )
                        func_metadata['newly_created_file'] = True
                        seed_nodes.append({
                            'id': func_id,
                            'type': 'test_function' if child.name.startswith('test_') else 'method',
                            'label': child.name,
                            'metadata': func_metadata,
                        })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_id = _make_id(f"func_{repo}_{rel_path}_{node.name}")
                func_metadata = _build_func_metadata(
                    node, rel_path, repo, import_map=import_map, source_lines=source_lines,
                )
                func_metadata['newly_created_file'] = True
                seed_nodes.append({
                    'id': func_id,
                    'type': 'test_function' if node.name.startswith('test_') else 'function',
                    'label': node.name,
                    'metadata': func_metadata,
                })

        return TestContext(
            seeds=seed_nodes,
            context_nodes=[],
            edges=[],
            test_nodes=[],
            repo=instance['repo'],
            base_commit=instance['base_commit'],
        )

    def _add_seed_imports(
        self,
        seed_ids: List[str],
        nodes: List[Dict],
        edges: List[Dict],
    ) -> tuple:
        """Add depends_on edges from seed nodes only (for import validation).

        Includes import edges where the source is a seed node. This enables
        validation rules to check whether seeds use things they import, without
        bloating the subgraph with all module dependencies.

        Args:
            seed_ids: List of seed node IDs to extract imports from
            nodes: Subgraph node list (will be extended with imported modules)
            edges: Subgraph edge list (will be extended with import edges)

        Returns:
            (nodes, edges) with import edges added
        """
        seen_node_ids = {n['id'] for n in nodes}
        seen_edge_keys = {(e['source'], e['target'], e['relation']) for e in edges}

        for seed_id in seed_ids:
            for import_edge in self.engine.edges_by_source.get(seed_id, []):
                if import_edge['relation'] in ('depends_on', 'module_depends_on'):
                    target_id = import_edge['target']
                    edge_key = (import_edge['source'], target_id, import_edge['relation'])

                    # Avoid duplicates and ensure target exists
                    if edge_key not in seen_edge_keys and target_id in self.engine.nodes_by_id:
                        edges.append(import_edge)
                        seen_edge_keys.add(edge_key)

                        # Also include the imported node if not already present
                        if target_id not in seen_node_ids:
                            nodes.append(self.engine.nodes_by_id[target_id])
                            seen_node_ids.add(target_id)

        return nodes, edges

    def _bfs(
        self,
        seed_ids: List[str],
        depth: int = 2,
        edge_filter: Optional[Set[str]] = None,
    ) -> tuple:
        """BFS traversal from seeds, filtered by edge type.

        Uses GraphTraversal utility for generic BFS. Expands outward from
        seed nodes up to `depth` hops, including only edges whose relation
        is in edge_filter. Both directions are traversed (incoming and outgoing).

        Args:
            seed_ids: Starting node IDs.
            depth: Maximum hop distance.
            edge_filter: Set of edge relations to include.

        Returns:
            (nodes, edges) — lists of node and edge dicts in the subgraph.
        """
        if edge_filter is None:
            edge_filter = set()

        # Filter seed IDs to those that exist in the KG
        valid_seed_ids = [nid for nid in seed_ids if nid in self.engine.nodes_by_id]

        # Use GraphTraversal for BFS
        traversal = GraphTraversal()
        visited_node_ids, traversed_edges = traversal.bfs(
            valid_seed_ids,
            self.engine.nodes_by_id,
            self.engine.kg['edges'],
            depth=depth,
            edge_filter=edge_filter,
            directions={"outgoing", "incoming"},
        )

        # Convert visited node IDs to node dicts
        nodes_list = [self.engine.nodes_by_id[nid] for nid in visited_node_ids]

        return nodes_list, traversed_edges
