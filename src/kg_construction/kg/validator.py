"""
kg_validator.py

Post-extraction validation for Knowledge Graphs.

Runs after full KG construction but before subgraph extraction to catch
Phase I extraction bugs early:
  ✓ Every edge connects two real nodes (no dangling edges)
  ✓ No orphaned nodes (indicates bad extraction)
  ✓ No self-loops (indicates edge artifact)
  ✓ No cycles in dependency graph (indicates circular imports)
  ✓ Consistent metadata (indicates uniform extraction)
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from kg_construction.validation.base import ValidationBase


class KGValidator(ValidationBase):
    """Validate Knowledge Graph structure and metadata consistency."""

    def __init__(self, kg: Dict):
        """
        Args:
            kg: KG dict as returned by RepoKGBuilder.build().
        """
        super().__init__()
        self.kg = kg
        self.nodes = kg.get('nodes', [])
        self.edges = kg.get('edges', [])
        self.repo = kg.get('metadata', {}).get('repo', 'unknown')

        # Build lookup structures
        self.node_ids: Set[str] = {n['id'] for n in self.nodes}
        self.node_types: Dict[str, str] = {n['id']: n['type'] for n in self.nodes}
        self.nodes_by_id: Dict[str, Dict] = {n['id']: n for n in self.nodes}
        self.edges_by_source: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_target: Dict[str, List[Dict]] = defaultdict(list)

        for edge in self.edges:
            self.edges_by_source[edge['source']].append(edge)
            self.edges_by_target[edge['target']].append(edge)

    def validate(self) -> Tuple[bool, str]:
        """Run all validations and return (success, report).

        Returns:
            (is_valid, report_string) where is_valid is True if all checks pass.
        """
        self.errors.clear()
        self.warnings.clear()

        self._check_dangling_edges()
        self._check_orphaned_nodes()
        self._check_self_loops()
        self._check_cycles()
        self._check_metadata_consistency()

        return self._format_report()

    def _check_dangling_edges(self) -> None:
        """Flag edges whose source or target is not a node.

        Everything downstream looks nodes up by id, so a dangling edge is
        skipped or raises a KeyError, and the relation it carried is lost
        (#225). The report counts them per relation and shows a few examples.
        """
        dangling = [e for e in self.edges
                    if e['source'] not in self.node_ids or e['target'] not in self.node_ids]
        if not dangling:
            return

        by_relation: Dict[str, int] = defaultdict(int)
        for edge in dangling:
            by_relation[edge['relation']] += 1
        counts = ', '.join(f"{relation} {count}" for relation, count in sorted(by_relation.items()))
        examples = [
            f"{edge['relation']}: {self._node_label(edge['source'])} -> "
            f"{self._node_label(edge['target'])}"
            for edge in dangling[:3]
        ]
        self.errors.append(
            f"Dangling edges ({len(dangling)}; {counts}): {'; '.join(examples)}"
            f"{'...' if len(dangling) > 3 else ''}"
        )

    def _check_orphaned_nodes(self) -> None:
        """Flag nodes with no incoming or outgoing edges (except files/imports).

        Looks nodes up via nodes_by_id rather than a linear scan over
        self.nodes -- at real repo scale (e.g. django's ~40-46k nodes),
        a scan per orphaned node found is quadratic exactly when orphaning
        is widespread, the failure mode this check exists to catch.
        """
        exclude_types = {'file', 'test_file', 'import'}
        orphaned = []

        for node_id in self.node_ids:
            node_type = self.node_types[node_id]
            if node_type in exclude_types:
                continue

            has_edges = bool(self.edges_by_source.get(node_id) or
                           self.edges_by_target.get(node_id))
            if not has_edges:
                node = self.nodes_by_id[node_id]
                orphaned.append(f"{node['label']} ({node_type})")

        if orphaned:
            self.errors.append(
                f"Orphaned nodes ({len(orphaned)}): {', '.join(orphaned[:5])}"
                f"{'...' if len(orphaned) > 5 else ''}"
            )

    def _check_self_loops(self) -> None:
        """Flag unexpected self-loops (except recursive calls which are valid)."""
        # Self-loops are OK for 'calls' edges (recursive calls) but suspicious for others
        # Note: Some edge types may legitimately have self-loops in extracted code
        self_loops = [e for e in self.edges if e['source'] == e['target'] and e['relation'] != 'calls']
        if self_loops:
            labels = [self._node_label(e['source']) for e in self_loops[:3]]
            self.warnings.append(
                f"Self-loops ({len(self_loops)}, non-call): {', '.join(labels)}"
                f"{'...' if len(self_loops) > 3 else ''}"
            )

    def _check_cycles(self) -> None:
        """Detect cycles in dependency graph using DFS.

        Only checks edges: calls, accesses, inherits, uses, overrides,
        module_depends_on. Excludes 'contains' (parent-child, not circular),
        'tests' (orthogonal), 'depends_on' and 'imports' (allowed to form
        cycles, resolved at runtime).
        """
        dep_relations = {'calls', 'accesses', 'inherits', 'uses', 'overrides', 'module_depends_on'}

        # Build adjacency list for dependency edges
        dep_graph: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges:
            if edge['relation'] in dep_relations:
                dep_graph[edge['source']].append(edge['target'])

        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(start_id: str) -> None:
            """Iterative DFS (explicit stack, not real recursion); a
            real repo's dependency chain can run deep enough to hit
            Python's default recursion limit, which would crash this
            whole validation pass instead of just reporting a warning.
            Each stack frame tracks its own position in dep_graph.get(
            node_id, []) so a node can be revisited after its neighbors
            are exhausted, matching the original recursive version's
            path/rec_stack bookkeeping exactly.
            """
            stack: List[Tuple[str, List[str], int]] = [(start_id, [], 0)]
            visited.add(start_id)
            rec_stack.add(start_id)

            while stack:
                node_id, path, neighbor_idx = stack[-1]
                if neighbor_idx == 0:
                    path = path + [node_id]

                neighbors = dep_graph.get(node_id, [])
                if neighbor_idx >= len(neighbors):
                    rec_stack.discard(node_id)
                    stack.pop()
                    continue

                stack[-1] = (node_id, path, neighbor_idx + 1)
                neighbor = neighbors[neighbor_idx]
                if neighbor not in visited:
                    visited.add(neighbor)
                    rec_stack.add(neighbor)
                    stack.append((neighbor, path, 0))
                elif neighbor in rec_stack:
                    cycle = path[path.index(neighbor):] + [neighbor]
                    cycles.append(cycle)

        for node_id in dep_graph:
            if node_id not in visited:
                dfs(node_id)

        if cycles:
            cycle_strs = []
            for cycle in cycles[:3]:
                labels = [self._node_label(n) for n in cycle]
                cycle_strs.append(" → ".join(labels))
            self.warnings.append(
                f"Cycles in dependency graph ({len(cycles)} found): "
                f"{'; '.join(cycle_strs)}"
                f"{'...' if len(cycles) > 3 else ''}"
            )

    def _check_metadata_consistency(self) -> None:
        """Verify each node type's metadata has its expected keys, AND that
        those keys actually hold real content (not just present-but-empty).

        Previously checked only a hardcoded {'filepath'} against every node
        type, ignoring _expected_metadata_keys entirely -- which already
        correctly declares 'signature' as expected for function/method/
        test_function nodes, but nothing ever consulted it. That's exactly
        the shape of kg-test-generation#51's bug (signature/source_code
        never computed for any function/method -- an empty string, not a
        missing key, so even a present-vs-missing check would have missed
        it): this validator had the right expectations table sitting right
        next to the fix location the whole time and never used it.

        Also: previously warned only if >50% of nodes in a type were
        affected, which would catch a repo-wide bug like #51 (100%
        affected) but miss a narrower one affecting a minority of nodes.
        Any affected node now produces a warning (capped to the first 3
        reported, to keep the report readable) -- there's no threshold
        below which missing a function's real signature/source_code is
        fine to stay silent about.
        """
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for node in self.nodes:
            by_type[node['type']].append(node)

        for node_type, nodes in by_type.items():
            if not nodes:
                continue

            expected_keys = self._expected_metadata_keys(node_type)
            if not expected_keys:
                continue

            inconsistent = []
            for node in nodes:
                meta = node.get('metadata', {})
                # A key that exists but holds an empty/falsy value (e.g.
                # signature: "") is exactly as useless to the LLM as a
                # missing key -- both mean "no real information here" --
                # so check truthiness, not just key presence.
                bad_keys = {k for k in expected_keys if not meta.get(k)}
                if bad_keys:
                    inconsistent.append((node['label'], bad_keys))

            if inconsistent:
                labels_missing = [f"{l}: {m}" for l, m in inconsistent[:3]]
                self.warnings.append(
                    f"{node_type} metadata missing/empty expected keys "
                    f"({len(inconsistent)}/{len(nodes)}): {', '.join(labels_missing)}"
                    f"{'...' if len(inconsistent) > 3 else ''}"
                )

    def _expected_metadata_keys(self, node_type: str) -> Set[str]:
        """Return expected metadata keys for a given node type.

        'file'/'test_file' nodes use metadata['path'] (see
        RepoASTParser.parse_repo's file-node construction), while
        function/method/class nodes use metadata['filepath'] -- two
        different conventions for two different node categories, not a
        typo in one or the other. Declaring 'filepath' for file/test_file
        here (as this previously did) meant _check_metadata_consistency
        would have flagged every single file/test_file node in every KG,
        100% of the time, the moment it was actually wired up to check
        real expected keys instead of a hardcoded {'filepath'} -- caught
        by running the strengthened check against the real dataset.
        """
        expectations = {
            'function': {'signature', 'filepath'},
            'method': {'signature', 'filepath'},
            'test_function': {'signature', 'filepath'},
            'class': {'filepath'},
            'file': {'path'},
            'test_file': {'path'},
        }
        return expectations.get(node_type, set())

    def _node_label(self, node_id: str) -> str:
        """Get human-readable label for a node ID.

        Falls back to the full node_id, not a truncated prefix. A
        truncated prefix risks two different nodes looking identical in
        a report at real repo scale.
        """
        node = self.nodes_by_id.get(node_id)
        return node['label'] if node else node_id

    def _format_report(self) -> Tuple[bool, str]:
        """Format validation report using base class formatter."""
        stats = f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}"
        return super()._format_report(
            title=f"KG Validation Report: {self.repo}",
            stats_line=stats,
        )
