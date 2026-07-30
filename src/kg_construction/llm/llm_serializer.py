"""
llm_serializer.py

Converts flat subgraph JSON (Phase 4) into hierarchical JSON (Phase 5) optimized for LLM consumption.

The hierarchical format organizes information into three semantic sections:
- Seed: The modified function(s) with metadata and source code
- Context: Surrounding functions, classes, and tests that provide execution context
- Instructions: Task directives for test generation (coverage targets, conventions)

This structure is designed to:
1. Minimize token usage (~2KB for typical subgraph)
2. Enable the LLM to prioritize relevant information
3. Mirror how developers reason about code changes
"""

import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


def _filepath_to_module(filepath: str) -> str:
    """Convert a repo-relative filepath (e.g. "requests/sessions.py") to a
    dotted module path (e.g. "requests.sessions") -- same derivation
    kg.builder.py uses internally to resolve qualified names, but that
    result was never carried through to the LLM-facing serialization,
    leaving the model to guess import paths (see issue #6: this caused
    the model to fabricate placeholder imports like
    "from your_module import X").
    """
    if not filepath:
        return ""
    return filepath.removesuffix(".py").replace("/", ".")


@dataclass
class LLMInput:
    """Hierarchical JSON structure passed to LLM.

    Attributes:
        seed: Section containing the modified function(s).
        context: Section providing execution context.
        instructions: Section with task directives.
    """
    seed: Dict
    context: Dict
    instructions: Dict


class LLMSerializer:
    """Converts TestContext (flat subgraph) to LLM-friendly hierarchical JSON."""

    def __init__(self, repo: str = ""):
        self.repo = repo

    def serialize(self, test_context: Dict) -> Dict:
        """Convert flat subgraph to hierarchical JSON for LLM.

        Args:
            test_context: Flat subgraph dict with seeds, context_nodes, edges, test_nodes.
                         (e.g., output from TestContext.save())

        Returns:
            Hierarchical dict with 'seed', 'context', 'instructions' sections.
        """
        seeds = test_context.get("seeds", [])
        context_nodes = test_context.get("context_nodes", [])
        edges = test_context.get("edges", [])
        test_nodes = test_context.get("test_nodes", [])

        # Build maps for efficient lookup
        node_by_id = {}
        for node in seeds + context_nodes + test_nodes:
            node_by_id[node["id"]] = node

        # Build Seed section from seed nodes
        seed_section = self._build_seed_section(seeds, node_by_id)

        # Build Context section from context_nodes, edges, and test_nodes
        context_section = self._build_context_section(
            seeds, context_nodes, test_nodes, edges, node_by_id
        )

        # Build Instructions section with coverage targets and conventions
        instructions_section = self._build_instructions_section(seeds)

        return {
            "seed": seed_section,
            "context": context_section,
            "instructions": instructions_section,
        }

    def _build_seed_section(self, seeds: List[Dict], node_by_id: Dict) -> Dict:
        """Extract and structure metadata from seed nodes.

        Returns dict with:
            - function_name: Name of the modified function
            - module: Dotted import path (e.g. "requests.sessions"), derived
                      from the node's filepath -- so the LLM can write a
                      real import instead of guessing/fabricating one.
            - class_name: Owning class name (e.g. "Session") when type is
                          "method", else "" -- a method isn't importable by
                          name on its own (e.g. `from requests.sessions
                          import resolve_redirects` doesn't exist); the LLM
                          needs the class to import and instantiate instead
                          (see kg-test-generation issue #14).
            - signature: Function signature
            - docstring: Function docstring
            - exceptions: Exception expressions this function itself raises
                          (metadata['raises'] -- what a test generator
                          needs to know to write pytest.raises assertions;
                          metadata['catches'] is a separate, less directly
                          test-relevant signal, not included here)
            - source_code: Complete function source
        """
        if not seeds:
            return {}

        # For now, assume single seed (most common case)
        seed_node = seeds[0]
        metadata = seed_node.get("metadata", {})

        return {
            "function_name": seed_node.get("label", ""),
            "type": seed_node.get("type", "function"),
            "module": _filepath_to_module(metadata.get("filepath", "")),
            "filepath": metadata.get("filepath", ""),
            "class_name": metadata.get("class", ""),
            "signature": metadata.get("signature", ""),
            "docstring": metadata.get("docstring", ""),
            "exceptions": metadata.get("raises", []),
            "source_code": metadata.get("source_code", ""),
            "type_hints": metadata.get("type_hints", {}),
        }

    def _build_context_section(
        self,
        seeds: List[Dict],
        context_nodes: List[Dict],
        test_nodes: List[Dict],
        edges: List[Dict],
        node_by_id: Dict,
    ) -> Dict:
        """Extract and structure execution context from subgraph.

        Returns dict with:
            - callers: Functions that call the seed
            - callees: Functions called by the seed
            - related: Classes the seed itself, or the seed's own class,
              inherits from or instantiates (not any inherits/uses edge
              anywhere in the subgraph -- restricted to edges sourced
              from the seed or its containing class, so this reflects
              the seed's own relationships, not an unrelated
              caller/callee/sibling's). Each entry's "source" field is
              "seed" (the seed function/method itself) or "seed_class" (the
              seed's owning class, e.g. via __init__) so the model can tell
              the two apart.
            - sibling_methods: Other methods of the seed's own class (e.g.
              __init__, or a setup method like prepare()) -- context a
              flat single-function extraction (the baseline arm) can
              never provide, since it has no notion of "what else does
              this class define" (see issue #50 in kg-test-generation).
            - existing_tests: Test functions that reference the seed
            - patterns: Control flow, type hints, error handling patterns
        """
        seed_ids = {s["id"] for s in seeds}

        # The class (if any) that directly contains a seed -- found via a
        # 'contains' edge from a class node to the seed itself. Needed to
        # tell "sibling method of the seed's own class" apart from any
        # other class->method contains edge that might appear in the
        # subgraph (e.g. via an unrelated 'related' class reached through
        # inheritance/instantiation).
        seed_class_ids = {
            edge["source"]
            for edge in edges
            if edge.get("relation") == "contains"
            and edge["target"] in seed_ids
            and node_by_id.get(edge["source"], {}).get("type") == "class"
        }

        callers = []
        callees = []
        related = []
        sibling_methods = []
        existing_tests = []

        # Extract relationships from edges
        for edge in edges:
            src_id, tgt_id = edge["source"], edge["target"]
            relation = edge.get("relation", "")

            src_node = node_by_id.get(src_id)
            tgt_node = node_by_id.get(tgt_id)

            if not src_node or not tgt_node:
                continue

            # Edges pointing to seed = callers
            if tgt_id in seed_ids and relation == "calls":
                callers.append(self._node_to_snippet(src_node))

            # Edges from seed = callees
            elif src_id in seed_ids and relation == "calls":
                callees.append(self._node_to_snippet(tgt_node))

            # Inheritance and composition -- restricted to edges sourced from
            # the seed itself OR the seed's own class (src_id in seed_ids or
            # seed_class_ids), not just any inherits/uses/instantiates edge
            # that happens to appear in the subgraph. Without this
            # restriction, a caller/callee/sibling method that itself
            # instantiates some class gets that fact attributed to "related"
            # as if it were about the seed -- e.g. Session and Request both
            # instantiate PreparedRequest somewhere in their own bodies,
            # which produced 3 near-identical-looking "instantiation:
            # PreparedRequest" entries for a seed (prepare_body) that
            # doesn't itself instantiate PreparedRequest at all (see
            # kg-test-generation#49's investigation -- the KG data/edges
            # were correct, this was a serialization scoping bug, not real
            # duplication or a bad graph). The seed's own class is included
            # (not just the seed function/method itself) because a method
            # inherits/instantiates via its class, not typically in its own
            # body -- e.g. PreparedRequest.__init__ instantiating
            # CaseInsensitiveDict is relevant to every method on
            # PreparedRequest, not just __init__. `source` records which of
            # the two applies so the model can tell "the seed does this
            # directly" from "the seed's class does this elsewhere."
            elif relation == "inherits" and (src_id in seed_ids or src_id in seed_class_ids):
                related.append(
                    {
                        "type": "parent_class",
                        "name": tgt_node.get("label", ""),
                        "module": _filepath_to_module(tgt_node.get("metadata", {}).get("filepath", "")),
                        "source_code": tgt_node.get("metadata", {}).get("source_code", ""),
                        "source": "seed" if src_id in seed_ids else "seed_class",
                    }
                )
            elif relation == "uses" and (src_id in seed_ids or src_id in seed_class_ids):
                related.append(
                    {
                        "type": "instantiation",
                        "name": tgt_node.get("label", ""),
                        "module": _filepath_to_module(tgt_node.get("metadata", {}).get("filepath", "")),
                        "source": "seed" if src_id in seed_ids else "seed_class",
                    }
                )

            # Other methods on the seed's own class -- e.g. __init__ or a
            # setup method (prepare()) whose side effects the seed's own
            # body depends on but doesn't itself set up. Excludes the
            # seed's own containment edge (tgt_id in seed_ids) so the
            # seed doesn't list itself as its own sibling.
            elif (
                relation == "contains"
                and src_id in seed_class_ids
                and tgt_id not in seed_ids
                and tgt_node.get("type") in ("method", "function")
            ):
                sibling_methods.append(self._node_to_snippet(tgt_node))

        # Extract test functions
        for test_node in test_nodes:
            existing_tests.append(
                {
                    "name": test_node.get("label", ""),
                    "source_code": test_node.get("metadata", {}).get("source_code", ""),
                }
            )

        # Extract patterns from seed nodes
        patterns = self._extract_patterns(seeds, context_nodes)

        return {
            "callers": callers,
            "callees": callees,
            "related": related,
            "sibling_methods": sibling_methods,
            "existing_tests": existing_tests,
            "patterns": patterns,
        }

    def _build_instructions_section(self, seeds: List[Dict]) -> Dict:
        """Generate task directives for test generation.

        Returns dict with:
            - coverage_targets: What to test
            - conventions: How to test (naming, assertions, patterns)
        """
        coverage_targets = [
            "boundary conditions (null, empty, negative values)",
            "happy path (normal inputs, expected outputs)",
            "error cases (exceptions, invalid inputs)",
            "edge cases (type variations, zero/max values)",
        ]

        conventions = {
            "naming": "test_<function>_<scenario> (e.g., test_send_with_timeout)",
            "assertions": "Use assert statements; pytest.raises for exceptions",
            "mocking": "Mock external dependencies only; test real logic",
            "isolation": "Each test should be independent",
        }

        return {
            "coverage_targets": coverage_targets,
            "conventions": conventions,
            "task": "Generate comprehensive unit tests for the seed function",
        }

    def _node_to_snippet(self, node: Dict) -> Dict:
        """Convert node to a brief snippet for inclusion in context."""
        metadata = node.get("metadata", {})
        return {
            "name": node.get("label", ""),
            "type": node.get("type", "function"),
            "module": _filepath_to_module(metadata.get("filepath", "")),
            "class_name": metadata.get("class", ""),
            "signature": metadata.get("signature", ""),
            "docstring": metadata.get("docstring", ""),
            "source_code": metadata.get("source_code", ""),
        }

    def _extract_patterns(self, seeds: List[Dict], context_nodes: List[Dict]) -> Dict:
        """Extract patterns from seed and context nodes (control flow, type hints, etc)."""
        patterns = {
            "control_flow": [],
            "type_hints": {},
            "error_handling": [],
        }

        for node in seeds + context_nodes:
            metadata = node.get("metadata", {})

            # Control flow branches
            if "branch_count" in metadata:
                patterns["control_flow"].append(f"Branches: {metadata['branch_count']}")

            # Type hints
            if "type_hints" in metadata:
                patterns["type_hints"].update(metadata["type_hints"])

            # Error handling
            if "exceptions" in metadata and metadata["exceptions"]:
                patterns["error_handling"].extend(metadata["exceptions"])

        return patterns
