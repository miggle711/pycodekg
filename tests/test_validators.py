"""Smoke tests for validators."""

import pytest
from kg_construction.kg.validator import KGValidator
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.extraction.context import TestContext


class TestKGValidator:
    """Test KGValidator for knowledge graphs."""

    @pytest.fixture
    def valid_kg(self):
        """Create a valid minimal KG."""
        return {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function',
                 'metadata': {'filepath': 'a.py', 'signature': 'def func_a():'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function',
                 'metadata': {'filepath': 'b.py', 'signature': 'def func_b():'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n2', 'relation': 'calls'},
            ],
            'metadata': {'repo': 'test/repo', 'commit': 'abc123'},
        }

    @pytest.fixture
    def kg_with_orphans(self):
        """Create KG with orphaned nodes."""
        return {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function', 'metadata': {'filepath': 'b.py'}},
                {'id': 'n3', 'label': 'func_c', 'type': 'function', 'metadata': {'filepath': 'c.py'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n2', 'relation': 'calls'},
            ],
            'metadata': {'repo': 'test/repo'},
        }

    def test_validate_valid_kg(self, valid_kg):
        """Validate a valid KG."""
        validator = KGValidator(valid_kg)
        is_valid, report = validator.validate()
        assert is_valid is True
        assert '✅' in report or 'passed' in report.lower()

    @pytest.mark.parametrize("edge", [
        {'source': 'n1', 'target': 'missing', 'relation': 'depends_on'},
        {'source': 'missing', 'target': 'n2', 'relation': 'calls'},
    ], ids=["missing_target", "missing_source"])
    def test_an_edge_to_a_missing_node_is_an_error(self, valid_kg, edge):
        valid_kg['edges'].append(edge)
        is_valid, report = KGValidator(valid_kg).validate()
        assert is_valid is False
        assert "Dangling edges (1; " in report and edge['relation'] in report and "missing" in report

    def test_validate_kg_with_orphans(self, kg_with_orphans):
        """Detect orphaned nodes."""
        validator = KGValidator(kg_with_orphans)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'Orphaned' in report or 'orphan' in report.lower()

    def test_validate_self_loops(self):
        """Detect self-loops (non-recursive calls)."""
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n1', 'relation': 'uses'},
            ],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        # Self-loop with non-'calls' relation should warn
        assert 'Self-loops' in report or 'self-loop' in report.lower()

    def test_detects_a_real_cycle(self):
        """A -> B -> C -> A must be reported as a cycle."""
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function', 'metadata': {'filepath': 'b.py'}},
                {'id': 'n3', 'label': 'func_c', 'type': 'function', 'metadata': {'filepath': 'c.py'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n2', 'relation': 'calls'},
                {'source': 'n2', 'target': 'n3', 'relation': 'calls'},
                {'source': 'n3', 'target': 'n1', 'relation': 'calls'},
            ],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'Cycles' in report or 'cycle' in report.lower()

    def test_no_false_positive_cycle_on_a_dag(self, valid_kg):
        """A plain acyclic call graph (n1 -> n2) must not be flagged."""
        validator = KGValidator(valid_kg)
        is_valid, report = validator.validate()
        assert 'Cycles' not in report

    def test_deep_chain_does_not_raise_recursion_error(self):
        """kg_construction#review: _check_cycles' DFS previously used real
        Python recursion with no depth guard -- a sufficiently deep real
        dependency chain could raise RecursionError and crash the whole
        validation pass. Converted to an iterative DFS; this is a direct
        regression test at a depth well past Python's default recursion
        limit (1000).
        """
        depth = 5000
        nodes = [
            {'id': f'n{i}', 'label': f'func_{i}', 'type': 'function', 'metadata': {'filepath': f'{i}.py'}}
            for i in range(depth)
        ]
        edges = [
            {'source': f'n{i}', 'target': f'n{i + 1}', 'relation': 'calls'}
            for i in range(depth - 1)
        ]
        kg = {'nodes': nodes, 'edges': edges, 'metadata': {'repo': 'test/repo'}}

        validator = KGValidator(kg)
        is_valid, report = validator.validate()  # must not raise RecursionError
        assert 'Cycles' not in report

    def test_report_includes_stats(self, valid_kg):
        """Report includes node/edge statistics."""
        validator = KGValidator(valid_kg)
        is_valid, report = validator.validate()
        assert 'Nodes:' in report or 'nodes' in report.lower()
        assert 'Edges:' in report or 'edges' in report.lower()

    def test_missing_signature_key_is_flagged(self):
        """kg-test-generation#51: signature/source_code were never computed
        for any function/method -- this must be caught, not silently
        passed, even though 'filepath' (the only key checked before this
        fix) was present the whole time #51 was live.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'signature' in report

    def test_empty_string_signature_is_flagged_same_as_missing(self):
        """A present-but-empty signature (metadata['signature'] == '') is
        exactly as useless to the LLM as a missing key -- both must be
        flagged, since key-presence alone doesn't mean real content.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function',
                 'metadata': {'filepath': 'a.py', 'signature': ''}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'signature' in report

    def test_single_affected_node_is_flagged_not_just_majority(self):
        """Previously only warned if >50% of nodes in a type were missing
        a critical key -- a bug affecting a MINORITY of nodes (e.g. one
        function out of ten with a real parsing edge case) would have
        been silently ignored. Any affected node must be reported.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function',
                 'metadata': {'filepath': 'a.py', 'signature': 'def func_a():'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function',
                 'metadata': {'filepath': 'b.py', 'signature': 'def func_b():'}},
                {'id': 'n3', 'label': 'func_c', 'type': 'function',
                 'metadata': {'filepath': 'c.py'}},  # the one bad node, 1/3
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'func_c' in report
        assert '1/3' in report

    def test_file_node_with_path_key_is_not_flagged(self):
        """file/test_file nodes use metadata['path'], not metadata['filepath']
        (a different convention than function/method/class nodes) --
        _expected_metadata_keys previously declared 'filepath' for these
        too, which would have flagged every real file/test_file node in
        every KG the moment this check was strengthened to actually look
        at real expected keys (caught via a real-dataset sweep).
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'mod.py', 'type': 'file', 'metadata': {'path': 'mod.py'}},
                {'id': 'n2', 'label': 'test_mod.py', 'type': 'test_file', 'metadata': {'path': 'test_mod.py'}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'file metadata missing' not in report
        assert 'test_file metadata missing' not in report

    def test_class_node_without_signature_is_not_flagged(self):
        """Classes aren't expected to have a 'signature' key
        (_expected_metadata_keys only requires 'filepath' for classes) --
        this must not produce a false-positive warning.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'MyClass', 'type': 'class', 'metadata': {'filepath': 'a.py'}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'signature' not in report


class TestTestContextValidator:
    """Test TestContextValidator for subgraph validation."""

    @pytest.fixture
    def valid_context(self):
        """Create a valid TestContext."""
        return TestContext(
            seeds=[
                {'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'session.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[
                {'id': 't1', 'label': 'test_send', 'type': 'test_function', 'metadata': {'filepath': 'test_session.py'}},
            ],
            repo='test/repo',
            base_commit='abc123def456',
        )

    @pytest.fixture
    def context_with_orphans(self):
        """Create TestContext with orphaned nodes."""
        return TestContext(
            seeds=[
                {'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'session.py'}},
                {'id': 'c2', 'label': 'orphan', 'type': 'function', 'metadata': {'filepath': 'other.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )

    def test_validate_valid_context(self, valid_context):
        """Validate a valid TestContext."""
        validator = TestContextValidator(valid_context)
        is_valid, report = validator.validate()
        assert is_valid is True

    def test_validate_context_with_orphans(self, context_with_orphans):
        """Detect orphaned nodes in TestContext."""
        validator = TestContextValidator(context_with_orphans)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'Orphaned' in report or 'orphan' in report.lower()

    def test_validate_broken_edges(self):
        """Detect broken edges in TestContext."""
        context = TestContext(
            seeds=[{'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}}],
            context_nodes=[],
            edges=[
                {'source': 's1', 'target': 'nonexistent', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is False

    def test_validate_disconnected_seeds(self):
        """Detect seeds with no outgoing edges."""
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'session.py'}},
            ],
            edges=[],  # No edges
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'Disconnected' in report or 'disconnected' in report.lower()

    def test_seed_with_only_incoming_edges_is_not_flagged_as_disconnected(self):
        """kg_construction#91: a seed with real INCOMING edges (callers)
        but no outgoing edges (e.g. a leaf-like function whose own calls
        aren't resolvable) has real, usable context -- must not be flagged.
        """
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': 'leaf', 'type': 'function', 'metadata': {'filepath': 'mod.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'caller', 'type': 'function', 'metadata': {'filepath': 'mod.py'}},
            ],
            edges=[
                {'source': 'c1', 'target': 's1', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is True

    def test_newly_created_file_seed_with_zero_edges_is_not_flagged(self):
        """kg_construction#93: a seed from a file the patch itself creates
        has genuinely zero edges (nothing at base_commit could reference a
        file that doesn't exist yet) -- this is expected, not a bug.
        """
        context = TestContext(
            seeds=[
                {
                    'id': 's1', 'label': 'brand_new', 'type': 'function',
                    'metadata': {'filepath': 'new_mod.py', 'newly_created_file': True},
                },
            ],
            context_nodes=[],
            edges=[],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is True

    def test_report_includes_context_stats(self, valid_context):
        """Report includes context statistics."""
        validator = TestContextValidator(valid_context)
        is_valid, report = validator.validate()
        assert 'Seeds:' in report or 'seeds' in report.lower()
        assert 'Context:' in report or 'context' in report.lower()

    def test_same_named_seeds_in_different_classes_is_not_flagged(self):
        """Two DIFFERENT classes each having their own real, correctly
        resolved same-named method (e.g. both KMeans.fit and
        MiniBatchKMeans.fit genuinely changed by the same patch) is not a
        collision -- each is already unambiguously scoped to its own class.
        PatchParser's real AST line ranges (kg_construction#75) can no
        longer produce the original #63 false-positive-sweep scenario at
        all (confirmed directly: a real single-class patch only ever
        resolves to that one class's method). Confirmed via a full
        TestGenEval validation run that flagging this shape wrongly
        rejected ~8% of real instances (django, matplotlib, astropy,
        pytest, scikit-learn, pylint, xarray) -- most were genuine
        multi-class patches like this one.
        """
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': 'fit', 'type': 'method',
                 'metadata': {'filepath': 'k_means_.py', 'class': 'KMeans'}},
                {'id': 's2', 'label': 'fit', 'type': 'method',
                 'metadata': {'filepath': 'k_means_.py', 'class': 'MiniBatchKMeans'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'helper', 'type': 'function', 'metadata': {'filepath': 'k_means_.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
                {'source': 's2', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is True

    def test_same_class_same_named_seeds_is_still_flagged(self):
        """The genuine remaining collision case: the SAME label on the
        SAME class (or with no class hint at all) resolving to more than
        one distinct node is a real, unresolved ambiguity -- seed lookup
        genuinely cannot tell the two apart, and LLMSerializer's
        seeds[0]-trusting behavior means the model could be shown either
        one arbitrarily.
        """
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': 'aclose', 'type': 'method',
                 'metadata': {'filepath': 'client.py', 'class': 'AsyncClient'}},
                {'id': 's2', 'label': 'aclose', 'type': 'method',
                 'metadata': {'filepath': 'client.py', 'class': 'AsyncClient'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'helper', 'type': 'function', 'metadata': {'filepath': 'client.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
                {'source': 's2', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'aclose' in report
        assert 'Ambiguous' in report or 'ambiguous' in report.lower()

    def test_genuine_multi_function_patch_is_not_flagged(self):
        """A real multi-function patch (different labels, e.g.
        api_multi_verb_2012 changing patch/put/request/post together)
        must NOT be flagged -- the signal is specifically the SAME label
        resolving to multiple nodes, not 'more than one seed' in general.
        """
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': 'patch', 'type': 'method', 'metadata': {'filepath': 'api.py'}},
                {'id': 's2', 'label': 'put', 'type': 'method', 'metadata': {'filepath': 'api.py'}},
                {'id': 's3', 'label': 'post', 'type': 'method', 'metadata': {'filepath': 'api.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'api.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
                {'source': 's2', 'target': 'c1', 'relation': 'calls'},
                {'source': 's3', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is True

    def test_newly_created_file_same_name_seeds_are_not_flagged(self):
        """kg_construction#93 follow-up: a wholly new file with multiple
        classes each defining their own '__init__' produces a genuine same
        -label collision, but every seed is already correctly, individually
        resolved -- there's no "which one is real" ambiguity to flag, since
        for a wholly new file EVERY symbol genuinely is changed. Confirmed
        on a real instance (pytest-dev/pytest,
        src/_pytest/mark/expression.py) with Token/Scanner/ParseError each
        defining their own '__init__'.
        """
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': '__init__', 'type': 'method',
                 'metadata': {'filepath': 'expression.py', 'class': 'Token', 'newly_created_file': True}},
                {'id': 's2', 'label': '__init__', 'type': 'method',
                 'metadata': {'filepath': 'expression.py', 'class': 'Scanner', 'newly_created_file': True}},
                {'id': 's3', 'label': '__init__', 'type': 'method',
                 'metadata': {'filepath': 'expression.py', 'class': 'ParseError', 'newly_created_file': True}},
            ],
            context_nodes=[],
            edges=[],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is True
