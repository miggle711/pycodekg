"""Tests for llm_serializer.py: converting flat subgraph JSON into the
hierarchical {seed, context, instructions} dict consumed by kg-test-generation.

Focus: the "module" field derived from each node's metadata["filepath"] --
without it, the LLM has no real import path for the seed function or its
callers/callees/related classes and can fabricate placeholder imports
(kg-test-generation issue #6).
"""

from kg_construction.llm.llm_serializer import LLMSerializer, _filepath_to_module


class TestFilepathToModule:
    def test_converts_nested_filepath(self):
        assert _filepath_to_module("requests/sessions.py") == "requests.sessions"

    def test_converts_top_level_filepath(self):
        assert _filepath_to_module("setup.py") == "setup"

    def test_empty_filepath_returns_empty_string(self):
        assert _filepath_to_module("") == ""


class TestSerializeSeedSection:
    """result["seed"] is a list, one dict per seed. A patch can change
    more than one function/class in the same file, so a single-dict shape
    would silently drop every seed but the first.
    """

    def test_seed_includes_module_and_filepath(self):
        seed_node = {
            "id": "n1",
            "label": "send",
            "type": "method",
            "metadata": {
                "filepath": "requests/sessions.py",
                "signature": "def send(self, request, **kwargs)",
                "source_code": "def send(self, request, **kwargs):\n    ...",
            },
        }
        result = LLMSerializer(repo="psf/requests").serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert len(result["seed"]) == 1
        assert result["seed"][0]["module"] == "requests.sessions"
        assert result["seed"][0]["filepath"] == "requests/sessions.py"
        assert result["seed"][0]["function_name"] == "send"

    def test_seed_with_no_filepath_metadata_gets_empty_module(self):
        seed_node = {"id": "n1", "label": "send", "type": "method", "metadata": {}}
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"][0]["module"] == ""
        assert result["seed"][0]["filepath"] == ""

    def test_seed_does_not_include_decorators(self):
        """decorators is redundant with source_code (it's the line directly
        above the 'def', already included verbatim there) and isn't read by
        kg-test-generation's prompt builder -- dropped from the seed
        section entirely rather than serialized and ignored
        (kg_construction#108).
        """
        seed_node = {
            "id": "n1",
            "label": "send",
            "type": "method",
            "metadata": {"decorators": ["staticmethod"], "source_code": "def send(self): ..."},
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert "decorators" not in result["seed"][0]

    def test_seed_exceptions_reads_from_raises_metadata_key(self):
        """_build_func_metadata (ast/helpers.py) stores raised exceptions
        under the 'raises' key (never 'exceptions') -- the seed section's
        exceptions field must read from that real key, not a key that
        never actually appears in real node metadata (previously
        metadata.get("exceptions", []) always silently returned [], since
        real metadata never has an 'exceptions' key at all).
        """
        seed_node = {
            "id": "n1",
            "label": "send",
            "type": "function",
            "metadata": {
                "raises": ["ValueError('bad input')", "TypeError"],
                "catches": ["KeyError"],
            },
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert set(result["seed"][0]["exceptions"]) == {"ValueError('bad input')", "TypeError"}

    def test_seed_with_no_raises_metadata_gets_empty_exceptions(self):
        seed_node = {"id": "n1", "label": "send", "type": "function", "metadata": {}}
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"][0]["exceptions"] == []

    def test_method_seed_includes_class_name(self):
        """A method's class name (e.g. "Session") must be surfaced so the
        LLM knows to import/instantiate the class rather than attempting a
        bare "from module import method_name" import, which doesn't exist
        for methods (see kg-test-generation issue #14).
        """
        seed_node = {
            "id": "n1",
            "label": "resolve_redirects",
            "type": "method",
            "metadata": {
                "filepath": "requests/sessions.py",
                "class": "Session",
            },
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"][0]["class_name"] == "Session"

    def test_function_seed_has_empty_class_name(self):
        seed_node = {
            "id": "n1",
            "label": "get",
            "type": "function",
            "metadata": {"filepath": "requests/api.py"},
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"][0]["class_name"] == ""

    def test_no_seeds_returns_empty_list(self):
        result = LLMSerializer().serialize(
            {"seeds": [], "context_nodes": [], "edges": [], "test_nodes": []}
        )
        assert result["seed"] == []

    def test_multiple_seeds_are_all_included_not_just_the_first(self):
        seed_node_1 = {
            "id": "n1", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py", "class": "Session"},
        }
        seed_node_2 = {
            "id": "n2", "label": "get", "type": "function",
            "metadata": {"filepath": "requests/api.py"},
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node_1, seed_node_2], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert len(result["seed"]) == 2
        names = {s["function_name"] for s in result["seed"]}
        assert names == {"send", "get"}


class TestSerializeContextSection:
    def _instance(self, seed_node, other_node, relation):
        return {
            "seeds": [seed_node],
            "context_nodes": [other_node],
            "edges": [{"source": other_node["id"], "target": seed_node["id"], "relation": relation}],
            "test_nodes": [],
        }

    def test_caller_includes_module(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        caller_node = {
            "id": "caller", "label": "request", "type": "method",
            "metadata": {"filepath": "requests/sessions.py", "signature": "def request(...)"},
        }
        result = LLMSerializer().serialize(self._instance(seed_node, caller_node, "calls"))

        assert len(result["context"]["callers"]) == 1

    def test_caller_includes_class_name_when_a_method(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        caller_node = {
            "id": "caller", "label": "request", "type": "method",
            "metadata": {"filepath": "requests/sessions.py", "class": "Session"},
        }
        result = LLMSerializer().serialize(self._instance(seed_node, caller_node, "calls"))

        assert result["context"]["callers"][0]["class_name"] == "Session"
        assert result["context"]["callers"][0]["type"] == "method"
        assert result["context"]["callers"][0]["module"] == "requests.sessions"
        assert result["context"]["callers"][0]["name"] == "request"

    def test_callee_includes_module(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        callee_node = {
            "id": "callee", "label": "get_adapter", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [callee_node],
            "edges": [{"source": seed_node["id"], "target": callee_node["id"], "relation": "calls"}],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        assert len(result["context"]["callees"]) == 1
        assert result["context"]["callees"][0]["module"] == "requests.sessions"

    def test_parent_class_includes_module(self):
        seed_node = {
            "id": "seed", "label": "Session", "type": "class",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        parent_node = {
            "id": "parent", "label": "SessionRedirectMixin", "type": "class",
            "metadata": {"filepath": "requests/sessions.py", "source_code": "class SessionRedirectMixin: ..."},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [parent_node],
            "edges": [{
                "source": seed_node["id"], "target": parent_node["id"], "relation": "inherits",
                "metadata": {"confidence": "exact"},
            }],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        related = result["context"]["related"]
        assert len(related) == 1
        assert related[0]["type"] == "parent_class"
        assert related[0]["module"] == "requests.sessions"
        assert related[0]["source"] == "seed"

    def test_ambiguous_inherits_edge_is_excluded(self):
        # A base class name that isn't unique repo-wide (e.g. "Meta") gets
        # one inherits edge per same-named candidate, confidence='ambiguous'
        # -- at most one candidate can be the real parent, so these edges
        # are excluded rather than surfaced as unreliable "parent_class"
        # context.
        seed_node = {
            "id": "seed", "label": "Session", "type": "class",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        parent_node = {
            "id": "parent", "label": "Meta", "type": "class",
            "metadata": {"filepath": "unrelated/models.py", "source_code": "class Meta: ..."},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [parent_node],
            "edges": [{
                "source": seed_node["id"], "target": parent_node["id"], "relation": "inherits",
                "metadata": {"confidence": "ambiguous"},
            }],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        assert result["context"]["related"] == []

    def test_instantiation_includes_module(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        used_node = {
            "id": "used", "label": "HTTPAdapter", "type": "class",
            "metadata": {"filepath": "requests/adapters.py"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [used_node],
            "edges": [{
                "source": seed_node["id"], "target": used_node["id"], "relation": "uses",
                "metadata": {"confidence": "exact"},
            }],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        related = result["context"]["related"]
        assert len(related) == 1
        assert related[0]["type"] == "instantiation"
        assert related[0]["module"] == "requests.adapters"
        assert related[0]["source"] == "seed"

    def test_ambiguous_uses_edge_is_excluded(self):
        # An instantiated-class candidate name that collides with another
        # real class or function elsewhere in the repo gets confidence
        # downgraded to 'ambiguous' (see test_uses_edge_confidence.py) --
        # not a confirmed instantiation, so excluded from related.
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        used_node = {
            "id": "used", "label": "Config", "type": "class",
            "metadata": {"filepath": "unrelated/models.py"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [used_node],
            "edges": [{
                "source": seed_node["id"], "target": used_node["id"], "relation": "uses",
                "metadata": {"confidence": "ambiguous"},
            }],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        assert result["context"]["related"] == []


class TestRelatedScopedToSeedOrSeedClass:
    """Issue #49: 'related' previously iterated ALL inherits/uses/
    instantiates edges in the subgraph, so a caller/callee/sibling method
    that itself instantiated some class had that fact misattributed to the
    seed -- e.g. Session and Request both instantiate PreparedRequest
    somewhere in their own bodies, producing repeated "instantiation:
    PreparedRequest" entries for a seed (prepare_body) that doesn't itself
    instantiate PreparedRequest at all. Restricted to edges sourced from
    the seed itself OR the seed's own class (a method's class typically
    does inheritance/instantiation in __init__, not the method's own body).
    """

    def _method_seed_instance(self, extra_edges, extra_nodes=None):
        seed_node = {
            "id": "seed", "label": "prepare_body", "type": "method",
            "metadata": {"filepath": "requests/models.py", "class": "PreparedRequest"},
        }
        class_node = {
            "id": "class_preparedrequest", "label": "PreparedRequest", "type": "class",
            "metadata": {"filepath": "requests/models.py"},
        }
        return {
            "seeds": [seed_node],
            "context_nodes": [class_node] + (extra_nodes or []),
            "edges": [
                {"source": class_node["id"], "target": seed_node["id"], "relation": "contains"},
            ] + extra_edges,
            "test_nodes": [],
        }

    def test_instantiation_sourced_from_seed_class_is_included(self):
        used_node = {
            "id": "used", "label": "CaseInsensitiveDict", "type": "class",
            "metadata": {"filepath": "requests/structures.py"},
        }
        instance = self._method_seed_instance(
            extra_edges=[{
                "source": "class_preparedrequest", "target": "used", "relation": "uses",
                "metadata": {"confidence": "exact"},
            }],
            extra_nodes=[used_node],
        )
        result = LLMSerializer().serialize(instance)

        related = result["context"]["related"]
        assert len(related) == 1
        assert related[0]["name"] == "CaseInsensitiveDict"
        assert related[0]["source"] == "seed_class"

    def test_inherits_sourced_from_seed_class_is_included(self):
        parent_node = {
            "id": "parent", "label": "RequestEncodingMixin", "type": "class",
            "metadata": {"filepath": "requests/models.py"},
        }
        instance = self._method_seed_instance(
            extra_edges=[{
                "source": "class_preparedrequest", "target": "parent", "relation": "inherits",
                "metadata": {"confidence": "exact"},
            }],
            extra_nodes=[parent_node],
        )
        result = LLMSerializer().serialize(instance)

        related = result["context"]["related"]
        assert len(related) == 1
        assert related[0]["name"] == "RequestEncodingMixin"
        assert related[0]["source"] == "seed_class"

    def test_instantiation_sourced_from_unrelated_node_is_excluded(self):
        """A caller/callee/sibling's own instantiation must NOT leak into
        the seed's 'related' list just because it appears in the subgraph
        -- this is the exact bug #49 identified.
        """
        caller_node = {
            "id": "caller", "label": "prepare", "type": "method",
            "metadata": {"filepath": "requests/sessions.py", "class": "Session"},
        }
        used_node = {
            "id": "used", "label": "PreparedRequest", "type": "class",
            "metadata": {"filepath": "requests/models.py"},
        }
        instance = self._method_seed_instance(
            extra_edges=[
                {"source": "caller", "target": "seed", "relation": "calls"},
                {"source": "caller", "target": "used", "relation": "instantiates"},
            ],
            extra_nodes=[caller_node, used_node],
        )
        result = LLMSerializer().serialize(instance)

        assert result["context"]["related"] == []


class TestSiblingMethods:
    """Issue #50: sibling methods reached via 'contains' BFS were present
    in context_nodes but silently dropped at serialization -- e.g. a
    method's own required setup (PreparedRequest.prepare()) was never
    visible to the model, only the method actually under test. This is
    context a flat single-function extraction (the baseline arm) can
    never provide, since it has no notion of "what else does this class
    define" -- see kg-test-generation#28's precondition-visibility gap.
    """

    def _class_seed_instance(self, sibling_relation="contains", sibling_target_is_seed=False):
        seed_node = {
            "id": "seed", "label": "prepare_content_length", "type": "method",
            "metadata": {"filepath": "requests/models.py", "class": "PreparedRequest"},
        }
        class_node = {
            "id": "class_preparedrequest", "label": "PreparedRequest", "type": "class",
            "metadata": {"filepath": "requests/models.py"},
        }
        sibling_node = {
            "id": "sibling", "label": "prepare", "type": "method",
            "metadata": {
                "filepath": "requests/models.py", "class": "PreparedRequest",
                "source_code": "def prepare(self, ...):\n    self.headers = {}\n",
            },
        }
        sibling_edge_target = seed_node["id"] if sibling_target_is_seed else sibling_node["id"]
        return {
            "seeds": [seed_node],
            "context_nodes": [class_node, sibling_node],
            "edges": [
                {"source": class_node["id"], "target": seed_node["id"], "relation": "contains"},
                {"source": class_node["id"], "target": sibling_edge_target, "relation": sibling_relation},
            ],
            "test_nodes": [],
        }

    def test_sibling_method_of_seed_class_is_included(self):
        result = LLMSerializer().serialize(self._class_seed_instance())

        siblings = result["context"]["sibling_methods"]
        assert len(siblings) == 1
        assert siblings[0]["name"] == "prepare"
        assert siblings[0]["module"] == "requests.models"
        assert "self.headers = {}" in siblings[0]["source_code"]

    def test_seed_does_not_list_itself_as_its_own_sibling(self):
        """The class->seed 'contains' edge itself must not cause the seed
        to appear in its own sibling_methods list.
        """
        result = LLMSerializer().serialize(
            self._class_seed_instance(sibling_target_is_seed=True)
        )

        # Only the class->seed edge exists in this instance (both edges
        # point at the seed) -- sibling_methods must be empty, not contain
        # the seed itself.
        assert result["context"]["sibling_methods"] == []

    def test_contains_edge_from_unrelated_class_is_not_treated_as_sibling(self):
        """A 'contains' edge from a DIFFERENT class (e.g. one reached via
        an unrelated 'related' relationship elsewhere in the subgraph)
        must not be mistaken for a sibling of the seed's own class.
        """
        seed_node = {
            "id": "seed", "label": "handle_401", "type": "method",
            "metadata": {"filepath": "requests/auth.py", "class": "HTTPDigestAuth"},
        }
        seed_class_node = {
            "id": "class_httpdigestauth", "label": "HTTPDigestAuth", "type": "class",
            "metadata": {"filepath": "requests/auth.py"},
        }
        unrelated_class_node = {
            "id": "class_unrelated", "label": "SomeOtherClass", "type": "class",
            "metadata": {"filepath": "requests/other.py"},
        }
        unrelated_method_node = {
            "id": "unrelated_method", "label": "some_method", "type": "method",
            "metadata": {"filepath": "requests/other.py", "class": "SomeOtherClass"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [seed_class_node, unrelated_class_node, unrelated_method_node],
            "edges": [
                {"source": seed_class_node["id"], "target": seed_node["id"], "relation": "contains"},
                {"source": unrelated_class_node["id"], "target": unrelated_method_node["id"], "relation": "contains"},
            ],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        assert result["context"]["sibling_methods"] == []

    def test_no_sibling_methods_is_an_empty_list_not_missing_key(self):
        seed_node = {"id": "seed", "label": "f", "type": "function", "metadata": {}}
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["context"]["sibling_methods"] == []

    def test_existing_tests_are_not_included_even_when_test_nodes_are_present(self):
        # kg_only must not get existing-test content instruct's full-setting
        # prompt has no equivalent of (pycodekg#130).
        seed_node = {"id": "seed", "label": "f", "type": "function", "metadata": {}}
        test_node = {
            "id": "t1", "label": "test_f", "type": "function",
            "metadata": {"source_code": "def test_f(): assert f() == 1"},
        }
        result = LLMSerializer().serialize(
            {
                "seeds": [seed_node],
                "context_nodes": [],
                "edges": [{"source": test_node["id"], "target": seed_node["id"], "relation": "tests"}],
                "test_nodes": [test_node],
            }
        )

        assert "existing_tests" not in result["context"]


class TestContextItemsAreCapped:
    """A high fan-in/fan-out seed (e.g. sklearn's check_array, called from
    hundreds of sites) produced uncapped callers/callees/sibling_methods
    lists under an unbounded depth-2 BFS, real prompts up to 2.7MB for a
    single instance (kg_construction#141). Each category is capped so one
    densely-connected seed can't blow out the prompt size.
    """

    def test_callers_are_capped(self):
        seed_node = {"id": "seed", "label": "check_array", "type": "function", "metadata": {}}
        caller_nodes = [
            {"id": f"caller{i}", "label": f"fn{i}", "type": "function", "metadata": {}}
            for i in range(15)
        ]
        edges = [
            {"source": n["id"], "target": seed_node["id"], "relation": "calls"}
            for n in caller_nodes
        ]
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": caller_nodes, "edges": edges, "test_nodes": []}
        )

        assert len(result["context"]["callers"]) == 10

    def test_callees_are_capped(self):
        seed_node = {"id": "seed", "label": "f", "type": "function", "metadata": {}}
        callee_nodes = [
            {"id": f"callee{i}", "label": f"fn{i}", "type": "function", "metadata": {}}
            for i in range(15)
        ]
        edges = [
            {"source": seed_node["id"], "target": n["id"], "relation": "calls"}
            for n in callee_nodes
        ]
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": callee_nodes, "edges": edges, "test_nodes": []}
        )

        assert len(result["context"]["callees"]) == 10

    def test_sibling_methods_are_capped(self):
        seed_node = {"id": "seed", "label": "send", "type": "method", "metadata": {}}
        seed_class_node = {"id": "cls", "label": "Session", "type": "class", "metadata": {}}
        sibling_nodes = [
            {"id": f"sib{i}", "label": f"method{i}", "type": "method", "metadata": {}}
            for i in range(15)
        ]
        edges = [
            {"source": seed_class_node["id"], "target": seed_node["id"], "relation": "contains"},
        ] + [
            {"source": seed_class_node["id"], "target": n["id"], "relation": "contains"}
            for n in sibling_nodes
        ]
        result = LLMSerializer().serialize(
            {
                "seeds": [seed_node],
                "context_nodes": [seed_class_node] + sibling_nodes,
                "edges": edges,
                "test_nodes": [],
            }
        )

        assert len(result["context"]["sibling_methods"]) == 10

    def test_cap_keeps_first_encountered_order_not_arbitrary(self):
        # Determinism matters here: a rebuild of kg_prompts.json must
        # keep selecting the same 10 callers, not some other subset,
        # since edges are stored in a fixed order in the KG, not a set.
        seed_node = {"id": "seed", "label": "f", "type": "function", "metadata": {}}
        caller_nodes = [
            {"id": f"caller{i}", "label": f"fn{i}", "type": "function", "metadata": {}}
            for i in range(15)
        ]
        edges = [
            {"source": n["id"], "target": seed_node["id"], "relation": "calls"}
            for n in caller_nodes
        ]
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": caller_nodes, "edges": edges, "test_nodes": []}
        )

        kept_names = [c["name"] for c in result["context"]["callers"]]
        assert kept_names == [f"fn{i}" for i in range(10)]
