# KG Improvements: Validation, Packaging, and the Neurosymbolic Interface

Discussion notes on three open questions raised by Mig and Miguel on where the knowledge-graph pipeline goes next, read against the current state of the codebase.

## 1. Structural vs. logical validation

Two validators exist today — `kg/validator.py`'s `KGValidator` and `extraction/validator.py`'s `TestContextValidator` — and both are entirely structural: orphaned nodes, self-loops, cycles, broken or duplicate edges, seed connectivity, coverage counts. Nothing checks whether the graph's claims about the code are actually *true*, only whether the graph is well-formed as a graph.

**Counterpoint holds — mostly.** Python's grammar rules out a whole class of logical errors before the KG is ever built: a `class` node can't inherit from something that isn't a class expression, a `contains` edge can't skip the file→class→method hierarchy. Anything the parser already guarantees doesn't need re-checking.

**One blind spot the counterpoint doesn't cover.** Some claims require semantic *resolution*, not just syntax parsing — and the builder can be structurally perfect while being substantively wrong. `KGValidator._check_cycles` already excludes `depends_on`/`imports` edges because they're "resolved at runtime" — an implicit admission that structural checks alone don't cover import semantics. More concretely: call edges carry an `exact` / `ambiguous` confidence tag because name-based resolution can attach a `calls` edge to the wrong same-named function in another module. That edge looks fine — real source, real target, valid relation — and is simply incorrect.

**Recommendation:** Don't build general semantic validation — type inference or control-flow proof is a large lift for uncertain payoff. Add one targeted check: for `exact`-confidence `calls` edges, cross-reference the resolved target against the caller's `depends_on` edges to confirm it's actually reachable. That tests the one place the pipeline already admits uncertainty, instead of adding validation uniformly everywhere.

## 2. Neurosymbolic packaging for the LLM

This is further along than the framing suggests. `llm/llm_serializer.py` already performs the flat→hierarchical transform: `TestContext`'s flat `seeds` / `context_nodes` / `edges` / `test_nodes` get reshaped into a `{seed, context, instructions}` document, separating the changed function from its callers, callees, related classes, existing tests, and task directives. The docstring already claims a "~2KB for typical subgraph" token target.

**Gap 1 — the claim is unmeasured.** There's no actual before/after token count anywhere in the repo backing that 2KB figure. The first move is empirical: take a real `TestContext.save()` output, tokenize both the raw flat JSON and the `LLMSerializer` output, and get real numbers before optimizing further.

**Gap 2 — JSON is still expensive relative to what it encodes.** Repeated keys (`name`, `signature`, `docstring`, `source_code` on every entry in `callers` / `callees`) spend tokens on structure, not content:

```python
# llm_serializer.py — _node_to_snippet — every caller/callee gets full source
def _node_to_snippet(self, node: Dict) -> Dict:
    metadata = node.get("metadata", {})
    return {
        "name": node.get("label", ""),
        "signature": metadata.get("signature", ""),
        "docstring": metadata.get("docstring", ""),
        "source_code": metadata.get("source_code", ""),  # full body, every time
    }
```

Two changes, ordered by expected payoff:

- **Measure** — Tokenize flat JSON vs. serialized output on a real subgraph to get a baseline instead of an estimate.
- **Trim** — Stop embedding full `source_code` for every caller/callee; signature + docstring is usually enough context for a neighbor, reserve full source for the seed itself. Likely the single largest token sink in the payload today.
- **Reshape** — Consider compact prose/markdown per node instead of repeated JSON keys — e.g. `def send(self, request, **kwargs) — called by Session.request (L123)` carries the same relational signal in fewer tokens.

## 3. Pip/conda packaging & API surface

`pyproject.toml` already makes the project installable: `pip install -e .` works today, there's a `kg-run` console entry point, and `packages.find` is correctly pointed at `src/`. Publishing to PyPI or conda-forge later is mechanical — version bump, build, upload, maybe a release workflow — and is not the hard part.

| | |
|---|---|
| Empty `__init__.py` files | 6 |
| Declared public exports | 0 |
| Lines of source, no stable surface | 3,942 |

**The real gap is the API surface, not packaging config.** Every `__init__.py` in the package — `kg_construction/`, `ast/`, `extraction/`, `kg/`, `llm/`, `validation/` — is empty. A consumer has to reach into internal module paths like `kg_construction.extraction.context.TestContextExtractor`, which means every internal refactor is a breaking change downstream. That defeats the stated goal: test-gen shouldn't need to know the internal module layout.

**Proposed export surface** (`src/kg_construction/__init__.py`):

```python
from kg_construction.kg.builder import RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine
from kg_construction.extraction.context import TestContextExtractor, TestContext
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.kg.validator import KGValidator
from kg_construction.llm.llm_serializer import LLMSerializer, LLMInput

__all__ = [
    "RepoKGBuilder", "KGQueryEngine",
    "TestContextExtractor", "TestContext",
    "TestContextValidator", "KGValidator",
    "LLMSerializer", "LLMInput",
]
```

That single change lets the test-generation package depend on `from kg_construction import TestContextExtractor` and nothing else — genuine separation of concerns — without touching packaging config at all. PyPI/conda publishing is worth doing, but it's a lower-stakes, later step once the API has settled enough to version.

## Suggested next step

Pick one of the three and make it concrete: the exact/ambiguous call-edge cross-check, a token-count script comparing flat JSON against `LLMSerializer` output, or the `__init__.py` export draft above.
