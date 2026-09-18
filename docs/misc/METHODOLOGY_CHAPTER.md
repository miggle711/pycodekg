# Chapter 7: Methodology

## 7.1 Knowledge Graph Construction

Repositories are first cloned into a bare mirror which will serve as the base of KG construction. The construction of our knowledge graph occurs in 2 phases: Parallel Extraction and Sequential Resolution.

In Parallel Extraction, each `.py` file is converted into an Abstract Syntax Tree. The tree's top-level nodes are iterated through:
- Imports are converted into Import Nodes
- Classes are converted into Class nodes; each child method is iterated through
  - For methods: emit call edges (capturing which functions/methods are invoked)
  - For classes: extract base classes, attributes, and class instantiations
- Top-level functions are processed similarly to methods, emitting function-level call edges

Each node stores metadata: function signatures, parameter types, docstrings, decorators, raised exceptions, and branch counts. Edges are initially unresolved (call targets stored as name strings).

In Sequential Resolution, all nodes from all workers are aggregated and a global index is built mapping function/class names to node IDs. Unresolved edges are resolved by matching call targets against this index with confidence tagging: `exact` (single match), `ambiguous` (multiple matches), or dropped (external libraries). Call context is added (caller count, direct callers per function).

The output is a deterministic KG with node IDs derived from MD5 hashes of qualified names, enabling reproducible, commit-specific code structure representation.

---

## 7.2 Patch Parsing and Seed Identification

Given a unified diff patch from a GitHub issue and a target code file, the patch parser identifies which functions/methods were modified. The parser:

1. Parses the unified diff into per-file change hunks
2. Locates hunks for the target code file
3. Extracts function/class definitions appearing in added or modified lines (regex matching `def function_name` and `class ClassName`)
4. Returns a set of changed function/class names

These changed names are then queried against the KG to locate corresponding seed nodes. Seeds are the starting points for subgraph extraction; they represent the modified functions that need test coverage. If a test file exists in the repo, it is also added as a seed to enable test discovery.

---

## 7.3 Subgraph Extraction

From seed nodes, a Breadth-First Search (BFS) extracts surrounding context up to a configurable depth. The algorithm traverses both outgoing edges (function calls, inheritance, instantiation) and incoming edges (callers, subclasses), collecting related nodes and edges.

Edge filtering excludes low-signal edges (e.g., `imports`, `module_depends_on`) which scatter context across unrelated code. Default filter includes: `contains`, `calls`, `inherits`, `tests`, `uses`.

The extracted subgraph includes:
- **Seed nodes**: the modified functions
- **Context nodes**: related functions (callers, callees), parent/child classes, test functions
- **Edges**: relationships within the subgraph

Depth parameter balances specificity (depth=1: tight focus) vs. context richness (depth=2+: broader understanding).

---

## 7.4 Subgraph Validation

The extracted subgraph is validated against two-layer criteria:

**Must-haves** (blocking errors):
- No orphaned nodes (every node has ≥1 edge)
- No broken edges (source and target both in subgraph)
- Seeds connected via subgraph edges
- Closed subgraph (all edge sources/targets present)
- No duplicate edges

**Should-haves** (warnings):
- Test coverage: ≥1 test function references seed
- Seed types: seeds are functions/methods (not files)
- Context density: edges ≥ 0.5 × nodes (indicates good coverage)

Validation ensures subgraphs are well-formed and suitable for LLM consumption.

---

## 7.5 LLM-Based Test Generation

The subgraph extraction process involves three distinct serialization points, each serving a different purpose:

For efficient LLM consumption, the validated subgraph is reformatted into a hierarchical JSON structure organized by semantic relationships. Recent research on structured prompting demonstrates 40% performance variance based on format (He et al., 2024), with hierarchical structures significantly outperforming flat representations for code reasoning tasks.

The hierarchical format organizes information into three semantic sections:

- **Seed Section**: Contains the modified function(s) at the center of analysis. Includes function signature, full docstring, declared exceptions, and complete source code. This section establishes what changed and what the LLM must generate tests for.

- **Context Section**: Provides execution context surrounding the seed. Includes direct callers (functions that invoke the seed), direct callees (functions the seed invokes), parent/child classes (for inheritance relationships), existing test functions (to infer testing patterns and conventions), and execution patterns (control flow branches, type hints, error handling). This section answers "how is the seed used?" and "what does it depend on?"

- **Instructions Section**: Explicit task directives for test generation. Specifies coverage targets (boundary conditions, happy paths, error cases, edge cases), naming conventions, assertion patterns expected by the repository, and any repository-specific testing idioms. This section acts as both domain knowledge (what matters to this codebase) and task specification (what the LLM should produce).

This compact ~2KB representation respects LLM token budgets while preserving essential context. Hierarchical organization enables the LLM to prioritize relevant code relationships and infer execution paths without traversing a flat list of nodes. The semantic grouping mirrors how developers reason about code: understand what changed (Seed), understand its context (Context), then generate appropriate tests (Instructions).

The prompt instructs the LLM to:

1. Analyze the seed function's signature, docstring, and exceptions
2. Review related functions (callers, callees) for execution context
3. Examine existing tests for patterns and conventions
4. Generate test cases covering:
   - Boundary conditions (from extracted control flow statements)
   - Happy path scenarios (normal inputs, expected returns)
   - Error paths (exceptions, invalid inputs)
   - Edge cases (None, empty containers, type variations)

The LLM outputs test code following repository conventions, with clear docstrings explaining each test's purpose. Post-generation, tests are executed against the patched code to verify functional correctness.

---

## 7.6 Test Execution and Validation

Generated tests are executed against the patched code to verify:
1. **Syntactic correctness**: tests parse as valid Python
2. **Executability**: tests run without import errors or crashes
3. **Assertion validity**: assertions pass on patched code
4. **Isolation**: tests do not interfere with each other or global state

Failed tests are logged with error traces for debugging. Passing tests indicate the LLM successfully understood the code change and generated appropriate test scenarios.

---

## 7.7 Evaluation Metrics

Test generation quality is evaluated across multiple dimensions:

**Coverage metrics**:
- Line coverage: % of code paths in seed functions exercised by generated tests
- Branch coverage: % of if/else branches covered
- Exception coverage: % of documented exceptions tested

**Test quality metrics**:
- Assertion count: number of assertions per test (target: ≥2)
- Assertion diversity: distinct assertion types used (assert, pytest.raises, mock assertions)
- Docstring presence: % of tests with explaining docstrings

**Efficiency metrics**:
- KG build time: wall-clock time for AST parsing and edge resolution
- Extraction time: BFS subgraph extraction latency
- LLM latency: time from prompt submission to test code generation
- Total pipeline time: end-to-end (repo clone → test generation)

**Validation metrics**:
- Subgraph validity: % of extracted subgraphs passing must-haves
- Test execution success: % of generated tests that pass without error
- Test relevance: % of tests that exercise seed function code (measured via coverage)

---

**End of Chapter 7**
