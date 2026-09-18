# pycodekg

Build structural Knowledge Graphs (KGs) from Python repository source code via real AST parsing — not embeddings, not text search. Every function, class, call, and inheritance relationship becomes a queryable node or edge, resolved with a real name index across the whole repo (not per-file guessing).

Point it at a GitHub repo + commit, or a local directory (including a working tree with uncommitted changes), and get back a JSON graph you can query for callers/callees, file contents, and more — or use as structural context for LLM-based test generation.

## Quick Start

```bash
pip install -e ".[all]"  # or just `.` for core-only, see Installation below

# Build a KG for a real GitHub repo at a specific commit
pkg-run build psf/requests --commit a0df2cbb

# ...or build one from a local directory (no git required)
pkg-run build . --name my-project

# Query it
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --callers send
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --callees send
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --file sessions.py
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --tests send
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --class Session
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --list-files
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --list-functions
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --list-classes
pkg-run query kg_output/kg_psf_requests_a0df2cbb.json --export send,resolve_redirects
```

All query subcommands print a human-readable listing by default; pass `--json` for the raw dict.

Running `pkg-run` with no arguments falls back to an interactive wizard that walks you through building a KG and extracting/validating an LLM-ready subgraph for a specific code change (repo, commit, patch file, code file, test file).

## Using it as a library

```python
from kg_construction.kg.builder import RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.extraction.validator import TestContextValidator
import json

# Build from a specific commit
builder = RepoKGBuilder()
kg = builder.build('psf/requests', 'a0df2cbb')
builder.save('psf/requests', kg)

# Load and query
with open('kg_output/kg_psf_requests.json') as f:
    kg = json.load(f)

engine = KGQueryEngine(kg)

# Find a file and explore its contents
files = engine.find_file_by_path('sessions.py')
contents = engine.get_file_contents(files[0]['id'])
print(contents['classes'], contents['functions'])

# Find what calls a function
callers = engine.find_callers(contents['functions'][0]['id'])

# Extract a subgraph for test generation
instance = {
    'repo': 'psf/requests',
    'base_commit': 'a0df2cbb',
    'patch': '...',  # unified diff
    'code_file': 'requests/sessions.py',
    'test_file': 'tests/test_sessions.py',
}

extractor = TestContextExtractor(engine)
context = extractor.extract(instance, depth=2)

# Validate the subgraph
validator = TestContextValidator(context)
is_valid, report = validator.validate()
print(report)
```

## Optional: pyright-backed type inference

The uppercase-heuristic that detects `uses` edges (`x = SomeClass()`) can't see class
instances returned by a lowercase factory function (`session = requests.session()`). Pass
`infer_types=True` to resolve these via a real type checker (pyright) instead:

```python
builder = RepoKGBuilder(infer_types=True)
kg = builder.build('psf/requests', 'a0df2cbb')
```

This is opt-in and off by default — it adds a `pyright` dependency and a per-repo
subprocess cost (roughly 1-3s for a mid-sized package). If pyright isn't installed, or
resolution fails for any reason, the KG build still succeeds; you just get the
uppercase-heuristic `uses` edges only. Resolved edges carry `metadata.source == 'pyright'`
so you can distinguish them from heuristic-derived ones. Requires the `types` extra (see
Installation).

## Package Structure

```text
src/kg_construction/
├── kg/
│   ├── builder.py         # Clone repo, parse AST, emit KG nodes and edges
│   ├── query.py           # In-memory query engine
│   ├── validator.py       # Full KG validation (post-extraction sanity checks)
│   ├── repo_manager.py    # Git clone and archive extraction
│   ├── traversal.py       # Shared BFS graph traversal
│   └── type_inference.py  # Optional pyright-backed type resolution for factory calls
├── ast/
│   └── helpers.py         # Pure AST-in/data-out utilities
├── extraction/
│   ├── context.py         # Subgraph extraction (TestContext, TestContextExtractor)
│   ├── validator.py       # Subgraph validation for LLM test generation
│   └── patch.py           # Unified diff parsing (changed-function detection)
├── llm/
│   └── llm_serializer.py  # Flat subgraph -> hierarchical JSON for LLM prompts
├── validation/
│   └── base.py            # Shared base class for kg/extraction validators
├── pipeline.py            # Extract and validate orchestration (extract_and_validate)
└── cli.py                 # pkg-run: build/query subcommands + interactive fallback

tests/                      # Flat pytest suite (no subdirectories)
```

## Graph Structure

### Node types

| Type | Represents |
|------|-----------|
| `file` | A `.py` source file |
| `test_file` | A test file (`test_*.py` or `*_test.py`) |
| `class` | A class definition |
| `function` | A top-level function |
| `method` | A method inside a class |
| `test_function` | A `test_*` function or method |
| `import` | An imported module or name |

### Edge types

| Relation | Meaning |
|----------|---------|
| `contains` | File/class contains a class, function, or method |
| `imports` | File imports a module or name |
| `calls` | Function calls another function, constructs a class instance (`SomeClass(...)`), or dispatches an operator to a dunder method (`a \| b` → `__or__`) (confidence: `exact`/`ambiguous`/`qualified`) |
| `accesses` | Function reads an `@property`-decorated attribute (no call syntax; confidence: `qualified`) |
| `inherits` | Class inherits from another class |
| `tests` | Test function targets a specific function, class, or dunder method (also derived automatically from any resolved `calls` edge sourced from a test function) |
| `uses` | Class instantiates another class (or, with `infer_types=True`, a lowercase factory-function call resolved via pyright — see below) |
| `overrides` | Method overrides a parent class method |
| `raises` | Function raises or catches an exception class defined elsewhere in the repo |
| `decorated_by` | Function is decorated by another function or class defined in the repo (bare, unqualified decorators only) |
| `depends_on` | Function uses a specific import |
| `module_depends_on` | File depends on another file via imports |

### Node metadata

Every function/method node carries: parameter list with defaults and annotations, return type annotation, decorators, docstring, raised and caught exceptions, branch count, and whether it is async. Test functions additionally store assert patterns. Class nodes include base classes, decorators, docstring, and class-level attributes. File nodes include module constants and `__all__` exports.

## Query Engine

```python
engine = KGQueryEngine(kg)

# Node accessors
engine.get_files()                          # all file/test_file nodes
engine.get_functions()                      # all function/method/test_function nodes

# Structural queries
engine.get_file_contents(file_id)           # {file, classes, functions}
engine.get_class_methods(class_id)          # list of method nodes

# Call graph
engine.find_callers(func_id)                # functions that call this one
engine.find_callees(func_id)                # functions this one calls
engine.find_test_functions_for(func_id)     # test functions covering this function

# Search
engine.find_file_by_path('sessions.py')     # substring match on path
engine.find_function_by_name('send')        # exact label match
engine.find_class_by_name('Session')        # exact label match, class nodes

# Export
engine.export_subgraph([node_id, ...])      # nodes + 1-hop edges as dict
```

## Running Tests

```bash
# Install in editable mode first
pip install -e .

# Run the full suite
pytest tests/ -v

# Run a single test file
pytest tests/test_uses_edge_confidence.py -v
```

Tests run entirely on synthetic Python source written to `tmp_path` fixtures — no git
clone or network access required. A handful of tests that exercise `infer_types=True`
require the optional `pyright` package (see Installation) and skip automatically if it
isn't installed.

## Installation

```bash
# Core install (KG building, extraction, validation — stdlib only)
pip install -e .

# Optional extras, as needed:
pip install -e ".[datasets]"  # SWE-bench dataset examples
pip install -e ".[types]"     # pyright-backed type inference (infer_types=True)
pip install -e ".[all]"       # everything
```

Core functionality requires only Python 3.10+. No other dependencies beyond the standard library.

## See Also

- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) — the benchmark dataset used as input
