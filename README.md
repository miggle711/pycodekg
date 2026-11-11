# pycodekg

pycodekg turns a Python repository into a knowledge graph (KG) you can query. It reads the source with Python's `ast` module and records every file, class, function and import as a node, and every call, inheritance link and import as an edge. Names are resolved against an index of the whole repo, not file by file, so a call to `send()` is linked to the right `send` and not just the first one with that name.

You can point it at a GitHub repo and a commit, or at a local directory, including a working tree with uncommitted changes. What you get back is a JSON graph. Use it to look up callers and callees, or as structural context for LLM test generation.

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
with open('kg_output/kg_psf_requests_a0df2cbb.json') as f:  # save() names it kg_{repo}_{commit[:8]}.json
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

## Context for LLM prompts

`llm/llm_serializer.py` turns an extracted subgraph into the JSON that goes into a prompt. It leaves test code out of the callers, callees and related classes, so a prompt for writing tests does not show the model the tests that already exist.

## Optional: pyright-backed type inference

The uppercase-heuristic that detects `uses` edges (`x = SomeClass()`) can't see class
instances returned by a lowercase factory function (`session = requests.session()`). Pass
`infer_types=True` to resolve these via a real type checker (pyright) instead:

```python
builder = RepoKGBuilder(infer_types=True)
kg = builder.build('psf/requests', 'a0df2cbb')
```

This is opt-in and off by default. It adds a `pyright` dependency and a per-repo
subprocess cost (roughly 1 to 3 seconds for a mid-sized package). If pyright isn't
installed, or resolution fails for any reason, the KG build still succeeds and you just
get the uppercase-heuristic `uses` edges. Resolved edges carry `metadata.source == 'pyright'`
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
scripts/                    # Dataset and build helpers, and kg_diff.py (compare KGs before and after a change)
```

## Graph Structure

### Node types

| Type | Represents |
|------|-----------|
| `file` | A `.py` source file |
| `test_file` | A test file: `test_*.py`, `*_test.py`, `conftest.py`, anything under a `tests` directory, or under a `test` or `testing` directory at the repo root |
| `class` | A class definition |
| `function` | A top-level function |
| `method` | A method inside a class |
| `test_function` | A `test_*` function or method |
| `import` | An imported module or name |

### Edge types

| Relation | Meaning |
|----------|---------|
| `contains` | File/class contains a class, function, or method |
| `imports` | File imports a module or name, wherever the import sits: at module level, under `try`/`except` or `if TYPE_CHECKING:`, or inside a function |
| `calls` | Function calls another function, constructs a class instance (`SomeClass(...)`), or dispatches an operator to a dunder method (`a \| b` calls `__or__`). Has a `confidence` label (see below). A call made only inside a closure the function defines is kept and tagged `via_closure` |
| `accesses` | Function reads an `@property`-decorated attribute (no call syntax; confidence: `qualified`) |
| `inherits` | Class inherits from another class |
| `tests` | Test function targets a specific function, class, or dunder method (also derived automatically from any resolved `calls` edge sourced from a test function) |
| `uses` | Class instantiates another class (or, with `infer_types=True`, a lowercase factory-function call resolved by pyright, see above) |
| `overrides` | Method overrides a parent class method |
| `raises` | Function raises or catches an exception class defined elsewhere in the repo |
| `decorated_by` | Function is decorated by another function or class defined in the repo (bare, unqualified decorators only) |
| `depends_on` | Function uses a specific import |
| `module_depends_on` | File depends on another file through its imports |

### How edges are resolved

When a file is parsed, an edge is only a name ("this function calls `save`"). A second pass, once every file has been read, links each name to a node using the whole repo. If it can't tell which node is meant, it drops the edge. It never links a name to every candidate.

Resolved edges carry a `confidence` label in their metadata.

For `calls`:
* `qualified`: pinned down by something concrete, like the class of `self`, a variable assigned from `SomeClass()`, or the import the name came through.
* `exact`: the name is unique in the repo, and the call is a bare name or goes through an import from the repo.
* `name_only`: the name is unique in the repo, but the type of the object is unknown (`x.save()`). A reasonable guess, not a certainty.

For `inherits`, `uses`, `raises` and `decorated_by`, the label is `exact` when one definition matches and `ambiguous` when several do.

A few rules that explain results you might not expect:
* Imports are followed in order. A call goes through the import that has run by that point in the function, so a rebound name counts as each import in turn.
* In `try: import a as c` / `except ImportError: import b as c`, calls go to `a`, the `try` body. If a name is imported in each branch of an `if`/`else`, calls and `depends_on` edges go to every branch.
* Imports are collected wherever they sit: at module level, under `try`/`except` or `if TYPE_CHECKING:`, or inside a function. `module_depends_on` follows `src/` layouts, packages, `from pkg import submodule` and relative imports.
* `v.name()` on an object is never linked to an imported function called `name`. Only a bare `name()` or a call on an imported module is.
* A name imported from outside the repo is not linked to a repo definition of the same name for `inherits`, `uses`, `raises` and `decorated_by`. `from functools import cached_property` does not point at a repo class called `cached_property`.
* Code outside the tests is never linked to a definition in a test file.

### Node metadata

Every function/method node carries: parameter list with defaults and annotations, return type annotation, decorators, docstring, raised and caught exceptions, branch count, and whether it is async. Test functions additionally store assert patterns. Class nodes include base classes, decorators, docstring, and class-level attributes. File nodes include module constants and `__all__` exports.

### A small example

For this pair of files:

```python
# models.py
class Model:
    def save(self): ...

# app.py
from models import Model

def create():
    m = Model()
    m.save()
```

`create` gets two `calls` edges, one to the class `Model` and one to its method `save`:

```json
{"source": "create", "target": "Model", "relation": "calls",
 "metadata": {"confidence": "qualified", "import_resolved": "models.Model"}}
{"source": "create", "target": "save", "relation": "calls",
 "metadata": {"confidence": "qualified", "import_resolved": null}}
```

(The real edges use node ids as `source` and `target`. Names are shown here to keep it readable.) Both are `qualified`, the first through the import and the second because `m` was assigned from `Model()`.

Each KG is stamped with the `schema_version` it was built with. A cached KG from an older version is rebuilt instead of reused, so results never mix two versions of the rules.

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

Tests run entirely on synthetic Python source written to `tmp_path` fixtures, so no git
clone or network access is needed. A handful of tests that exercise `infer_types=True`
require the optional `pyright` package (see Installation) and skip automatically if it
isn't installed.

## Installation

```bash
# Core install (KG building, extraction and validation, stdlib only)
pip install -e .

# Optional extras, as needed:
pip install -e ".[datasets]"  # SWE-bench dataset examples
pip install -e ".[types]"     # pyright-backed type inference (infer_types=True)
pip install -e ".[all]"       # everything
```

Core functionality requires only Python 3.10+. No other dependencies beyond the standard library.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR process. A change that alters which nodes or edges a KG contains needs a `scripts/kg_diff.py` table in the PR description and a `SCHEMA_VERSION` bump.

## See Also

* [SWE-bench](https://github.com/princeton-nlp/SWE-bench), the benchmark dataset used as input
* [docs/BUILDER_DEEP_DIVE.md](docs/BUILDER_DEEP_DIVE.md), how the builder parses files and resolves edges
* [docs/misc/AST_HELPERS_DEEP_DIVE.md](docs/misc/AST_HELPERS_DEEP_DIVE.md), the AST helpers
* [docs/misc/LLM_SERIALIZER_DEEP_DIVE.md](docs/misc/LLM_SERIALIZER_DEEP_DIVE.md), how subgraphs become prompts
