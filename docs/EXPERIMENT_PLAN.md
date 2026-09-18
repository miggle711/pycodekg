# Experiment Plan: `instruct` vs `kg_only` on TestGenEval

This document describes the current design of the comparison experiment for
the Patch-Aware Knowledge Graph Retrieval for LLM-Based Repository-Level
Test Generation project: a KG-augmented test-generation approach (`kg_only`)
against a standard, retrieval-free baseline (`instruct`), evaluated on
TestGenEval (TGE).

## Scope

The experiment targets **test generation, not test completion**, 
no existing test content is shown to either arm.
Concretely, this means using **TestGenEval's `full` setting**: given a code
file, generate a complete test file from scratch. TGE's other three
settings (`first`/`last`/`extra`) are completion tasks that show part of an
existing test file and ask for one more test to be added; they don't fit a
generation-only scope and are not part of the primary result.

**Open decision:** whether the completion settings are dropped from the
codebase entirely or kept available as a secondary/appendix comparison.
Current direction is `full`-only as the primary and, likely, only result.

## Research questions

> **RQ2 (Patch-Based Subgraph Retrieval):** How effectively can code
> patches identify affected entities and retrieve relevant KG subgraphs?

> **RQ3 (Test Generation Correctness and Coverage):** To what extent does
> patch-aware KG subgraph retrieval improve the correctness and coverage of
> LLM-generated test cases compared to retrieval-free baselines?

TestGenEval instances already
come with `code_file` and `test_file` pre-selected by the benchmark itself,
before either arm ever sees a prompt. The "find the relevant file in a
large, unbounded repository" problem the original motivation describes is
already solved by the benchmark's own construction, for both arms equally.
RQ2 is therefore scoped, within this experiment, to a narrower and more
concrete question than "search a whole repository": given the file TGE
already selected, how well does patch-based analysis identify the specific
function(s) within it that changed, and how well does KG traversal from
that point recover genuinely relevant structural context (callers, callees,
class relationships) versus missing it. RQ3 is the direct behavioural
question: does the structured context `kg_only` retrieves lead to better
generated tests than `instruct` reading the file directly.

## Stage 1: Knowledge graph construction

A structural knowledge graph is built for each repository at the exact
commit a dataset instance references (`base_commit`). Every `.py` file is
parsed via AST into graph nodes (files, classes, methods, functions, test
functions) and directed edges (containment, calls, attribute access,
inheritance, existing-test relationships, usage, imports). This produces
one whole-repository graph per commit, reused across every instance that
shares a commit.

## Stage 2: Per-instance subgraph extraction

For each instance, the patch's changed function(s)/class(es) are identified
by resolving the patch's changed lines against real AST-derived line
ranges, checked against both the file's pre-patch and reconstructed
post-patch state. The patch itself is never shown to either arm as prompt
content, it is used only to locate which part of the file the instance is
about. The resolved function/class names, matched to real KG nodes, become
the **seed(s)** for that instance (an instance can have more than one seed
when a patch touches multiple functions).

A bounded breadth-first search from the seed(s): depth 2, over
containment/calls/access/inheritance edges, import edges excluded by
default, collects the surrounding structural context: the seed's own
source, its callers, its callees, and sibling methods on the same class.
The seed's own source is taken from the post-patch version of the file, so
both arms are evaluated against, and shown, the same corrected code for the
function actually under test. The resulting subgraph passes a set of
structural validation checks (no isolated nodes, no dangling edges, seed
connectivity, no duplicate edges) before use.

Existing-test content is deliberately **not** part of either arm's
context, under the `full`-only scope, generation is meant to be a
from-scratch task for both arms, so neither is shown any pre-existing test
material, KG-retrieved or otherwise.

## Stage 3: Prompt construction

Both arms are asked to generate a complete test file for the same target
function(s), with no existing test content shown to either.

- **`instruct`** (baseline): the whole code file, presented as flat text,
  with an instruction naming the specific function(s) the patch changed as
  the focus of testing.
- **`kg_only`**: the seed function's own source, docstring, and structural
  context (callers, callees, sibling methods, and any classes it inherits
  from or instantiates), presented as labelled, KG-derived sections rather
  than a flat file dump.

Both arms are told to focus on the same named function(s), the KG
pipeline's own patch-seeded identification is the single source of truth
for this, shared by both arms rather than reimplemented independently on
each side. This keeps the comparison about *representation* (structured
KG sections vs. a flat file) rather than the two arms silently being asked
to do different jobs.

One deliberate, retained asymmetry: `kg_only`'s structural context is not
restricted to the seed's own file. In the sample checked, the large
majority of `kg_only`'s context nodes are drawn from files other than the
seed's, this is treated as the approach's actual value proposition
(surfacing real cross-file relationships a single-file baseline has no way
to access), not an imbalance to correct. Restricting it to same-file
content to make the comparison "fairer" would remove the one capability
that distinguishes a KG-based approach from `instruct` in the first place.

## Stage 4: Inference

Both arms' prompts are sent to each model under evaluation across a
temperature sweep of **0, 0.5, and 1.0**, to check sensitivity to sampling
temperature as well as model choice. Generated test files are stored for
evaluation.

Models are drawn from open-source coding LLMs, spanning small, medium, and
large size tiers, with all candidates below confirmed to exist and be
accessible:

**Small (~7-8B params)**
- Qwen2.5-Coder-7B-Instruct
- DeepSeek-Coder-6.7B-Instruct
- Meta-Llama-3.1-8B-Instruct (gated, requires accepting Meta's license)
- CodeGemma-7B-IT

**Medium (~15-32B params)**
- Qwen2.5-Coder-32B-Instruct
- StarCoder2-15B-Instruct-v0.1
- Codestral-22B-v0.1
- DeepSeek-Coder-V2-Lite-Instruct (16B MoE, ~2.4B active)

**Large (~70-73B params)**
- Meta-Llama-3.1-70B-Instruct (gated)
- Qwen2.5-72B-Instruct (~145GB at float16)

Two substantially larger models (DeepSeek-Coder-V2-Instruct at 235.7B
params, Qwen3-Coder-480B-A35B-Instruct at 480.2B params) were considered
and set aside, their storage footprint at float16 (roughly 470GB and
960GB respectively) is well beyond what the available compute/storage
setup is expected to support. Qwen2.5-72B-Instruct is listed above as a
same-tier substitute.

**Open decision:** the exact final model shortlist and how many samples
are generated per configuration is not yet locked down as a concrete run
plan. (Sheryl to do)

Inference is planned to run on Monash's M3 HPC cluster. M3 does not
support Docker, so evaluation (Stage 5) will either run in a ported Apptainer image or 
run in a separate environment.

## Stage 5: Evaluation

Each generated test file is evaluated in an isolated environment against
the real, current repository state, along three dimensions:

- **Functional correctness**: whether generated tests execute
  successfully, reported both as "at least one test passes" and "every
  test passes", to distinguish partial-suite failures from total ones.
- **Structural coverage**: the proportion of source lines exercised by
  passing tests.
- **Mutation score**: synthetic bugs are injected into the source and the
  generated suite is re-run against each mutant; the fraction of injected
  bugs caught measures whether the tests meaningfully validate behaviour,
  not just execute without error.

Coverage and mutation score are each computed at two scopes: the
conventional whole-file measurement, and a function-scoped measurement
restricted to the lines of the specific function the instance's patch
touched. Both are recorded; the function-scoped figures are intended as the
primary comparison numbers for RQ2/RQ3, since `kg_only` is structurally
only able to generate tests for the function it was given, while `instruct`
has the whole file available and could pick up incidental coverage or
mutation kills elsewhere in the file unrelated to the function actually
under test. Whole-file figures are kept as secondary/contextual numbers
rather than dropped, since they remain informative about overall generated
test quality on the file as a whole.

**Open decision:** whether the function-scoped numbers are reported as the
sole primary result or presented alongside whole-file numbers as two
named, distinct comparisons in the eventual write-up.

Since Docker is unavailable on M3, running evaluation has two options: a local workstation, or a refactor of the TestGenEval code to port the Docker image to Apptainer. 

Test execution, mutation testing in particular, is comparatively slow per instance, so the choice affects how
long a full evaluation pass takes; not yet decided which is the better fit
once the run matrix (Stage 4) is finalised.

## Known limitations

- **Knowledge graph completeness.** Static analysis currently misses a
  handful of call-resolution patterns: certain dynamic-dispatch/registry
  patterns, chained attribute access on inferred types, some framework
  dispatch conventions (e.g. Django's class-based-view `.as_view()`),
  constructor calls, and dunder/operator-overload calls (e.g. `Q() | Q()`
  routing through `__or__`). These affect how complete the retrieved
  call-graph context is for a given seed, independent of the fairness
  design above, and are tracked as separate KG-quality issues rather than
  correctness bugs in the experiment design itself.
- **Call-graph richness varies per instance.** For a seed with few real
  call-graph connections, `kg_only`'s retrieved context leans more heavily
  on sibling methods and structural relationships than on callers/callees.
  How "rich" a given instance's retrieved context is is not uniform across
  the dataset, and is worth reporting as a property of the sample rather
  than treated as a uniform baseline.
- **RQ2's "issue reports" clause does not apply to this dataset.** TGE's
  schema has no issue-report/problem-statement field at all, so RQ2 as
  originally worded (patches *and* issue reports) is answered here using
  only the patch signal, which is the only one the dataset provides.

## Open decisions requiring sign-off

1. Whether the completion settings (`first`/`last`/`extra`) are dropped
   from the codebase entirely, or retained as an available secondary
   comparison.
2. The concrete run matrix for Stage 4, final model shortlist and sample
   counts per configuration.
3. Whether Stage 5 evaluation runs on a local workstation or a shared,
   Docker-capable cluster.
4. Whether Stage 5's function-scoped coverage/mutation numbers are reported
   as the sole primary metric, or alongside whole-file numbers as two
   named comparisons.
