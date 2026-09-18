# Experiment Design: RQ2/RQ3, Comparison Arms, and Fairness

This document captures a long design discussion about how to fairly compare
a KG-augmented test-generation approach against a standard ("retrieval-free")
baseline on TestGenEval (TGE), and how that comparison relates to the
project's RQ2/RQ3 research questions. Written to preserve context across
sessions -- nothing here is implemented yet unless explicitly marked DONE.

## Research questions (as given)

> **RQ2 (Patch-Based Subgraph Retrieval):** How effectively can code patches
> and issue reports identify affected entities and retrieve relevant KG
> subgraphs?

> **RQ3 (Test Generation Correctness and Coverage):** To what extent does
> patch-aware KG subgraph retrieval improve the correctness and coverage of
> LLM-generated test cases compared to retrieval-free baselines?

## The paper's stated motivation (for context)

The project's original motivation text describes three failure modes of
existing approaches, framed around *retrieval over a large, unbounded
repository*:

- **Untargeted Retrieval** -- broad, full-repository context introduces
  irrelevant code that degrades model performance.
- **Context-Change Mismatch** -- static context fails to capture the
  precise boundaries and semantic implications of code modifications.
- **Absence of Semantic Grounding** -- existing retrieval mechanisms don't
  explicitly anchor context to specific changes, failing to support
  accurate oracle generation.

**Key realization:** none of these failure modes actually apply to TGE's
task setup. TGE never presents a full, unbounded repository to search --
every instance already comes with `code_file` and `test_file` pre-selected
by the benchmark's own construction, before either comparison arm ever sees
a prompt. The "find the needle in a large repo" problem your paper motivates
has already been solved by TGE itself, for every arm equally. This means:

- **RQ2** (does patch-aware retrieval correctly identify affected entities
  in a large repo) is **not measurable on TGE at all**. It needs a
  different evaluation entirely -- see "RQ2 is out of scope for this
  experiment" below.
- **RQ3** (does KG-structured context improve generation quality) **is**
  measurable on TGE, but only if the comparison isolates *representation*
  (flat text vs. KG-derived structure), not *retrieval* or *information
  quantity*.

## The core design tension: X vs Y, or X vs X+Y?

Throughout this discussion we used shorthand:

- **X** = what the `instruct` (baseline) arm receives: `code_src` (the
  whole changed source file, in full) + `test_src` (the existing test
  file, at varying truncation per setting -- see below).
- **Y** = KG-derived structural context: the seed function's own source,
  callers, callees, sibling methods, and existing tests already linked via
  a real KG edge.

Two candidate designs were debated at length:

### Design A: "X vs Y" (substitution comparison)
`instruct` = X. `kg_only` = Y only, no `code_src`/`test_src` at all.
Answers: "can KG context *replace* raw file dumping?" A real, useful
question, but **not a clean ablation** -- `kg_only` is missing information
`instruct` has, by design, so any score difference conflates "structure
helped/hurt" with "having less information helped/hurt."

### Design B: "X vs X+Y" (pure additive ablation)
`instruct` = X (unchanged). `kg_only` = X + Y (KG content appended on top
of the *exact same* `code_src`/`test_src` instruct gets).
Answers: "does adding KG context on top of the standard baseline help?" A
true single-variable ablation -- but **redundant**: the KG's code-side
content (the seed's own source) is already fully visible inside `code_src`
for anything within the same file. The only genuinely new information X+Y
adds is out-of-file callers/callees/siblings -- a much narrower question
than the paper's actual value proposition, and it still doesn't test
"structure vs flat" cleanly since instruct's contribution is 100% flat and
kg_only's is "100% flat + a little structured."

**Neither A nor B was accepted as final.** Design A was rejected because it
isn't fair (confounds structure with information quantity). Design B was
rejected because it's redundant and doesn't isolate the structure variable
either.

## Where we landed: Design C

**`kg_only` should NOT receive raw `code_src`/`test_src` text.** Instead,
it should receive **KG-derived structural equivalents of the same
underlying information** `instruct` gets as flat text -- i.e., match
`instruct` on *information content*, but keep *everything* on the KG side
represented as structured KG output (labeled sections, real node data),
not as a raw file dump.

This makes RQ3 a clean, defensible comparison: **same underlying
information available to both arms, presented as flat text (instruct) vs.
KG-structured content (kg_only)**. Any measured difference in
correctness/coverage/mutation-score is then attributable to
*representation*, not to *missing information* (Design A's problem) or to
*redundant duplication* (Design B's problem).

### Code side: already correctly designed

`kg_only`'s existing code-side content (seed's own source, callers,
callees, siblings) already matches this principle -- it's KG-derived,
structured, and roughly informationally comparable to what a model could
infer from `code_src` for the *relevant* parts of that file. **No change
needed here.**

### A real asymmetry worth naming explicitly: patch-awareness itself

Neither `patch` nor `test_patch` is ever shown to either arm as prompt
content -- confirmed by grepping `inference/` in the testgeneval fork:
both fields are referenced only in `swebench_docker/` (evaluation-side:
`evaluate_instance.py` applies both as real patches to set up the eval
container's ground truth before the generated test runs; `swebench_utils
.py`'s `get_test_directives` regexes `test_patch`'s `diff --git` headers
just to get a test-runner target). Also confirmed: `code_src` is already
the POST-patch file content (e.g. instance `django__django-17087-15`'s
`code_src` contains `klass.__qualname__`, the patch's fixed version, not
the pre-patch `klass.__name__`) -- so `instruct` never sees a diff, a
before/after contrast, or any signal about which lines changed. It reads
one complete, already-correct file and has to find what's relevant to
test by reading, the same way a human unfamiliar with the specific change
would.

`kg_only`, by contrast, IS patch-aware from its very first step:
`TestContextExtractor.extract()` calls `PatchParser
.extract_changed_functions_with_scope()` directly on `instance['patch']`
to identify exactly which function(s)/class(es) changed (via real AST
line-range matching against the pre-patch source, not diff-text
guessing), and that becomes the seed the entire subgraph is built around.
This is not something `instruct` could replicate by reading `code_src`
alone -- post-patch file content carries no signal about which lines were
touched; that information only exists in `patch`, which `kg_only`
consumes and `instruct` never sees.

**This is a real, structural difference between the arms, not an
implementation gap to fix.** Checked against RQ3's own wording: "does
**patch-aware** KG subgraph retrieval improve ... compared to
retrieval-free baselines" -- patch-awareness IS the named independent
variable, not a confound to eliminate. `instruct` is deliberately the
retrieval-free baseline (must discover relevance by reading); `kg_only`
is deliberately the patch-aware arm (told where to look, then retrieves
structurally around that point). Design C's "match on information
content, differ on representation" principle applies to the CONTENT once
each arm has decided what's relevant -- it was never meant to also
equalize *how* each arm decides what's relevant, since that decision
process is the thing RQ3 asks about.

### Both arms are testing already-correct code, not "the bug"

A related, important framing point, checked directly against real data:
**`code_src` is the POST-patch (fixed) file content, for both arms, on
essentially every instance.** Verified at scale, not just the one
instance above: comparing each patch's added vs. removed lines against
`code_src` across all 66 django instances, 46/66 clearly match the
post-patch version, only 1/66 appeared to lean pre-patch (and that one
was a line-count artifact of the comparison method, not a real
exception -- manually confirmed `code_src` contains the real fixed line,
`relabeled_clone`, for that instance too), 19/66 inconclusive (too few
distinguishing lines in the diff to tell either way, not evidence of the
opposite).

**Neither `patch` nor `test_patch` is ever shown to the model in either
arm** (confirmed by grep -- both fields only appear in
`swebench_docker/`, purely for evaluation-container setup and
test-target extraction, never in `inference/`'s prompt-building code).
So from the model's perspective in both arms, there is no "bug" and no
"fix" -- it only ever sees one complete, already-correct file (or a
KG-derived view of it) and is asked to write test(s) for it. This isn't
"write a regression test that would have caught this bug" in either
arm -- that framing doesn't apply to TGE's actual task structure at all.

**What this means for interpreting `patch`'s role and any eventual
results:** `patch` is not "the bug" from either arm's perspective --
it's purely a **localization signal**, used internally by `kg_only`'s
retrieval logic (and separately by the eval harness, for container setup
and test-target extraction) to identify which part of a possibly large
file is recently-relevant. The actual task, for both arms, is
coverage/quality-of-test-suite generation against a fixed snapshot of
code -- scored via `baseline_covs`-style coverage and mutation score,
not via "did this test previously fail against the buggy version."

If `kg_only` outperforms `instruct`, the correct interpretation is
**"patch-aware retrieval helps a model triage which part of a large file
is worth testing"** -- not "patch-aware retrieval helps a model catch
bugs" or "helps generate regression tests." The latter framings aren't
supported by TGE's task structure and would be a mischaracterization of
whatever the comparison actually measures. Worth stating this precisely
in any eventual write-up, not just here.

### Test side: the real, confirmed gap

`kg_only` currently gets **zero** test-file information, for any setting.
`instruct` gets real, substantial existing-test content:

Verified directly against a real instance (`django__django-17087-15`,
`tests/migrations/test_writer.py`, 40,029 real chars):

| Setting | `preds_context` key | Content | % of full file |
|---|---|---|---|
| `first` | `preamble` | Imports/helpers only, no test bodies yet | ~4.4% (1,769 chars) |
| `last` | `last_minus_one` | Entire file MINUS the final test method | ~98.6% (39,478 chars) |
| `extra` | `last` | The complete, unmodified file | 100% (40,029 chars) |

**Important correction from initial assumption:** this is not "just the
target test class" -- it's the **entire test file**, including every
other unrelated test class also defined there (confirmed: the real file
also contains `DeconstructibleInstances`, `Money`, `TestModel1`,
`TextEnum`, and others alongside the actual target class
`OperationWriterTests(SimpleTestCase)`). For `last`/`extra`, `instruct`
sees essentially the whole real file's test content, not a filtered slice.

### Confirmed: this test file is already in the KG

Checked directly against the real built KG
(`kg_django_django_4a72da71.json`): **80 real nodes** already exist for
`tests/migrations/test_writer.py` -- real `class` and `method` nodes,
including `OperationWriterTests` and its real methods, with full
`source_code` already captured. This is not new data to generate -- the
full-repo KG build already parses test files exactly like any other `.py`
file, no special-casing excludes them.

**However:** `TestContextExtractor.extract()` currently never reads
`instance['test_file']` at all (confirmed via grep -- zero references).
The only test content it ever surfaces is via real `'tests'`-relation
edges discovered incidentally during BFS traversal from the seed -- which
is a different, narrower signal than "the specific test file this
instance targets." This is the real, scoped gap to fix.

## RQ2 is out of scope for this experiment

Since TGE never presents an unbounded repository to search (every instance
pre-selects `code_file`/`test_file`), **no comparison run on TGE, under any
design (A/B/C), produces evidence about retrieval precision/recall.** RQ2
needs its own, separate evaluation methodology -- likely: run
`PatchParser`/`TestContextExtractor` against a real, full repo (not a
TGE-pre-cut instance) and measure whether the retrieved subgraph
corresponds to the real affected entities, using some ground truth (e.g.
the patch's own touched files/functions). **Not designed yet -- future
work, separate from the current instruct/kg_only comparison.**

## Open implementation questions (not yet resolved)

1. **Granularity of retrieved test content: full method bodies, or
   structure only (names/signatures)?**
   **Resolved in discussion: full bodies, for true parity with `instruct`**
   (which gives full real bodies for `last`/`extra`). Structure-only would
   reintroduce a scope gap on the test side even after fixing the
   "zero test content" gap.

2. **Which test nodes to include, and how to match TGE's own truncation
   boundary per setting?**
   Confirmed instruct gives the *whole file* (all classes/methods, not
   just the target class), truncated only at the tail (0 tests held back
   for `extra`, 1 held back for `last`, effectively all held back for
   `first`). For the KG arm to match this precisely per-setting, we need a
   way to determine, from the KG's test-file nodes, which specific test
   method corresponds to "the one TGE held back" for `last`.

   **Per-setting mapping, worked out:**

   | Setting | What instruct sees | What the KG arm should retrieve |
   |---|---|---|
   | `first` | Near-empty file (imports/helpers only, no test bodies) | Nothing needed -- no real test content exists to map either arm to. Both arms legitimately start from "empty file." |
   | `extra` | The complete, current test file, unmodified | All real test-file nodes for `instance['test_file']`, full bodies, no filtering -- maps 1:1, no held-back test to identify. |
   | `last` | The whole file MINUS exactly one test method | All real test-file nodes MINUS the one specific node TGE held back -- **needs identifying which one.** |

   **`last`'s held-back test is directly identifiable -- VERIFIED against
   all 66 real instances, RESOLVED:**

   First attempt used a naive line-based diff (`difflib.unified_diff` on
   `last` vs `last_minus_one`, regex-matching `def test_` in the added
   lines). Result: 60/66 clean, 6/66 "messy" (no `def` line detected in the
   diff -- caused by the line-diff algorithm sometimes windowing the change
   such that the real `def` line landed in the *unchanged* region, only
   showing a tail fragment as "added"). Manually inspecting one messy case
   confirmed the real method was present in both fragments' text; the
   line-diff heuristic just failed to bound it correctly. 2 of the 6 also
   turned out to be trivial (1-char, trailing-newline-only) diffs with no
   real held-back test at all.

   **Fix: parse both fragments as real Python (`ast.parse` -- both are
   valid, complete-enough Python on their own) and extract every
   `(containing_class, method_name)` pair for methods whose name starts
   with `test`, at any nesting level under a class. Set-difference the two
   method-sets** (`methods_in(last) - methods_in(last_minus_one)`) instead
   of diffing raw text lines. This directly answers "which named test
   method exists in one but not the other" rather than inferring a method
   boundary from a line diff.

   Re-run against all 66 real instances with this fix:

   | Result | Count |
   |---|---|
   | Exactly 1 held-back method found (clean) | 61 |
   | No real held-back method (trivial/no-op diff) | 5 |
   | More than 1 held-back method | 0 |
   | Unparseable | 0 |

   **100% resolve correctly** -- either to exactly one held-back method
   (61/66, the normal case) or correctly identified as "nothing meaningful
   was held back, treat like `extra`" (5/66 -- up from 2 found by manual
   inspection; the AST approach caught 3 more genuine no-op cases the
   line-diff missed entirely). Zero remaining edge cases on the django
   subset.

   **Extended validation: re-ran against the FULL 160-instance
   `kjain14/testgenevallite` dataset, all 11 repos** (not just the 66
   django instances) -- important because other repos use genuinely
   different test-writing conventions (e.g. `pytest-dev/pytest` itself
   uses bare top-level `test_*` functions, not Django's `TestCase`-style
   methods; sympy, sklearn, matplotlib, astropy, xarray, seaborn, sphinx,
   pylint, flask all included too).

   | Repo | n | Clean | No real held-back method |
   |---|---|---|---|
   | django/django | 66 | 61 | 5 |
   | sympy/sympy | 42 | 39 | 3 |
   | scikit-learn/scikit-learn | 20 | 20 | 0 |
   | pytest-dev/pytest | 12 | 10 | 2 |
   | matplotlib/matplotlib | 7 | 7 | 0 |
   | astropy/astropy | 3 | 2 | 1 |
   | pydata/xarray | 3 | 3 | 0 |
   | mwaskom/seaborn | 2 | 2 | 0 |
   | sphinx-doc/sphinx | 2 | 2 | 0 |
   | pylint-dev/pylint | 2 | 1 | 1 |
   | pallets/flask | 1 | 1 | 0 |
   | **Total** | **160** | **148 (92.5%)** | **12 (7.5%)** |

   **0 multi-method, 0 unparseable, across every repo.** The
   `(class_name_or_None, method_name)` tuple design (not just
   `method_name` alone) is what makes this work for both class-bound
   (Django) and bare top-level (pytest) test functions without special-
   casing either style. This closes the open question with confidence at
   full-dataset scale, not just the 66-instance subset actually being used
   for the current experiment run.

   Verification scripts (scratch, not committed to the repo):
   `scratchpad/verify_held_back_test.py` (v1, line-diff, superseded),
   `scratchpad/verify_held_back_test_v2.py` (v2, AST-based, django-only),
   `scratchpad/verify_held_back_test_v3.py` (v3, AST-based, full 160-instance/
   11-repo dataset) in the session scratchpad directory.

3. **The `first` setting is a natural non-issue.**
   Since `instruct`'s own `preamble` has no real test bodies (just
   imports/helpers), there's nothing substantive for the KG arm to
   retrieve either for symmetry on `first` -- both arms legitimately start
   from "near-empty test file, write the first test." No special handling
   needed here beyond what already exists.

4. **Does the target test class always cleanly resolve to a real KG node?**
   Confirmed true for the one instance checked in depth
   (`OperationWriterTests` -> real `class` node with real `method`
   children), and true in aggregate for 51/66 django instances checked in
   the connectivity study below (15/66 failed node lookup or held-back
   detection for unrelated reasons -- see that section). Node resolution
   itself is not the open risk; **which nodes to include once resolved is**
   -- see below.

5. **Do the retrieved test-side nodes actually connect, in the graph, to
   the changed code -- or just share a file?** **Checked directly against
   real data. Answer: mostly no.**

   The plan up to this point (open question #2, "which test nodes to
   include") assumed *file-scope* retrieval: all nodes in
   `instance['test_file']`, full bodies, minus the one held-back method.
   That's what was verified end-to-end for `django__django-17087-15`
   (AST diff -> `find_file_by_path` -> `get_file_contents` ->
   `get_class_methods` -> real matching `source_code`, ~95% char parity
   with TGE's own `last_minus_one`). That check confirmed the *retrieval
   mechanism* works, but not that its *output* is meaningfully connected to
   the seed -- a separate question.

   Checked directly: for that same instance, does each of the 52
   `WriterTests` methods have a real graph edge (any relation) to the seed
   node `MigrationWriter.register_serializer`? **Only 2/52.** Both are
   `calls` + `tests` edges (`confidence: qualified`), correctly resolved by
   the KG's own static analysis:
   - `test_register_serializer` -> `register_serializer`
   - `test_register_non_serializer` (the held-back test) -> `register_serializer`

   The other 50 -- including 49 other test methods plus helpers like
   `safe_exec`/`assertSerializedEqual` -- are in the file-scope retrieval
   set purely by file/class co-location, with zero verified graph
   relationship to the actual changed code.

   **This was re-checked across 51 analyzable django instances (of 66;
   15 skipped -- 6 for target-method-node-not-found, 4 for
   zero-held-back-methods, both orthogonal to this question and re-uses
   the same seed-identification path `build_kg_prompts.py` already relies
   on, `TestContextExtractor.extract()`, not a re-derived heuristic).
   Result, per-instance "% of sibling test methods with a direct edge to
   the seed(s)":**

   | Stat | Value |
   |---|---|
   | Mean | 6.7% |
   | Median | 0.0% |
   | Instances with 0% connected siblings | 43/51 (84%) |
   | Instances with 100% connected siblings | 3/51 (all tiny classes, 2-5 methods total) |

   **Conclusion: file-scope retrieval is not KG-selected content.** The
   graph edges are real and correctly resolved (the KG isn't broken), but
   the retrieval logic as validated so far doesn't use them to select
   anything -- it would produce the same content as "dump the test file,"
   just reformatted into KG node sections. This doesn't make Design C
   unfair relative to `instruct` (which is *equally* indiscriminate --
   whole file, unfiltered, same noise). But it does narrow what a result
   from this comparison could actually support: not "the KG identified
   relevant context and that helped," only "restructuring the same flat
   content as typed nodes helped/hurt/did nothing" -- a real but much
   weaker version of RQ3's premise. Worth stating this limitation
   explicitly if/when results are written up, rather than implying
   graph-driven selection happened when it didn't.

   Verification script (scratch, not committed):
   `scratchpad/verify_test_seed_connectivity.py`.

## Concrete next step (revised)

Two options, given the connectivity finding above:

**Option 1 -- keep file-scope retrieval, but state the limitation.**
Simplest: implement retrieval exactly as originally planned (read
`instance['test_file']`, pull all real KG nodes, full bodies, minus the
held-back method), matching TGE's per-setting truncation boundary. Cheap,
mirrors `instruct`'s own indiscriminate scope, and is still a valid (if
narrower) test of "does structured representation help." Must be
documented as testing *representation*, not *retrieval quality* --
RQ3's "patch-aware ... retrieval" framing would not be earned by this
version.

**Option 2 -- edge-driven retrieval, closer to what RQ3 actually claims.**
Instead of "all nodes in the test file," walk outward from the seed
node(s) (`context.seeds`, already computed by `TestContextExtractor`) via
`tests`/`calls`/`depends_on` edges (capped at some depth, e.g. depth 2,
matching the existing `--depth` extractor param) to find test methods that
are graph-connected to the change, and retrieve only those -- falling back
to file-scope only when the walk finds nothing (per the data above, that
fallback would trigger on the majority of instances, e.g. django's 84%
zero-connected rate, so it needs to be a real, working fallback, not an
edge case). This is a materially different retrieval algorithm than what
was validated end-to-end for `django__django-17087-15` earlier in this
doc -- that validation covered node *resolution*, not edge-driven
*selection*, and would need to be re-verified once written.

**Not decided yet which option to build.** Option 1 is a smaller change
and still answers a legitimate (narrower) question. Option 2 is more
faithful to the "patch-aware retrieval" framing but is unproven at this
scale (only 2-5 connected methods per instance on average -- may be too
sparse a context to be a fair swap for `instruct`'s full-file scope, which
is itself an open question once built).

### Option 2 tried directly, and root-caused

Built and ran real edge-driven retrieval against all 51 analyzable django
instances: BFS from `context.seeds` (the same seed(s)
`TestContextExtractor.extract()` already resolves) outward via
`tests`/`calls`/`depends_on` edges, either direction, capped at depth 2 --
collecting whichever sibling test methods (same class as the held-back
test) are reachable, then measuring recovered char content against TGE's
own `last_minus_one` fragment.

**v1 result (seed(s) only as BFS start points):**

| Metric | Value |
|---|---|
| Held-back test itself reachable via edges | 13/51 (25%) |
| Instances reaching <=1 test method (~nothing) | 35/51 (69%) |
| % of TGE's `last_minus_one` content recovered | mean 8.6%, median 0.0%, max 87.0% |

Worse than file-scope on the metric that matters most: most instances
recover essentially nothing, and the walk fails to even find the one test
method already confirmed relevant (the held-back one) 75% of the time.

**Root cause, checked directly (not assumed):** for the unreachable
instances, is the seed edge-poor overall, or just not connected to *this*
test specifically?

| Category | Count | Meaning |
|---|---|---|
| `seed_isolated` | 8/48 (17%) | Seed has zero incoming `calls`/`tests` edges from anywhere -- a real KG-coverage gap (dynamic dispatch, decorators, unresolved imports -- consistent with issues #112/#113/#114). |
| `seed_connected_elsewhere` | 40/48 (83%) | Seed HAS real incoming `calls`/`tests` edges (one instance had 102), just not from the held-back test. |

**This `seed_connected_elsewhere` bucket turned out to be conflating two
very different phenomena -- split apart by checking, for each instance,
whether the seed function's name even appears literally in the held-back
test's source:**

| Sub-category | Count | Meaning |
|---|---|---|
| `IRRELEVANT_HELD_BACK` | 38/51 (75%) | Held-back test's source doesn't mention the seed at all -- TGE's `last`-setting truncation boundary just happens to land on an unrelated test in a large, multi-class test file. **Not a KG gap.** There is genuinely nothing to connect; no amount of KG completeness fixes this. |
| `NAME_PRESENT_BUT_UNLINKED` | 10/51 (20%) | Seed name literally appears in the held-back test's source, but no `calls`/`tests` edge exists. **Real, fixable static-analysis gap.** |
| `reachable` | 3/51 (6%) | Direct edge already exists. |

Confirmed `django__django-11620-38` as a concrete `IRRELEVANT_HELD_BACK`
example: the patch only touches `technical_404_response()` (an
import-exception-type fix), but the AST-diff's held-back test in this
instance's `last` setting is `HelperFunctionTests
.test_cleanse_setting_recurses_in_dictionary` -- a wholly unrelated test
in the same file, testing `cleanse_setting()`. No connection to find.

Manually traced one real `NAME_PRESENT_BUT_UNLINKED` case
(`django__django-17087-15`) to a concrete, fixable mechanism: the patch's
`code_file` is `serializer.py`, and `TestContextExtractor` correctly
resolves the seed as `FunctionTypeSerializer.serialize()`. The held-back
test (`WriterTests.test_register_non_serializer`) calls
`MigrationWriter.register_serializer()`, which invokes `serialize()` only
indirectly at runtime via a serializer registry (dynamic dispatch) --
static analysis doesn't resolve this into an edge, even though `serialize`
literally appears in the test. **5 of the 10 `NAME_PRESENT_BUT_UNLINKED`
cases share `seed='serialize'`**, suggesting this specific
registry/dynamic-dispatch pattern recurs across multiple instances in
django's migration-serializer code, not a one-off. The other 5
(`migrate`, `delete` x2, `settings_to_cmd_args_env`, `_js`/`_css`/`merge`)
are unconfirmed individually but plausibly similar indirect-call patterns
-- worth a case-by-case check before filing as a single issue.

**Revised takeaway: the earlier "40/51 is a KG undercounting problem"
framing was too broad.** Checked directly, only ~20% (10/51) of the
originally-unreachable cases are a real, fixable KG gap; the other ~75%
(38/51) are a held-back-test-selection artifact (TGE's own truncation
picking an unrelated test), not a KG limitation at all -- fixing KG
static analysis, however completely, cannot recover these, since there
is no real relationship to capture. Verification script (scratch, not
committed): `scratchpad/root_cause_v2_split.py`.

**Fix tried: widen the BFS start set to the seed(s) PLUS their direct
callers** (one extra backward hop before the main walk), on the
hypothesis that the real connection is usually one hop away on the
caller side.

**v2 result (seed + direct callers as BFS start points):**

| Metric | v1 (seed only) | v2 (seed + callers) |
|---|---|---|
| Held-back test reachable | 13/51 (25%) | 17/51 (33%) |
| <=1 test method reached | 35/51 (69%) | 27/51 (53%) |
| % of TGE content recovered | mean 8.6%, median 0.0% | mean 19.7%, **median 0.0%** |

Real, non-trivial improvement (mean roughly doubled, "nothing recovered"
bucket shrank by a third) -- confirms the caller-side hypothesis was
partly right. **But median stayed at 0%: more than half of instances still
recover nothing, even with the extra hop.** This means the gap isn't
purely "seed too narrow" either -- for most instances, the missing edge
is invisible to static analysis at any reasonable depth (decorators,
fixtures, `assertRaises`-style indirection, dynamic dispatch), not just
one hop further out than currently walked. Going deeper (depth 3+)
would not fix this category, since the missing link isn't a longer
*static* chain, it's an edge type the analyzer never created.

**Verification scripts (scratch, not committed):**
`scratchpad/verify_edge_driven_retrieval.py` (v1),
`scratchpad/root_cause_unreachable.py` (root-cause breakdown),
`scratchpad/verify_edge_driven_retrieval_v2.py` (v2, widened seeds).

### Correcting the measurement itself: wrong target, not just wrong depth

The `IRRELEVANT_HELD_BACK` finding above (75% of "misses" were checking
connectivity against a test unrelated to the patch) means the v1/v2
reachability numbers (13/51, 17/51) were measuring the wrong thing: "is
the seed connected to whichever test TGE's `last` setting happens to hold
back" rather than "is the seed connected to a test that's actually
relevant." Re-ran the check against a better target: for each instance,
find every test method in the FULL real test file (via AST, independent
of the KG's own edges) whose body literally references the seed
function's name, then check whether ANY of those are KG-connected to the
seed.

**Result, all 66 django instances:**

| Category | Count |
|---|---|
| No test in the file references the seed name at all | 22/66 (33%) |
| Has a relevant test, connected via a real KG edge | 26/66 (39%) |
| Has a relevant test, but none connected | 18/66 (27%) |
| **Of instances WITH a genuinely relevant test: KG-connected** | **26/44 = 59.1%** |

**59.1% is far higher than the earlier 13/51 (25%) / 17/51 (33%)
held-back-test-targeted numbers** -- confirming most of the earlier
"disconnection" was an artifact of checking the wrong test, not a real KG
gap. This is a meaningfully different picture for the retrieval-design
decision.

**Caveat: this 59.1% is noisy, likely an underestimate of true relevant
connectivity.** The "references the seed name" proxy over-recovers for
generic names -- e.g. `serialize` matched 18-23 methods across
`WriterTests`/`OperationWriterTests` per instance (most of that test
class's suite happens to mention `serialize` in some form, not all of
which are really testing this specific change), and `delete` matched 42-52
methods in `django/tests/delete.py`. Instances dominated by a generic seed
name skew toward "NONE connected" simply because the true target is
diluted among many false-positive "relevant" matches. A stricter proxy
(e.g. "seed name is the direct callee of an assertion," not just anywhere
in the method body) would likely raise the true rate further by removing
this noise. Verification script (scratch, not committed):
`scratchpad/verify_connectivity_correct_test.py`.

**Conclusion: pure edge-driven retrieval (Option 2) is more viable than
the earlier (mismeasured) numbers suggested, but still not reliable
enough to be `kg_only`'s SOLE test-side content on its own** -- roughly
somewhere in the 41-73% miss range even under the corrected, more
generous measurement (100% - 59.1% at the low end, treating all "no
relevant test found" cases as true negatives at the high end), which
still leaves a large minority of instances with an empty or thin test
section if used standalone. Two paths remain, revised in light of the
corrected number:

- **Option 1 (file-scope), with the limitation now precisely stated:**
  not "selected" content, but reliably produces enough test-side context
  to be usable across the dataset.
- **Option 3 (hybrid, new):** edge-driven retrieval as the primary
  signal when it fires (the 33% reachable / mean 19.7% cases are
  genuinely meaningful, graph-grounded content), falling back to
  file-scope when the walk comes up empty (the majority case) --
  crediting real connections where they exist without leaving most
  instances empty-handed. Still needs its own fairness framing: the
  *arm* would then contain a mix of "truly retrieved" and "fallback
  dumped" content per instance, which should be tracked/reported
  per-instance (e.g. a flag on whether edge-driven retrieval fired) so
  results can be sliced by retrieval quality later, not presented as
  uniformly "KG-retrieved."

Either option still needs, from the original plan:
- Read `instance['test_file']` (already passed in, currently unused).
- Look up real KG nodes (classes/methods, already captured with full
  `source_code`).
- Match TGE's own per-setting truncation boundary (open question #2
  above).
- Present as KG-structured sections (matching the existing
  `Callers:`/`Callees:`/`Sibling methods:` pattern in
  `scripts/build_kg_prompts.py`'s `_snippet_section` helper), not a raw
  file dump -- to preserve the "flat vs structured" distinction Design C
  depends on.

### Open question: should Option 3's fallback exist at all, for a
faithful RQ3 comparison?

Not resolved -- flagging explicitly rather than deciding by default,
since the reasoning cuts against the framing used everywhere else in
this doc.

Everywhere above, "fair" has meant *informationally comparable content,
differing only in representation* (Design C's core principle) -- and
under that framing, `kg_only` going test-content-empty on ~41%+ of
instances while `instruct` still gets the whole file looks like an
unjustified asymmetry, which is why Option 3's fallback was proposed: to
avoid starving `kg_only` on instances where edge-driven retrieval simply
comes up empty.

**But that framing may not actually apply to the test side the way it
applied to the code side.** On the code side, the KG's coverage of
callers/callees/siblings was never in question -- any gap there would
have been an arbitrary, fixable shortfall unrelated to what's being
tested, so equalizing it was just removing noise. On the test side, the
thing RQ3 names as the independent variable IS the retrieval mechanism
itself: "patch-aware KG subgraph retrieval" vs. "retrieval-free
baseline." If the KG's retrieval genuinely cannot find a given test's
connection to the seed (dynamic dispatch, operator overloading, etc. --
see pycodekg#117/#118/#119/#120/#122), failing to retrieve it isn't
noise obscuring the variable being tested -- **it may BE the variable,
expressing itself.** A real retrieval mechanism's coverage/recall is a
constitutive part of what "KG-based retrieval" means as an approach, not
an accident to paper over.

Under this reading, Option 3's fallback doesn't make the comparison
fairer -- it silently swaps in a *different, easier task* (reformat the
whole file) wherever the real task (retrieve via the graph) fails, which
would flatter the approach on exactly the instances where it's weakest,
rather than testing it honestly. A more faithful design might report
retrieval-failure instances as failures/non-results for RQ3's core
claim (or exclude them from that specific comparison, reporting the
exclusion rate as its own finding -- "KG-based retrieval had no
test-side signal for N% of instances"), rather than backfilling with
`instruct`'s own strategy under the same arm's name. Under that view, a
fallback might still be worth building for separate, system-viability
reasons (would a real deployed tool need one? almost certainly) -- but
as an explicitly distinct variant (e.g. `kg_hybrid`), not folded into
whatever result is presented as evidence for or against RQ3's actual
claim.

**Not decided.** Worth resolving before the final analysis is written up
(even if #121's implementation includes the fallback for now, given it's
already scoped that way) -- the choice changes what a `kg_only`-beats-
`instruct` (or doesn't) result would actually mean.

## Current run status (as of this doc)

Both prior experiment runs (the 51/66-generated `kg_only` Groq run, and
the partial local-MLX `instruct` run) were **stopped** once this design gap
was identified -- their results are invalid under Design C and need a
fresh run once the test-side retrieval fix above is implemented.

`kg_prompts_66.json` (built by `scripts/build_kg_prompts.py`) also needs
regenerating once the fix lands -- the current artifact has no test-side
content at all.

## Sanity check against the original FYP paper

Checked everything above against `docs/FYP__Patch_aware_Knowledge_Graph_
based_Test_Generation.pdf` (the project's original design document) to
confirm this session's work is still aligned with the stated methodology,
not drifting from it silently.

| Paper claim | Checked against real code | Verdict |
|---|---|---|
| Subgraph validation: pruning isolated nodes, no broken edges, seed connectivity, no duplicate edges | All 4 exist in `TestContextValidator` (`_check_no_orphaned_nodes`, `_check_no_broken_edges`, `_check_seed_connectivity`, `_check_no_duplicate_edges`), plus 5 more not mentioned in the paper (`_check_closed_subgraph`, `_check_no_ambiguous_seed_names`, `_check_test_coverage`, `_check_seed_types`, `_check_context_coverage`) | Matches, exceeds |
| BFS depth-2, edge filtering excludes low-signal edges (imports/`depends_on`) | Matches `TestContextExtractor.extract()`'s default `edge_filter` exactly | Matches |
| Seed = modified functions; Context = callers/callees/parent-child classes/existing tests | Matches `TestContext`'s actual fields (`seeds`, `context_nodes`, `test_nodes`) | Matches |
| RQ3: "patch-aware KG subgraph retrieval... compared to retrieval-free baselines" | Matches Design C and the patch-awareness section above exactly | Matches |
| Seed identification via "regex pattern matching" on diff hunks (Section II-B) | Actual implementation (`PatchParser.extract_changed_functions_with_scope`) uses real AST line-range matching against the pre-patch source, explicitly built to replace an earlier regex/hunk-scanning approach that caused 4 real bugs (kg_construction#14, #43, #63, #71) | **Paper text is stale** -- describes the superseded approach; code is a documented upgrade, not a deviation to fix |
| "If a test file exists in the repository, it is also added as a seed to enable test discovery" (Section II-B) | Deliberately removed from `TestContextExtractor.extract()` (see the code comment at context.py:269-285): caused non-deterministic seed ordering that sometimes placed the test file at `seeds[0]`, starving the LLM arm of the real target function's source on ~1/3 of runs (kg_construction#54) | **Deliberate, justified deviation** -- undocumented in the paper; worth reflecting the fix back into the paper text at some point, not a bug to fix in code |
| RQ2: "How effectively can code patches **and issue reports** identify affected entities" | `testgenevallite`'s schema has no issue-report field at all (confirmed: `problem_statement` and all its variants are absent from the row keys) | **Reinforces the existing RQ2-out-of-scope-for-TGE conclusion, for a second, more basic reason** -- not just that TGE pre-scopes `code_file`/`test_file` (already documented above), but that half of RQ2's named input signal doesn't exist in this dataset at all |
| Test-side retrieval, Design C, Option 1/2/3, `instruct` vs `kg_only` as named arms | Not addressed at this level of detail anywhere in the paper -- Section II-C's Context Section only lists "existing test functions" as one bullet, with no discussion of *how* they're selected, TGE's `preds_context` truncation settings, or a fairness comparison design | Not a contradiction -- the paper is upstream design intent; this session did downstream, TGE-specific implementation reasoning the paper doesn't cover at this depth |

**Net assessment: the core architecture (KG construction, BFS-based
subgraph extraction, edge filtering, validation) is faithfully
implemented and in some respects exceeds the paper's description.** The
two real gaps are documentation staleness (seed-identification method)
and an undocumented-but-justified implementation deviation (test-file-
as-seed removal) -- neither invalidates anything built this session, but
both are worth folding back into the paper's methodology section at some
point for accuracy.
