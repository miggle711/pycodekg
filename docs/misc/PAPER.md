# Patch-Aware Knowledge Graph Retrieval for LLM-Based Repository-Level Test Generation

Joesiah Liu, Miguel Varilla, Sheryl Lee, and Thong Wan Mun

Joesiah Liu and Thong Wan Mun are with the Faculty of Engineering, Monash University, Clayton, VIC 3800, Australia. (e-mail: jliu0290, wtho0016@monash.edu)

Miguel Varilla and Sheryl Lee are with the Faculty of Engineering, Monash University Malaysia, Subang Jaya, Selangor, Malaysia. (e-mail: mvar0010, wlee0060@monash.edu)

## Abstract

While repository-level Large Language Models (LLMs) show promise for automated test generation, more than half of their generated unit tests are invalid. This stems from inadequate contextual grounding, which leads to hallucinations, incorrect API usage, and unresolved dependencies. Existing retrieval-augmented approaches fail to isolate precise, change-centric code relationships necessary to ensure behavioural correctness during oracle generation. To address this, we propose a patch-aware KG retrieval framework for LLM-based test generation. The framework constructs a repository-level KG mapping structural dependencies and API relationships, identifies patch-affected entities, and extracts change-anchored subgraphs via bounded breadth-first search. To ensure scalability, an incremental update mechanism is employed rather than full graph reconstruction per commit. We evaluate our approach against no-retrieval and text-RAG baselines using execution-based metrics, including test correctness, coverage, and mutation scores. This architecture mitigates model hallucinations and improves oracle correctness for targeted test generation.

**Index Terms** — automated test generation, code knowledge graphs, hallucination mitigation, large language models, neurosymbolic integration, patch-aware retrieval, repository-level software engineering, test oracle problem

## I. Introduction

The emergence of LLMs ecosystem tools such as GitHub Copilot, Cursor, and Claude Code, has shifted how repository-level software engineering tasks are approached. These systems increasingly support automated test generation, debugging, and code review across entire code-bases [1]. Among these tasks, automated test generation has attracted particular attention, as manually writing and maintaining comprehensive test suites remains costly and time-consuming [2]. Nevertheless, its practical value depends on output quality: generated tests must be syntactically valid, compile-able, and behaviourally meaningful representations of intended software functionality. Consequently, improving the reliability of repository-level LLM-based test generation has become an important research problem.

### A. Background and Motivation

Test generation is more complex than code generation. Where code generation produces implementation logic, test generation requires implicit understanding of intended behaviour to construct accurate assertions and test oracles [3].

Although LLMs excel at self-contained tasks, their performance degrades at the repository level due to fragmented project-specific context, intricate cross-file dependencies, and evolving API interfaces [4], [5]. Consequently, LLM hallucinations such as non-existent API calls, unresolved dependencies, and inconsistent cross-file logic can cause generated tests to fail compilation, induce false positives, or diverge from intended behaviour [6].

At the core of these limitations lies the test oracle problem: current methodologies predominantly ensure syntactic validity while failing to guarantee behavioural correctness, historically necessitating expensive human review to verify developer intent [7].

### B. Related Works and Gaps

Retrieval-Augmented Generation (RAG) has been used to reduce hallucinations by supplying LLMs with relevant code snippets as additional context [8], but existing retrieval mechanisms are often not anchored to localised code changes, making retrieved context broad and noisy. While RAG can improve syntactic and API-level grounding, it mainly resurfaces existing code rather than inferring developer intent, providing limited support for patch-specific behavioural expectations or resolving the test oracle problem.

To capture structural context more effectively, KGs have been used to model relationships such as function calls, imports, class hierarchies, and API usage [9]. However, many KG approaches rely on static, full-repository context rather than localised code deltas, which introduces irrelevant context and computational overhead in dynamic test generation. Moreover, structural topology alone does not resolve the oracle problem, as graph-based representations do not directly infer developer intent or expected behaviour.

To bridge this gap between rigid code structure and flexible behavioural inference, recent paradigms turn toward neurosymbolic integration, broadly categorised by integration direction: neural → symbolic, where LLM outputs construct symbolic representations, and symbolic → neural, where structured context is injected into LLM inputs to ground generation [10]. This work adopts the symbolic → neural direction, using verifiable symbolic constraints to guide and bound stochastic LLM behavior.

Rather than relying on the LLM to construct or update the underlying KG, this work uses validated symbolic context as input to the model, reducing the risk of hallucinated dependencies or incorrect structural assumptions. Because symbolic context can be checked before inference, the symbolic → neural direction grounds test generation in deterministic repository facts rather than model-inferred assumptions. As summarised in Table I, no existing approach combines KG, RAG, and LLM specifically for test generation.

**Table I — Coverage of Related Approaches Across Key Dimensions**

| Method | KG | LLM | RAG | Test/Code Gen |
|---|---|---|---|---|
| GraphEval [11] | ✓ | ✓ | — | — |
| SemanticForge [9] | ✓ | ✓ | — | Code |
| GraphCoder [12] | ✓ | ✓ | ✓ | Code |
| CGM [13] | ✓ | ✓ | ✓ | Code |
| TESTGENEVAL [3] | — | ✓ | — | Test |
| SWT-Bench [14] | — | ✓ | — | Test |
| **Ours** | ✓ | ✓ | ✓ | **Test** |

While prior work combines some of these components, existing retrieval and KG-based methods remain weakly connected to active code modifications and are often optimised for general code generation rather than repository-level test synthesis. This motivates a patch-aware KG retrieval framework that extracts validated, change-centric subgraphs for LLM-based test generation.

### C. Problem Statement and Research Questions

Formally translating this literature gap into a concrete research problem, the core deficiency of current language-model-based test generation is its limited awareness of local code modifications. Because contemporary methods lack change-aware, localised retrieval and sufficient semantic grounding to infer expected software behaviour, they may produce incorrect, incomplete, or misleading test suites characterised by faulty or absent oracles. This problem is driven by three architectural limitations:

- **Untargeted Retrieval**: Broad repository context can introduce irrelevant code structures that degrade model performance.
- **Context-change Mismatch**: Static, full-repository context may fail to reflect the precise boundaries and semantic implications of what has changed.
- **Absence of Semantic Grounding**: Existing retrieval mechanisms do not explicitly anchor context to specific code modifications, limiting their ability to support correct oracle generation.

Repository-level test generation therefore requires localised, change-aware context that is explicitly grounded in the patch signal. To evaluate the feasibility and effectiveness of this direction, this study addresses four research questions:

- **RQ1 (KG Construction Quality)**: How accurately does the repository-level KG capture code structure, dependencies, and API relationships?
- **RQ2 (Patch-Based Subgraph Retrieval)**: How effectively can code patches and issue reports identify affected entities and retrieve relevant KG subgraphs?
- **RQ3 (Test Generation Quality)**: To what extent does patch-aware KG subgraph retrieval improve the quality of LLM-generated test cases compared to retrieval-free baselines?
- **RQ4 (Efficiency)**: What is the computational cost of the proposed approach relative to the quality gains achieved?

### D. Aim and Objectives

Driven by these research questions, this research aims to develop a patch-aware KG retrieval framework that improves the correctness, coverage, and oracle accuracy of repository-level LLM-generated tests. This aim is addressed through four technical objectives:

1. **Repository-Level Knowledge Graph Construction** (Sem. 1, Weeks 8–10): Construct a repository-level KG capturing structural, dependency, and API relationships via Abstract Syntax Tree (AST), featuring native support for incremental updates.
2. **Change-Aware Subgraph Extraction** (Sem. 2, Weeks 1–4): Design and validate an algorithmic extraction mechanism anchored to patch-affected entities utilising bounded Breadth-First Search (BFS) expansion.
3. **Neuro-Symbolic Prompt Integration** (Sem. 2, Weeks 1–4): Integrate structurally validated subgraphs as structured context into language model prompts to tightly constrain test and oracle generation.
4. **Empirical Framework Evaluation** (Sem. 2, Weeks 5–8): Evaluate the framework against language-model-only and untargeted retrieval baselines across four core metrics: test correctness, code coverage, hallucination rate, and mutation score.

## II. Methodology

As shown in Fig. 1, our methodology constructs a repository-wide KG, extracts a focused subgraph around patch-modified functions, and formats this localised structure as context for LLM test generation. Generated tests are subsequently executed to evaluate coverage and correctness.

**Fig. 1 — Pipeline overview**: Repository + Patch → (1. Build KG, 2. Find Seeds, 3. Extract Subgraph) → Subgraph → (4. Validate, 5. LLM Test Gen) → Test Code → (6. Execute, 7. Evaluate) → Report

### A. Knowledge Graph Construction

To build the KG, repositories are cloned as bare mirrors. Construction proceeds in two distinct phases: Parallel Extraction and Sequential Resolution.

**a) Parallel Extraction**: Each `.py` file is parsed into an AST to extract top-level nodes. Abstract nodes are mapped to corresponding graph counterparts: imports become Import nodes, while classes, methods, and functions form structural nodes. Abstract relations, including class inheritances, instantiations, and function calls, are recorded to emit directed edges.

**b) Sequential Resolution**: Extracted nodes are aggregated into a global index mapping definitions to node IDs. Unresolved call targets are matched against this index and tagged by confidence: exact (single match), ambiguous (multiple matches), or dropped (external libraries). Finally, call contexts such as caller counts and direct callers are attached to each node.

The resulting deterministic graph uses node IDs derived from Message-Digest algorithm 5 (MD5) hashes of component names, providing a reproducible, commit-specific representation persisted as JSON.

### B. Patch Parsing and Seed Identification

Given a unified patch from a GitHub issue and a target code file, the patch parser identifies which functions/methods were modified. The parser:

1. Parses the unified diff into per-file change hunks.
2. Locates hunks for the target code file.
3. Extracts function/class definitions appearing in added or modified lines (regex matching `def {function_name}` and `class ClassName`).
4. Returns a set of changed function/class names.

These changes are then queried against the KG to locate corresponding seed nodes and represent the modified functions that need test coverage. If a test file exists in the repository, it is also added as a seed to enable test discovery.

### C. Subgraph Extraction

From seed nodes, a Breadth-First Search (BFS) extracts surrounding context up to a configurable depth which has shown to be effective in capturing immediate dependencies within a repository [15]. The algorithm traverses both outgoing edges (function calls, inheritance, instantiation) and incoming edges (callers, subclasses), collecting related nodes and edges.

Edge filtering excludes low-signal edges (e.g., imports, `module_depends_on`) which scatter context across unrelated code. This is consistent with established practice in graph-based subgraph extraction [16]. The extracted subgraph includes:

- **Seed nodes**: the modified functions
- **Context Nodes**: related functions (callers and callees), parent/child classes, test functions
- **Edges**: relationship within the subgraph

### D. Subgraph Validation

The extracted subgraph is validated to ensure it does not contain the following structural violations:

**a) Pruning Isolated Nodes**: Isolated nodes in the KG provide no relational context, breaking link-based reasoning. Including these orphaned nodes adds noise without signal, as the LLM cannot reason about their callers, dependencies, or callees.

**b) No Broken Edges**: Dangling edges create logical inconsistencies. If function B calls function A, but A is not in the subgraph, the LLM cannot reason about A's behaviour, leading to incomplete or incorrect test generation.

**c) Seeds Connected via Subgraph Edges**: The seed represents the code entity under test. If it has no edges connecting it to the rest of the subgraph, the LLM receives no context about how the code integrates with callers, dependencies, or existing tests which severely degrades test quality. Seeds must maintain at least one edge connection to the broader subgraph.

**d) No Duplicate Edges**: Duplicate edges add no information and only consume tokens in the LLM's limited context window, reducing the effective density of useful context. They inflate edge counts used in metrics and debugging. All duplicate edges must be removed from the subgraph.

### E. Neuro-Symbolic Integration

For efficient consumption of LLM, the validated subgraph is reformatted into a hierarchical JSON structure organised by semantic relationships. The hierarchical format organises information into three semantic sections:

- **Seed Section**: Contains the modified function(s) at the centre of analysis. This section establishes what changed and what the LLM must generate tests for.
- **Context Section**: Provides execution context surrounding the seed. Includes direct callers (functions that invoke the seed), direct callees (functions the seed invokes), parent/child classes (for inheritance relationships), existing test functions, and execution patterns. This section answers "how is the seed used?" and "what does it depend on?"
- **Instructions Section**: Explicit task directives for test generation. Specifies coverage targets (boundary conditions, happy paths, error cases, edge cases), naming conventions, assertion patterns expected by the repository, and any repository-specific testing idioms.

Hierarchical organisation enables LLMs to prioritise relevant code relationships and infer execution paths without traversing a flat list of nodes. Ego-graph retrieval centred on a seed node, capturing immediate caller/callee dependencies and interactions has been shown to be crucial for understanding functional context, with LLM generation conditioned on such subgraph context producing more nuanced, repository-aware outputs [15].

### F. Test Execution and Validation

To evaluate the capability and robustness of LLM-generated tests, we implement an execution-based framework modelled after the TESTGENEVAL [3] benchmark pipeline. This approach moves beyond static analysis by integrating sandbox execution, dynamic coverage tracking, and mutation testing across three distinct dimensions:

1. **Functional Correctness (Pass Rates)**: To balance test suite verbosity against strictness, correctness is evaluated using two metrics:
   - **Any Pass@1**: Assesses if at least one generated test in the suite executes successfully, preventing models with large test suites from being heavily penalised for a single faulty method.
   - **All Pass@1**: Evaluates whether every test case in the generated suite passes flawlessly, strictly penalising syntax errors or assertion hallucinations.
2. **Structural Code Coverage**: We measure the proportion of source code lines executed by the tests. To isolate valid testing behaviours from broken or failing assertions, this is done strictly via passing tests (Coverage@pass).
3. **Semantic Fault-Detection (Mutation Testing)**: To ensure the tests actively validate logic, we inject synthetic bugs (mutants) into the source code using an automated mutation engine. The test suite is re-executed against each mutant. The mutation score, a measurement of generated test code performance, is calculated as:

```
Mutation Score = (Mutations Discovered / Total Mutants Injected) × 100%
```

### G. Datasets and Baselines

The evaluation utilises the TESTGENEVAL [3] and SWT-Bench [14] datasets, with sample characteristics summarised in Table II. To ensure robust evaluation, samples are filtered to satisfy four prerequisites: an executable environment, identifiable changed entities, traceable dependency/API context, and the presence of baseline human-written tests.

**Table II — Evaluation Datasets**

| Dataset | Task | Evaluation use |
|---|---|---|
| TESTGENEVAL [3] | Unit test generation and completion | Pass@1, Coverage@pass, Mutation Score |
| SWT-Bench [14] | Patch-based bug reproduction | Fail-before-fix/pass-after-fix behaviour |

### H. Initial Works

The implementation is available on GitHub, with the following implemented and validated:

- **Knowledge Graph construction** from Python source files parsed into AST-based nodes and edges.
- **Patch parsing and seed identification** from unified diff patches to map changed code entities to KG seed nodes.
- **Patch-centered subgraph extraction/validation** using bounded BFS with checks for orphaned nodes, broken edges, seed connectivity, and duplicate edges.

## III. Conclusion

This paper introduces a patch-aware KG retrieval framework to mitigate LLM hallucinations and address the test oracle problem in repository-level test generation. Using a symbolic → neural paradigm, the architecture addresses the weak connection between existing retrieval methods and active code modifications, extracting validated, change-anchored subgraphs for repository-level test synthesis. This deterministic context ensures the LLM receives dependency and behavioural boundaries centred on code deltas.

## References

[1] J. Liu, K. Wang, Y. Chen, X. Peng, Z. Chen, L. Zhang, and Y. Lou, "Large language model-based agents for software engineering: A survey," *ACM Transactions on Software Engineering and Methodology*, 2024.

[2] M. Boukhlif, N. Kharmoum, and M. Hanine, "Llms for intelligent software testing: a comparative study," in *Proceedings of the 7th International Conference on Networking, Intelligent Systems and Security*, 2024, pp. 1–8.

[3] K. Jain, G. Synnaeve, and B. Rozière, "Testgeneval: A real world unit test generation and test completion benchmark," in *International Conference on Learning Representations*, vol. 2025, 2025, pp. 14947–14999.

[4] X. Deng, J. Da, E. Pan, Y. Y. He, C. Ide, K. Garg, N. Lauffer, A. Park, N. Pasari, C. Rane et al., "Swe-bench pro: Can ai agents solve long-horizon software engineering tasks?" *arXiv preprint arXiv:2509.16941*, 2025.

[5] S. Haroon, M. T. Khan, and M. A. Gulzar, "Evaluating llm-based test generation under software evolution," *arXiv preprint arXiv:2603.23443*, 2026.

[6] H. Taherkhani, A. DaghighFarsoodeh, M. Chowdhury, H. V. Pham, and H. Hemmati, "Consistency meets verification: Enhancing test generation quality in large language models without ground-truth solutions," *arXiv preprint arXiv:2602.10522*, 2026.

[7] E. T. Barr, M. Harman, P. McMinn, M. Shahbaz, and S. Yoo, "The oracle problem in software testing: A survey," *IEEE transactions on software engineering*, vol. 41, no. 5, pp. 507–525, 2014.

[8] Z. Zhang, C. Wang, Y. Wang, E. Shi, Y. Ma, W. Zhong, J. Chen, M. Mao, and Z. Zheng, "Llm hallucinations in practical code generation: Phenomena, mechanism, and mitigation," *Proceedings of the ACM on Software Engineering*, vol. 2, no. ISSTA, pp. 481–503, 2025.

[9] H. Wang, Y. L. Gao, Z. Feng, and X. Wang, "Semanticforge: Knowledge graph-enriched code generation for repository-level software development," in *Proceedings of the ACM International Conference on the Foundations of Software Engineering (FSE)*, 2025, pp. 812–825.

[10] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, "Unifying large language models and knowledge graphs: A roadmap," *IEEE Transactions on Knowledge and Data Engineering*, vol. 36, no. 7, pp. 3580–3599, 2024.

[11] H. Sansford, N. Richardson, H. P. Maretic, and J. N. Saada, "Grapheval: A knowledge-graph based llm hallucination evaluation framework," *arXiv preprint arXiv:2407.10793*, 2024.

[12] W. Liu, A. Yu, D. Zan, B. Shen, W. Zhang, H. Zhao, Z. Jin, and Q. Wang, "Graphcoder: Enhancing repository-level code completion via code context graph-based retrieval and language model," *arXiv preprint arXiv:2406.07003*, 2024.

[13] N. Poolsup, N. Suksomboon, and A. M. Kyaw, "Systematic review and meta-analysis of the effectiveness of continuous glucose monitoring (cgm) on glucose control in diabetes," *Diabetology & metabolic syndrome*, vol. 5, no. 1, p. 39, 2013.

[14] N. Mündler, M. N. Müller, J. He, and M. Vechev, "Swt-bench: Testing and validating real-world bug-fixes with code agents," *Advances in Neural Information Processing Systems*, vol. 37, pp. 81857–81887, 2024.

[15] S. Ouyang, W. Yu, K. Ma, Z. Xiao, Z. Zhang, M. Jia, J. Han, H. Zhang, and D. Yu, "Repograph: Enhancing ai software engineering with repository-level code graph," in *International Conference on Learning Representations*, vol. 2025, 2025, pp. 30098–30121.

[16] M. Gardner and T. Mitchell, "Efficient and expressive knowledge base completion using subgraph feature extraction," in *Proceedings of the 2015 conference on empirical methods in natural language processing*, 2015, pp. 1488–1498.
