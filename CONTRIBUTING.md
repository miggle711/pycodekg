# Contributing

## Setup

Python 3.10 or newer (CI runs 3.10, 3.11 and 3.12).

```
pip install -e ".[all]" pytest
pytest tests/ -v
```

Without the editable install, run `PYTHONPATH=src pytest tests/ -v`, as CI does. `pytest.ini` takes precedence over the `pythonpath` setting in `pyproject.toml`, so a bare `pytest` cannot import `kg_construction` unless the package is installed or `PYTHONPATH=src` is set.

Before opening a PR, run the suite under a few hash seeds. Ordering bugs only show up for some seeds (see #149):

```
for s in 0 1 42; do PYTHONHASHSEED=$s pytest tests/ -q; done
```

## Branches and commits

- Branch from a fresh `origin/main`. Do not branch from an unrelated feature branch, or its commits end up in your PR.
- Name the branch with a prefix that says what it is: `fix/`, `feat/`, `docs/`, `chore/`, `refactor/` or `test/`.
- Start the commit title with the same kind of prefix: `fix: ...`, `feat: ...`.
- Keep one concern per PR.

## Tests

- Write one test per real behaviour and parametrize variants of the same behaviour. Skip tests that only restate the code.
- A regression test must fail without the fix. Check this by stashing the source change and running the test.

## Pull requests

Link the issue in the first line. Use `Fixes #N` when the PR completes the issue, and `Part of #N` when it does not, so merging does not close an issue that still has work left.

The description should show the change on real input, not describe it:

1. **Problem:** a minimal repro, with the output you actually got.
2. **Fix:** the changed lines, as a diff.
3. **After:** the same repro with the new output.
4. **Tests:** what is covered, and the suite result.

#184 is a good example.

### PRs that change KG contents

If a PR changes which nodes, edges or metadata a KG contains, do two more things.

1. **Validate on real code.** Run `scripts/kg_diff.py` and paste its output into the description, including the `before:` and `repos:` lines under it:

   ```
   python scripts/kg_diff.py --before origin/main --after HEAD --repos requests pytest django
   ```

   It builds a KG from the same repositories with the code at each ref and reports how many edges were removed and added per relation, which nodes changed type, and a labelled sample. The `repos:` line names the commit of each repository that was measured. Two caches at different commits give different numbers, so if your table and a reviewer's disagree, compare those commits first. Look at the removed edges: each should be a false link going away or an edge being retargeted, and the description should say which.
2. **Bump `SCHEMA_VERSION`** in `src/kg_construction/kg/builder.py` and add a line to the comment block above it describing the change. A cached KG with an older version is rejected by `load()`. If a KG rebuild is already queued behind other PRs, say so in the description instead of bumping again.

A change that only affects the shape of the KG (new fields, node types or relations) needs the bump too.

### Stacked PRs

When a PR depends on another that is still open:

- Base it on the other PR's branch, so the diff shows only its own change.
- List the stack and the merge order in the description.
- Merge in order. After each squash merge, retarget the next PR to `main` and rebase it.
