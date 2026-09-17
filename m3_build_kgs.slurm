#!/bin/bash
#SBATCH --job-name=build-kgs
#SBATCH --partition=comp
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=build-kgs-%j.out

# Runs scripts/build_kgs_for_instances.py directly on M3, no GitHub
# Actions involved. cpus-per-task feeds RepoASTParser's
# ProcessPoolExecutor directly (more cores = more files parsed in
# parallel), tune it up if a smaller request queues but runs too slowly,
# or down further if this still queues too long -- it's not a hard
# requirement, just parallelism, a single core still works, just slower.
#
# KG construction is depth-independent -- BFS depth only affects the
# separate prompt-building step (see m3_build_kg_prompts.slurm), so this
# only needs to run once per (INSTANCES, DATASET) pair, not once per
# depth you want to compare.
#
# Configure via env vars, same pattern as m3_run_inference.slurm:
#
#   INSTANCES="django__django-10390-15718,sympy__sympy-11400-17079" \
#   DATASET="kjain14/testgeneval" \
#     sbatch m3_build_kgs.slurm
#
# INSTANCES empty (default) builds every row in DATASET.
#
# Submit this from a checkout under scratch space (e.g.
# ~/al49_scratch/.../repo-kg-construction), not $HOME -- RepoManager's
# bare git clones land in ./repo_cache, relative to wherever this runs
# from, and django/sympy's real clone sizes would otherwise eat $HOME's
# small quota the same way HF_HOME/HF_DATASETS_CACHE would.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

INSTANCES="${INSTANCES:-}"
DATASET="${DATASET:-kjain14/testgenevallite}"

# $HOME has its own small, easy-to-exhaust quota on M3 (confirmed real
# this session on the testgeneval side, same load_dataset default cache
# location applies here). Point both at scratch space.
export HF_HOME="${HF_HOME:-$HOME/al49_scratch/$USER/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/al49_scratch/$USER/hf-datasets-cache}"

module load miniforge3

# conda create -n <name> (no --prefix) puts the env under ~/.conda/envs,
# inside $HOME, same quota problem as HF_HOME/HF_DATASETS_CACHE above,
# confirmed real: a real run hit "OSError: [Errno 122] Disk quota
# exceeded" installing dependencies into ~/.conda/envs/pycodekg on a
# $HOME already at 20G/20G (100%) used. --prefix puts the env on scratch
# space instead, same as HF_HOME/HF_DATASETS_CACHE.
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/al49_scratch/$USER/conda-envs/pycodekg}"

# Create the env once, reused on every subsequent submission.
# requires-python is >=3.10 (see pyproject.toml). Dependencies are
# installed every run regardless, pip install is a fast no-op when
# already satisfied, and skipping it on an existing env once left a
# real dependency (datasets) missing after an interrupted/cancelled
# earlier attempt had created the env but not finished installing.
if [ ! -d "$CONDA_ENV_DIR" ]; then
  echo "---- Creating pycodekg conda env at $CONDA_ENV_DIR (first run) ----"
  conda create -y --prefix "$CONDA_ENV_DIR" python=3.11
fi
source activate "$CONDA_ENV_DIR"
pip install -e ".[datasets]"

echo "---- Building KGs ----"
KG_BUILD_START=$(date +%s)
PYTHONPATH=src python3 scripts/build_kgs_for_instances.py \
  --instances "$INSTANCES" \
  --dataset "$DATASET"
KG_BUILD_END=$(date +%s)
echo "KG build time: $((KG_BUILD_END - KG_BUILD_START))s"

echo "Done. kg_output/ is in $SLURM_SUBMIT_DIR"

# Non-fatal: aggregate_build_times.py summarizes real per-repo build
# timing already saved in kg_output/'s own KG files (build_time_parsing_s/
# build_time_resolution_s, RQ4 instrumentation), nothing this job's own
# success depends on. Piped through `|| true` deliberately -- this
# script's own sys.exit(...) on "no timing data found" would otherwise
# propagate under this file's `set -euo pipefail` above and mark the
# whole job FAILED in sacct, even though the real, valuable work (KG
# building) already succeeded cleanly.
echo "---- Aggregating build times ----"
PYTHONPATH=src python3 scripts/aggregate_build_times.py \
  --kg-dir kg_output --out build_times.csv || echo "  (aggregation skipped/failed, not fatal -- run it manually if needed)"

echo "Next: submit m3_build_kg_prompts.slurm (once per depth you want)."
