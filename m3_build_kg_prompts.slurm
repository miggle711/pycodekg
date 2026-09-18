#!/bin/bash
#SBATCH --job-name=build-kg-prompts
#SBATCH --partition=comp
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=build-kg-prompts-%j.out

# Runs scripts/build_kg_prompts.py against KGs already built by
# m3_build_kgs.slurm (must be run first, same kg_output/ directory).
# Cheap relative to KG construction, so this is the script to submit
# once per BFS depth you want to compare (e.g. a depth-1 vs depth-2
# ablation) -- rerunning m3_build_kgs.slurm per depth would waste real
# compute rebuilding depth-independent KGs.
#
# Configure via env vars:
#
#   DATASET="kjain14/testgeneval" DEPTH="1" sbatch m3_build_kg_prompts.slurm
#   DATASET="kjain14/testgeneval" DEPTH="2" sbatch m3_build_kg_prompts.slurm
#
# Output is depth-named (kg_prompts_depth1.json, kg_prompts_depth2.json,
# ...) so two depths' outputs never collide, unlike a fixed
# kg_prompts.json filename would.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DATASET="${DATASET:-kjain14/testgenevallite}"
DEPTH="${DEPTH:-2}"
OUTPUT="${OUTPUT:-kg_prompts_depth${DEPTH}.json}"

if [ ! -d kg_output ] || [ -z "$(ls -A kg_output 2>/dev/null)" ]; then
  echo "kg_output/ is missing or empty. Run m3_build_kgs.slurm first."
  exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/al49_scratch/$USER/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/al49_scratch/$USER/hf-datasets-cache}"

module load miniforge3

CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/al49_scratch/$USER/conda-envs/pycodekg}"
if [ ! -d "$CONDA_ENV_DIR" ]; then
  echo "pycodekg conda env not found at $CONDA_ENV_DIR. Run m3_build_kgs.slurm first (it creates it)."
  exit 1
fi
source activate "$CONDA_ENV_DIR"

echo "---- Saving dataset to disk ----"
python3 -c "
from datasets import load_dataset
ds = load_dataset('$DATASET')
ds.save_to_disk('dataset_on_disk')
"

echo "---- Building kg_only prompts (depth=$DEPTH) ----"
PROMPTS_BUILD_START=$(date +%s)
PYTHONPATH=src python3 scripts/build_kg_prompts.py \
  --kg-dir kg_output \
  --dataset-path dataset_on_disk \
  --output "$OUTPUT" \
  --depth "$DEPTH"
PROMPTS_BUILD_END=$(date +%s)
echo "Prompt build time (depth=$DEPTH): $((PROMPTS_BUILD_END - PROMPTS_BUILD_START))s"

echo "Done. $OUTPUT is in $SLURM_SUBMIT_DIR"
