# Reproducing the experiments

The release keeps the accepted FlowVN model, stage-aligned loss, and revised
velocity metrics. The command entry point removes unrelated model branches;
paths and logging defaults are portable. This is a cleaned implementation,
not a byte-identical archive of a historical run.

## Environment and scope

Archived training records identify Python 3.10.20, PyTorch 2.12.0 with CUDA
13.0, and PyTorch Lightning 2.6.5. On Linux, run:

```bash
export UV_PYTHON=3.10.20
uv sync --locked --extra cuda
```

The dependency versions and hashes are in `uv.lock`. Some supporting pins came
from the surviving environment rather than a complete historical lockfile;
the lock is a reproducible release environment, not proof of an identical
historical environment. The default Python 3.12.13 / SciPy 1.18.0 combination
supports the Apple Silicon CPU checks. Linux retains SciPy 1.15.3.

Training used a 24 GB NVIDIA MIG allocation. Full-volume validation used a
48 GB allocation. Complete-case timing used an RTX 3090. Memory needs depend
on coil count and volume shape; first try a small authorized sample on your
hardware. This release preparation runs CPU checks only, with no retraining
or new CUDA performance claims.

## Matched training

The paper uses four S8 recipes and two seeds, `12345` and `23456`. All arms use
the same patient splits, S16-derived initialization, Adam optimizer, learning
rate `1e-4`, cosine schedule, batch size one, and ten epochs. The training patch
has depth five and 15 cardiac phases. Acceleration is sampled from
`10, 20, 30, 40, 50` by the loader.

The ground-truth complex L1 loss has weight one. Teacher losses are normalized
by mean target magnitude plus `1e-6`. The teacher scale over the first five
epochs is `1.0, 1.0, 0.8, 0.4, 0.2`; the final five epochs use ground truth only.
The teacher is frozen and excluded from the student optimizer and checkpoints.

Prepare [data and masks](data.md) and the [teacher checkpoint](checkpoints.md).
For example:

```bash
export FLOWVN_MASK_BACKEND=challenge
export FLOWVN_CHALLENGE_MASK_ARCHIVE="$PWD/external/CMRx4DFlowMaskGeneration.zip"

uv run --locked --extra cuda python main.py --config configs/s8_supervised.yaml \
  --seed 12345 --save_dir outputs/s8_supervised/seed12345

uv run --locked --extra cuda python main.py --config configs/s8_full.yaml \
  --seed 12345 --save_dir outputs/s8_full/seed12345
```

Run `s8_final.yaml` and `s8_trajectory.yaml` the same way, then repeat all four
with seed `23456` and distinct output directories. The original run had 13,738
steps per epoch; another dataset or split changes the update budget even if the
epoch count is unchanged. TF32, cuDNN benchmarking, and nondeterministic kernels
are enabled in the historical training configs. A seed does not guarantee
bitwise-identical training.

Resume interrupted training using the same recipe and run directory:

```bash
uv run --locked --extra cuda python main.py --config configs/s8_full.yaml \
  --seed 12345 --save_dir outputs/s8_full/seed12345 \
  --resume_from_checkpoint outputs/s8_full/seed12345/last.ckpt
```

Use the actual checkpoint location printed by the run if it differs. Select the
end-of-epoch-10 checkpoint for the primary comparison, not a validation-selected
best epoch. Fixed averaging of epochs 8–10 was a secondary analysis only.

## Common evaluation

The same evaluator and deterministic masks must be used for every arm. Keep the
mask environment variables above set, then evaluate a selected checkpoint:

```bash
uv run --locked --extra cuda python main.py --config configs/validate.yaml \
  --ckpt_path checkpoints/s8_full_seed12345.ckpt \
  --save_dir outputs/evaluation/full_kd/seed12345 \
  --val_metrics_output outputs/evaluation/full_kd/seed12345/validation_metrics.csv
```

Use output arm names `supervised`, `final_only`, `final_trajectory`, and `full_kd`
for the four recipes, for both seeds. Override `--num_stages 16` for an S16
reference. In validation mode the loader uses full depth and all cardiac phases.

Each run should produce 80 case/acceleration groups: 16 cases × five rates,
with all four velocity encodings in each group. Metrics are magnitude nRMSE,
SSIM, phase-domain relative error, angular error in degrees, normalized complex
L1, and VENC-scaled velocity-vector RMSE in cm/s. Missing VENC leaves the physical
metric unavailable; the analysis rejects incomplete metrics.

The original validation mask seed contains the case path. Read the
[relocation limitation](data.md#masks-and-relocation) before claiming exact
reproduction. Checkpoint and split access, the original seed inputs, and a
qualified GPU environment remain prerequisites for reproducing paper numbers.

## Case-level analysis

`scripts/analyze.py` accepts newly generated validation CSVs. It calls the
same numerical analysis functions used for the revision. Example for the full
two-seed matrix:

```bash
uv run --locked --extra cpu python scripts/analyze.py \
  --run 12345 supervised outputs/evaluation/supervised/seed12345/validation_metrics.csv \
  --run 12345 final_only outputs/evaluation/final_only/seed12345/validation_metrics.csv \
  --run 12345 final_trajectory outputs/evaluation/final_trajectory/seed12345/validation_metrics.csv \
  --run 12345 full_kd outputs/evaluation/full_kd/seed12345/validation_metrics.csv \
  --run 23456 supervised outputs/evaluation/supervised/seed23456/validation_metrics.csv \
  --run 23456 final_only outputs/evaluation/final_only/seed23456/validation_metrics.csv \
  --run 23456 final_trajectory outputs/evaluation/final_trajectory/seed23456/validation_metrics.csv \
  --run 23456 full_kd outputs/evaluation/full_kd/seed23456/validation_metrics.csv \
  --output-dir outputs/analysis
```

The analysis checks the complete arm/seed matrix, finite metrics, matching
case/acceleration keys, and case-level aggregation. It averages acceleration
levels within each case, computes matched effects within seed, averages those
effects across seeds, and uses 10,000 case-bootstrap samples. Patients, not
encodings or acceleration masks, are the independent units.

The output includes JSON, CSV, and LaTeX tables, input hashes, and paired
uncertainty. Existing analysis files are not overwritten. This portable CLI
does not certify the data, training provenance, or publication eligibility.
The retained `analyze_flowvn_revision_ablation.py` also contains the historical
audit-bound loader for owners of those original audit records.

## Timing

Keep two comparisons separate: S16 versus S8 at a fixed input shape, and eager
versus optimized execution of the same S8 checkpoint on a complete case.
Distillation does not alter the S8 inference graph.

For model-call timing, the benchmark reports CUDA-event and synchronized wall
times. Fix the device, checkpoint, precision, input shape, warm-up, and repeat
count. An illustrative synthetic CUDA command is:

```bash
uv run --locked --extra cuda python scripts/benchmark_flowvn.py \
  --device cuda:0 --num-stages 8 --ckpt-path checkpoints/s8_full_seed12345.ckpt \
  --velocity-encodings 1 --coils 8 --depth 5 --time-frames 15 --spatial-size 32 \
  --num-warmup 10 --num-iters 50 --output-json outputs/model_timing.json
```

Those illustrative dimensions are not a claim to match the paper's fixed-shape
manifest. The complete-case script additionally measures loading, preprocessing,
transfer, reconstruction, and output writing:

```bash
uv run --locked --extra cuda python scripts/benchmark_flowvn_e2e.py \
  --case-dir data/ValidationSet/site_a/scanner_a/case_a \
  --ckpt-path checkpoints/s8_full_seed12345.ckpt --num-stages 8 \
  --device cuda:0 --usrate 10 --out-base-dir outputs/timing/eager \
  --output-json outputs/timing/eager.json
```

For the optimized comparison, use another output directory and add:

```text
--no-flowvn-activation-lowmem
--compile-flowvn-activations --compile-flowvn-regularizer
--loader-mode dataloader --num-workers 2 --prefetch-factor 2
--test-skip-gt-precompute --no-deterministic-algorithms
--test-gpu-preprocess-adjoint --flowvn-transpose-conv-as-conv
```

Run each mode in a fresh process, use a fresh `TORCHINDUCTOR_CACHE_DIR` for each
cold optimized run, and compare saved complex outputs before interpreting speed
or memory differences. The original complete-case study used three runs per
mode on one case; that does not establish dataset-wide latency.
