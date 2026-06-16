# PLAN.md — RWKV TF-Domain Speech Separation: Production Migration Blueprint

## Document Purpose

This plan converts the current flat-script layout (`train_rwkv_TFtest.py`, `rwkv_separator_Final.py`, `infer_rwkv_TFDomain.py`, `make_csv.py`) into a configuration-driven, modular, reproducible ML repository aligned with modern MLOps practice.

**Scope:** RWKV-v7 time–frequency (STFT) speech separation on Libri2Mix-style CSV datasets, with SI-SDR + PIT training loss and multi-metric offline evaluation.

---

## 1. Current State Analysis

### 1.1 Repository Inventory

| Asset | Lines (approx.) | Role | Migration Priority |
|---|---|---|---|
| `train_rwkv_TFtest.py` | ~661 | Monolith: STFT, dataset, loss, train/val loop, TB, checkpoints | Decompose first |
| `rwkv_separator_Final.py` | ~447 | RWKV-v7 grouped bi-directional separator + factory | Move to `src/models/` |
| `infer_rwkv_TFDomain.py` | ~877 | Duplicate STFT/dataset/metrics + evaluation | Refactor after shared modules exist |
| `make_csv.py` | ~44 | Libri2Mix CSV builder | Move to `scripts/` |

**Missing today:** `pyproject.toml`, tests, CI, centralized config, shared modules, dependency lockfile, README/runbook.

### 1.2 Architectural Characteristics (Pipeline-Specific)

1. **Two-stage forward pass:** Fixed STFT → RWKV predicts separated magnitudes → iSTFT with **mixture phase** → time-domain PIT SI-SDR loss.
2. **Derived model dimension:** `codec_dim = n_fft // 2 + 1` (frequency bins `F`) is inferred at runtime from a batch STFT probe in `main()`. This must become an explicit config-derived value with validation, not a silent runtime probe.
3. **RWKV CUDA coupling:** `BiTimeMixFull` requires `RUN_CUDA_RWKV7g` from external package `RWKV.RWKV_v7.train_temp.src.model`. Training fails on CPU-only paths.
4. **Import-order sensitivity:** Five `RWKV_*` environment variables are set via `os.environ.setdefault` at module import time in both train and infer scripts.
5. **Hidden hyperparameters:** `num_workers` (4/2), `weight_decay=0.01`, `drop_last=True`, AdamW-only, `num_sources=2` hardcoded, subset seed `42`, optimizer scheduler state not restored on resume.
6. **Code duplication:** `STFTProcessor`, CSV loading, collate logic, and SI-SDR/PIT utilities are duplicated between train and infer with minor divergences (eval dataset returns `row` metadata; SI-SDR implementations differ slightly).

### 1.3 Existing Partial Abstractions (Reuse, Don't Rewrite)

- `SeparatorV7Config` dataclass and `build_rwkv7_separator()` factory in `rwkv_separator_Final.py` are already config-friendly — extend rather than replace.
- TensorBoard is wired (`SummaryWriter`) with batch-level and epoch-level scalars.
- Checkpoints already store `config`, `n_fft`, `hop_length` — good foundation for artifact versioning.

### 1.4 Known Technical Debt to Address During Migration

| Issue | Location | Plan Action |
|---|---|---|
| Resume skips scheduler state | `train_rwkv_TFtest.py` | Persist/restore full training state |
| `SeparatorV7Config.dropout` is class-level, not instance | `rwkv_separator_Final.py` | Move to config instance |
| Early-stop counter logic fragile | train loop | Encapsulate in `EarlyStopping` callback |
| Infer/train SI-SDR formulas differ | both scripts | Single canonical implementation |
| RWKV import failure silently passes | `rwkv_separator_Final.py` | Fail fast at startup with actionable error |

---

## 2. Target Repository Layout

```
RWKV_SS/
├── PLAN.md                          # This document
├── README.md                        # Setup, data prep, train/eval commands
├── pyproject.toml                   # Poetry: locked deps + optional CUDA extras
├── poetry.lock
├── configs/                         # Hydra config tree (primary config system)
│   ├── config.yaml                  # Root defaults + composition
│   ├── data/
│   │   └── libri2mix.yaml
│   ├── model/
│   │   └── rwkv_v7_separator.yaml
│   ├── training/
│   │   └── default.yaml
│   ├── stft/
│   │   └── default.yaml
│   ├── logging/
│   │   └── tensorboard.yaml       # + wandb.yaml, mlflow.yaml variants
│   └── experiment/
│       └── baseline_tf.yaml       # Named experiment overrides
├── src/
│   └── rwkv_ss/                     # Installable package namespace
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets/
│       │   │   ├── libri2mix.py   # Libri2MixDataset (train) + Eval variant
│       │   └── datamodule.py      # DataLoader factory (no training logic)
│       ├── transforms/
│       │   └── stft.py            # STFTProcessor (single source of truth)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── registry.py        # Model factory by config name
│       │   └── rwkv_v7/
│       │       ├── config.py      # SeparatorV7Config (Pydantic or dataclass)
│       │       ├── separator.py   # From rwkv_separator_Final.py
│       │       └── blocks/        # Optional: split large file incrementally
│       ├── training/
│       │   ├── __init__.py
│       │   ├── engine.py          # Train/val epoch loops
│       │   ├── trainer.py         # Orchestrator (epochs, callbacks, ckpt)
│       │   ├── losses/
│       │   │   └── pit_si_sdr.py
│       │   ├── optim/
│       │   │   └── factory.py     # Optimizer + scheduler from config
│       │   ├── callbacks/
│       │   │   ├── checkpoint.py
│       │   │   ├── early_stopping.py
│       │   │   └── logging.py     # TB/W&B/MLflow adapters
│       │   └── pipeline.py        # STFT→model→iSTFT→loss forward (shared train/infer)
│       ├── evaluation/
│       │   ├── metrics.py         # SI-SDR, PESQ, STOI, etc.
│       │   └── evaluator.py       # Batch evaluation runner
│       └── utils/
│           ├── device.py          # CPU/CUDA/MPS selection
│           ├── seed.py            # Global reproducibility
│           ├── rwkv_env.py        # RWKV_* env bootstrap (pre-import)
│           └── checkpoint.py      # Save/load with schema versioning
├── scripts/
│   ├── train.py                   # Hydra entry: `@hydra.main`
│   ├── evaluate.py                # Inference + metrics
│   └── make_csv.py                # Moved from root
├── tests/
│   ├── conftest.py                # Fixtures: dummy audio, tiny config
│   ├── test_stft_shapes.py
│   ├── test_model_forward.py
│   ├── test_loss_pit.py
│   ├── test_datamodule.py
│   └── test_config_validation.py
├── .github/
│   └── workflows/
│       └── ci.yaml                # Lint + CPU-only unit tests
└── artifacts/                     # Gitignored: checkpoints, logs, runs
```

---

## 3. Configuration Strategy (Configuration-Driven, Zero Hardcoding)

### 3.1 Recommended Stack: Hydra + Pydantic Validation Layer

**Why Hydra (primary):** Nested ML configs (data/model/training/logging), experiment overrides (`experiment=baseline_tf`), CLI overrides (`training.lr=5e-4`), and run directory management (`outputs/YYYY-MM-DD/HH-MM-SS`).

**Why Pydantic (secondary):** Validate composed config at runtime — e.g., `n_embd % n_groups == 0`, `codec_dim == n_fft // 2 + 1`, enum constraints on `head_mode`.

### 3.2 Config Namespace Mapping (Every Current CLI Arg → Config Key)

| Current CLI / Hardcoded Value | Proposed Config Path | Notes |
|---|---|---|
| `--train_csv`, `--valid_csv` | `data.train_csv`, `data.valid_csv` | Paths via env var interpolation |
| `--sample_rate` | `data.sample_rate` | Default 16000 |
| `--seg_sec` | `data.segment_seconds` | Train-only random crop |
| `--subset_frac` | `data.subset_frac` | Shared train/valid |
| subset seed `42` | `data.subset_seed` | Currently hidden |
| `--n_fft`, `--hop_length` | `stft.n_fft`, `stft.hop_length` | Drives derived `model.codec_dim` |
| STFT window `'hann'` | `stft.window` | Currently implicit |
| `--n_layer`, `--n_embd`, etc. | `model.*` | Maps to `SeparatorV7Config` |
| `num_sources=2` | `model.num_sources` | Currently hardcoded |
| `--batch_size` | `data.batch_size` | |
| `num_workers=4/2` | `data.num_workers.train`, `.valid` | Currently hardcoded |
| `drop_last=True` | `data.drop_last` | |
| `--lr`, `weight_decay=0.01` | `training.optimizer.lr`, `.weight_decay` | |
| `--grad_clip` | `training.grad_clip` | |
| LR scheduler flags | `training.scheduler.*` | Include `enabled` bool |
| Early stop flags | `training.early_stopping.*` | |
| `--epochs` | `training.epochs` | |
| `--device` | `hardware.device` | `auto` \| `cuda` \| `cpu` \| `mps` |
| `--save_dir`, `--log_dir` | `paths.checkpoint_dir`, `paths.log_dir` | Hydra can override |
| `--resume_checkpoint` | `training.resume_from` | |
| RWKV env vars | `rwkv.jit_on`, `rwkv.cuda_on`, etc. | Set in `rwkv_env.py` before model import |
| TensorBoard | `logging.backend=tensorboard` | Pluggable |

### 3.3 Derived-Value Resolution (Critical for This Pipeline)

Introduce a **config resolver** step (run once at startup):

1. Compute `model.codec_dim = stft.n_fft // 2 + 1`.
2. Compute `data.segment_samples = int(data.segment_seconds * data.sample_rate)`.
3. Optionally compute expected TF frames for sanity checks: `T_tf ≈ floor(segment_samples / hop_length) + 1` (accounting for `center=True` padding).
4. Validate divisibility: `model.n_embd % model.n_groups == 0`.
5. Persist **resolved config** alongside every checkpoint (`resolved_config.yaml`).

This replaces the current pattern of probing `train_loader` + STFT in `main()` to discover `F`.

### 3.4 Environment Variable Isolation

Use Hydra `defaults` with optional `${oc.env:LIBRI2MIX_ROOT}` for data paths. Never hardcode absolute paths in YAML — use `.env` (gitignored) + documented env vars in README.

---

## 4. Experiment Tracking & Checkpointing

### 4.1 Logging Architecture (Callback Injection Points)

Define a `LoggerProtocol` (or lightweight ABC) with methods: `log_scalar`, `log_scalars`, `log_hyperparams`, `log_audio` (optional), `close`.

| Metric / Event | When | Current | Target Injection Point |
|---|---|---|---|
| `batch/train_loss` | Every train batch | TensorBoard | `BatchLoggingCallback` |
| `loss/train`, `loss/val` | End of epoch | TensorBoard | `EpochLoggingCallback` |
| `lr` | End of epoch | TensorBoard | `OptimizerCallback` |
| Model param count | Startup | stdout | `Trainer.setup()` → all loggers |
| Resolved config | Startup | none | `log_hyperparams(resolved_cfg)` |
| Best checkpoint path | On improvement | stdout | `CheckpointCallback` + logger |

**Backend adapters:** `TensorBoardLogger` (port existing), `WandBLogger`, `MLflowLogger` — selected by `logging.backend` config. All implement same interface; `Trainer` remains backend-agnostic.

### 4.2 Checkpoint Schema (Versioned Artifacts)

Standardize checkpoint dict (version `1.0`):

- `schema_version`
- `epoch`, `global_step`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict` (if enabled)
- `early_stopping_state` (best metric, counter)
- `metrics`: `{val_loss, train_loss}`
- `resolved_config` (full YAML dict)
- `git_sha` (via `subprocess` at save time, best-effort)
- `random_states`: `{python, numpy, torch, cuda}` for reproducible resume

**File naming:** `{checkpoint_dir}/best.pt` (symlink/copy) + `{checkpoint_dir}/epoch_{epoch:03d}_val_{val_loss:.4f}.pt` (optional retention policy via config: `keep_last_n`).

**Safety:** Atomic writes (save to `.tmp` then rename). Never overwrite best without validation metric improvement.

### 4.3 Artifact Layout Per Run

```
artifacts/runs/{run_id}/
├── resolved_config.yaml
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   └── tensorboard/   # or wandb offline, mlflow/
└── metrics/
    └── epoch_metrics.jsonl
```

Hydra's `hydra.run.dir` can point here directly.

---

## 5. Reproducibility & Testing

### 5.1 Poetry Dependency Isolation

**Core dependencies (exact pins in `poetry.lock`):**

- `torch`, `torchaudio` — CUDA variant via Poetry extras group: `[tool.poetry.group.cuda.dependencies]`
- `soundfile`, `numpy`, `tqdm`
- `tensorboard`
- `hydra-core`, `omegaconf`, `pydantic` (v2)
- `pytest`, `pytest-cov` (dev group)

**Optional evaluation extras:** `pesq`, `pystoi`, `torchmetrics` — group `eval` so CI doesn't require them.

**External RWKV CUDA package:** Document as one of:
- Git submodule at `vendor/RWKV/` with path dependency in `pyproject.toml`
- Private/internal package with pinned commit hash
- Install script that compiles CUDA extension

Add `RWKV_SS_CUDA=1` gate in tests — CPU tests mock/skip CUDA kernel paths.

**CUDA locking strategy:** Document in README: install PyTorch via Poetry with explicit index URL matching CUDA version (e.g., cu124). Commit `poetry.lock`; CI uses CPU-only torch.

### 5.2 Reproducibility Utilities (`src/utils/`)

| Utility | Responsibility |
|---|---|
| `seed.py` | `set_seed(seed, deterministic=True)` — Python, NumPy, PyTorch, CUDA, cuDNN flags |
| `device.py` | `resolve_device("auto")` → cuda > mps > cpu; warn if RWKV model requested on non-CUDA |
| `rwkv_env.py` | Apply all `RWKV_*` vars from config **before** importing model modules |

**DataLoader reproducibility:** Pass `generator` with fixed seed; set `worker_init_fn` when `num_workers > 0`.

### 5.3 Pytest Validation Plan

| Test File | What It Validates | Hardware |
|---|---|---|
| `test_config_validation.py` | Pydantic rejects invalid `n_embd/n_groups`; `codec_dim` derivation | CPU |
| `test_stft_shapes.py` | STFT/iSTFT round-trip shape; `[B,1,T]` → `[B,F,T_tf]` | CPU |
| `test_model_forward.py` | Dummy `[B,T_tf,F]` through model → `[B,T_tf,S,F]` | CPU mock or CUDA skip |
| `test_loss_pit.py` | PIT SI-SDR sign, permutation invariance on synthetic signals | CPU |
| `test_datamodule.py` | Collate padding; segment crop/pad logic with synthetic CSV | CPU |
| `test_pipeline.py` | End-to-end micro-batch: wav → STFT → model → iSTFT → loss finite | CUDA optional |
| `test_checkpoint.py` | Save/load round-trip preserves weights | CPU |

**Fixtures (`conftest.py`):**
- `tiny_config` — minimal epochs, `n_layer=1`, `n_embd=64`, `n_groups=2`
- `dummy_waveform` — `[2, 1, 48000]` random tensor
- `synthetic_csv` — tmp paths to 2–3 generated WAV files

**CI policy:** GitHub Actions runs CPU tests on every PR; optional nightly workflow with `[self-hosted, gpu]` label for CUDA integration test.

---

## 6. Migration Phases

### Phase 1 — Data Layer (`src/data/`, `src/transforms/`)

**Goal:** Isolate all data I/O and preprocessing from training logic.

#### Step 1.1 — Extract STFTProcessor
- Move `STFTProcessor` from `train_rwkv_TFtest.py` to `src/rwkv_ss/transforms/stft.py`.
- Delete duplicate from `infer_rwkv_TFDomain.py` after migration.
- Unit test shape contracts immediately.

#### Step 1.2 — Extract Libri2Mix Dataset
- Create `Libri2MixDataset` (train: random segment crop/pad) in `datasets/libri2mix.py`.
- Create `Libri2MixEvalDataset` (full utterance + metadata) as subclass or separate module sharing `_load_mono`.
- Extract `collate_fn` / `collate_eval_fn` to same module.

#### Step 1.3 — DataModule Factory
- Implement `Libri2MixDataModule` with methods:
  - `setup()` — validate CSV paths exist
  - `train_dataloader()`, `val_dataloader()`, `test_dataloader()`
- All DataLoader params from config (batch_size, num_workers, shuffle, drop_last, pin_memory).
- **No imports from `training/` or `models/`.**

#### Step 1.4 — Relocate make_csv
- Move `make_csv.py` → `scripts/make_csv.py`.
- Add config-driven wrapper optional later; keep script simple.

#### Step 1.5 — Phase 1 Exit Criteria
- [x] Train script can import dataloaders without model code
- [x] STFT tests pass
- [x] Dataset tests pass with synthetic CSV
- [x] No duplicated STFT/dataset code in infer script (stub infer to import shared modules)

---

### Phase 2 — Model Layer (`src/models/`)

**Goal:** Encapsulate RWKV separator with config-driven instantiation.

#### Step 2.1 — Relocate Architecture
- Move `rwkv_separator_Final.py` → `src/rwkv_ss/models/rwkv_v7/separator.py`.
- Move `SeparatorV7Config` → `config.py`; align fields with Hydra YAML.

#### Step 2.2 — Model Registry
- Implement `build_model(cfg: ModelConfig) -> nn.Module`:
  - Computes `codec_dim` from STFT config if not explicitly set
  - Calls existing `build_rwkv7_separator()` internally
- Support `head_mode` enum: `residual | softmax_mask | direct`

#### Step 2.3 — RWKV Environment Bootstrap
- Create `src/rwkv_ss/utils/rwkv_env.py`:
  - Called at top of `scripts/train.py` and `scripts/evaluate.py` **before** model imports
  - Sets all `RWKV_*` env vars from config
- Replace scattered `os.environ.setdefault` calls.

#### Step 2.4 — CUDA Dependency Handling
- On import, if `RUN_CUDA is None` and `hardware.require_cuda=true`, raise clear error with install instructions.
- Document CUDA extension build in README.

#### Step 2.5 — Optional Incremental Refactor
- Split `separator.py` into `blocks/` submodules only if file remains unwieldy (>600 lines); not required for initial migration.

#### Step 2.6 — Phase 2 Exit Criteria
- [ ] Model builds from YAML-only config (no train_loader probe)
- [ ] Forward pass test: dummy input `[B, T_tf, codec_dim]` → `[B, T_tf, num_sources, codec_dim]`
- [ ] `n_embd % n_groups != 0` fails at config validation

---

### Phase 3 — Training Loop (`src/training/`)

**Goal:** Clean separation engine with shared forward pipeline for train/infer.

#### Step 3.1 — Loss Module
- Move `si_sdr`, `pit_si_sdr_loss` → `training/losses/pit_si_sdr.py`.
- Canonicalize on one SI-SDR implementation (reconcile train vs infer differences).
- Add `pit_si_sdr_with_perm` for evaluation reuse.

#### Step 3.2 — Forward Pipeline (Critical Abstraction)
- Implement `SeparationPipeline` in `training/pipeline.py`:
  - Inputs: `mix_wav`, `ref_sources`, `model`, `stft_processor`
  - Steps: STFT → transpose → model → ReLU magnitudes → iSTFT(mix phase) → align lengths
  - Output: `sep_wav`, `loss` (if refs provided)
- Used by both `train_one_epoch` and `validate` — eliminates duplicated 30-line blocks.

#### Step 3.3 — Training Engine
- `engine.py`:
  - `train_one_epoch(trainer_state, dataloader, pipeline, callbacks)`
  - `validate_epoch(...)`
- Handles: `model.train()/eval()`, AMP if configured, grad clip, progress bar.

#### Step 3.4 — Optimizer Factory
- `optim/factory.py`: build AdamW + optional `ReduceLROnPlateau` from config.
- Expose all currently hidden values (`weight_decay`, scheduler params).

#### Step 3.5 — Trainer Orchestrator
- `trainer.py`:
  - `setup()` — seed, device, model, optim, loggers, callbacks
  - `fit()` — epoch loop
  - `teardown()` — close loggers
- Callbacks:
  - `CheckpointCallback` — best/last saving with full state
  - `EarlyStoppingCallback` — patience from config
  - `LoggingCallback` — epoch metrics to TB/W&B/MLflow

#### Step 3.6 — Entry Point
- `scripts/train.py`:
  - Hydra `@hydra.main(config_path="../configs", config_name="config")`
  - Instantiate DataModule, Model, Trainer, call `trainer.fit()`

#### Step 3.7 — Inference / Evaluation
- `src/evaluation/evaluator.py` + `scripts/evaluate.py`:
  - Load checkpoint + resolved config
  - Reuse `SeparationPipeline`, `Libri2MixEvalDataset`, shared metrics
  - Optional metric flags from config (`evaluation.metrics: [si_sdr, pesq, stoi]`)

#### Step 3.8 — Phase 3 Exit Criteria
- [ ] Single training command reproduces prior baseline (± numerical tolerance)
- [ ] Resume from checkpoint restores optimizer + scheduler + epoch
- [ ] TensorBoard scalars match previous naming (or documented migration)
- [ ] Infer script uses zero duplicated training logic
- [ ] Legacy scripts kept as thin wrappers emitting deprecation warnings (optional, 1 release cycle)

---

### Phase 4 — Config, CI, and Production Hardening

**Goal:** Reproducible installs, automated validation, experiment hygiene.

#### Step 4.1 — Poetry Project Bootstrap
- Initialize `pyproject.toml` with package `rwkv_ss` (src layout).
- Define dependency groups: main, dev, cuda, eval.
- Add `poetry install --with dev` documented in README.

#### Step 4.2 — Hydra Config Tree
- Create full config tree per Section 3.2.
- Add `configs/experiment/baseline_tf.yaml` reproducing current default invocation:
  - `n_fft=512`, `hop_length=128`, `seg_sec=3.0`, `n_layer=8`, etc.
- Add `configs/config.yaml` composition root.

#### Step 4.3 — CI Pipeline (`.github/workflows/ci.yaml`)
- Jobs:
  1. **lint** — `ruff check` (optional `mypy` on config/models)
  2. **test** — `pytest tests/ -m "not cuda"` on Ubuntu + Python 3.10/3.11
  3. **config-smoke** — `python scripts/train.py training.epochs=1 data.subset_frac=0.001` (dry run / 1 epoch smoke on self-hosted GPU if available)

#### Step 4.4 — Pre-commit Hooks (Optional)
- `ruff format`, `ruff check`, trailing whitespace.
- Do not block on GPU tests locally.

#### Step 4.5 — Documentation
- README sections: Environment setup, RWKV CUDA install, Data preparation (`make_csv`), Training, Evaluation, Config overrides, Checkpoint resume.
- `docs/architecture.md` — pipeline diagram (waveform → STFT → RWKV → iSTFT → PIT loss).

#### Step 4.6 — Deprecation & Cutover
- Mark root-level `train_rwkv_TFtest.py` as deprecated wrapper calling `scripts/train.py` OR remove after validation.
- Tag release `v0.1.0-migrated` when baseline metrics match.

#### Step 4.7 — Phase 4 Exit Criteria
- [ ] Fresh clone → `poetry install` → `pytest` green on CPU
- [ ] Training launchable via single Hydra command with no required CLI args beyond data paths
- [ ] CI green on PR
- [ ] Checkpoint contains resolved config sufficient to run inference without guessing hyperparameters

---

## 7. Cross-Cutting Concerns

### 7.1 MPS / CPU Fallback Policy

RWKV CUDA kernels are **CUDA-only** today. Config should expose:
- `hardware.device=auto` with explicit warning when falling back
- `hardware.require_cuda=true` (default for training) to fail fast

MPS may work for STFT/data components but not full model training — document clearly.

### 7.2 Multi-GPU (Future, Out of Initial Scope)

Structure `Trainer` to accept optional `accelerator` wrapper (HuggingFace Accelerate or Lightning-style) without implementing DDP in Phase 1–4. Keep `engine.py` device-agnostic.

### 7.3 Metric Naming for Production Monitoring

Standardize scalar keys for downstream dashboards:
- `train/loss`, `val/loss`, `train/lr`, `val/si_sdr` (if computed during val)
- Prefix with project: `rwkv_ss/train/loss` if using shared W&B project

---

## 8. Migration Execution Order (Summary)

```
Week 1: Phase 1 (Data + STFT) + pytest for data/stft
Week 2: Phase 2 (Model + registry + rwkv_env) + model forward tests
Week 3: Phase 3 (Pipeline + Trainer + callbacks) + parity run vs old script
Week 4: Phase 4 (Poetry + Hydra + CI + docs) + infer migration + deprecation
```

**Parity validation gate:** Before deleting legacy scripts, run old and new pipelines on identical `subset_frac=0.01` seed and compare:
- Initial loss (first batch)
- Val loss at epoch 1 (±1e-4 tolerance for bf16/non-determinism)
- Checkpoint load → infer SI-SDR on 10 utterances

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| RWKV CUDA extension not pip-installable | Blocks training | Submodule + build docs; CI GPU smoke test |
| bf16 nondeterminism | Parity test flakiness | Document tolerance; offer fp32 debug config |
| Derived `codec_dim` mismatch at inference | Silent quality collapse | Embed resolved config in checkpoint; validate on load |
| Libri2Mix path drift | Broken dataloaders | Env-var-based paths; CSV validation in DataModule.setup |
| Large infer script metrics deps | CI/install bloat | Optional `eval` Poetry group |
| Hydra learning curve | Slower onboarding | Ship `baseline_tf` experiment + README examples |

---

## 10. Success Definition

The migration is complete when:

1. **No hyperparameters remain hardcoded** in Python training code — all flow from YAML + env.
2. **Modules are independently testable** — data, model, training, utils have no circular imports.
3. **One command trains, one command evaluates**, both reading the same config schema.
4. **Checkpoints are self-describing** and restorable with full training state.
5. **CI validates** shape/loss/config correctness on every change without a GPU.
6. **Baseline separation quality** matches the monolithic script within agreed tolerance.

---

*Generated from analysis of: `train_rwkv_TFtest.py`, `rwkv_separator_Final.py`, `infer_rwkv_TFDomain.py`, `make_csv.py`.*
