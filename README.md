RWKV-SS — TF-domain speech separation (migrated)

Quickstart (CPU-only, minimal):

1) Create virtualenv and activate

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies (CPU Torch example)

```powershell
pip install -U pip
# Recommended: use Poetry to manage dependencies
poetry install --with dev

# CPU-only PyTorch example (alternative to poetry)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Optional evaluation extras (PESQ/STOI) can be installed via:
poetry install --with eval
```

3) Run a Hydra-configured training dry-run

```powershell
python -m scripts.train --config configs/experiment/baseline_tf.yaml
```

See `configs/` for composition and override examples.
