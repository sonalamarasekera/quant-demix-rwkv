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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt  # if provided
```

3) Run a Hydra-configured training dry-run

```powershell
python -m scripts.train --config configs/experiment/baseline_tf.yaml
```

See `configs/` for composition and override examples.
