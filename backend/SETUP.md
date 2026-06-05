# AccentShift Backend — Setup (Windows, 8GB+ NVIDIA GPU)

Architecture A pipeline (Seed-VC V2 + Vevo ensemble). All audio 16kHz mono internally.

## 1. Python env (uv venv)

Tested on: RTX 4060 Laptop 8GB, CUDA 11.8 (nvcc), Windows 11, Python 3.10. Torch is pinned
to the cu118 build to match the local toolkit — change the index URL if your CUDA differs.

```powershell
cd backend
uv venv --python 3.10
# Torch matching local CUDA 11.8 (separate index, NOT in pyproject):
uv pip install --python .\.venv\Scripts\python.exe `
    torch==2.6.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# Everything else (Seed-VC + Vevo deps included; their requirements.txt pins are skipped on
# purpose because they downgrade torch/numpy):
uv pip install --python .\.venv\Scripts\python.exe `
    numpy "scipy>=1.10" "librosa>=0.10.2" soundfile pyloudnorm pyworld silero-vad transformers `
    jiwer speechbrain scikit-learn joblib pyyaml click huggingface-hub hf_xet truststore tqdm `
    hydra-core omegaconf einops munch accelerate pydub `
    "fastapi>=0.111" "uvicorn[standard]>=0.29" python-multipart
```

> `truststore` is required on this network: a proxy injects a self-signed root CA, so plain
> certifi-based TLS (huggingface_hub, torch.hub, speechbrain) fails with
> `CERTIFICATE_VERIFY_FAILED`. The entrypoints call `truststore.inject_into_ssl()` to use the
> Windows cert store. For `uv` itself, always pass `--native-tls`.

Do NOT run `uv pip install -r third_party/seed-vc/requirements.txt` — it pins torch==2.4.0
(+cu126 nightly) and numpy==1.26.4 and will break the working GPU stack.

Verify CUDA:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Keep all model downloads inside the project folder

Set HF cache under `backend/` before running download or inference:

```powershell
$env:HF_HOME = "D:\designathon_2\backend\.hf_cache"
```

`model_manager.py` auto-sets `HF_HOME` to `.hf_cache` if not already set, so inference works
without manually exporting it. Set it explicitly for `download_models.py`.

Whisper large-v3 is stored in HF cache only (no copy to `checkpoints/` — too large to duplicate).
SER, BigVGAN, ECAPA are copied to `checkpoints/` for explicit local control.

## 2. Download models + clone VC repos

```powershell
uv run python scripts/download_models.py
```

Pulls (into `checkpoints/`): Whisper large-v3 (GPU on droplets ≥12GB VRAM; auto-falls to CPU on 8GB cards), audeering wav2vec2 SER, BigVGAN-v2, ECAPA.
Clones (into `third_party/`): Seed-VC, Amphion. Prefetches UTMOS via torch.hub.

> Seed-VC and Amphion may need their own checkpoint downloads on first inference run
> (they auto-pull from HF). See their READMEs in `third_party/`.

## 3. Seed-VC install + Windows fallbacks

```powershell
cd third_party\seed-vc
uv pip install -r requirements.txt
cd ..\..
```

Windows fallbacks if install fails:
- `flash-attn` — skip it (CPU/Windows has no wheel). Comment out in their requirements; Seed-VC runs without it (slower attention).
- `deepspeed` — not needed for inference. Remove from requirements if it blocks.
- If a CUDA op is missing, confirm torch CUDA build from step 1 matches.

Confirm Seed-VC inference entrypoint:
```powershell
uv run python third_party\seed-vc\inference_v2.py --help
```
Note the exact `--source/--target/--output` flag names; update `pipeline/converter.py` SeedVCBackend if they differ.

## 4. Vevo (Amphion) install + Windows fallbacks

```powershell
cd third_party\Amphion
uv pip install -r requirements.txt
cd ..\..
```

- Amphion is heavy; many deps are training-only. For Vevo inference you mainly need the VC submodule deps.
- On an 8GB card Vevo runs on CPU (`vevo.device: cpu` in `configs/pipeline_config.yaml`) — slower but fits VRAM budget.
- Confirm the Vevo inference module path:
```powershell
uv run python -m models.vc.vevo.infer_vevovoice --help
```
Update `pipeline/converter.py` VevoBackend module path/flags if they differ.

## 5. Reference clips

Add 5–10s WAVs of native target-accent speakers under:
```
references/indian_english/  chinese_english/  japanese_english/
references/british_english/  american_english/
```
First clip per folder is used by default.

## 6. Run the pipeline (end-to-end)

```powershell
uv run python run_pipeline.py --input sample.wav --target-accent indian_english `
    --output out.wav --metrics-out metrics.json
```

## 7. Evaluation

```powershell
uv run python -m evaluation.eval_pipeline --test-dir data\l2arctic_test `
    --target-accent indian_english --out-json eval_results.json
```

## 8. Fine-tuning (A100 ONLY — do not run on the 8GB box)

```powershell
# On the A100 box, after datasets are in place:
uv run python scripts\prepare_l2arctic.py --l2arctic-dir data\l2arctic --cmu-dir data\cmu_arctic `
    --out data\l2arctic_pairs\manifest.jsonl
uv run python scripts\finetune_style.py --accent all
```

`finetune_style.py` is a guarded skeleton — wire the Seed-VC trainer imports (step 1 in that
file) against the cloned repo on the A100 box before running. After training, copy the
checkpoint to `checkpoints/seedvc_finetuned/` and point `seed_vc` in
`configs/pipeline_config.yaml` at it.

## 9. Start the API server

```powershell
$env:HF_HOME = "D:\designathon_2\backend\.hf_cache"
cd backend
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Once you see `AccentShift is ready.` in the logs, the server accepts requests.
Test with:
```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/accents
```
