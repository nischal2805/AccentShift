# Vevo backend — working configuration & patches

Vevo (Amphion Vevo-Voice) gives stronger accent than Seed-VC. This branch captures
the fixes that made it reliable. Run with `--config configs/pipeline_vevo.yaml`.

## How to run
```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_TORCH=1 \
  uv run python run_pipeline.py --config configs/pipeline_vevo.yaml \
  --input <wav/mp4> --target-accent <accent> --output out.wav
```
- `HF_HUB_OFFLINE=1` avoids transient HF-client errors (models are cached).
- Run ONE pipeline at a time; verify `nvidia-smi` ~0 MiB before launch (zombies oversubscribe the 8GB GPU → 100x slowdown).

## Extra Python deps (installed via `uv pip install`, NOT in pyproject)
- `ipython`, `json5`  — required by Amphion's vevo_utils import chain.
  Reinstall on a fresh env: `uv pip install ipython json5`.

## Patches applied to third_party/Amphion (gitignored embedded git repos)
`third_party/` is gitignored AND Amphion/seed-vc are embedded git repos, so their
files can't be tracked in our repo. The patches are preserved in `vevo_patches/`:
- `vevo_patches/amphion_vevo.patch` — reapply: `git -C third_party/Amphion apply ../../vevo_patches/amphion_vevo.patch`
- `vevo_patches/vevo_utils.py`, `vevo_patches/ar_model.py` — full patched copies (drop-in replacements)

The exact changes (for reference):

### models/vc/vevo/vevo_utils.py — in `inference_ar_and_fm`, the `ar_model.generate(...)` call:
```python
        predicted_hubert_codecs = self.ar_model.generate(
            input_ids=ar_input_ids,
            prompt_mels=self.extract_prompt_mel_feature(style_ref_speech16k),
            prompt_output_ids=prompt_output_ids,
            max_length=3500,        # allow 12s segments -> fewer seams
            temperature=0.7,
            repeat_penalty=1.3,     # reduces loops/emotion-shift gibberish
            min_new_tokens=8,       # was 50: stop near real speech end -> no trailing noise
        )
        # CRASH GUARD: AR can emit special tokens (>=8192) mid-sequence that overflow
        # the diffusion codec embedding -> CUDA device-side assert (kills process).
        # Truncate at first out-of-range token, clamp the rest.
        _VOCAB = 8192
        _bad = (predicted_hubert_codecs[0] >= _VOCAB) | (predicted_hubert_codecs[0] < 0)
        if bool(_bad.any()):
            _cut = int(_bad.nonzero()[0].item())
            if _cut >= 1:
                predicted_hubert_codecs = predicted_hubert_codecs[:, :_cut]
        predicted_hubert_codecs = predicted_hubert_codecs.clamp(0, _VOCAB - 1)
```

### models/vc/autoregressive_transformer/ar_model.py — both `self.model.generate(...)` calls:
- `do_sample=True` -> `do_sample=False` (greedy: deterministic, fewer hallucinations/crashes).

## Pipeline-side changes (tracked normally)
- `pipeline/converter.py` VevoBackend.convert: trims the style/timbre reference to
  `vevo.max_ref_s` (8s) — long refs overflow the AR's ~2000-token input cap.
- `run_pipeline.py` build_backends: honors `seed_vc.enabled` (false in vevo config) so
  Vevo runs alone on the GPU.
- `configs/pipeline_vevo.yaml`: seed_vc.enabled=false, vevo.enabled=true, vevo.device=cuda,
  max_segment_s=12 (fewer seams), max_ref_s=8, vram_gb_threshold=0 (force Vevo on GPU).

## Known remaining limitation
Vevo's AR still garbles on the most emotionally-extreme segments (big pitch swings);
softened via greedy + repeat_penalty but not eliminated. Next ideas to try:
larger Whisper, cascade Seed-VC after Vevo, or per-segment backend selection.
