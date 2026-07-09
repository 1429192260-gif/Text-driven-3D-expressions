# Mac Official Render Phase 2 Verification - 2026-07-09

This document records the macOS Phase 2 verification state for the official EmoAva / DECA / FLAME rendering assets and inputs.

## Scope

Verified on Mac:

- Official DECA / FLAME asset restoration.
- Official render input restoration.
- Input regeneration via `scripts/export_emoava_official_render_inputs.py`.
- CPU environment check via `scripts/check_emoava_official_3d_env.py --device cpu`.
- Rendered video presence check.

Not attempted on Mac:

- CUDA rendering.
- PyTorch3D installation/debugging.
- `--instantiate-deca`, because PyTorch3D and other DECA dependencies are missing.
- Official `.mp4` generation.

## Repository

```text
branch: codex/3d-expression-backend
HEAD: 4aa0257448d524fb3c05e90da91134fe3683ef6c
commit: 4aa0257 Add official render phase 2 handoff
```

Mac checkout:

```text
/Users/cyz/Documents/Codex/2026-07-09/mac-codex-windows-macos-clone-checkout/work/Text-driven-3D-expressions
```

## Transfer Package

The Phase 2 package arrived on Mac as an already-unpacked directory:

```text
/Users/cyz/Downloads/project_mac_transfer_official_render_phase2
```

Package summary:

```text
size: 320M
files: 39
```

`SHA256SUMS.txt` uses Windows path separators, CRLF line endings, and a BOM. Verification passed with a temporary normalization pipe:

```bash
perl -pe 's/^\xEF\xBB\xBF// if $.==1; s/\r$//; s|\\|/|g' SHA256SUMS.txt | shasum -a 256 -c -
```

All listed files returned `OK`.

## Restored Assets

Official DECA / FLAME assets:

```text
external/EmoAva/src/data/default_code_trevor_emoca2.pkl
external/EmoAva/src/data/fixed_displacement_256.npy
external/EmoAva/src/data/generic_model.pkl
external/EmoAva/src/data/head_template.obj
external/EmoAva/src/data/landmark_embedding.npy
external/EmoAva/src/data/mean_texture.jpg
external/EmoAva/src/data/texture_data_256.npy
external/EmoAva/src/data/uv_face_eye_mask.png
external/EmoAva/src/data/uv_face_mask.png
```

Official render inputs:

```text
outputs/emoava_official_render_inputs/v6_showcase/gold_selected_exps.pkl
outputs/emoava_official_render_inputs/v6_showcase/baseline_selected_exps.pkl
outputs/emoava_official_render_inputs/v6_showcase/v6_sample_selected_exps.pkl
outputs/emoava_official_render_inputs/v6_showcase/manifest.json
outputs/emoava_official_render_inputs/v6_showcase/manifest.md
```

Inputs required to regenerate selected render inputs:

```text
external/EmoAva/dataset/test_stage1_exps.pkl
external/EmoAva/dataset/test_stage1_text.pkl
external/EmoAva/dataset/stage1_mean.npy
external/EmoAva/dataset/stage1_std.npy
outputs/when_words_smile_repro/test_parallel_full.pt
outputs/when_words_smile_prior_v6/test_mixture_gate_sample.pt
outputs/when_words_smile_prior_v6/test_mixture_gate.pt
```

## Export Regeneration

Command:

```bash
.venv/bin/python scripts/export_emoava_official_render_inputs.py \
  --methods gold,baseline,v6_sample \
  --cases 967,354,295,428,84 \
  --output-dir outputs/emoava_official_render_inputs/v6_showcase_mac_check
```

Result: succeeded.

Generated files:

```text
outputs/emoava_official_render_inputs/v6_showcase_mac_check/gold_selected_exps.pkl
outputs/emoava_official_render_inputs/v6_showcase_mac_check/baseline_selected_exps.pkl
outputs/emoava_official_render_inputs/v6_showcase_mac_check/v6_sample_selected_exps.pkl
outputs/emoava_official_render_inputs/v6_showcase_mac_check/manifest.md
outputs/emoava_official_render_inputs/v6_showcase_mac_check/manifest.json
```

Manifest summary:

```text
records: 5
methods: gold, baseline, v6_sample
sequence_lengths:
  gold: [16, 11, 18, 9, 17]
  baseline: [16, 11, 18, 9, 17]
  v6_sample: [16, 11, 18, 9, 17]
```

## CPU Environment Check

Command:

```bash
.venv/bin/python scripts/check_emoava_official_3d_env.py --device cpu
```

Output summary:

```text
torch: 2.13.0
cuda available: False
torch cuda version: None
```

Python modules:

```text
cv2: missing
yacs: missing
kornia: missing
skimage: missing
tqdm: ok
pytorch3d: missing
```

Official files:

```text
external/EmoAva/src/data/generic_model.pkl: ok
external/EmoAva/src/data/head_template.obj: ok
external/EmoAva/src/data/landmark_embedding.npy: ok
external/EmoAva/src/data/mean_texture.jpg: ok
external/EmoAva/src/data/uv_face_eye_mask.png: ok
external/EmoAva/src/data/uv_face_mask.png: ok
external/EmoAva/src/data/default_code_trevor_emoca2.pkl: ok
external/EmoAva/src/data/deca_model.tar: optional missing
external/EmoAva/src/data/FLAME_albedo_from_BFM.npz: optional missing
```

Default code:

```text
default_code: ok
shape: (1, 100)
tex: (1, 50)
exp: (1, 50)
pose: (1, 6)
cam: (1, 3)
light: (1, 9, 3)
detail: (1, 128)
images: (1, 3, 224, 224)
```

## Rendered Videos

No rendered video files were found under `outputs` or `external`:

```bash
find outputs external -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' \)
```

Output was empty. This matches the Phase 2 handoff note that cloud-rendered official videos were not included.

## Conclusion

Mac is ready to:

- Store and verify official EmoAva / DECA / FLAME rendering assets.
- Regenerate official `.pkl` render inputs from 53D expression sequences.
- Hand these inputs to a Linux GPU/cloud environment for rendering.

Mac is not currently ready to produce official rendered videos locally because:

- CUDA is unavailable.
- PyTorch3D is missing.
- Required DECA visualization dependencies are missing: `cv2`, `yacs`, `kornia`, `skimage`.

Recommended next step: run official rendering on a Linux GPU/cloud machine with CUDA, PyTorch3D, and DECA dependencies installed.
