# Official Render Phase 2 Handoff - 2026-07-09

This document defines Phase 2 of the Windows-to-Mac migration: moving the official EmoAva / DECA / FLAME rendering assets and inputs.

## Status

Phase 1 is complete. The Mac can run the core text-to-expression pipeline.

Phase 2 focuses on the official 3D rendering path:

```text
53D expression sequence -> official EmoAva / DECA / FLAME backend -> 3D expression video
```

The original Windows workspace does not currently contain the cloud-rendered official `.mp4` videos. It does contain the official render inputs and the DECA/FLAME assets needed to reproduce rendering on a GPU machine.

## Transfer Package

Windows prepared:

```text
C:\Users\蔡\Desktop\project_mac_transfer_official_render_phase2.zip
```

Approx size:

```text
286.62 MB
```

Unpacked source directory:

```text
C:\Users\蔡\Desktop\project_mac_transfer_official_render_phase2
```

The package includes `SHA256SUMS.txt`.

## Included Assets

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

Inputs for regenerating selected render inputs:

```text
external/EmoAva/dataset/test_stage1_exps.pkl
external/EmoAva/dataset/test_stage1_text.pkl
external/EmoAva/dataset/stage1_mean.npy
external/EmoAva/dataset/stage1_std.npy
outputs/when_words_smile_repro/test_parallel_full.pt
outputs/when_words_smile_prior_v6/test_mixture_gate_sample.pt
outputs/when_words_smile_prior_v6/test_mixture_gate.pt
```

Analysis and reference files:

```text
outputs/emoava_official_render_analysis/
external/EmoAva/resource/
external/EmoAva/src/requirements.txt
external/EmoAva/README.md
external/EmoAva/license_agreement.pdf
docs/official_3d_backend_integration.md
```

## Not Included

Not included in this package:

```text
external/EmoAva/checkpoints/model.chkpt
models/bert-base-cased/
full outputs/
cloud-rendered official mp4 videos
```

Reason:

- Official DECA rendering from exported `.pkl` inputs does not need `external/EmoAva/checkpoints/model.chkpt`.
- The checkpoint is only needed if we want to rerun the original EmoAva text-to-53D model.
- The cloud-rendered videos are not present in the current Windows workspace.

If Phase 2 later needs the official model checkpoint, create a separate checkpoint transfer package.

## Mac Phase 2 Goals

On Mac, Phase 2 should verify asset restoration and script readiness, not necessarily complete GPU rendering locally.

Expected Mac tasks:

1. Unzip `project_mac_transfer_official_render_phase2.zip` into the repository root.
2. Confirm official render input `.pkl` files exist.
3. Confirm DECA/FLAME asset files exist.
4. Run `scripts/export_emoava_official_render_inputs.py` to regenerate selected `.pkl` inputs.
5. Run `scripts/check_emoava_official_3d_env.py --device cpu` and report missing render dependencies.
6. Do not spend time forcing PyTorch3D/CUDA rendering on Mac if it becomes the blocker.
7. Keep actual official video rendering as GPU-cloud execution unless Mac has a working PyTorch3D setup.

## Mac Verification Commands

After unzipping package into repo root:

```bash
test -f external/EmoAva/src/data/generic_model.pkl
test -f external/EmoAva/src/data/default_code_trevor_emoca2.pkl
test -f outputs/emoava_official_render_inputs/v6_showcase/v6_sample_selected_exps.pkl
test -f outputs/when_words_smile_prior_v6/test_mixture_gate_sample.pt
```

Regenerate official render inputs:

```bash
.venv/bin/python scripts/export_emoava_official_render_inputs.py \
  --methods gold,baseline,v6_sample \
  --cases 967,354,295,428,84 \
  --output-dir outputs/emoava_official_render_inputs/v6_showcase_mac_check
```

Check official backend environment:

```bash
.venv/bin/python scripts/check_emoava_official_3d_env.py --device cpu
```

If Mac has PyTorch3D installed and the DECA stack imports successfully, try:

```bash
.venv/bin/python scripts/check_emoava_official_3d_env.py --device cpu --instantiate-deca
```

Do not treat missing `FLAME_albedo_from_BFM.npz` as a blocker for geometry-only rendering.

## GPU / Cloud Render Commands

On a Linux GPU machine with CUDA and PyTorch3D:

```bash
python scripts/check_emoava_official_3d_env.py --device cuda --instantiate-deca
```

Render V6 sample:

```bash
python scripts/render_emoava_official_deca.py \
  --exp-path outputs/emoava_official_render_inputs/v6_showcase/v6_sample_selected_exps.pkl \
  --output-dir outputs/emoava_official_render_videos/v6_sample \
  --device cuda \
  --geometry-only \
  --no-detail
```

Render baseline:

```bash
python scripts/render_emoava_official_deca.py \
  --exp-path outputs/emoava_official_render_inputs/v6_showcase/baseline_selected_exps.pkl \
  --output-dir outputs/emoava_official_render_videos/baseline \
  --device cuda \
  --geometry-only \
  --no-detail
```

Render gold:

```bash
python scripts/render_emoava_official_deca.py \
  --exp-path outputs/emoava_official_render_inputs/v6_showcase/gold_selected_exps.pkl \
  --output-dir outputs/emoava_official_render_videos/gold \
  --device cuda \
  --geometry-only \
  --no-detail
```

## Decision Point

After Mac verifies the Phase 2 package:

- If we only need Mac to manage render inputs and research files, Phase 2 is complete.
- If we need official mp4 videos locally, retrieve them from the cloud machine if they still exist.
- If cloud outputs are unavailable, rerun the GPU render commands on a Linux GPU environment.

