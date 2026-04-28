# ReferDINO Reproduction Manual

This handoff is designed for Bunya because the local workstation does not expose a GPU. The commands below assume:

- repo root: `/scratch/user/uqqnguy9/segmentation`
- inner ReferDINO repo: `/scratch/user/uqqnguy9/segmentation/ReferDINO`
- sibling dataset root: `/scratch/user/uqqnguy9/dataset`

## 1. Pull the latest code

```bash
cd /scratch/user/uqqnguy9/segmentation
git pull

cd /scratch/user/uqqnguy9/segmentation/ReferDINO
git pull
```

The second pull matters because `ReferDINO/` is its own git repo.

## 2. Start a GPU shell on Bunya

Use your normal Bunya account and partition values. A generic interactive example is:

```bash
srun \
  --account <account> \
  --partition <gpu_partition> \
  --gres=gpu:a100:1 \
  --mem=128G \
  --cpus-per-task=16 \
  --time=08:00:00 \
  --pty bash -l
```

If your site workflow requires modules, load them before the Python setup.

## 3. Create the ReferDINO environment

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO

conda create -n referdino python=3.10 -y
conda activate referdino

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Build the deformable attention extension:

```bash
cd models/GroundingDINO/ops
python setup.py build install
python test.py
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
```

## 4. Download the required checkpoints

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
bash ckpt.sh
```

This creates only the required local files:

- `pretrained/groundingdino_swinb_cogcoor.pth`
- `ckpt/ryt_swinb.pth`
- `ckpt/mevis_swinb.pth`

If Bunya cannot download from the internet, place those files manually in the same locations and skip `ckpt.sh`.

## 5. Prepare the local dataset layout

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
bash tools/prepare_local_rvos_data.sh --dataset-root /scratch/user/uqqnguy9/dataset
```

This builds the ReferDINO runtime layout under `ReferDINO/data/` from the sibling dataset tree and normalizes MeViS metadata names.

## 6. Optional smoke test on one video

DAVIS smoke test:

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
conda activate referdino

PYTHONPATH=. python eval/inference_davis.py \
  -c configs/davis_swinb.yaml \
  -ng 1 \
  -ckpt ckpt/ryt_swinb.pth \
  --version smoke_davis \
  --video_first_n 1 \
  --overlay_video_first_n 1
```

MeViS smoke test:

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
conda activate referdino

PYTHONPATH=. python eval/inference_mevis.py \
  --split valid_u \
  -c configs/mevis_swinb.yaml \
  -ng 1 \
  -ckpt ckpt/mevis_swinb.pth \
  --version smoke_mevis \
  --video_first_n 1 \
  --overlay_video_first_n 1
```

## 7. Run the full DAVIS reproduction

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
conda activate referdino

PYTHONPATH=. python eval/inference_davis.py \
  -c configs/davis_swinb.yaml \
  -ng 1 \
  -ckpt ckpt/ryt_swinb.pth \
  --version swinb_repro \
  --overlay_video_first_n 5
```

Evaluate DAVIS:

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
conda activate referdino

PYTHONPATH=. python eval/eval_davis.py \
  --results_path output/davis/swinb_repro/Annotations
```

Paper target for comparison:

- `J = 65.1`
- `F = 72.9`
- `J&F = 68.9`

## 8. Run the full MeViS reproduction

Use `valid_u`, because that is the split with public GT required by `eval_mevis.py`.

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
conda activate referdino

PYTHONPATH=. python eval/inference_mevis.py \
  --split valid_u \
  -c configs/mevis_swinb.yaml \
  -ng 1 \
  -ckpt ckpt/mevis_swinb.pth \
  --version swinb_repro \
  --overlay_video_first_n 5
```

Evaluate MeViS:

```bash
cd /scratch/user/uqqnguy9/segmentation/ReferDINO
conda activate referdino

PYTHONPATH=. python eval/eval_mevis.py \
  --mevis_pred_path output/mevis/swinb_repro/valid_u/Annotations \
  --mevis_exp_path data/mevis/valid_u/meta_expressions.json \
  --mevis_mask_path data/mevis/valid_u/mask_dict.json
```

Paper target for comparison:

- `J = 44.7`
- `F = 53.9`
- `J&F = 49.3`

Small deviations are acceptable. Focus on `J`, `F`, and `J&F`.

## 9. Overlay-video outputs

The inference runs above export green mask-overlay MP4s. Use `--overlay_video_first_n 0`
to disable MP4 export, a positive number for a prefix subset, or `-1` for all selected videos.

DAVIS overlay videos:

- root: `output/davis/swinb_repro/overlay_videos/`
- layout: `<video>/exp_<exp_id>.mp4`
- each `<video>` folder also contains `manifest.json` entries of `[relative_path, query]`

DAVIS predictions:

- root: `output/davis/swinb_repro/Annotations/`
- layout: `anno_<annotator>/<video>/<frame>.png`
- DAVIS evaluation CSVs are written in each `anno_<annotator>` folder.

MeViS overlay videos:

- root: `output/mevis/swinb_repro/valid_u/overlay_videos/`
- layout: `<video>/exp_<exp_id>.mp4`
- each `<video>` folder also contains `manifest.json` entries of `[relative_path, query]`

Metric plots:

- root: `output/visualize/<dataset>/swinb_repro/<video>/`
- layout: `<exp_id>.png`
- each PNG contains per-frame `J`, `F`, and `J&F` lines titled by the query expression.

## 10. Notes

- `--overlay_video_first_n 0` disables MP4 export.
- `--video_first_n 0` uses the full dataset.
- `--visualize` still writes per-frame PNG overlays if you want them, but it is not required for the MP4 export.
- Do not run these jobs on a login node without a visible GPU.
