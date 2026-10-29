# TCB Detection

Training data and reproducible MambaVision-T code for transverse cirrus band (TCB) classification in GK2A imagery. The four inputs are RGB, RGB+CTH, RGB+CTT and RGB+CTT+CTH. Samples are 512 × 512 pixels at 1 km resolution.

## Dataset

The complete matched catalog contains 6,378 labelled samples from 2019–2024: 1,530 TCB and 4,848 non-TCB. Each sample has a linear RGB float32 NPY `(512,512,3)` and matched CTH/CTT float32 NPYs `(512,512)`. Cloud retrieval NaNs are retained, including fully missing cloud channels. The RGB arrays are original linear radiometric values, not display PNGs divided by 255. CTH/CTT retain the scaled product values used by the study.

`data/catalog.csv` lists every sample using dataset-relative paths. `data/splits/*.csv` reconstructs the original balancing procedure and row order from the supplied catalog with seed 1004. The test list and labels match the saved paper test predictions:

| Split | Years | TCB | non-TCB | Total |
|---|---|---:|---:|---:|
| Train |2019–2021|851|2,553|3,404|
| Validation |2022|360|1,079|1,439|
| Test |2023–2024|319|957|1,276|

The catalog includes 259 additional negative samples not selected by the paper balancing step. The 2022 catalog has 1,079 negatives; the released validation split preserves that actual count rather than duplicating a sample to obtain 1,080. These CSVs and counts describe the supplied matched catalog.

The three modality roots after extraction are `RGB/`, `CTH/` and `CTT/`. For example:

```text
dataset/
  RGB/False/npy/2019/201907010200_00_01.npy
  CTH/False/2019/201907010200_00_01.npy
  CTT/False/2019/201907010200_00_01.npy
```

### Download

The full uncompressed NPY data are about 31 GiB and are distributed as independent ZIP assets in GitHub Release `v1.0.0`, rather than Git objects. The release must be uploaded by the authors before these links become available. Once published:

```bash
python tools/download_dataset.py --output-dir ./dataset
```

Every archive and extracted NPY is checked with SHA256. Downloads/verified extractions can be resumed by repeating the command. The destination should be empty or contain the same verified dataset version.

## Environment

Use Linux, an NVIDIA CUDA GPU and Python 3.10. The model implementation used by the study is included with its original license in `third_party/MambaVision`; it supports the 3/4/5-channel inputs. A reference installation is:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

`mamba-ssm` may require a matching CUDA build toolchain. The released scripts have been checked for preserved preprocessing/splits; GPU training has not been rerun as part of this packaging step.

## Train

```bash
python train.py --scheme RGB_CTH --data-root ./dataset --gpu 0
```

All four configurations can be trained separately:

```bash
for scheme in RGB RGB_CTH RGB_CTT RGB_CTT_CTH; do
  python train.py --scheme "$scheme" --data-root ./dataset --gpu 0
done
```

The original paper training operations are retained in `training/`. Defaults: ImageNet1K initialization, Adam 1e-4, BCEWithLogitsLoss, batch size 16, 500 epochs, validation TCB F1 checkpoint selection, horizontal/vertical flip probability 0.5, training seed 1004, sampling seed 1004. This entry point reproduces the single-run paper training protocol; it does not add early stopping. `configs/paper.json` contains the settings.

Mean/std are computed from training data only. Normalization applies z-score first, then replaces NaN/Inf with zero. The input order for the five-channel scheme is **R,G,B,CTT,CTH**.

Outputs are written to `outputs/<scheme>/`. Set `--output-dir`, `--epochs`, `--sampling-seed`, `--training-seed` or `--pretrained-checkpoint` as needed; changing these changes the experiment. If the ImageNet checkpoint is absent, the original MambaVision loader downloads it to the chosen checkpoint path. The 1,000-class ImageNet head is replaced for binary classification, and extra-channel initialization follows the original training scripts. `--check-only` validates all catalog paths without training or writing results.

## Patch inference and evaluation

Use the validation-selected checkpoint and saved training normalization from the same model:

```bash
python predict.py --scheme RGB_CTH --data-root ./dataset \
  --checkpoint outputs/RGB_CTH/best_f1_epoch123.pth \
  --manifest data/splits/test.csv --gpu 0
```

The epoch in the example is the paper checkpoint name; a new training run can select another epoch. No weights are retrained during inference. Default patch inference uses batch size 1 and classifies probabilities **>0.5** as TCB. Outputs are sample-level predictions and metrics including TCB precision/recall/F1, AUROC and average precision (`AUPRC_AP`, as in the GK2A evaluation). The NASA trapezoidal AUPRC experiment is a separate analysis.

## Full-scene RGB+CTH inference

This entry point retains the paper full-scene patch classifier. Each timestamp folder must contain linear `RGB.npy` in CHW format and a 2D `CTH.npy` on the matching 1 km grid:

```bash
python predict_scenes.py --scene-root /path/to/prepared_scenes \
  --checkpoint outputs/RGB_CTH/best_f1_epoch123.pth --gpu 0
```

Scenes are centrally cropped to 512-pixel tile multiples and classified in batches of 16. The paper time selection is hourly 01–06 UTC. The original full-scene script uses probability **≥0.5**; the patch evaluation script uses **>0.5**. Outputs are patch classifications, not pixel-level segmentation or a validated climatology. Existing completed timestamps are skipped when the command is repeated.

## Source and licensing

Source hashes and the scope of the public adaptations are recorded in `configs/source_provenance.json`. Paths were parameterized, post-training visualization code was separated, and scoped PyTorch 2.6 checkpoint compatibility was added. Numerical preprocessing, augmentation and the training loops were retained.

MambaVision is by NVIDIA; see https://github.com/NVlabs/MambaVision and the retained `third_party/MambaVision/LICENSE` (NVIDIA Source Code License-NC). See `THIRD_PARTY_NOTICES.md`. The project-specific code and dataset license has not yet been specified by the authors.

For the authors' Linux upload steps, see `UPLOAD_FROM_LINUX_KO.md`.
