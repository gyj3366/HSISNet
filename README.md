# HSISNet Evaluation

This repository provides evaluation scripts for paired cover/stego and secret/recovered RGB images, as well as statistical analysis of results across independently trained models.

## Repository Structure

- `evaluate.py`: Computes PSNR, SSIM, MAE, CLIP similarity, and LPIPS. Exports per-image metrics, dataset-level summaries, and run metadata.
- `summarize_runs.py`: Aggregates results across independent runs and calculates the mean, sample standard deviation, and two-sided Student's t confidence interval.
- `requirements.txt`: Lists the required Python packages.
- `.gitignore`: Specifies files and directories excluded from version control.

## Installation

Python 3.9 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

The evaluation uses pretrained CLIP and LPIPS models. Their weights are downloaded during the first evaluation if they are not locally cached.

## Input Data

The evaluation manifest is provided as a CSV file through `--manifest`, containing the following columns:

| Column | Description |
|---|---|
| `pair_id` | Unique image-pair identifier |
| `cover_path` | Cover image path |
| `secret_path` | Ground-truth secret image path |
| `stego_path` | Generated stego image path |
| `recovered_secret_path` | Recovered secret image path |

Reference image paths are resolved relative to `--reference-root`, while generated image paths are resolved relative to `--output-root`. Absolute paths are also supported.

## Evaluation

The evaluation is performed using:

```bash
python evaluate.py \
  --manifest /path/to/test_pairs.csv \
  --reference-root /path/to/reference_images \
  --output-root /path/to/model_outputs \
  --out-dir /path/to/results/seed_1
```

Optional arguments include `--device auto|cpu|cuda`, `--clip-model`, `--clip-pretrained`, `--lpips-net alex|vgg|squeeze`, and `--batch-size`.

Images are loaded in RGB format and clipped to \([0,255]\). The evaluator retains their original dimensions without additional resizing. Reference and generated images are evaluated in corresponding pairs.

## Evaluation Metrics

- **PSNR:** Calculated on the BT.601 luminance (Y) channel:
  \[
  Y=0.299R+0.587G+0.114B
  \]
  with a dynamic range of 255.
- **SSIM:** Calculated on RGB images with a data range of 255, an \(11\times11\) window, Gaussian weights, and \(\sigma=1.5\).
- **MAE:** Calculated across all pixels and three RGB channels, expressed in 8-bit intensity levels.
- **CLIP similarity:** Calculated as the cosine similarity between L2-normalized image embeddings using the configured OpenCLIP model.
- **LPIPS:** Measures perceptual differences between RGB image tensors normalized to \([-1,1]\), using the selected pretrained backbone.

Each metric is first calculated for individual image pairs. Dataset-level results are obtained by averaging the per-image scores. Cover/stego and secret/recovered results are reported separately.

## Evaluation Outputs

The following files are generated in the specified output directory:

- `per_image_metrics.csv`: Per-image evaluation results for each image pair and pair type.
- `summary_metrics.csv`: Dataset-level mean and standard deviation across test images.
- `run_metadata.json`: Evaluation configuration, manifest hash, model settings, and software versions.

## Repeated-Run Uncertainty Analysis

Five independent training runs with different random seeds are evaluated using the same test manifest and evaluation protocol. Each run produces its own `per_image_metrics.csv`.

The results are aggregated using:

```bash
python summarize_runs.py \
  --runs /path/to/results/seed_1 /path/to/results/seed_2 /path/to/results/seed_3 \
  --out-dir /path/to/results/uncertainty \
  --confidence 0.95
```

The analysis calculates the mean and sample standard deviation across independent run-level test-set means, together with two-sided Student's t confidence intervals.

The following files are generated:

- `run_level_means.csv`: Test-set mean metrics for each independent run.
- `uncertainty_summary.csv`: Overall mean, sample standard deviation, and confidence intervals across runs.
- `uncertainty_metadata.json`: Run identifiers and statistical analysis settings.

The statistical analysis uses independently trained checkpoints as experimental units. Externally reported baseline results are presented separately from repeated-run uncertainty estimates.
