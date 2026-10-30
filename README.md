# HSISNet Evaluation Code

This repository contains scripts for evaluating paired cover/stego and secret/recovered RGB images and summarizing uncertainty across independently trained runs.

## Files

- `evaluate.py`: computes PSNR on the BT.601 Y channel, RGB SSIM, RGB MAE, CLIP image-embedding cosine similarity, and LPIPS. It exports per-image metrics, dataset-level summaries, and run metadata.
- `summarize_runs.py`: aggregates results from independent runs and reports the mean, sample standard deviation, and two-sided Student-t confidence interval across run-level test-set means.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: excludes caches, virtual environments, result folders, and common model checkpoint files from Git.

## Requirements and installation

Python 3.9 or newer is recommended. Install the listed packages with:

```bash
python -m pip install -r requirements.txt
```

The first evaluation may download the selected pretrained CLIP and LPIPS weights. Internet access may be required unless the weights are already cached locally.

## Input manifest

Pass a CSV file to `evaluate.py` using `--manifest`. The CSV must contain these columns:

- `pair_id`: unique identifier for each cover-secret pair.
- `cover_path`: path to the cover image.
- `secret_path`: path to the ground-truth secret image.
- `stego_path`: path to the generated stego image.
- `recovered_secret_path`: path to the recovered secret image.

Paths in `cover_path` and `secret_path` are resolved relative to `--reference-root`; paths in `stego_path` and `recovered_secret_path` are resolved relative to `--output-root`. Absolute paths are also accepted. Keep the manifest, image IDs, and pairing rules fixed across methods and runs. The manifest should refer to your actual evaluation data; do not use invented or placeholder image IDs in reported experiments.

## Run evaluation

```bash
python evaluate.py \
  --manifest /path/to/test_pairs.csv \
  --reference-root /path/to/reference_images \
  --output-root /path/to/model_outputs \
  --out-dir /path/to/results/seed_1
```

Optional arguments include `--device auto|cpu|cuda`, `--clip-model`, `--clip-pretrained`, `--lpips-net alex|vgg|squeeze`, and `--batch-size`.

The evaluator does not resize images. Perform all required preprocessing before evaluation and ensure each reference/output pair has identical dimensions. Images are loaded as RGB and clipped to `[0, 255]`.

## Metric definitions

- **PSNR:** computed per image on the BT.601 luma channel, `Y = 0.299R + 0.587G + 0.114B`, with a data range of 255.
- **SSIM:** computed on RGB images with `data_range=255`, an `11 x 11` window, Gaussian weights, `sigma=1.5`, and channel-axis aggregation as implemented by scikit-image.
- **MAE:** mean absolute error across all pixels and all three RGB channels, expressed in 8-bit intensity levels.
- **CLIP similarity:** cosine similarity between L2-normalized image embeddings from the configured OpenCLIP model. OpenCLIP's configured image preprocessing is applied for this metric.
- **LPIPS:** perceptual distance between RGB images represented as tensors in `[-1, 1]`, using the selected LPIPS backbone.

For each metric, the dataset-level score is the arithmetic mean of the per-image scores. The script reports metrics separately for `cover_stego` and `secret_recovered` pairs. CLIP and LPIPS are complementary semantic/perceptual measures and should not be interpreted as substitutes for pixel-level metrics.

## Output files

`evaluate.py` writes the following files to `--out-dir`:

- `per_image_metrics.csv`: one row per image pair and pair type, with all metrics.
- `summary_metrics.csv`: mean and standard deviation across test images for each metric and pair type. This standard deviation describes image-to-image variation, not training-run uncertainty.
- `run_metadata.json`: manifest hash, metric settings, model configuration, and software versions.

## Repeated-run uncertainty

Train model independently using different random seeds, evaluate each resulting checkpoint on the same fixed test manifest, and save each run to its own output directory. 

At least two independent run directories are required; more runs are preferable for a stable uncertainty estimate. Each run directory must contain `per_image_metrics.csv` produced by `evaluate.py`.

```bash
python summarize_runs.py \
  --runs /path/to/results/seed_1 /path/to/results/seed_2 /path/to/results/seed_3 \
  --out-dir /path/to/results/uncertainty \
  --confidence 0.95
```

The script verifies that all runs contain the same `pair_id` and `pair_type` rows, then writes:

- `run_level_means.csv`: test-set mean for each metric in each run.
- `uncertainty_summary.csv`: mean of run-level means, sample standard deviation across runs, and the requested two-sided Student-t confidence interval.
- `uncertainty_metadata.json`: the run list and statistical procedure.

The confidence interval describes uncertainty across the supplied independent run-level means. It is not valid to use repeated evaluations of the same fixed checkpoint as independent training runs. Externally reported baseline values cannot be assigned run-level uncertainty unless the corresponding methods are independently reproduced.
