#!/usr/bin/env python3
"""Evaluate paired cover/stego and secret/recovered RGB images.

The script intentionally does not resize images. Preprocessing and target-resolution
selection must be completed before evaluation, and all paired images must have equal
spatial dimensions. See README.md for the exact metric definitions.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
import open_clip
import lpips
import skimage
from skimage.metrics import structural_similarity

REQUIRED_COLUMNS = [
    "pair_id",
    "cover_path",
    "secret_path",
    "stego_path",
    "recovered_secret_path",
]
METRICS = ["psnr_y_db", "ssim_rgb", "mae_rgb", "clip_similarity", "lpips"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(root: Path, relative_path: str) -> Path:
    p = Path(str(relative_path))
    return p if p.is_absolute() else root / p


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    return np.clip(arr, 0.0, 255.0)


def rgb_to_y_bt601(rgb: np.ndarray) -> np.ndarray:
    """Return the Y' (luma) channel using the BT.601 RGB coefficients."""
    x = np.asarray(rgb, dtype=np.float32)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def psnr_y(reference: np.ndarray, output: np.ndarray) -> float:
    y_ref = rgb_to_y_bt601(reference)
    y_out = rgb_to_y_bt601(output)
    mse = float(np.mean(np.square(y_ref - y_out), dtype=np.float64))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def ssim_rgb(reference: np.ndarray, output: np.ndarray) -> float:
    if min(reference.shape[0], reference.shape[1]) < 11:
        raise ValueError(
            f"SSIM requires images at least 11x11 with win_size=11; got {reference.shape[1]}x{reference.shape[0]}"
        )
    return float(
        structural_similarity(
            reference.astype(np.uint8),
            output.astype(np.uint8),
            data_range=255,
            channel_axis=-1,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def mae_rgb(reference: np.ndarray, output: np.ndarray) -> float:
    return float(np.mean(np.abs(reference - output), dtype=np.float64))


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def check_same_shape(reference: np.ndarray, output: np.ndarray, pair_id: str, pair_type: str) -> None:
    if reference.shape != output.shape:
        raise ValueError(
            f"Shape mismatch for pair_id={pair_id}, pair_type={pair_type}: "
            f"reference={reference.shape}, output={output.shape}. "
            "This evaluator does not resize images; correct preprocessing upstream."
        )


def clip_batch_similarity(ref_images, out_images, preprocess, clip_model, device) -> np.ndarray:
    """Compute cosine similarity of normalized CLIP image embeddings."""
    ref_tensors = [preprocess(Image.fromarray(to_uint8(x))) for x in ref_images]
    out_tensors = [preprocess(Image.fromarray(to_uint8(x))) for x in out_images]
    batch = torch.stack(ref_tensors + out_tensors, dim=0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n = len(ref_images)
    sims = (features[:n] * features[n:]).sum(dim=-1)
    return sims.detach().cpu().numpy().astype(float)


def lpips_batch_distance(ref_images, out_images, lpips_model, device) -> np.ndarray:
    """Compute LPIPS for RGB images represented in [-1, 1]."""
    ref = np.stack(ref_images, axis=0).astype(np.float32) / 127.5 - 1.0
    out = np.stack(out_images, axis=0).astype(np.float32) / 127.5 - 1.0
    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).contiguous().to(device)
    out_t = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().to(device)
    with torch.no_grad():
        distances = lpips_model(ref_t, out_t, normalize=False)
    return distances.reshape(distances.shape[0], -1).mean(dim=1).detach().cpu().numpy().astype(float)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest contains no rows.")
    if (df["pair_id"].str.strip() == "").any():
        raise ValueError("Every manifest row must have a non-empty pair_id.")
    if df["pair_id"].duplicated().any():
        duplicates = df.loc[df["pair_id"].duplicated(), "pair_id"].tolist()
        raise ValueError(f"pair_id values must be unique. Duplicates: {duplicates[:10]}")
    for col in REQUIRED_COLUMNS[1:]:
        if (df[col].str.strip() == "").any():
            raise ValueError(f"Manifest column '{col}' contains empty paths.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest; see the Manifest format section in README.md")
    parser.add_argument("--reference-root", type=Path, default=Path("."), help="Root for cover_path and secret_path")
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Root for stego_path and recovered_secret_path")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for per-image and summary outputs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device for CLIP and LPIPS")
    parser.add_argument("--clip-model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for CLIP and LPIPS")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = read_manifest(manifest_path)
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model = clip_model.to(device).eval()
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    records = []
    pair_ids = df["pair_id"].tolist()
    for start in range(0, len(df), args.batch_size):
        batch_df = df.iloc[start : start + args.batch_size]
        cover_refs, stego_outs, secret_refs, recovered_outs = [], [], [], []
        batch_ids = batch_df["pair_id"].tolist()
        paths_for_batch = []

        for _, row in batch_df.iterrows():
            paths = {
                "cover": resolve_path(reference_root, row["cover_path"]),
                "secret": resolve_path(reference_root, row["secret_path"]),
                "stego": resolve_path(output_root, row["stego_path"]),
                "recovered": resolve_path(output_root, row["recovered_secret_path"]),
            }
            imgs = {k: load_rgb(v) for k, v in paths.items()}
            check_same_shape(imgs["cover"], imgs["stego"], row["pair_id"], "cover_stego")
            check_same_shape(imgs["secret"], imgs["recovered"], row["pair_id"], "secret_recovered")
            if imgs["cover"].shape != imgs["secret"].shape:
                raise ValueError(
                    f"Cover and secret sizes differ for pair_id={row['pair_id']}: "
                    f"cover={imgs['cover'].shape}, secret={imgs['secret'].shape}"
                )
            cover_refs.append(imgs["cover"])
            stego_outs.append(imgs["stego"])
            secret_refs.append(imgs["secret"])
            recovered_outs.append(imgs["recovered"])
            paths_for_batch.append(paths)

        clip_cs = clip_batch_similarity(cover_refs, stego_outs, clip_preprocess, clip_model, device)
        clip_sr = clip_batch_similarity(secret_refs, recovered_outs, clip_preprocess, clip_model, device)
        lpips_cs = lpips_batch_distance(cover_refs, stego_outs, lpips_model, device)
        lpips_sr = lpips_batch_distance(secret_refs, recovered_outs, lpips_model, device)

        for i, pair_id in enumerate(batch_ids):
            cover, stego = cover_refs[i], stego_outs[i]
            secret, recovered = secret_refs[i], recovered_outs[i]
            h, w = cover.shape[:2]
            records.append({
                "pair_id": pair_id,
                "pair_type": "cover_stego",
                "width": w,
                "height": h,
                "psnr_y_db": psnr_y(cover, stego),
                "ssim_rgb": ssim_rgb(cover, stego),
                "mae_rgb": mae_rgb(cover, stego),
                "clip_similarity": float(clip_cs[i]),
                "lpips": float(lpips_cs[i]),
            })
            records.append({
                "pair_id": pair_id,
                "pair_type": "secret_recovered",
                "width": w,
                "height": h,
                "psnr_y_db": psnr_y(secret, recovered),
                "ssim_rgb": ssim_rgb(secret, recovered),
                "mae_rgb": mae_rgb(secret, recovered),
                "clip_similarity": float(clip_sr[i]),
                "lpips": float(lpips_sr[i]),
            })
        print(f"Evaluated {min(start + len(batch_df), len(df))}/{len(df)} image pairs", flush=True)

    per_image = pd.DataFrame(records)
    per_image_path = out_dir / "per_image_metrics.csv"
    per_image.to_csv(per_image_path, index=False, float_format="%.10g")

    summary_rows = []
    for pair_type, group in per_image.groupby("pair_type", sort=True):
        for metric in METRICS:
            vals = group[metric].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            mean = float(np.mean(vals)) if len(finite) == len(vals) else float("inf")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 and len(finite) == len(vals) else float("nan")
            summary_rows.append({
                "pair_type": pair_type,
                "metric": metric,
                "n_images": int(len(vals)),
                "mean_across_images": mean,
                "std_across_images": std,
                "note": "std_across_images describes image-to-image variation, not repeated-run uncertainty",
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False, float_format="%.10g")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_image_pairs": int(len(df)),
        "device": device,
        "metric_protocol": {
            "input_colour": "RGB",
            "input_range": "clipped to [0,255]",
            "psnr": "BT.601 Y' = 0.299R + 0.587G + 0.114B; data range 255; per-image PSNR then arithmetic mean across images",
            "ssim": "RGB; data_range=255; channel_axis=-1; win_size=11; gaussian_weights=True; sigma=1.5; use_sample_covariance=False; per-image then mean",
            "mae": "mean absolute error across all pixels and RGB channels; 8-bit intensity levels; per-image then mean",
            "clip_similarity": "cosine similarity of L2-normalized open_clip image embeddings",
            "lpips": "LPIPS on RGB tensors normalized to [-1,1]; reported backbone recorded below",
            "dataset_average": "arithmetic mean of per-image metric values; no pooled-pixel aggregation",
            "automatic_resize": False,
        },
        "models": {
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "lpips_net": args.lpips_net,
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "open_clip_torch": getattr(open_clip, "__version__", "unknown"),
            "lpips": getattr(lpips, "__version__", "unknown"),
            "scikit_image": skimage.__version__,
        },
        "outputs": ["per_image_metrics.csv", "summary_metrics.csv", "run_metadata.json"],
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {per_image_path}")
    print(f"Wrote: {out_dir / 'summary_metrics.csv'}")
    print(f"Wrote: {out_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
