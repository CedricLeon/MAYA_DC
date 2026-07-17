#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.maya4_dir_datamodule import MAYA4DirDataModule
from src.models.components.losses import (
    complex_correlation_metric,
    phase_preservation_metric,
    psnr_amplitude,
    ssim_amplitude,
)
from src.utils.sarpyx_azimuth_compression import full_azimuth_compress_batch


@dataclass(frozen=True)
class CodecSpec:
    name: str
    pil_format: str
    default_qualities: tuple[int, ...]


CODECS: dict[str, CodecSpec] = {
    "jpeg": CodecSpec("JPEG", "JPEG", (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)),
    "webp": CodecSpec("WebP", "WEBP", (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)),
    "jpeg2000": CodecSpec("JPEG2000", "JPEG2000", (1, 2, 4, 8, 16, 32, 64)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate classical image codecs as MAYA_DC RD baselines."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing MAYA4 *.zarr products.")
    parser.add_argument(
        "codec",
        choices=sorted(CODECS),
        help="Codec to run. Implemented with Pillow in this script.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to the aggregated summary CSV used by the RD notebook.",
    )
    parser.add_argument(
        "--qualities",
        type=str,
        default=None,
        help="Comma-separated quality settings. Uses codec defaults when omitted.",
    )
    parser.add_argument(
        "--representation",
        choices=("channels", "amplitude"),
        default="channels",
        help="Encode the 2 complex channels separately or encode amplitude only.",
    )
    parser.add_argument(
        "--eval-domain",
        choices=("rcmc", "slc"),
        default="slc",
        help=(
            "Domain on which metrics are computed. "
            "'slc' requires --representation channels because phase must be preserved."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        metavar=("AZ", "RG"),
        default=(512, 512),
        help="Core patch size used by the datamodule.",
    )
    parser.add_argument(
        "--azimuth-buffer",
        type=int,
        default=512,
        help="Azimuth buffer size used by the datamodule and rate denominator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Datamodule batch size. Small values are safer when eval-domain=slc.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Datamodule worker count.",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=-1,
        help="Maximum number of products to read from input_dir after ephemeris filtering.",
    )
    parser.add_argument(
        "--samples-per-prod",
        type=int,
        default=0,
        help="Number of patches per product. 0 means all patches.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional hard stop after N batches, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--dataset-short",
        type=str,
        default="MAYA4",
        help="Label stored in the summary CSV for plotting.",
    )
    parser.add_argument(
        "--group-label",
        type=str,
        default=None,
        help="Optional legend label override for the summary CSV.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Optional explicit path for the per-patch CSV. Defaults next to output_csv.",
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Optional explicit path for a small metadata JSON dump.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch.set_num_threads value for CPU-only runs.",
    )
    return parser.parse_args()


def parse_qualities(spec: CodecSpec, quality_arg: str | None) -> list[int]:
    if quality_arg is None:
        return list(spec.default_qualities)
    values = [int(part.strip()) for part in quality_arg.split(",") if part.strip()]
    if not values:
        raise ValueError("No quality settings were parsed from --qualities.")
    return values


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")


def default_group_label(codec: str, representation: str, eval_domain: str) -> str:
    rep = "2xGray" if representation == "channels" else "AmpGray"
    return f"{CODECS[codec].name} ({rep}, eval={eval_domain})"


def codec_save_kwargs(codec: str, quality: int) -> dict[str, Any]:
    if codec == "jpeg":
        return {"quality": int(quality)}
    if codec == "webp":
        return {"quality": float(quality), "method": 4}
    if codec == "jpeg2000":
        return {
            "quality_mode": "rates",
            "quality_layers": [float(quality)],
            "irreversible": True,
        }
    raise KeyError(f"Unsupported codec: {codec}")


def normalize_to_uint8(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(arr, dtype=np.float32)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if math.isclose(arr_min, arr_max):
        return np.zeros(arr.shape, dtype=np.uint8), arr_min, arr_max
    scaled = (arr - arr_min) / (arr_max - arr_min)
    quantized = np.clip(np.rint(scaled * 255.0), 0.0, 255.0).astype(np.uint8)
    return quantized, arr_min, arr_max


def denormalize_from_uint8(arr_uint8: np.ndarray, arr_min: float, arr_max: float) -> np.ndarray:
    arr_uint8 = np.asarray(arr_uint8, dtype=np.float32)
    if math.isclose(arr_min, arr_max):
        return np.full(arr_uint8.shape, fill_value=arr_min, dtype=np.float32)
    scaled = arr_uint8 / 255.0
    return (scaled * (arr_max - arr_min) + arr_min).astype(np.float32)


def encode_decode_grayscale(
    arr: np.ndarray, codec: str, quality: int
) -> tuple[np.ndarray, int, float, float]:
    arr_uint8, arr_min, arr_max = normalize_to_uint8(arr)
    image = Image.fromarray(arr_uint8, mode="L")
    tmp = io.BytesIO()

    start = time.perf_counter()
    image.save(tmp, format=CODECS[codec].pil_format, **codec_save_kwargs(codec, quality))
    enc_time = time.perf_counter() - start
    bits = tmp.getbuffer().nbytes * 8

    tmp.seek(0)
    start = time.perf_counter()
    rec_image = Image.open(tmp)
    rec_image.load()
    dec_time = time.perf_counter() - start

    rec_uint8 = np.asarray(rec_image, dtype=np.uint8)
    rec = denormalize_from_uint8(rec_uint8, arr_min, arr_max)
    return rec, bits, enc_time, dec_time


def amplitude_from_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x[:, 0] ** 2 + x[:, 1] ** 2, min=0.0))


def amplitude_psnr(
    pred_amp: torch.Tensor, target_amp: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    pred_amp = pred_amp.float()
    target_amp = target_amp.float()
    batch = pred_amp.shape[0]
    data_range = target_amp.reshape(batch, -1).amax(dim=1)
    mse = torch.mean((pred_amp - target_amp) ** 2, dim=(1, 2))
    return (10.0 * torch.log10(data_range**2 / (mse + eps))).mean()


def amplitude_ssim(
    pred_amp: torch.Tensor, target_amp: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    pred_amp = pred_amp.float().unsqueeze(1)
    target_amp = target_amp.float().unsqueeze(1)
    data_range = float(torch.clamp(target_amp.max() - target_amp.min(), min=eps).item())
    result = ssim_fn(pred_amp, target_amp, data_range=data_range, return_full_image=False)
    return result if isinstance(result, torch.Tensor) else result[0]


def encode_batch_channelwise(
    batch: torch.Tensor,
    codec: str,
    quality: int,
) -> tuple[torch.Tensor, list[int], list[float], list[float]]:
    batch_np = batch.detach().cpu().numpy()
    recon_np = np.empty_like(batch_np, dtype=np.float32)
    bits_per_sample: list[int] = []
    enc_times: list[float] = []
    dec_times: list[float] = []

    for sample_np in batch_np:
        sample_bits = 0
        sample_enc = 0.0
        sample_dec = 0.0
        recon_channels = []
        for channel_idx in range(sample_np.shape[0]):
            rec_channel, bits, enc_time, dec_time = encode_decode_grayscale(
                sample_np[channel_idx], codec, quality
            )
            recon_channels.append(rec_channel)
            sample_bits += bits
            sample_enc += enc_time
            sample_dec += dec_time
        bits_per_sample.append(sample_bits)
        enc_times.append(sample_enc)
        dec_times.append(sample_dec)
        recon_np[len(bits_per_sample) - 1] = np.stack(recon_channels, axis=0)

    recon = torch.from_numpy(recon_np).to(dtype=batch.dtype)
    return recon, bits_per_sample, enc_times, dec_times


def encode_batch_amplitude(
    batch: torch.Tensor,
    codec: str,
    quality: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[float], list[float]]:
    batch_amp = amplitude_from_channels(batch).detach().cpu().numpy()
    recon_amp_np = np.empty_like(batch_amp, dtype=np.float32)
    bits_per_sample: list[int] = []
    enc_times: list[float] = []
    dec_times: list[float] = []

    for idx, amp_np in enumerate(batch_amp):
        rec_amp, bits, enc_time, dec_time = encode_decode_grayscale(amp_np, codec, quality)
        recon_amp_np[idx] = rec_amp
        bits_per_sample.append(bits)
        enc_times.append(enc_time)
        dec_times.append(dec_time)

    target_amp = torch.from_numpy(batch_amp.astype(np.float32))
    recon_amp = torch.from_numpy(recon_amp_np)
    return recon_amp, target_amp, bits_per_sample, enc_times, dec_times


def core_slice(full_height: int, azimuth_buffer: int) -> slice:
    return slice(azimuth_buffer, full_height - azimuth_buffer)


def compute_complex_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    corr_mean, corr_std = complex_correlation_metric(pred, target)
    phase_mean, phase_std = phase_preservation_metric(pred, target)
    return {
        "psnr_amp": float(psnr_amplitude(pred, target).item()),
        "ssim_amp": float(ssim_amplitude(pred, target).item()),
        "complex_corr_mean": float(corr_mean.item()),
        "complex_corr_std": float(corr_std.item()),
        "phase_err_mean": float(phase_mean.item()),
        "phase_err_std": float(phase_std.item()),
    }


def compute_amplitude_metrics(
    pred_amp: torch.Tensor, target_amp: torch.Tensor
) -> dict[str, float]:
    return {
        "psnr_amp": float(amplitude_psnr(pred_amp, target_amp).item()),
        "ssim_amp": float(amplitude_ssim(pred_amp, target_amp).item()),
    }


def build_summary(
    raw_df: pd.DataFrame,
    codec: str,
    representation: str,
    eval_domain: str,
    dataset_short: str,
    group_label: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for quality, quality_df in raw_df.groupby("quality", sort=True):
        record: dict[str, Any] = {
            "codec": codec,
            "codec_name": CODECS[codec].name,
            "representation": representation,
            "eval_domain": eval_domain,
            "dataset_short": dataset_short,
            "group_label": group_label,
            "model_short": CODECS[codec].name,
            "activation_short": "2xGray" if representation == "channels" else "AmpGray",
            "training_mode": eval_domain,
            "lmbda": quality,
            "num_runs": int(len(quality_df)),
            "num_seed_values": pd.NA,
            "rate_mean": float(quality_df["bpp"].mean()),
            "rate_std": float(quality_df["bpp"].std(ddof=0)),
            "rate_min": float(quality_df["bpp"].min()),
            "rate_max": float(quality_df["bpp"].max()),
            "quality_mean": float(quality_df["psnr_amp"].mean()),
            "quality_std": float(quality_df["psnr_amp"].std(ddof=0)),
            "quality_min": float(quality_df["psnr_amp"].min()),
            "quality_max": float(quality_df["psnr_amp"].max()),
            "psnr_amp_mean": float(quality_df["psnr_amp"].mean()),
            "psnr_amp_std": float(quality_df["psnr_amp"].std(ddof=0)),
            "ssim_amp_mean": float(quality_df["ssim_amp"].mean()),
            "ssim_amp_std": float(quality_df["ssim_amp"].std(ddof=0)),
            "encoding_time_mean_s": float(quality_df["encoding_time_s"].mean()),
            "decoding_time_mean_s": float(quality_df["decoding_time_s"].mean()),
            "resolved_rate_metric": "bpp",
            "resolved_quality_metric": "psnr_amp",
            "curve_source": "classical_codec",
        }
        if "complex_corr_mean" in quality_df.columns:
            record["complex_corr_mean_mean"] = float(quality_df["complex_corr_mean"].mean())
            record["complex_corr_mean_std"] = float(quality_df["complex_corr_mean"].std(ddof=0))
        if "phase_err_mean" in quality_df.columns:
            record["phase_err_mean_mean"] = float(quality_df["phase_err_mean"].mean())
            record["phase_err_mean_std"] = float(quality_df["phase_err_mean"].std(ddof=0))
        records.append(record)

    summary_df = pd.DataFrame.from_records(records)
    return summary_df.sort_values("lmbda").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    spec = CODECS[args.codec]
    qualities = parse_qualities(spec, args.qualities)

    if args.eval_domain == "slc" and args.representation != "channels":
        raise ValueError("--eval-domain slc currently requires --representation channels.")

    torch.set_num_threads(max(1, int(args.torch_threads)))

    output_csv = args.output_csv.resolve()
    raw_csv = (
        args.raw_csv.resolve()
        if args.raw_csv is not None
        else output_csv.with_name(f"{output_csv.stem}_per_patch{output_csv.suffix}")
    )
    meta_json = (
        args.meta_json.resolve()
        if args.meta_json is not None
        else output_csv.with_name(f"{output_csv.stem}_meta.json")
    )

    ensure_output_path(output_csv, args.overwrite)
    ensure_output_path(raw_csv, args.overwrite)
    ensure_output_path(meta_json, args.overwrite)

    group_label = args.group_label or default_group_label(
        args.codec, args.representation, args.eval_domain
    )

    datamodule = MAYA4DirDataModule(
        train_dir=None,
        val_dir=None,
        test_dir=str(args.input_dir),
        patch_size=tuple(args.patch_size),
        azimuth_buffer=args.azimuth_buffer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        max_products_train=-1,
        max_products_val=-1,
        max_products_test=args.max_products,
        samples_per_prod=args.samples_per_prod,
        online=False,
    )
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    raw_records: list[dict[str, Any]] = []
    batches_processed = 0
    samples_processed = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break

        rcmc, slc, metadata_list, ephemeris_list, coords_list = batch
        full_height = int(rcmc.shape[2])
        full_width = int(rcmc.shape[3])
        core = core_slice(full_height, args.azimuth_buffer)

        for quality in qualities:
            if args.representation == "channels":
                recon_rcmc, bits_list, enc_times, dec_times = encode_batch_channelwise(
                    rcmc, args.codec, quality
                )
                if args.eval_domain == "slc":
                    with torch.no_grad():
                        recon_eval = full_azimuth_compress_batch(
                            recon_rcmc,
                            metadata_list,
                            ephemeris_list,
                            buffer_size=args.azimuth_buffer,
                            device="cpu",
                            coords_batch=coords_list,
                        )
                    target_eval = slc[:, :, core, :].cpu()
                else:
                    recon_eval = recon_rcmc[:, :, core, :].cpu()
                    target_eval = rcmc[:, :, core, :].cpu()

                for sample_offset in range(rcmc.shape[0]):
                    metrics = compute_complex_metrics(
                        recon_eval[sample_offset : sample_offset + 1],
                        target_eval[sample_offset : sample_offset + 1],
                    )
                    coords = coords_list[sample_offset]
                    raw_records.append(
                        {
                            "batch_idx": batch_idx,
                            "sample_idx": samples_processed + sample_offset,
                            "quality": quality,
                            "codec": args.codec,
                            "codec_name": spec.name,
                            "representation": args.representation,
                            "eval_domain": args.eval_domain,
                            "bpp": float(bits_list[sample_offset] / (full_height * full_width)),
                            "bits_total": int(bits_list[sample_offset]),
                            "encoding_time_s": float(enc_times[sample_offset]),
                            "decoding_time_s": float(dec_times[sample_offset]),
                            "zfile": coords["zfile"],
                            "y": int(coords["y"]),
                            "x": int(coords["x"]),
                            **metrics,
                        }
                    )
            else:
                recon_amp, target_amp, bits_list, enc_times, dec_times = encode_batch_amplitude(
                    rcmc[:, :, core, :], args.codec, quality
                )
                for sample_offset in range(rcmc.shape[0]):
                    metrics = compute_amplitude_metrics(
                        recon_amp[sample_offset : sample_offset + 1],
                        target_amp[sample_offset : sample_offset + 1],
                    )
                    coords = coords_list[sample_offset]
                    raw_records.append(
                        {
                            "batch_idx": batch_idx,
                            "sample_idx": samples_processed + sample_offset,
                            "quality": quality,
                            "codec": args.codec,
                            "codec_name": spec.name,
                            "representation": args.representation,
                            "eval_domain": args.eval_domain,
                            "bpp": float(bits_list[sample_offset] / (full_height * full_width)),
                            "bits_total": int(bits_list[sample_offset]),
                            "encoding_time_s": float(enc_times[sample_offset]),
                            "decoding_time_s": float(dec_times[sample_offset]),
                            "zfile": coords["zfile"],
                            "y": int(coords["y"]),
                            "x": int(coords["x"]),
                            **metrics,
                        }
                    )

        batches_processed += 1
        samples_processed += int(rcmc.shape[0])
        print(
            f"Processed batch {batch_idx + 1} | samples so far: {samples_processed} | "
            f"qualities: {len(qualities)}"
        )

    raw_df = pd.DataFrame.from_records(raw_records)
    if raw_df.empty:
        raise RuntimeError(
            "No records were produced. Check the input directory and sampling args."
        )

    summary_df = build_summary(
        raw_df,
        codec=args.codec,
        representation=args.representation,
        eval_domain=args.eval_domain,
        dataset_short=args.dataset_short,
        group_label=group_label,
    )

    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(output_csv, index=False)

    meta = {
        "input_dir": str(args.input_dir.resolve()),
        "codec": args.codec,
        "codec_name": spec.name,
        "qualities": qualities,
        "representation": args.representation,
        "eval_domain": args.eval_domain,
        "patch_size": list(args.patch_size),
        "azimuth_buffer": args.azimuth_buffer,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_products": args.max_products,
        "samples_per_prod": args.samples_per_prod,
        "limit_batches": args.limit_batches,
        "group_label": group_label,
        "batches_processed": batches_processed,
        "samples_processed": samples_processed,
        "raw_csv": str(raw_csv),
        "summary_csv": str(output_csv),
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    print(f"Saved per-patch results to {raw_csv}")
    print(f"Saved summary results to {output_csv}")
    print(f"Saved metadata to {meta_json}")


if __name__ == "__main__":
    main()
