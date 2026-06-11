#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Plot-input CSVs are tracked next to this script in notebooks/paper_results/.
_CSV_DIR = Path(__file__).resolve().parent / "paper_results"
DEFAULT_AI_RUNS_CSV = _CSV_DIR / "wandb_runs_cedric-leonard_MAYA_DC.csv"
DEFAULT_BASELINE_SUMMARY_CSVS = (
    _CSV_DIR / "jpeg_baseline_rcmc_channels_slc_summary.csv",
    _CSV_DIR / "webp_baseline_rcmc_channels_slc_summary.csv",
    _CSV_DIR / "jpeg2000_baseline_rcmc_channels_slc_summary.csv",
)
DEFAULT_METRICS = ("ssim_amp", "psnr_amp")

PRETTY_METRIC_NAMES = {
    "rate": "Bit-rate [bpp]",
    "bpp": "Bit-rate [bpp]",
    "ssim_amp": "SSIM amplitude",
    "psnr_amp": "PSNR amplitude [dB]",
}

DISPLAY_ORDER = ["AI sq", "AI nsq", "JPEG", "WebP", "JPEG2000"]
DISPLAY_STYLES = {
    "AI sq": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.2,
        "annot_offset": (5, 5),
    },
    "AI nsq": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2.2,
        "annot_offset": (5, -12),
    },
    "JPEG": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.9,
        "annot_offset": (5, 5),
    },
    "WebP": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 1.9,
        "annot_offset": (5, -12),
    },
    "JPEG2000": {
        "color": "#9467bd",
        "marker": "v",
        "linestyle": "--",
        "linewidth": 1.9,
        "annot_offset": (5, 5),
    },
}

AI_FILTERS = {
    "seed_min": 0,
    "seed_max": 10,
    "blocked_tags": ("debug", "crashed", "bart"),
    "training_mode": "slc",
    "model_short": "SHyp",
    "activation_short": "GDN",
    "dataset_short": "MAYA4DirDataModule",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot AI sq/nsq RD curves together with classical codec baselines "
            "for SSIM and/or PSNR, with point labels and a variance report."
        )
    )
    parser.add_argument(
        "--ai-runs-csv",
        type=Path,
        default=DEFAULT_AI_RUNS_CSV,
        help="Cached WandB runs CSV used to aggregate the AI RD curves.",
    )
    parser.add_argument(
        "--baseline-summary-csv",
        type=Path,
        action="append",
        default=None,
        help="Classical codec summary CSV. Repeat to add multiple files.",
    )
    parser.add_argument(
        "--metric",
        choices=DEFAULT_METRICS,
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="One or more quality metrics to plot.",
    )
    parser.add_argument(
        "--variance-mode",
        choices=("none", "sem", "std", "minmax"),
        default="sem",
        help=(
            "How to visualize spread. 'sem' is the default because it reflects uncertainty "
            "of the mean more consistently across AI seeds and codec patches."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Directory that will receive the plots, combined CSVs, and metadata.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="RD Curves: AI sq/nsq + Classical Codecs",
        help="Prefix for each plot title. The metric name is appended automatically.",
    )
    parser.add_argument(
        "--no-annotate-points",
        action="store_true",
        help="Disable point labels. Labels are enabled by default.",
    )
    parser.add_argument(
        "--copy-stable-files",
        action="store_true",
        help="Copy the generated plots and CSVs to stable top-level filenames in the workspace.",
    )
    return parser.parse_args()


def pretty_metric_name(metric_key: str) -> str:
    return PRETTY_METRIC_NAMES.get(metric_key, metric_key)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    as_float = float(value)
    rounded = round(as_float)
    return str(int(rounded)) if abs(as_float - rounded) < 1e-9 else f"{as_float:g}"


def load_json_like(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if not text:
        return default
    return json.loads(text)


def ordered_labels(labels: list[str]) -> list[str]:
    order_index = {label: idx for idx, label in enumerate(DISPLAY_ORDER)}
    return sorted(labels, key=lambda label: (order_index.get(label, len(order_index)), label))


def compute_basic_stats(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    count = int(len(values))
    if count == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "min": math.nan,
            "max": math.nan,
        }

    std = float(values.std(ddof=0))
    sem = float(std / math.sqrt(count))
    return {
        "count": count,
        "mean": float(values.mean()),
        "std": std,
        "sem": sem,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def metric_key(metric_name: str) -> str:
    return f"test/{metric_name}"


def load_ai_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    runs["summary"] = runs["summary"].apply(lambda value: load_json_like(value, {}))
    runs["tags"] = runs["tags"].apply(lambda value: load_json_like(value, []))
    runs["seed"] = pd.to_numeric(runs["seed"], errors="coerce")
    runs["lmbda"] = pd.to_numeric(runs["lmbda"], errors="coerce")
    runs["created_at"] = pd.to_datetime(runs["created_at"], utc=True, errors="coerce")
    return runs


def filter_and_dedupe_ai_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = runs.copy()
    before_count = len(work)

    blocked_tags = AI_FILTERS["blocked_tags"]
    seed_mask = work["seed"].between(
        AI_FILTERS["seed_min"], AI_FILTERS["seed_max"], inclusive="both"
    )
    blocked_mask = work["tags"].apply(
        lambda tags: any(
            blocked in str(tag).lower() for tag in (tags or []) for blocked in blocked_tags
        )
    )
    work = work.loc[seed_mask & ~blocked_mask].copy()
    work = work.loc[
        (work["training_mode"] == AI_FILTERS["training_mode"])
        & (work["model_short"] == AI_FILTERS["model_short"])
        & (work["activation_short"] == AI_FILTERS["activation_short"])
        & (work["dataset_short"] == AI_FILTERS["dataset_short"])
    ].copy()

    after_filter_count = len(work)
    dedupe_keys = ["group_label", "lmbda", "seed"]
    duplicate_rows = int(work.duplicated(dedupe_keys, keep=False).sum())
    work = work.reset_index(drop=True)

    report = {
        "rows_before_filter": before_count,
        "rows_after_filter": after_filter_count,
        "duplicate_logical_runs_detected": duplicate_rows,
        "group_labels": ordered_labels(work["group_label"].dropna().unique().tolist()),
        "rows_per_group": work.groupby("group_label").size().sort_index().to_dict(),
    }
    return work, report


def ai_display_label(row: pd.Series) -> str:
    return f"AI {'nsq' if bool(row['non_square_kernels']) else 'sq'}"


def aggregate_ai_stats(runs: pd.DataFrame, quality_metric: str) -> pd.DataFrame:
    quality_key = metric_key(quality_metric)
    rate_key = metric_key("rate")
    work = runs.copy()
    work["has_required_metrics"] = work["summary"].apply(
        lambda summary: rate_key in summary and quality_key in summary
    )
    work = work.loc[work["has_required_metrics"]].copy()
    work.sort_values(["group_label", "lmbda", "seed", "created_at"], inplace=True)
    work = work.drop_duplicates(["group_label", "lmbda", "seed"], keep="last").reset_index(
        drop=True
    )

    records: list[dict[str, Any]] = []
    for _, group_df in work.groupby(["group_label", "non_square_kernels"], sort=False):
        group_df = group_df.copy()
        display_label = ai_display_label(group_df.iloc[0])
        for lmbda, lambda_df in group_df.groupby("lmbda", sort=True):
            rate_values = lambda_df["summary"].apply(lambda summary: summary.get(rate_key))
            quality_values = lambda_df["summary"].apply(lambda summary: summary.get(quality_key))
            rate_stats = compute_basic_stats(rate_values)
            quality_stats = compute_basic_stats(quality_values)
            if rate_stats["count"] == 0 or quality_stats["count"] == 0:
                continue

            records.append(
                {
                    "curve_source": "ai_model",
                    "group_label": group_df.iloc[0]["group_label"],
                    "display_label": display_label,
                    "point_label": f"L={format_value(lmbda)}",
                    "codec_name": pd.NA,
                    "non_square_kernels": group_df.iloc[0]["non_square_kernels"],
                    "lmbda": float(lmbda),
                    "sample_count": int(len(lambda_df)),
                    "num_seed_values": int(lambda_df["seed"].nunique(dropna=True)),
                    "rate_mean": rate_stats["mean"],
                    "rate_std": rate_stats["std"],
                    "rate_sem": rate_stats["sem"],
                    "rate_min": rate_stats["min"],
                    "rate_max": rate_stats["max"],
                    "quality_mean": quality_stats["mean"],
                    "quality_std": quality_stats["std"],
                    "quality_sem": quality_stats["sem"],
                    "quality_min": quality_stats["min"],
                    "quality_max": quality_stats["max"],
                }
            )

    result = pd.DataFrame.from_records(records)
    return result.sort_values(["display_label", "lmbda"], kind="stable").reset_index(drop=True)


def infer_per_patch_csv(summary_csv: Path) -> Path:
    return summary_csv.with_name(f"{summary_csv.stem}_per_patch{summary_csv.suffix}")


def aggregate_baseline_from_raw(summary_csv: Path, quality_metric: str) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        return summary_df

    raw_csv = infer_per_patch_csv(summary_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Expected per-patch baseline CSV next to {summary_csv}: {raw_csv}"
        )

    raw_df = pd.read_csv(raw_csv)
    quality_column = quality_metric

    records: list[dict[str, Any]] = []
    for quality, quality_df in raw_df.groupby("quality", sort=True):
        rate_stats = compute_basic_stats(quality_df["bpp"])
        quality_stats = compute_basic_stats(quality_df[quality_column])
        summary_row = summary_df.loc[summary_df["lmbda"] == quality]
        if summary_row.empty:
            raise KeyError(
                f"Could not match quality={quality} between {summary_csv} and {raw_csv}"
            )
        summary_row = summary_row.iloc[0]

        records.append(
            {
                "curve_source": "classical_codec",
                "group_label": summary_row["group_label"],
                "display_label": str(summary_row["codec_name"]),
                "point_label": f"Q={format_value(quality)}",
                "codec_name": summary_row["codec_name"],
                "non_square_kernels": pd.NA,
                "lmbda": float(quality),
                "sample_count": rate_stats["count"],
                "num_seed_values": pd.NA,
                "rate_mean": rate_stats["mean"],
                "rate_std": rate_stats["std"],
                "rate_sem": rate_stats["sem"],
                "rate_min": rate_stats["min"],
                "rate_max": rate_stats["max"],
                "quality_mean": quality_stats["mean"],
                "quality_std": quality_stats["std"],
                "quality_sem": quality_stats["sem"],
                "quality_min": quality_stats["min"],
                "quality_max": quality_stats["max"],
                "baseline_summary_csv": str(summary_csv.resolve()),
                "baseline_raw_csv": str(raw_csv.resolve()),
            }
        )

    result = pd.DataFrame.from_records(records)
    return result.sort_values(["display_label", "lmbda"], kind="stable").reset_index(drop=True)


def build_combined_stats(ai_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([ai_df, baseline_df], ignore_index=True, sort=False)
    combined.sort_values(["display_label", "rate_mean", "lmbda"], inplace=True, kind="stable")
    return combined.reset_index(drop=True)


def uncertainty_columns(variance_mode: str) -> tuple[str | None, str | None]:
    if variance_mode == "std":
        return "rate_std", "quality_std"
    if variance_mode == "sem":
        return "rate_sem", "quality_sem"
    return None, None


def plot_rd_curve(
    stats_df: pd.DataFrame,
    quality_metric: str,
    title: str,
    annotate_points: bool,
    variance_mode: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    xerr_column, yerr_column = uncertainty_columns(variance_mode)

    for label in ordered_labels(stats_df["display_label"].dropna().unique().tolist()):
        group_df = stats_df.loc[stats_df["display_label"] == label].sort_values("rate_mean")
        style = DISPLAY_STYLES.get(label, {})

        ax.plot(
            group_df["rate_mean"],
            group_df["quality_mean"],
            label=label,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.8),
            markersize=6,
        )

        if variance_mode == "minmax":
            ax.fill_between(
                group_df["rate_mean"],
                group_df["quality_min"],
                group_df["quality_max"],
                color=style.get("color"),
                alpha=0.12,
            )
        elif xerr_column is not None and yerr_column is not None:
            ax.errorbar(
                group_df["rate_mean"],
                group_df["quality_mean"],
                xerr=group_df[xerr_column],
                yerr=group_df[yerr_column],
                fmt="none",
                ecolor=style.get("color"),
                elinewidth=1.0,
                capsize=2,
                alpha=0.9,
            )

        if annotate_points:
            offset = style.get("annot_offset", (5, 5))
            for _, point in group_df.iterrows():
                ax.annotate(
                    str(point["point_label"]),
                    (point["rate_mean"], point["quality_mean"]),
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=8,
                    color=style.get("color"),
                )

    ax.set_title(title)
    ax.set_xlabel(pretty_metric_name("rate"))
    ax.set_ylabel(pretty_metric_name(quality_metric))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def verify_combined_stats(stats_df: pd.DataFrame) -> dict[str, Any]:
    expected_labels = {"AI sq", "AI nsq", "JPEG", "WebP", "JPEG2000"}
    labels = set(stats_df["display_label"].dropna().tolist())
    missing_labels = sorted(expected_labels.difference(labels))
    null_rate = int(stats_df["rate_mean"].isna().sum())
    null_quality = int(stats_df["quality_mean"].isna().sum())

    return {
        "num_rows": int(len(stats_df)),
        "display_labels": ordered_labels(sorted(labels)),
        "rows_per_label": stats_df.groupby("display_label").size().sort_index().to_dict(),
        "missing_expected_labels": missing_labels,
        "null_rate_points": null_rate,
        "null_quality_points": null_quality,
        "passed": not missing_labels and null_rate == 0 and null_quality == 0,
    }


def build_variance_report(
    ai_runs: pd.DataFrame,
    ai_dedupe_report: dict[str, Any],
    combined_by_metric: dict[str, pd.DataFrame],
    variance_mode: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "variance_mode_used_for_plots": variance_mode,
        "ai_deduplication": ai_dedupe_report,
        "observations": [],
        "metrics": {},
    }

    report["observations"].append(
        "The old codec spread looked too large because codec summaries were aggregated over per-patch samples "
        "while the AI curves were aggregated over seed-level runs."
    )
    report["observations"].append(
        "The previous plot also used wide quality bands derived from patch-level min/max or std, which measures "
        "content heterogeneity, not uncertainty of the mean RD point."
    )
    report["observations"].append(
        "These plots use the selected variance mode consistently across AI and codec curves. The default is SEM."
    )

    for metric_name, df in combined_by_metric.items():
        metric_report = {
            "rows_per_label": df.groupby("display_label").size().sort_index().to_dict(),
            "sample_count_range_per_label": (
                df.groupby("display_label")["sample_count"]
                .agg(["min", "max"])
                .to_dict(orient="index")
            ),
            "mean_quality_std_per_label": df.groupby("display_label")["quality_std"]
            .mean()
            .to_dict(),
            "mean_quality_sem_per_label": df.groupby("display_label")["quality_sem"]
            .mean()
            .to_dict(),
        }

        codec_rows = df.loc[df["curve_source"] == "classical_codec"].copy()
        if not codec_rows.empty:
            top_spread = codec_rows.sort_values("quality_std", ascending=False).head(5)
            metric_report["largest_codec_spreads"] = top_spread[
                [
                    "display_label",
                    "lmbda",
                    "sample_count",
                    "quality_std",
                    "quality_sem",
                    "quality_min",
                    "quality_max",
                ]
            ].to_dict(orient="records")

        report["metrics"][metric_name] = metric_report

    return report


def save_metric_outputs(
    output_dir: Path,
    metric_name: str,
    combined_df: pd.DataFrame,
    title_prefix: str,
    annotate_points: bool,
    variance_mode: str,
    copy_stable_files: bool,
) -> dict[str, str]:
    csv_path = output_dir / f"combined_rd_stats_{metric_name}.csv"
    png_path = output_dir / f"combined_rd_curve_{metric_name}.png"
    meta_path = output_dir / f"combined_rd_curve_{metric_name}_meta.json"

    combined_df.to_csv(csv_path, index=False)
    fig, _ = plot_rd_curve(
        combined_df,
        quality_metric=metric_name,
        title=f"{title_prefix} ({pretty_metric_name(metric_name)})",
        annotate_points=annotate_points,
        variance_mode=variance_mode,
    )
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    meta_path.write_text(
        json.dumps(
            {
                "metric": metric_name,
                "variance_mode": variance_mode,
                "annotate_points": annotate_points,
                "output_csv": str(csv_path),
                "output_png": str(png_path),
            },
            indent=2,
        )
    )

    if copy_stable_files:
        shutil.copy2(csv_path, Path(f"combined_rd_stats_{metric_name}.csv"))
        shutil.copy2(png_path, Path(f"combined_rd_curve_{metric_name}.png"))

    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "meta": str(meta_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_runs = load_ai_runs(args.ai_runs_csv.resolve())
    filtered_ai_runs, ai_dedupe_report = filter_and_dedupe_ai_runs(ai_runs)

    baseline_paths = args.baseline_summary_csv or list(DEFAULT_BASELINE_SUMMARY_CSVS)
    baseline_paths = [path.resolve() for path in baseline_paths]

    outputs: dict[str, dict[str, str]] = {}
    combined_by_metric: dict[str, pd.DataFrame] = {}
    verification_by_metric: dict[str, dict[str, Any]] = {}

    for metric_name in args.metric:
        ai_df = aggregate_ai_stats(filtered_ai_runs, metric_name)
        baseline_frames = [
            aggregate_baseline_from_raw(path, metric_name) for path in baseline_paths
        ]
        baseline_df = pd.concat(baseline_frames, ignore_index=True, sort=False)
        combined_df = build_combined_stats(ai_df, baseline_df)

        verification = verify_combined_stats(combined_df)
        if not verification["passed"]:
            raise RuntimeError(
                f"Combined RD stats failed verification for {metric_name}: {verification}"
            )

        outputs[metric_name] = save_metric_outputs(
            output_dir=output_dir,
            metric_name=metric_name,
            combined_df=combined_df,
            title_prefix=args.title_prefix,
            annotate_points=not args.no_annotate_points,
            variance_mode=args.variance_mode,
            copy_stable_files=args.copy_stable_files,
        )
        combined_by_metric[metric_name] = combined_df
        verification_by_metric[metric_name] = verification

    variance_report = build_variance_report(
        ai_runs=filtered_ai_runs,
        ai_dedupe_report=ai_dedupe_report,
        combined_by_metric=combined_by_metric,
        variance_mode=args.variance_mode,
    )
    variance_report_path = output_dir / "variance_analysis.json"
    variance_report_path.write_text(json.dumps(variance_report, indent=2))

    run_meta = {
        "ai_runs_csv": str(args.ai_runs_csv.resolve()),
        "baseline_summary_csvs": [str(path) for path in baseline_paths],
        "metrics": list(args.metric),
        "variance_mode": args.variance_mode,
        "annotate_points": not args.no_annotate_points,
        "outputs": outputs,
        "verification": verification_by_metric,
        "variance_analysis_json": str(variance_report_path),
    }
    run_meta_path = output_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2))

    print(f"Saved run metadata to {run_meta_path}")
    print(f"Saved variance analysis to {variance_report_path}")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
