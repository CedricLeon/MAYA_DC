from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "RD-curve_plots.ipynb"
DEFAULT_OUTPUT_PNG = PROJECT_ROOT / "notebooks" / "RD-curve_plots.png"

# Run only the RD-curve portion by default. The later reconstruction cells need
# local checkpoints and fixed-patch data that may not be present on every node.
DEFAULT_LAST_CODE_CELL_TO_EXECUTE = 11


def load_notebook_code_cells(notebook_path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(notebook_path.read_text())
    code_cells: list[tuple[int, str]] = []
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_cells.append((cell_index, "".join(cell["source"])))
    return code_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute RD-curve_plots.ipynb as a script.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=NOTEBOOK_PATH,
        help="Notebook to execute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Base output path for saved figures.",
    )
    parser.add_argument(
        "--last-code-cell",
        type=int,
        default=DEFAULT_LAST_CODE_CELL_TO_EXECUTE,
        help="Last code cell index to execute.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return value.lower() or "figure"


def figure_title(fig: plt.Figure, fallback: str) -> str:
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            return title
    return fallback


def save_all_figures(base_output_path: Path) -> list[Path]:
    figure_numbers = plt.get_fignums()
    if not figure_numbers:
        raise RuntimeError("No matplotlib figures were created by the executed notebook cells.")

    base_output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    base_stem = base_output_path.stem
    suffix = base_output_path.suffix or ".png"

    for ordinal, figure_number in enumerate(figure_numbers, start=1):
        fig = plt.figure(figure_number)
        title_slug = slugify(figure_title(fig, f"figure_{ordinal:02d}"))
        numbered_path = base_output_path.with_name(f"{base_stem}_{ordinal:02d}_{title_slug}{suffix}")
        fig.savefig(numbered_path, dpi=200, bbox_inches="tight")
        saved_paths.append(numbered_path)

    # Keep the previous single-file behavior by mirroring the first figure to the
    # unsuffixed path. This also gives downstream tooling a stable filename.
    base_output_path.write_bytes(saved_paths[0].read_bytes())
    return saved_paths


def main() -> None:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    output_path = args.output.resolve()
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(notebook_path)}
    code_cells = load_notebook_code_cells(notebook_path)

    for cell_index, cell_source in code_cells:
        if cell_index > args.last_code_cell:
            break
        print(f"Executing notebook code cell {cell_index}...")
        exec(compile(cell_source, f"{notebook_path}::cell_{cell_index}", "exec"), namespace)

    saved_paths = save_all_figures(output_path)
    print(f"Saved {len(saved_paths)} figure(s):")
    for path in saved_paths:
        print(f"  - {path}")
    print(f"Updated stable output path: {output_path}")


if __name__ == "__main__":
    main()
