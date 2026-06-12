from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .processing_utils import log_info


def find_optimal_step_size(rows: int, cols: int, target_size: int) -> int:
    """Calculate a step size to subsample an array to approximately reach a size of target_size.

    Args:
        - rows (int): Number of rows in the original array.
        - cols (int): Number of columns in the original array.
        - target_size (int): Desired maximum size for the larger dimension after subsampling.
    Returns:
        - step size (int) to use for subsampling.
    """
    step_r = max(1, rows // target_size)
    step_c = max(1, cols // target_size)
    return max(step_r, step_c)


def save_image_subsampled(
    path: Path,
    data: np.ndarray,
    title: str | None = None,
    target_size: int | None = 500,
    patch_box: tuple[slice, slice] | None = None,
) -> None:
    """Saves an image downsampled to approx (target_size x target_size) with annotations. Rotates
    the image so Azimuth is X (top axis) and Range is Y (left axis, descending). Input data is
    expected to be (Azimuth, Range).

    Args:
        - path (Path): File path to save the image.
        - data (np.ndarray): 2D array of data to visualize (Azimuth x Range).
        - title (str | None): Optional title for the plot.
        - target_size (int | None): Target size for the larger dimension after subsampling. If None, no subsampling is done.
        - patch_box (tuple[slice, slice] | None): Optional box to highlight on the image, given as (az_slice, rg_slice).
    """
    assert data.ndim == 2, "Data must be 2D array"
    rows, cols = data.shape  # (Azimuth, Range)
    if target_size is None:
        target_size = max(rows, cols)
    step = find_optimal_step_size(rows, cols, target_size)
    subsampled = data[::step, ::step]

    # Transpose for visualization: Want Azimuth on X, Range on Y
    subsampled = subsampled.T

    # Determine value range
    vmin = float(np.percentile(subsampled, 1))
    vmax = float(np.percentile(subsampled, 99))

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Create figure with annotations
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # origin='upper': (0,0) at top-left. Y axis increases downwards.
    plt.imshow(subsampled, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")

    # Move X axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    if patch_box:
        import matplotlib.patches as patches

        # patch_box is (az_slice, rg_slice) i.e. (rows, cols) of original data
        az_sl, rg_sl = patch_box

        # Coordinates for plotting (after transpose)
        # X axis = Azimuth
        # Y axis = Range

        x_start = az_sl.start / step
        x_width = (az_sl.stop - az_sl.start) / step

        y_start = rg_sl.start / step
        y_height = (rg_sl.stop - rg_sl.start) / step

        rect = patches.Rectangle(
            (x_start, y_start), x_width, y_height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    if title:
        plt.title(f"{title}\nOriginal: {rows}x{cols} | Subsampled [::{step}]: {subsampled.shape}")
    else:
        plt.title(f"Original(Az,Rg): {rows}x{cols}", y=1.08)

    plt.xlabel(f"Azimuth (lines / {step})")  # X axis (Top)
    plt.ylabel(f"Range (samples / {step})")  # Y axis (Left)
    plt.colorbar(label="Magnitude", shrink=0.7)

    plt.tight_layout()
    plt.savefig(str(path))
    plt.close(fig)

    log_info(f"Saved {path} (Original: {rows}x{cols}, Plot(Rg,Az): {subsampled.shape})")


def save_comparison_plot(
    path: Path,
    images: list[np.ndarray],
    titles: list[str],
    main_title: str | None = None,
    target_size: int | None = None,
) -> None:
    """Saves a comparison plot of multiple images side-by-side. Rotates images so Azimuth is X
    (top) and Range is Y (left). Input data is expected to be (Azimuth, Range).

    Args:
        - path (Path): File path to save the comparison plot.
        - images (list[np.ndarray]): List of 2D arrays to visualize (Azimuth x Range).
        - titles (list[str]): List of titles for each image.
        - main_title (str | None): Optional main title for the entire figure.
        - target_size (int | None): Target size for the larger dimension after subsampling. If None, no subsampling is done.
    """
    n_imgs = len(images)
    if n_imgs == 0:
        raise ValueError("No images provided for comparison plot")
    if len(titles) != n_imgs:
        raise ValueError("Number of titles must match number of images")

    fig, axes = plt.subplots(1, n_imgs, figsize=(6 * n_imgs, 6))
    if n_imgs == 1:
        axes = [axes]

    step = 1
    # Assuming all images have similar shapes, calculate step from the first one if target_size is set
    if target_size is not None:
        step = find_optimal_step_size(images[0].shape[0], images[0].shape[1], target_size)

    for ax, img, title in zip(axes, images, titles):
        # Subsample
        subsampled = img[::step, ::step]
        # Transpose (Az, Rg) -> (Rg, Az) for plotting (Rows=Rg, Cols=Az)
        to_plot = subsampled.T

        vmin = float(np.percentile(to_plot, 1))
        vmax = float(np.percentile(to_plot, 99))

        # origin='upper': (0,0) at top-left. Y axis increases downwards.
        im = ax.imshow(
            to_plot, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper", aspect="auto"
        )
        ax.set_title(f"{title}\nOriginal: {img.shape}", y=1.08)

        # Axis setup
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(f"Azimuth (lines / {step})")
        ax.set_ylabel(f"Range (samples / {step})")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if main_title:
        plt.suptitle(main_title, y=0.98, fontsize=16)

    plt.tight_layout()
    plt.savefig(str(path))
    plt.close(fig)
    log_info(f"Saved comparison plot to {path}")
