from typing import Any, Dict, List, Union

import numpy as np


def print_images_statistics(
    images: Dict[str, Union[np.ndarray, None]],
    metrics: List[Union[str, Any]] = ["min", "max", "mean", "std"],
    title: str = "Image Statistics",
) -> None:
    """Print a formatted table of statistics for a dictionary of images.

    Args:
        images: Dictionary mapping image names to numpy arrays.
        metrics: List of metrics to compute. Can be strings ('min', 'max', 'mean', 'std')
                 or callables taking an array and returning a scalar.
    """
    if not images:
        print("No images provided for statistics.")
        return

    # 1. Determine Row Names and Column Widths
    # Max length of image names
    max_name_len = max(len(name) for name in images.keys())
    name_col_width = max(max_name_len, len("Image")) + 2  # +2 padding

    # Resolve metric functions and names
    metric_funcs = []
    metric_names = []
    for m in metrics:
        if isinstance(m, str):
            metric_names.append(m.capitalize())
            if m.lower() == "min":
                metric_funcs.append(np.min)
            elif m.lower() == "max":
                metric_funcs.append(np.max)
            elif m.lower() == "mean":
                metric_funcs.append(np.mean)
            elif m.lower() == "std":
                metric_funcs.append(np.std)
            else:
                metric_funcs.append(lambda x: float("nan"))
        elif callable(m):
            metric_names.append(m.__name__.capitalize())
            metric_funcs.append(m)
        else:
            metric_names.append(str(m))
            metric_funcs.append(lambda x: float("nan"))

    # Determine metric column widths (at least 10 chars, or name length)
    col_widths = [12] * len(metrics)  # Default width for floats like 1234.5678

    # Table Width
    total_width = name_col_width + sum(w + 3 for w in col_widths) + 1  # 3 for " | "

    print()
    print("-" * total_width)
    if title:
        print(f"{title:^{total_width}}")
        print("-" * total_width)

    # 2. Print Header
    header = f"{'Image':<{name_col_width}}"
    for name, w in zip(metric_names, col_widths):
        header += f" | {name:^{w}}"
    print(header)
    print("-" * total_width)

    # 3. Print Rows
    for img_name, img_data in images.items():
        row = f"{img_name:<{name_col_width}}"
        if img_data is not None:
            for func, w in zip(metric_funcs, col_widths):
                try:
                    val = func(img_data)
                    row += f" | {val:^{w}.4f}"
                except Exception:
                    row += f" | {'Err':^{w}}"
        else:
            # Handle None data
            for w in col_widths:
                row += f" | {'[Not Found]':^{w}}"
        print(row)
    print("-" * total_width)
    print()
