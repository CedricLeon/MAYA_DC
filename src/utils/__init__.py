"""Public utilities exported for convenient importing."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "RankedLogger": ("src.utils.pylogger", "RankedLogger"),
    "instantiate_callbacks": ("src.utils.template_utils", "instantiate_callbacks"),
    "instantiate_loggers": ("src.utils.template_utils", "instantiate_loggers"),
    "log_hyperparameters": ("src.utils.template_utils", "log_hyperparameters"),
    "print_config_tree": ("src.utils.template_utils", "print_config_tree"),
    "enforce_tags": ("src.utils.template_utils", "enforce_tags"),
    "early_wandb_initialization": ("src.utils.template_utils", "early_wandb_initialization"),
    "extras": ("src.utils.template_utils", "extras"),
    "get_metric_value": ("src.utils.template_utils", "get_metric_value"),
    "task_wrapper": ("src.utils.template_utils", "task_wrapper"),
    "compute_correlation": ("src.utils.processing_utils", "compute_correlation"),
    "compute_mag_correlation": ("src.utils.processing_utils", "compute_mag_correlation"),
    "load_partitioned_dataframe": ("src.utils.processing_utils", "load_partitioned_dataframe"),
    "log_info": ("src.utils.processing_utils", "log_info"),
    "log_step": ("src.utils.processing_utils", "log_step"),
    "scan_available_data_extent": ("src.utils.processing_utils", "scan_available_data_extent"),
    "find_optimal_step_size": ("src.utils.visualization_utils", "find_optimal_step_size"),
    "save_comparison_plot": ("src.utils.visualization_utils", "save_comparison_plot"),
    "save_image_subsampled": ("src.utils.visualization_utils", "save_image_subsampled"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "RankedLogger",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "enforce_tags",
    "early_wandb_initialization",
    "extras",
    "get_metric_value",
    "task_wrapper",
    "load_partitioned_dataframe",
    "scan_available_data_extent",
    "log_info",
    "log_step",
    "find_optimal_step_size",
    "save_image_subsampled",
    "save_comparison_plot",
]
