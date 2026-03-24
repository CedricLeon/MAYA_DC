"""MonitorFixedPatch -- F14 callback.

Runs the NIC model on a pre-selected fixed RCMC+SLC patch every N validation
epochs and logs a comparison figure to WandB.  At test end it saves a PNG
(log-intensity, quick look) and a NPY file (linear amplitude) for each
reconstructed domain, making cross-run comparisons easy.

Patch selection workflow
------------------------
1. Run ``notebooks/cherry_pick_patch.ipynb`` to scan all local zarr products,
   pick a region of interest, and save coordinates to a JSON file under
   ``data/fixed_patches/``.  The notebook checks download completeness,
   flags products with missing ephemeris, and lets you zoom into a candidate
   area before committing.
2. Point ``patch_json`` to that file -- it records the product path and the
   core patch extents (``az_start``, ``az_end``, ``rg_start``, ``rg_end``).
3. This callback adds the azimuth buffer (taken from ``pl_module.hparams``) at
   load time, so the same JSON works across buffer-size ablations.

Note on ephemeris
-----------------
The SLC reconstruction panel requires valid ephemeris.  The notebook flags
products with empty ephemeris.  If you cherry-pick such a product the callback
will only show RCMC panels; SLC recon is silently skipped.  Select a product
from PT1/PT4 with valid ephemeris for full 4-panel output.

Figure layout (logged to ``fixed_patch/reconstruction`` in WandB):

    [RCMC input] | [RCMC recon] | [SLC recon *] | [SLC target]

    * SLC recon only shown in ``"slc"`` training mode with valid ephemeris.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
import wandb
import zarr
from lightning import Callback, LightningModule, Trainer
from maya4 import RC_MAX, RC_MIN, minmax_inverse

from src.utils.processing_utils import (
    clip_mean_std_numpy,
    correct_swst_range_offset,
    phys_to_logI_np,
)
from src.utils.sarpyx_azimuth_compression import full_azimuth_compress_batch

# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _load_complex_patch(
    arr: zarr.Array,
    az_sl: slice,
    rg_sl: slice,
    norm_constants: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load a complex zarr patch as a ``(2, H, W)`` float32 array.

    Real and imaginary channels are stacked along axis 0.  If
    ``norm_constants=(vmin, vmax)`` is given, both channels are minmax-
    normalised: ``(x - vmin) / (vmax - vmin)``.
    """
    data = np.array(arr[az_sl, rg_sl], dtype=np.complex128)
    stack = np.stack([np.real(data), np.imag(data)], axis=0).astype(np.float32)
    if norm_constants is not None:
        vmin, vmax = norm_constants
        stack = (stack - vmin) / (vmax - vmin)
    return stack


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class MonitorFixedPatch(Callback):
    """Log reconstruction of a cherry-picked fixed patch to WandB every N epochs (F14).

    See module docstring for the patch selection workflow.

    Args:
        patch_json: Path to the patch JSON, relative to the project root or
            absolute.  E.g. ``"data/fixed_patches/fogo_s6_vh_20240502.json"``.
        log_every_n_epochs: How often to run and log the reconstruction.
        clip_factor: Std-deviation multiplier for logI contrast clipping.
        verbose: Print loading / shape info to stdout during ``on_fit_start``.
    """

    def __init__(
        self,
        patch_json: str,
        log_every_n_epochs: int = 10,
        clip_factor: float = 3.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.patch_json = patch_json
        self.log_every_n_epochs = log_every_n_epochs
        self.clip_factor = clip_factor
        self.verbose = verbose
        # Remaining attributes are set by on_fit_start; see that method's
        # docstring for their shapes.  Accessing them before on_fit_start
        # raises AttributeError -- intentional loud crash.
        self._patch_stem: str = ""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Load the fixed patch from zarr and cache everything for inference.

        Called once before training starts.  All data is kept on CPU; it is
        moved to the model device inside ``_forward`` on every call.

        Sets the following instance attributes:

            _rcmc_norm       Tensor  (1, 2, Az+2*buf, Rg) -- normalised RCMC input
            _rcmc_input_phys ndarray (2, Az_core, Rg)     -- physical RCMC core (display)
            _slc_target_phys ndarray (2, Az_core, Rg)     -- physical SLC core  (display)
            _metadata        DataFrame | None
            _ephemeris       DataFrame | None
            _coords          dict {"zfile", "y", "x"}     -- for the filter cache
            _buffer          int  -- azimuth buffer size
            _az_core         int  -- core azimuth lines
        """
        # -- Resolve JSON path -------------------------------------------
        json_path = Path(self.patch_json)
        if not json_path.is_absolute():
            project_root = rootutils.find_root(search_from=__file__, indicator=".project-root")
            json_path = project_root / json_path

        if not json_path.exists():
            raise FileNotFoundError(
                f"[MonitorFixedPatch] Patch JSON not found: {json_path}.\n"
                "Run notebooks/cherry_pick_patch.ipynb to create it."
            )

        with open(json_path) as f:
            info = json.load(f)

        zpath = Path(info["product"])
        az_start: int = info["az_start"]
        az_end: int = info["az_end"]
        rg_start: int = info["rg_start"]
        rg_end: int = info["rg_end"]

        buffer: int = pl_module.hparams.azimuth_buffer  # type: ignore[attr-defined]
        az_full_start = az_start - buffer
        az_full_end = az_end + buffer

        self._patch_stem = json_path.stem
        self._buffer = buffer
        self._az_core = az_end - az_start

        if az_full_start < 0:
            raise ValueError(
                f"[MonitorFixedPatch] az_start ({az_start}) - buffer ({buffer}) = "
                f"{az_full_start} < 0.  The patch is too close to the image top edge.  "
                "Reduce the buffer or pick a patch further from the edge."
            )

        if self.verbose:
            print(
                f"[MonitorFixedPatch] Loading '{self._patch_stem}' from {zpath.name}\n"
                f"  Az: {az_full_start}:{az_full_end}  "
                f"(core {az_start}:{az_end}, buffer={buffer})\n"
                f"  Rg: {rg_start}:{rg_end}"
            )

        # -- Open arrays -------------------------------------------------
        rcmc_arr: zarr.Array = zarr.open_array(str(zpath / "rcmc"), mode="r")
        az_arr: zarr.Array = zarr.open_array(str(zpath / "az"), mode="r")

        # RCMC input (buffered, normalised) -- used for model forward pass
        self._rcmc_norm = torch.from_numpy(
            _load_complex_patch(
                rcmc_arr,
                slice(az_full_start, az_full_end),
                slice(rg_start, rg_end),
                norm_constants=(RC_MIN, RC_MAX),
            )
        ).unsqueeze(
            0
        )  # (1, 2, Az+2*buf, Rg)

        # RCMC core (physical) -- for visualization only
        self._rcmc_input_phys = _load_complex_patch(
            rcmc_arr, slice(az_start, az_end), slice(rg_start, rg_end)
        )  # (2, Az_core, Rg)

        # SLC core (physical) -- for visualization only
        self._slc_target_phys = _load_complex_patch(
            az_arr, slice(az_start, az_end), slice(rg_start, rg_end)
        )  # (2, Az_core, Rg)

        # -- Metadata ----------------------------------------------------
        store_root = zarr.open_group(str(zpath), mode="r")
        meta_data_list: list = []
        eph_data_list: list = []
        if "metadata" in store_root.attrs:
            raw = store_root.attrs["metadata"]
            if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                meta_data_list = raw["data"]
        if "ephemeris" in store_root.attrs:
            raw = store_root.attrs["ephemeris"]
            if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                eph_data_list = raw["data"]

        meta_full = pd.DataFrame(meta_data_list)
        eph_df = pd.DataFrame(eph_data_list)

        # Azimuth-slice metadata to the buffered patch rows
        az_clip_end = min(az_full_end, len(meta_full))
        meta_patch = meta_full.iloc[az_full_start:az_clip_end].reset_index(drop=True).copy()
        correct_swst_range_offset(meta_patch, rg_start)

        self._metadata: pd.DataFrame | None = meta_patch if len(meta_patch) > 0 else None
        self._ephemeris: pd.DataFrame | None = eph_df if len(eph_df) > 0 else None
        self._coords = {"zfile": str(zpath), "y": az_full_start, "x": rg_start}

        if self.verbose:
            eph_status = (
                f"{len(eph_df)} rows"
                if len(eph_df) > 0
                else "key present but 0 rows — SLC recon will be skipped"
            )
            print(
                f"[MonitorFixedPatch] metadata rows={len(meta_patch)}, " f"ephemeris: {eph_status}"
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward(self, pl_module: LightningModule) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the model on the fixed patch.

        Returns:
            x_hat_core_phys: ``(2, Az_core, Rg)`` reconstructed RCMC, physical scale.
            slc_recon_phys:  ``(2, Az_core, Rg)`` reconstructed SLC, physical scale,
                             or ``None`` when azimuth compression is not available.
        """
        # Device is always read dynamically -- it can change between fit and test.
        device = pl_module.device

        with torch.no_grad():
            output = pl_module(self._rcmc_norm.to(device))
            x_hat = output.x_hat  # (1, 2, Az+2*buf, Rg)

        buf, azc = self._buffer, self._az_core
        x_hat_core = x_hat[:, :, buf : buf + azc, :]
        x_hat_core_phys = minmax_inverse(x_hat_core, RC_MIN, RC_MAX).squeeze(0).cpu().numpy()

        # SLC reconstruction -- only in "slc" mode with valid metadata + ephemeris
        slc_recon_phys = None
        if (
            pl_module.hparams.training_mode == "slc"  # type: ignore[attr-defined]
            and self._metadata is not None
            and self._ephemeris is not None
        ):
            with torch.no_grad():
                x_hat_phys = minmax_inverse(x_hat, RC_MIN, RC_MAX)
                slc_recon = full_azimuth_compress_batch(
                    x_hat_phys,
                    [self._metadata],
                    [self._ephemeris],
                    buffer_size=buf,
                    device=str(device),
                    coords_batch=[self._coords],
                )  # (1, 2, Az_core, Rg) physical SLC scale
            slc_recon_phys = slc_recon.squeeze(0).cpu().numpy()

        return x_hat_core_phys, slc_recon_phys

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reconstruct the fixed patch and log a comparison figure to WandB."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        x_hat_core_phys, slc_recon_phys = self._forward(pl_module)
        fig = self._build_figure(x_hat_core_phys, slc_recon_phys, trainer.current_epoch)

        if (
            pl_module.logger is not None
            and hasattr(pl_module.logger, "experiment")
            and hasattr(pl_module.logger.experiment, "log")  # type: ignore[union-attr]
        ):
            pl_module.logger.experiment.log(  # type: ignore[union-attr]
                {"fixed_patch/reconstruction": wandb.Image(fig)},
                step=trainer.global_step,
            )

        plt.close(fig)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save PNG (logI) and NPY (linear amplitude) for each reconstructed domain."""
        x_hat_core_phys, slc_recon_phys = self._forward(pl_module)

        # default_root_dir is always set via configs/paths/default.yaml
        log_dir = Path(trainer.default_root_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = self._patch_stem

        def _save(arr: np.ndarray, tag: str) -> None:
            logI = clip_mean_std_numpy(phys_to_logI_np(arr), self.clip_factor)
            plt.imsave(str(log_dir / f"fixed_patch_{stem}_{tag}_logI.png"), logI, cmap="viridis")
            linA = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
            np.save(str(log_dir / f"fixed_patch_{stem}_{tag}_linA.npy"), linA)

        _save(x_hat_core_phys, "rcmc_recon")
        if slc_recon_phys is not None:
            _save(slc_recon_phys, "slc_recon")

        if self.verbose:
            print(
                f"[MonitorFixedPatch] on_test_end: outputs saved to {log_dir}\n"
                f"  RCMC: fixed_patch_{stem}_rcmc_recon_{{logI.png,linA.npy}}"
                + (
                    f"\n  SLC:  fixed_patch_{stem}_slc_recon_{{logI.png,linA.npy}}"
                    if slc_recon_phys is not None
                    else "\n  SLC:  skipped (no ephemeris)"
                )
            )

    # ------------------------------------------------------------------
    # Figure builder
    # ------------------------------------------------------------------

    def _build_figure(
        self,
        x_hat_core_phys: np.ndarray,  # (2, Az, Rg)
        slc_recon_phys: np.ndarray | None,  # (2, Az, Rg) or None
        epoch: int,
    ) -> matplotlib.figure.Figure:
        """Build a 3- or 4-panel logI comparison figure.

        Panels: RCMC input | RCMC recon | [SLC recon] | SLC target
        """
        panels: list[tuple[str, np.ndarray]] = [
            ("RCMC input\n(logI)", self._rcmc_input_phys),
            ("RCMC recon\n(logI)", x_hat_core_phys),
        ]
        if slc_recon_phys is not None:
            panels.append(("SLC recon\n(logI)", slc_recon_phys))
        panels.append(("SLC target\n(logI)", self._slc_target_phys))

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), squeeze=False)

        for col, (title, arr) in enumerate(panels):
            # logI shape is (Az, Rg): rows = azimuth (vertical, top→bottom),
            # cols = range (horizontal, left→right). No transpose needed.
            logI = clip_mean_std_numpy(phys_to_logI_np(arr), self.clip_factor)
            ax = axes[0, col]
            ax.set_facecolor("black")
            im = ax.imshow(logI, cmap="viridis", aspect="auto", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Range →", fontsize=7)
            ax.set_ylabel("Azimuth ↓", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Fixed patch '{self._patch_stem}' -- epoch {epoch}", fontsize=11)
        plt.tight_layout()
        return fig
