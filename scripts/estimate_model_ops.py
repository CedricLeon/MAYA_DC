"""Estimate the computational cost (MACs/FLOPs) and parameter count of the ScaleHyperprior NIC
model used in MAYA_DC.

Two regimes are reported:

* **full**     — the complete forward pass g_a → h_a → h_s → g_s (encode + decode),
                 i.e. what runs during training / `forward()`.
* **compress** — only the operations needed to turn an input patch into a
                 bitstream: g_a + h_a + h_s (+ entropy coding). This is the
                 "sensor side" cost, the relevant number for an on-board /
                 edge deployment. `g_s` (image reconstruction decoder) is excluded.
* **decompress** — h_s + g_s, the "ground side" cost, for completeness.

The azimuth-compression step (`full_azimuth_compress_batch`) is a fixed FFT
signal-processing stage, NOT part of the trainable network; it is estimated
separately and analytically (see bottom of file).

Counting convention: PyTorch's FlopCounterMode counts a fused multiply-add as
ONE flop (i.e. it reports MACs). FLOPs ≈ 2 × MACs. Both are printed.

Run:
    conda run -n MAYA_DC python scripts/estimate_model_ops.py
    conda run -n MAYA_DC python scripts/estimate_model_ops.py --patch 512 512 --buffer 512 --non-square-kernels
"""

from __future__ import annotations

import argparse
import math

import rootutils
import torch
from torch.utils.flop_counter import FlopCounterMode

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.scale_hyperprior import ScaleHyperprior


def count(fn, display: bool = False) -> int:
    """Run `fn()` under FlopCounterMode and return total MACs."""
    counter = FlopCounterMode(display=display)
    with torch.no_grad(), counter:
        fn()
    return counter.get_total_flops()  # NOTE: this returns MACs, not 2x FLOPs


def human(n: float) -> str:
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}E"


def profile_net(net, x):
    """Return dict of per-submodule MACs and params for one forward pass."""
    y = net.g_a(x)
    z = net.h_a(torch.abs(y))
    z_hat, _ = net.entropy_bottleneck(z)
    scales = net.h_s(z_hat)
    y_hat, _ = net.gaussian_conditional(y, scales)

    macs = {
        "g_a": count(lambda: net.g_a(x)),
        "h_a": count(lambda: net.h_a(torch.abs(y))),
        "h_s": count(lambda: net.h_s(z_hat)),
        "g_s": count(lambda: net.g_s(y_hat)),
    }
    params = {
        n: sum(p.numel() for p in getattr(net, n).parameters())
        for n in ("g_a", "h_a", "h_s", "g_s")
    }
    return macs, params, tuple(y.shape), tuple(z.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, nargs=2, default=[512, 512], help="patch_size [Az, Rg]")
    ap.add_argument("--buffer", type=int, default=512, help="azimuth_buffer")
    ap.add_argument("--channels-main", type=int, default=128, help="N")
    ap.add_argument("--channels-latent", type=int, default=256, help="M")
    ap.add_argument("--activation", default="gdn")
    ap.add_argument("--non-square-kernels", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    az = args.patch[0] + 2 * args.buffer
    rg = args.patch[1]
    assert az % 16 == 0, f"(patch_az + 2*buffer)={az} must be divisible by 16"

    x = torch.randn(args.batch, 2, az, rg)
    pix = az * rg

    # Paper sweep = {square, non-square} kernels x lambda x seed0.
    # lambda/seed are loss-weight / RNG only -> identical compute, so we
    # profile just the two kernel variants. (--non-square-kernels forces a
    # single variant if you want only one.)
    variants = [args.non_square_kernels] if args.non_square_kernels else [False, True]
    rows = {}
    yz = {}
    for nsq in variants:
        net = ScaleHyperprior(
            nb_input_channels=2,
            nb_channels_main=args.channels_main,
            nb_channels_latent=args.channels_latent,
            activation=args.activation,
            non_square_kernels=nsq,
        ).eval()
        macs, params, yshape, zshape = profile_net(net, x)
        rows[nsq] = (macs, params)
        yz[nsq] = (yshape, zshape)

    # ---------------------------------------------------------------- header
    print("\n" + "=" * 70)
    print("MAYA_DC ScaleHyperprior — computational cost (paper sweep configs)")
    print("=" * 70)
    print(f"Input x        : {tuple(x.shape)}  (B, C, Az+2*buf, Rg)  =  {pix:,} px")
    print(f"N (main)       : {args.channels_main}")
    print(f"M (latent)     : {args.channels_latent}")
    print(f"activation     : {args.activation}")
    print("note           : lambda & seed do NOT affect OPs (loss weight / RNG only)")
    print("convention     : MACs = 1 multiply-add ;  FLOPs = 2 x MACs")
    print("compress       = g_a + h_a + h_s   |   decompress = h_s + g_s   |   full = all four")

    def label(nsq):
        return "non-square (5x15/3x9)" if nsq else "square (5x5/3x3)"

    for nsq in variants:
        macs, params = rows[nsq]
        yshape, zshape = yz[nsq]
        full = sum(macs.values())
        compress = macs["g_a"] + macs["h_a"] + macs["h_s"]
        decompress = macs["h_s"] + macs["g_s"]
        total_params = sum(params.values())

        print("\n" + "-" * 70)
        print(f"  KERNELS: {label(nsq)}    latent y={yshape}  z={zshape}")
        print("-" * 70)
        print(f"  {'submodule':20s} {'params':>10s} {'MACs':>12s} {'FLOPs':>12s}")
        order = [
            ("g_a", "encoder"),
            ("h_a", "hyper-enc"),
            ("h_s", "hyper-dec"),
            ("g_s", "decoder"),
        ]
        for key, desc in order:
            name = f"{key} ({desc})"
            print(
                f"  {name:20s} {human(params[key])+'p':>10s}"
                f" {human(macs[key]):>12s} {human(2*macs[key]):>12s}"
            )
        print("  " + "." * 56)
        print(
            f"  {'FULL':20s} {human(total_params)+'p':>10s} {human(full):>12s} {human(2*full):>12s}"
        )
        print(
            f"  {'COMPRESS':20s} {human(params['g_a']+params['h_a']+params['h_s'])+'p':>10s}"
            f" {human(compress):>12s} {human(2*compress):>12s}"
        )
        print(
            f"  {'DECOMPRESS':20s} {human(params['h_s']+params['g_s'])+'p':>10s}"
            f" {human(decompress):>12s} {human(2*decompress):>12s}"
        )
        print(f"  per input pixel (full): {human(full/pix)}MACs")

    # ---- azimuth compression (NOT part of the network) --------------------
    fft_macs = args.batch * 2 * rg * (az * math.log2(az)) * 5  # rough FFT+IFFT+filter mult
    print("\n" + "-" * 70)
    print("  Azimuth compression (FFT, fixed DSP, NOT trainable):")
    print(f"    ~{human(fft_macs)}MACs  (analytic O(N log N); ~3 orders below the net)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
