# Dependency de-vendoring notes

The vendored `Maya4/ srp/ s1isp/` clones were replaced with installable deps so a fresh clone
installs without manual cloning. This note's durable purpose is the two "switch back to PyPI"
follow-ups below.

## Current deps (`pyproject.toml`)

| Package | Source now | Why not plain PyPI | Last-good commit |
|---|---|---|---|
| `sarpyx` | PyPI `>=0.1.10` | — | `4024da1` |
| `maya4` | git `@pypi` | PyPI 0.1.2 uses zarr-v2 `zarr.hierarchy` (breaks on zarr 3) **and** has no HF-bucket support | `dc798a3` (vendored, older) → branch HEAD `c535776` |
| `compressai` | git `@main` | PyPI 1.2.8 caps `numpy<2`; cap removed on main (`1.2.9.dev0`) | — |
| `huggingface-hub` | `>=1.5` | bucket API (`download_bucket_files`) needed by maya4 | — |
| `numpy` | `>=2.0` | kept (compressai-main supports it) | — |

`s1isp` was dropped (demo-notebook only; not on PyPI). Also fixed a broken build backend
(`setuptools.backends.legacy:build` → `setuptools.build_meta`) that had silently prevented
`pip install -e .`.

## TODOs (switch git → PyPI when upstream releases)

- **maya4** — replace `maya4 @ git+https://github.com/sirbastiano/Maya4@pypi` with
  `maya4>=<ver>` once the zarr-v3 + bucket-aware line is published to PyPI.
- **compressai** — replace `compressai @ git+https://github.com/InterDigitalInc/CompressAI`
  with `compressai>=1.2.9` once the `numpy<2` cap removal ships to PyPI.

## Verified (env `MAYA_DC`, 2026-06-11)

`pip install -e ".[dev]"` resolves cleanly; all imports OK; pytest unchanged vs baseline;
azimuth compression numerically identical to the pre-bump baseline; full train→val→test smoke
passes; HF-bucket streaming works.
