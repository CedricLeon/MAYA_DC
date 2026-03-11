from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

SAR_REQUIRED_PACKAGES = (
    "torch",
    "lightning",
    "rootutils",
    "compressai",
    "pandas",
    "sarpyx",
)
SARPYX_VERIFIED_REVISION = "06971f0bef13fd14bf3e70f3da9165a2a66a2ae4"


def get_missing_packages(packages: Sequence[str] = SAR_REQUIRED_PACKAGES) -> list[str]:
    missing: list[str] = []
    for package in packages:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def build_sar_install_hint(project_root: Path | None = None) -> str:
    root = Path(project_root or Path.cwd()).resolve()
    sarpyx_path = root.parent / "sarpyx"
    return "\n".join(
        [
            f"git -C {sarpyx_path} checkout {SARPYX_VERIFIED_REVISION}",
            f"uv pip install --python {root / '.venv/bin/python'} -r requirements.txt",
            f"uv pip install --python {root / '.venv/bin/python'} -e {sarpyx_path}",
        ]
    )


def ensure_sar_environment(
    packages: Sequence[str] = SAR_REQUIRED_PACKAGES,
    project_root: Path | None = None,
) -> None:
    missing = get_missing_packages(packages)
    if not missing:
        return

    missing_list = ", ".join(missing)
    raise RuntimeError(
        "Missing required SAR training packages: "
        f"{missing_list}.\nInstall them with:\n{build_sar_install_hint(project_root)}"
    )


def main() -> int:
    try:
        ensure_sar_environment()
    except RuntimeError as exc:
        print(exc)
        return 1

    print("SAR environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
