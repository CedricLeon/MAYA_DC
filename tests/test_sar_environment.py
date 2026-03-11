from __future__ import annotations

import importlib.util

import pytest

from src.utils.sar_environment import ensure_sar_environment


def test_ensure_sar_environment_reports_clear_install_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name == "sarpyx":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="uv pip install"):
        ensure_sar_environment(packages=("rootutils", "sarpyx"))
