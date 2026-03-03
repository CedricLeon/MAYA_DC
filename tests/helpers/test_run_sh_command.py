import pytest

from tests.helpers.package_available import _SH_AVAILABLE
from tests.helpers.run_sh_command import run_sh_command


@pytest.mark.skipif(not _SH_AVAILABLE, reason="`sh` package is not available.")
def test_run_sh_command_fails_when_stderr_is_empty() -> None:
    """A failing command without stderr should still fail the test."""
    with pytest.raises(pytest.fail.Exception):
        run_sh_command(["-c", "import sys; sys.exit(1)"])


@pytest.mark.skipif(not _SH_AVAILABLE, reason="`sh` package is not available.")
def test_run_sh_command_reports_stderr() -> None:
    """Failure output should include stderr content when available."""
    with pytest.raises(pytest.fail.Exception, match="boom"):
        run_sh_command(["-c", "import sys; print('boom', file=sys.stderr); sys.exit(1)"])
