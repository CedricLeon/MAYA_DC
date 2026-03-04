import sys
from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    if not _SH_AVAILABLE:
        pytest.fail(reason="`sh` package is not available in this environment.")

    try:
        python_command = sh.Command(sys.executable)
        python_command(command)
    except sh.ErrorReturnCode as e:
        stderr_output = e.stderr.decode().strip() if e.stderr else ""
        stdout_output = e.stdout.decode().strip() if e.stdout else ""
        msg_output = stderr_output or stdout_output or str(e)
        pytest.fail(reason=msg_output)  # msg argument deprecated (use reason instead)
