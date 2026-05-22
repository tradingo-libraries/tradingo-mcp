"""Execute arbitrary Python snippets and capture their output."""

from __future__ import annotations

import concurrent.futures
import contextlib
import io
import traceback
from typing import Any


def _exec_in_namespace(code: str) -> dict[str, str]:
    import numpy as np
    import pandas as pd

    from tradingo_mcp.arctic import arctic_client

    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "arctic_client": arctic_client,
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            exec(code, namespace)  # noqa: S102
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
    except Exception:
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": traceback.format_exc(),
        }


def run_script(code: str, timeout: int = 60) -> dict[str, str]:
    """Execute *code* in an isolated namespace with timeout.

    The namespace pre-imports: ``pd`` (pandas), ``np`` (numpy),
    ``arctic_client`` (factory for the configured ArcticDB instance).

    Returns a dict with keys ``stdout``, ``stderr``, and optionally
    ``error`` (traceback string on exception or ``timeout``).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_exec_in_namespace, code)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {
                "stdout": "",
                "stderr": "",
                "error": f"Script timed out after {timeout}s",
            }
