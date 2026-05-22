"""Tests for tradingo_mcp.exec_python."""

from __future__ import annotations

from unittest.mock import patch

import arcticdb as adb
import pandas as pd
import pytest

from tradingo_mcp import exec_python


@pytest.fixture()
def mem_arctic():
    ac = adb.Arctic("mem://")
    lib = ac.create_library("prices")
    df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )
    lib.write("test.close", df)
    return ac


@pytest.fixture(autouse=True)
def patch_arctic(mem_arctic):
    with patch("tradingo_mcp.arctic.arctic_client", return_value=mem_arctic):
        yield


# ---------------------------------------------------------------------------
# basic execution
# ---------------------------------------------------------------------------


def test_print_captured():
    result = exec_python.run_script("print('hello world')")
    assert result["stdout"].strip() == "hello world"
    assert "error" not in result


def test_stderr_captured():
    result = exec_python.run_script("import sys; sys.stderr.write('err\\n')")
    assert "err" in result["stderr"]
    assert "error" not in result


def test_multiline_script():
    code = "x = 1 + 1\nprint(x)"
    result = exec_python.run_script(code)
    assert result["stdout"].strip() == "2"


def test_pandas_available():
    code = "import pandas as pd\nprint(type(pd.DataFrame()))"
    result = exec_python.run_script(code)
    assert "DataFrame" in result["stdout"]
    assert "error" not in result


def test_numpy_available():
    code = "print(np.array([1, 2, 3]).sum())"
    result = exec_python.run_script(code)
    assert result["stdout"].strip() == "6"


# ---------------------------------------------------------------------------
# arctic_client pre-injected
# ---------------------------------------------------------------------------


def test_arctic_client_injected():
    code = "ac = arctic_client()\nprint(type(ac).__name__)"
    result = exec_python.run_script(code)
    assert "error" not in result
    assert result["stdout"].strip() == "Arctic"


def test_arctic_read():
    code = """
ac = arctic_client()
lib = ac.get_library("prices")
df = lib.read("test.close").data
print(df.shape)
"""
    result = exec_python.run_script(code)
    assert "error" not in result
    assert "(3, 1)" in result["stdout"]


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_exception_captured():
    result = exec_python.run_script("raise ValueError('boom')")
    assert "error" in result
    assert "ValueError" in result["error"]
    assert "boom" in result["error"]


def test_stdout_before_exception():
    code = "print('before')\nraise RuntimeError('after')"
    result = exec_python.run_script(code)
    assert "before" in result["stdout"]
    assert "error" in result
    assert "RuntimeError" in result["error"]


def test_syntax_error_captured():
    result = exec_python.run_script("def broken(:\n    pass")
    assert "error" in result


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


def test_timeout():
    result = exec_python.run_script("import time; time.sleep(10)", timeout=1)
    assert "error" in result
    assert "timed out" in result["error"]
