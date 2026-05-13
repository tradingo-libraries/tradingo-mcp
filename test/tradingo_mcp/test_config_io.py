"""Tests for tradingo_mcp.config_io."""

from __future__ import annotations

import os
import tempfile

# Point config dirs at temp directories so tests don't touch NFS
os.environ.setdefault("TP_RESEARCH_CONFIG_HOME", tempfile.mkdtemp())
os.environ.setdefault("TP_RESEARCH_RUN_HOME", tempfile.mkdtemp())

from tradingo_mcp.config_io import make_run_id, validate_config


def _make_valid_yaml(run_id: str) -> str:
    prefix = f"research.{run_id}."
    return f"""
signals.{prefix}trend.test:
  depends_on: []
  function: "tradingo_quant.signals.signals.ewmac_signal"
  symbols_in:
    close: "prices/test.mid.close"
  symbols_out:
    - "signals/trend.test"
  publish_args:
    symbol_prefix: "{prefix}"
  params:
    speed1: 16
    speed2: 64

backtest.research.{run_id}:
  depends_on: ["signals.{prefix}trend.test"]
  function: "tradingo.backtest.backtest"
  symbols_in:
    portfolio: "portfolio/{prefix}position.unlimited"
    bid_close: "prices/test.bid.close"
    ask_close: "prices/test.ask.close"
  symbols_out:
    - "backtest/portfolio"
  publish_args:
    symbol_prefix: "{prefix}"
  params:
    price_ffill_limit: 5
"""


def test_make_run_id_format():
    rid = make_run_id("agent")
    parts = rid.split("_")
    assert parts[0] == "agent"
    assert len(parts) == 3
    assert len(parts[2]) == 4


def test_valid_config():
    run_id = "test_20240101T1200_abcd"
    result = validate_config(run_id, _make_valid_yaml(run_id))
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["dag_summary"]["n_tasks"] == 2


def test_missing_symbol_prefix():
    run_id = "test_20240101T1200_xyz1"
    yaml_text = f"""
backtest.research.{run_id}:
  depends_on: []
  function: "tradingo.backtest.backtest"
  symbols_in: {{}}
  symbols_out: []
  params: {{}}
"""
    result = validate_config(run_id, yaml_text)
    assert result["valid"] is False
    assert any("symbol_prefix" in e for e in result["errors"])


def test_wrong_symbol_prefix():
    run_id = "test_20240101T1200_wron"
    prefix = "research.other_run."  # wrong run_id
    yaml_text = f"""
backtest.research.{run_id}:
  depends_on: []
  function: "tradingo.backtest.backtest"
  symbols_in: {{}}
  symbols_out: []
  publish_args:
    symbol_prefix: "{prefix}"
  params: {{}}
"""
    result = validate_config(run_id, yaml_text)
    assert result["valid"] is False
    assert any("must start with" in e for e in result["errors"])


def test_disallowed_function():
    run_id = "test_20240101T1200_bad1"
    prefix = f"research.{run_id}."
    yaml_text = f"""
backtest.research.{run_id}:
  depends_on: []
  function: "os.system"
  symbols_in: {{}}
  symbols_out: []
  publish_args:
    symbol_prefix: "{prefix}"
  params: {{}}
"""
    result = validate_config(run_id, yaml_text)
    assert result["valid"] is False
    assert any("allowlist" in e for e in result["errors"])


def test_double_prefix_in_symbols_out():
    run_id = "test_20240101T1200_dbl1"
    prefix = f"research.{run_id}."
    yaml_text = f"""
backtest.research.{run_id}:
  depends_on: []
  function: "tradingo.backtest.backtest"
  symbols_in: {{}}
  symbols_out:
    - "backtest/research.somesymbol"
  publish_args:
    symbol_prefix: "{prefix}"
  params: {{}}
"""
    result = validate_config(run_id, yaml_text)
    assert result["valid"] is False
    assert any("research." in e for e in result["errors"])


def test_no_backtest_task():
    run_id = "test_20240101T1200_nob1"
    prefix = f"research.{run_id}."
    yaml_text = f"""
signals.{prefix}trend:
  depends_on: []
  function: "tradingo_quant.signals.signals.ewmac_signal"
  symbols_in: {{}}
  symbols_out: []
  publish_args:
    symbol_prefix: "{prefix}"
  params: {{}}
"""
    result = validate_config(run_id, yaml_text)
    assert result["valid"] is False
    assert any("backtest" in e for e in result["errors"])
