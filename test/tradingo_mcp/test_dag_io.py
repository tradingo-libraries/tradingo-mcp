"""Tests for tradingo_mcp.dag_io."""

from __future__ import annotations

import json

import pytest

from tradingo_mcp import dag_io

CONFIG = """
"download.{{ universe_name }}":
  depends_on: []
  function: tradingo.sampling.instruments.download_instruments
  symbols_out:
    - "instruments/{{ universe_name }}"
  publish_args:
    write_pickle: true
  params:
    epics: ["IX.D.DAX.DAILY.IP"]
  enabled: {{ TP_INCLUDE_INSTRUMENTS }}

"sample.{{ universe_name }}":
  depends_on: []
  function: tradingo.sampling.ig.create_universe
  symbols_out:
    - "prices/{{ universe_name }}.mid.close"
  params:
    pricelib: raw
    start_date: "{{ data_interval_start }}"
    end_date: "{{ data_interval_end }}"
"""

KW = {
    "variables": {"universe_name": "dax"},
    "params": {"TP_INCLUDE_INSTRUMENTS": False},
}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_RESEARCH_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_valid_dag_reports_tasks_and_runtime_enabled():
    result = dag_io.validate_dag("promoted_dax", CONFIG, **KW)
    assert result["errors"] == []
    assert result["tasks"] == ["download.dax", "sample.dax"]
    assert result["runtime_enabled"] == ["download.dax"]


def test_render_swaps_tokens_for_airflow_templates():
    spec = {"config": CONFIG, **KW}
    config, runtime_enabled = dag_io.render_config(spec)
    assert runtime_enabled == {"download.dax": "{{ params.TP_INCLUDE_INSTRUMENTS }}"}
    assert "enabled" not in config["download.dax"]
    params = config["sample.dax"]["params"]
    assert (
        params["start_date"]
        == "{{ params.data_interval_start or data_interval_start }}"
    )
    assert params["end_date"] == "{{ data_interval_end }}"


def test_environment_is_not_visible_to_the_config(monkeypatch):
    monkeypatch.setenv("SECRET_THING", "hunter2")
    bad = CONFIG.replace("pricelib: raw", "pricelib: '{{ SECRET_THING }}'")
    result = dag_io.validate_dag("promoted_dax", bad, **KW)
    assert not result["valid"]
    assert "render failed" in result["errors"][0]


def test_sandbox_blocks_attribute_escapes():
    bad = CONFIG.replace("pricelib: raw", "pricelib: \"{{ ''.__class__.__mro__ }}\"")
    assert not dag_io.validate_dag("promoted_dax", bad, **KW)["valid"]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (
            lambda c: c.replace("tradingo.sampling.ig.create_universe", "os.system"),
            "allowlist",
        ),
        (lambda c: c + "\ninc:\n  include: file:///etc/passwd\n", "include"),
        (
            lambda c: c.replace(
                "depends_on: []\n  function: tradingo.sampling.ig",
                "depends_on: [nope]\n  function: tradingo.sampling.ig",
            ),
            "nope",
        ),
    ],
)
def test_rejects_bad_configs(mutate, fragment):
    result = dag_io.validate_dag("promoted_dax", mutate(CONFIG), **KW)
    assert not result["valid"]
    assert fragment in " ".join(result["errors"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dag_id": "dax"},
        {"dag_id": "promoted_Dax"},
        {"schedule": "* * * * *"},
        {"schedule": "*/2 * * * *"},
        {"schedule": "every day"},
        {"start_date": "yesterday"},
        {"params": {"data_interval_start": ""}},
        {"params": {"a__b": 1}},
        {"variables": {"data_interval_end": "x"}},
    ],
)
def test_rejects_bad_spec_fields(kwargs):
    args = {"dag_id": "promoted_dax", "config_yaml": CONFIG, **KW, **kwargs}
    assert not dag_io.validate_dag(**args)["valid"]


def test_schedules_accepted():
    for schedule in (None, "@daily", "*/15 14-21 * * 1-5", "0 6 * * *"):
        assert dag_io.validate_dag("promoted_dax", CONFIG, schedule=schedule, **KW)[
            "valid"
        ]


def test_write_list_read_delete_roundtrip(home):
    written = dag_io.write_dag(
        "promoted_dax", CONFIG, schedule="@daily", description="d", **KW
    )
    assert written["valid"] and written["path"] == str(
        home / "dags" / "promoted_dax.json"
    )

    assert [d["dag_id"] for d in dag_io.list_dags()] == ["promoted_dax"]
    assert "config" not in dag_io.list_dags()[0]
    assert dag_io.read_dag("promoted_dax")["config"] == CONFIG

    created = dag_io.read_dag("promoted_dax")["created"]
    dag_io.write_dag("promoted_dax", CONFIG, schedule="@weekly", **KW)
    assert dag_io.read_dag("promoted_dax")["created"] == created

    assert dag_io.delete_dag("promoted_dax") is True
    assert dag_io.delete_dag("promoted_dax") is False
    assert dag_io.list_dags() == []


def test_invalid_write_leaves_nothing_behind(home):
    result = dag_io.write_dag("promoted_dax", CONFIG, schedule="* * * * *", **KW)
    assert not result["valid"] and result["path"] is None
    assert dag_io.list_dags() == []


def test_delete_rejects_path_traversal(home):
    (home / "victim.json").write_text("{}")
    assert dag_io.delete_dag("../victim") is False
    assert (home / "victim.json").exists()


def test_dag_limit(monkeypatch):
    monkeypatch.setattr(dag_io, "MAX_DAGS", 1)
    assert dag_io.write_dag("promoted_a", CONFIG, **KW)["valid"]
    assert not dag_io.write_dag("promoted_b", CONFIG, **KW)["valid"]
    assert dag_io.write_dag("promoted_a", CONFIG, **KW)["valid"]  # replace is fine


def test_load_specs_isolates_bad_files(home):
    dag_io.write_dag("promoted_good", CONFIG, **KW)
    dags = home / "dags"
    (dags / "promoted_broken.json").write_text("{not json")
    (dags / "promoted_mismatch.json").write_text(
        json.dumps(
            {"dag_id": "promoted_other", "config": "x", "start_date": "2026-01-01"}
        )
    )
    specs, errors = dag_io.load_specs()
    assert [s["dag_id"] for s in specs] == ["promoted_good"]
    assert set(errors) == {"promoted_broken", "promoted_mismatch"}
