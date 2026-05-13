"""tradingo-cli subprocess wrapper + run tracking."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

_RESEARCH_RUN_HOME = pathlib.Path(
    os.environ.get("TP_RESEARCH_RUN_HOME", "/opt/research/runs")
)
_RESEARCH_CONFIG_HOME = pathlib.Path(
    os.environ.get("TP_RESEARCH_CONFIG_HOME", "/opt/research/configs")
)

_LOG_TAIL_LINES = 50


def _manifest_path(run_id: str) -> pathlib.Path:
    return _RESEARCH_RUN_HOME / f"{run_id}.json"


def _save_manifest(run_id: str, data: dict[str, Any]) -> None:
    _RESEARCH_RUN_HOME.mkdir(parents=True, exist_ok=True)
    _manifest_path(run_id).write_text(json.dumps(data, default=str), encoding="utf-8")


def execute(
    config_name: str,
    task_name: str,
    start_date: str,
    end_date: str,
    with_deps: bool = True,
    batch_interval: str | None = None,
    batch_mode: str = "stepped",
    executor: str = "celery",
    wait: bool = True,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    """Run a research task via tradingo-cli subprocess."""
    config_path = _RESEARCH_CONFIG_HOME / f"{config_name}.yaml"
    if not config_path.exists():
        return {
            "run_id": config_name,
            "status": "error",
            "exit_code": -1,
            "duration_s": 0,
            "output_symbols": [],
            "log_tail": f"Config not found: {config_path}",
        }

    cmd = [
        "tradingo-cli",
        "--config",
        str(config_path),
        "task",
        "run",
        task_name,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--with-deps",
        "1" if with_deps else "0",
        "--executor",
        executor,
    ]

    if executor == "celery":
        broker_url = os.environ.get("TP_CELERY_BROKER_URL", "redis://redis:6379/0")
        cmd += ["--broker-url", broker_url]

    if batch_interval:
        cmd += [
            "--batch-interval",
            batch_interval,
            "--batch-mode",
            batch_mode,
            "--recover",
        ]

    run_id = f"{config_name}-{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    started = datetime.now(tz=timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "config_name": config_name,
        "task_name": task_name,
        "start_date": start_date,
        "end_date": end_date,
        "started": started,
        "status": "running",
        "exit_code": None,
        "duration_s": None,
        "output_symbols": [],
        "log_tail": "",
    }
    _save_manifest(run_id, manifest)

    t0 = time.monotonic()
    try:
        _MCP_ONLY_VARS = {"TP_RESEARCH_CONFIG_HOME", "TP_RESEARCH_RUN_HOME"}
        subprocess_env = {
            k: v for k, v in os.environ.items() if k not in _MCP_ONLY_VARS
        }
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s if wait else None,
            env=subprocess_env,
        )
        duration_s = time.monotonic() - t0
        combined = (proc.stdout or "") + (proc.stderr or "")
        log_tail = "\n".join(combined.splitlines()[-_LOG_TAIL_LINES:])
        status = "success" if proc.returncode == 0 else "failed"
        manifest.update(
            {
                "status": status,
                "exit_code": proc.returncode,
                "duration_s": round(duration_s, 1),
                "log_tail": log_tail,
                "finished": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    except subprocess.TimeoutExpired:
        manifest.update({"status": "timeout", "duration_s": timeout_s})
    except Exception as e:
        manifest.update({"status": "error", "log_tail": str(e)})

    _save_manifest(run_id, manifest)
    return manifest


def list_runs(config_name: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if not _RESEARCH_RUN_HOME.exists():
        return []
    manifests = sorted(
        _RESEARCH_RUN_HOME.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    result = []
    for p in manifests[: limit * 2]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if config_name and data.get("config_name") != config_name:
            continue
        result.append(data)
        if len(result) >= limit:
            break
    return result


def get_run_status(run_id: str) -> dict[str, Any]:
    path = _manifest_path(run_id)
    if not path.exists():
        return {"run_id": run_id, "status": "not_found"}
    return json.loads(path.read_text(encoding="utf-8"))
