"""Agent-authored Airflow DAG specs: write / list / read / delete, plus the shared render path.

An agent "promotes" a tradingo config into a DAG by writing a spec (a small JSON file) under
``<TP_RESEARCH_CONFIG_HOME>/dags/``. ``tradingo-plat/dags/promoted_dags.py`` loads every spec
and registers one Airflow DAG per file, so this module is also imported by the dag-processor and
the workers. It therefore must not import Airflow.

Two kinds of Jinja variable reach the config:

* ``variables`` are fixed in the spec and substituted when the DAG file is parsed (they may shape
  the graph, e.g. ``epics`` lists).
* ``params`` are Airflow DAG params: the default lives in the spec, a manual run can override it in
  the trigger ``conf``. They are substituted as ``{{ params.NAME }}`` and resolved per run, so they
  can only appear where a string is acceptable — plus ``enabled:``, which is evaluated per run
  (a disabled task becomes a no-op rather than being removed from the graph).

``data_interval_start`` is always available as a param (default ``""``): when set it replaces
Airflow's data interval start in every ``{{ data_interval_start }}`` of the config, so a manual run
can re-sample a longer window.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

import yaml

from tradingo_mcp.config_io import _is_allowed_function

DAG_ID_PREFIX = "promoted_"
MAX_DAGS = 50
MAX_CONFIG_BYTES = 256 * 1024
MIN_CRON_MINUTES = 5

# Always present on a promoted DAG; ``data_interval_start`` overrides Airflow's interval start.
START_OVERRIDE_PARAM = "data_interval_start"
START_OVERRIDE_PATTERN = r"^$|^\d{4}-\d{2}-\d{2}([T ].*)?$"

_DAG_ID_RE = re.compile(rf"^{DAG_ID_PREFIX}[a-z0-9_]{{1,60}}$")
_PARAM_NAME_RE = re.compile(
    r"^(?!.*__)[A-Za-z][A-Za-z0-9_]*$"
)  # no "__": it delimits tokens
_TOKEN_RE = re.compile(r"__rcparam_([A-Za-z0-9_]+?)__")
_PRESETS = {"@once", "@hourly", "@daily", "@weekly", "@monthly"}
_RESERVED_VARIABLES = {"data_interval_end", START_OVERRIDE_PARAM}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_DAG_HOME_ENV = "TP_RESEARCH_CONFIG_HOME"


def dag_home() -> pathlib.Path:
    """Directory holding the specs; ``dags/`` under the research config volume."""
    root = pathlib.Path(os.environ.get(_DAG_HOME_ENV, "/opt/research/configs"))
    return root / "dags"


def _spec_path(dag_id: str) -> pathlib.Path:
    return dag_home() / f"{dag_id}.json"


def param_token(name: str) -> str:
    """YAML-safe stand-in for a run-time param, so it survives the Jinja -> YAML round trip."""
    return f"__rcparam_{name}__"


def _template_for(name: str) -> str:
    if name == START_OVERRIDE_PARAM:
        return "{{ params.data_interval_start or data_interval_start }}"
    if name == "data_interval_end":
        return "{{ data_interval_end }}"
    return "{{ params." + name + " }}"


def builtin_tokens() -> dict[str, str]:
    """Variables every promoted DAG gets: the (overridable) interval start and the interval end."""
    return {
        START_OVERRIDE_PARAM: param_token(START_OVERRIDE_PARAM),
        "data_interval_end": param_token("data_interval_end"),
    }


def resolve_runtime(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Swap param tokens for Airflow templates.

    Returns ``(config, runtime_enabled)`` where ``runtime_enabled`` maps task name -> template for
    tasks whose ``enabled:`` depends on a param. Those tasks lose the key (so they stay in the
    graph) and the caller evaluates the template per run.
    """
    from tradingo.dag import collect_task_configs

    runtime_enabled: dict[str, str] = {}
    for name, task in collect_task_configs(config).items():
        enabled = task.get("enabled")
        if isinstance(enabled, str) and _TOKEN_RE.search(enabled):
            runtime_enabled[name] = _TOKEN_RE.sub(
                lambda m: _template_for(m.group(1)), enabled
            )
            del task["enabled"]

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return _TOKEN_RE.sub(lambda m: _template_for(m.group(1)), node)
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(config), runtime_enabled


def _contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        return key in node or any(_contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(v, key) for v in node)
    return False


def render_config(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Render a spec's config. Returns ``(config, runtime_enabled)`` as :func:`resolve_runtime`.

    Rendering is sandboxed and sees only the spec's variables and param tokens — never the process
    environment, so a promoted config cannot read the workers' credentials.
    """
    import jinja2
    from jinja2.sandbox import SandboxedEnvironment

    env = SandboxedEnvironment(undefined=jinja2.StrictUndefined)
    variables: dict[str, Any] = {
        **spec.get("variables", {}),
        **builtin_tokens(),
        **{name: param_token(name) for name in spec.get("params", {})},
    }
    text = env.from_string(spec["config"]).render(**variables)
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("config must render to a YAML mapping")
    if _contains_key(config, "include"):
        raise ValueError("'include' is not allowed in a promoted config")
    return resolve_runtime(config)


def _cron_errors(schedule: str) -> list[str]:
    fields = schedule.split()
    if len(fields) != 5:
        return [f"schedule '{schedule}' must be a preset (@daily...) or a 5-field cron"]
    minute = fields[0]
    if minute == "*":
        return [f"schedule runs every minute; minimum is every {MIN_CRON_MINUTES}m"]
    step = re.search(r"/(\d+)$", minute)
    if step and int(step.group(1)) < MIN_CRON_MINUTES:
        return [
            f"schedule steps minutes by {step.group(1)}; minimum is {MIN_CRON_MINUTES}"
        ]
    if not re.fullmatch(r"[\d*,/\-A-Za-z?LW#]+", "".join(fields)):
        return [f"schedule '{schedule}' contains unexpected characters"]
    return []


def _spec_errors(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _DAG_ID_RE.match(spec["dag_id"]):
        errors.append(
            f"dag_id must match {_DAG_ID_RE.pattern} (lowercase letters, digits, underscores)"
        )
    schedule = spec.get("schedule")
    if schedule is not None:
        if not isinstance(schedule, str):
            errors.append("schedule must be a string or null")
        elif schedule not in _PRESETS:
            errors.extend(_cron_errors(schedule))
    start = spec.get("start_date")
    if not isinstance(start, str) or not _DATE_RE.match(start):
        errors.append("start_date must be a YYYY-MM-DD string")
    variables = spec.get("variables", {})
    if not isinstance(variables, dict):
        errors.append("variables must be an object")
    else:
        for name in variables:
            if not _PARAM_NAME_RE.match(str(name)):
                errors.append(f"variable '{name}' is not a valid name")
            if name in _RESERVED_VARIABLES:
                errors.append(f"variable '{name}' is reserved")
    params = spec.get("params", {})
    if not isinstance(params, dict):
        errors.append("params must be an object of name -> default")
    else:
        for name, default in params.items():
            if not _PARAM_NAME_RE.match(str(name)) or name in _RESERVED_VARIABLES:
                errors.append(f"param '{name}' is reserved or not a valid name")
            elif isinstance(variables, dict) and name in variables:
                errors.append(f"'{name}' is both a variable and a param")
            if not isinstance(default, (bool, int, float, str)):
                errors.append(
                    f"param '{name}' default must be a bool, number or string"
                )
    if len(spec["config"].encode()) > MAX_CONFIG_BYTES:
        errors.append(f"config exceeds {MAX_CONFIG_BYTES} bytes")
    return errors


def validate_dag(
    dag_id: str,
    config_yaml: str,
    schedule: str | None = None,
    variables: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    start_date: str = "2026-01-01",
) -> dict[str, Any]:
    """Validate a would-be DAG spec. Returns ``{valid, errors, tasks, runtime_enabled}``."""
    spec = {
        "dag_id": dag_id,
        "config": config_yaml,
        "schedule": schedule,
        "variables": variables or {},
        "params": params or {},
        "start_date": start_date,
    }
    errors = _spec_errors(spec)
    result: dict[str, Any] = {
        "valid": False,
        "errors": errors,
        "tasks": [],
        "runtime_enabled": {},
    }
    if errors:
        return result

    try:
        config, runtime_enabled = render_config(spec)
    except (
        Exception
    ) as ex:  # jinja / yaml / sandbox errors all read the same to an agent
        errors.append(f"config render failed: {type(ex).__name__}: {ex}")
        return result

    from tradingo.dag import DAG, collect_task_configs

    tasks = collect_task_configs(config)
    if not tasks:
        errors.append("config defines no tasks (a task has 'depends_on' or 'stage')")
    stage_members = {m for t in tasks.values() for m in t.get("stage", [])}
    for name, task in tasks.items():
        func = task.get("function", "")
        if "stage" not in task:
            if not func:
                errors.append(f"Task '{name}': function is missing")
            elif not _is_allowed_function(func):
                errors.append(
                    f"Task '{name}': function '{func}' is not in the allowlist"
                )
        if name in runtime_enabled and name in stage_members:
            errors.append(
                f"Task '{name}': a param-driven 'enabled' is not supported inside a stage"
            )
        for dep in task.get("depends_on", []):
            if dep not in tasks:
                errors.append(
                    f"Task '{name}': depends_on '{dep}' is not a task in this config"
                )
    if errors:
        return {**result, "errors": errors}

    try:
        graph = DAG.from_config(config)
    except Exception as ex:
        errors.append(f"tradingo could not build the graph: {type(ex).__name__}: {ex}")
        return {**result, "errors": errors}

    return {
        "valid": True,
        "errors": [],
        "tasks": sorted(graph),
        "runtime_enabled": sorted(runtime_enabled),
    }


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def write_dag(
    dag_id: str,
    config_yaml: str,
    schedule: str | None = None,
    variables: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    description: str = "",
    start_date: str = "2026-01-01",
) -> dict[str, Any]:
    """Validate and persist a DAG spec (create or replace)."""
    result = validate_dag(dag_id, config_yaml, schedule, variables, params, start_date)
    if not result["valid"]:
        return {**result, "path": None}

    existing = _spec_path(dag_id)
    if not existing.exists() and len(list_dags()) >= MAX_DAGS:
        return {
            **result,
            "valid": False,
            "errors": [f"limit of {MAX_DAGS} promoted DAGs reached; delete one first"],
            "path": None,
        }

    now = datetime.now(tz=timezone.utc).isoformat()
    created = now
    if existing.exists():
        try:
            created = json.loads(existing.read_text(encoding="utf-8")).get(
                "created", now
            )
        except OSError, ValueError:
            pass
    spec = {
        "dag_id": dag_id,
        "description": description,
        "schedule": schedule,
        "start_date": start_date,
        "variables": variables or {},
        "params": params or {},
        "config": config_yaml,
        "created": created,
        "updated": now,
    }
    _atomic_write(existing, json.dumps(spec, indent=2))
    return {**result, "path": str(existing)}


def read_dag(dag_id: str) -> dict[str, Any]:
    path = _spec_path(dag_id)
    if not path.exists():
        raise FileNotFoundError(f"No promoted DAG '{dag_id}'")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def list_dags() -> list[dict[str, Any]]:
    """Summaries of every spec (no config body; use :func:`read_dag`)."""
    out: list[dict[str, Any]] = []
    for path in sorted(dag_home().glob(f"{DAG_ID_PREFIX}*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            out.append({"dag_id": path.stem, "error": f"unreadable: {ex}"})
            continue
        out.append(
            {
                k: spec.get(k)
                for k in (
                    "dag_id",
                    "description",
                    "schedule",
                    "start_date",
                    "variables",
                    "params",
                    "created",
                    "updated",
                )
            }
        )
    return out


def delete_dag(dag_id: str) -> bool:
    """Remove the spec; Airflow drops the DAG on its next parse of the loader file."""
    if not _DAG_ID_RE.match(dag_id):
        return False
    path = _spec_path(dag_id)
    if path.exists():
        path.unlink()
        return True
    return False


def load_specs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """For the Airflow loader: ``(specs, {dag_id: error})``; one bad file never hides the rest."""
    specs: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for path in sorted(dag_home().glob(f"{DAG_ID_PREFIX}*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            if spec.get("dag_id") != path.stem:
                raise ValueError(
                    f"dag_id {spec.get('dag_id')!r} does not match the file name"
                )
            problems = _spec_errors(spec)
            if problems:
                raise ValueError("; ".join(problems))
            specs.append(spec)
        except (OSError, ValueError, KeyError) as ex:
            errors[path.stem] = f"{type(ex).__name__}: {ex}"
    return specs, errors
