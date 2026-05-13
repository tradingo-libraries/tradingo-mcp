"""Research config read/write/validate."""

from __future__ import annotations

import os
import pathlib
import random
import string
from datetime import datetime, timezone
from typing import Any

import yaml

_ALLOWLISTED_PREFIXES = (
    "tradingo_quant.signals.",
    "tradingo_quant.limits.",
    "tradingo_quant.stops.",
    "tradingo.portfolio.aggregate_portfolio",
    "tradingo.portfolio.apply_dealing_rules",
    "tradingo.backtest.backtest",
    "tradingo.sampling.",
)

_RESEARCH_CONFIG_HOME = pathlib.Path(
    os.environ.get("TP_RESEARCH_CONFIG_HOME", "/opt/research/configs")
)


def _config_path(run_id: str) -> pathlib.Path:
    return _RESEARCH_CONFIG_HOME / f"{run_id}.yaml"


def _is_allowed_function(func: str) -> bool:
    return any(func.startswith(p) for p in _ALLOWLISTED_PREFIXES)


def _rand4() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


def make_run_id(user: str = "agent") -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M")
    return f"{user}_{ts}_{_rand4()}"


def render_template(
    template_name: str,
    variables: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Render a Jinja2 template from tradingo_mcp/templates/.

    Returns {run_id, prefix, yaml} — the agent only edits params.
    """
    import jinja2

    if run_id is None:
        run_id = make_run_id()

    # strip agent-supplied run_id/prefix so they can't override
    variables = {k: v for k, v in variables.items() if k not in ("run_id", "prefix")}
    prefix = f"research.{run_id}."
    variables["run_id"] = run_id
    variables["prefix"] = prefix

    templates_dir = pathlib.Path(__file__).parent / "templates"
    loader = jinja2.FileSystemLoader(str(templates_dir))
    env = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)

    fname = (
        template_name if template_name.endswith(".j2") else f"{template_name}.yaml.j2"
    )
    tmpl = env.get_template(fname)
    rendered = tmpl.render(**variables)

    return {"run_id": run_id, "prefix": prefix, "yaml": rendered}


def validate_config(run_id: str, yaml_text: str) -> dict[str, Any]:
    """Structural validation of a research config YAML.

    Returns {valid, errors, dag_summary}.
    """
    errors: list[str] = []

    # 1. Parse YAML
    try:
        config = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return {"valid": False, "errors": [f"YAML parse error: {e}"], "dag_summary": {}}

    if not isinstance(config, dict):
        return {
            "valid": False,
            "errors": ["Config must be a YAML mapping"],
            "dag_summary": {},
        }

    # 2. Walk tasks
    n_tasks = 0
    n_stages = 0
    leaves: list[str] = []
    all_task_names: set[str] = set()
    dep_targets: set[str] = set()

    for key, val in config.items():
        if not isinstance(val, dict):
            continue
        if "depends_on" not in val and "stage" not in val:
            continue
        n_tasks += 1
        all_task_names.add(key)

        if "stage" in val:
            n_stages += 1
            continue

        # Check function allowlist
        func = val.get("function", "")
        if func and not _is_allowed_function(func):
            errors.append(f"Task '{key}': function '{func}' is not in the allowlist")

        # Check publish_args.symbol_prefix
        publish_args = val.get("publish_args", {})
        prefix_val = publish_args.get("symbol_prefix", "")
        if not prefix_val:
            errors.append(f"Task '{key}': publish_args.symbol_prefix is missing")
        elif not prefix_val.startswith(f"research.{run_id}."):
            errors.append(
                f"Task '{key}': symbol_prefix '{prefix_val}' must start with 'research.{run_id}.'"
            )

        # Check symbols_out doesn't contain literal "research."
        for sym in val.get("symbols_out", []):
            if "research." in sym:
                errors.append(
                    f"Task '{key}': symbols_out entry '{sym}' must not contain"
                    " literal 'research.' — use symbol_prefix instead"
                )

        for dep in val.get("depends_on", []):
            dep_targets.add(dep)

    # Leaves = tasks that nothing depends on
    leaves = [t for t in all_task_names if t not in dep_targets]

    # 3. Require at least one backtest leaf
    backtest_leaves = [
        leaf for leaf in leaves if leaf.startswith(f"backtest.research.{run_id}")
    ]
    if not backtest_leaves:
        errors.append(
            f"No backtest task found starting with 'backtest.research.{run_id}'"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "dag_summary": {
            "n_tasks": n_tasks,
            "n_stages": n_stages,
            "leaves": leaves,
        },
    }


def write_config(run_id: str, yaml_text: str) -> dict[str, Any]:
    """Validate and write YAML to research_configs/<run_id>.yaml."""
    result = validate_config(run_id, yaml_text)
    if not result["valid"]:
        return {**result, "path": None}

    _RESEARCH_CONFIG_HOME.mkdir(parents=True, exist_ok=True)
    path = _config_path(run_id)
    path.write_text(yaml_text, encoding="utf-8")
    return {**result, "path": str(path)}


def read_config(run_id: str) -> str:
    path = _config_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No config for run_id '{run_id}'")
    return path.read_text(encoding="utf-8")


def list_configs() -> list[dict[str, Any]]:
    if not _RESEARCH_CONFIG_HOME.exists():
        return []
    return [
        {"run_id": p.stem, "path": str(p), "size": p.stat().st_size}
        for p in sorted(_RESEARCH_CONFIG_HOME.glob("*.yaml"))
    ]


def delete_config(run_id: str) -> bool:
    path = _config_path(run_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_templates() -> list[dict[str, Any]]:
    templates_dir = pathlib.Path(__file__).parent / "templates"
    result = []
    for p in sorted(templates_dir.glob("*.j2")):
        name = p.name.replace(".yaml.j2", "").replace(".j2", "")
        result.append({"name": name, "file": p.name, "description": _template_desc(p)})
    return result


def _template_desc(path: pathlib.Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").split("\n")[0]
        if first.startswith("{# ") and first.endswith(" #}"):
            return first[3:-3].strip()
    except Exception:
        pass
    return ""
