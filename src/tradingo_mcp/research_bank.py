"""Read-only queries over the research bank built by tradingo-plat ``dags/research_bank.py``.

The bank is a SQLite file (``<agent-research>/_bank/bank.sqlite``) holding a catalog of every research item, a
lessons ledger, theme stats and a knowledge graph (``nodes`` / ``edges``). It is replaced atomically on rebuild,
so every call opens a fresh immutable read-only connection.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sqlite3
from collections import deque
from typing import Any

SCHEMA_VERSION = "1"
ITEM_FIELDS = (
    "key",
    "run_date",
    "one_line",
    "verdict",
    "recommendation",
    "decision",
    "theme",
    "net_sharpe_1bp",
    "holdout_sharpe_1bp",
    "gross_bp_per_trade",
    "breakeven_cost_bp",
    "kill_causes",
    "tags",
    "instruments",
    "dsr",
)
JSON_FIELDS = {
    "tags",
    "instruments",
    "kill_causes",
    "data_window",
    "success_criteria",
    "results",
    "dsr_inputs",
    "scope",
    "evidence",
    "contradicts",
    "history",
    "verdicts",
    "decisions",
    "items",
    "attrs",
}
FACET_TYPES = (
    "mechanism",
    "trigger",
    "horizon",
    "asset_class",
    "data_source",
    "method",
    "instrument",
    "cause",
    "theme",
)


class BankUnavailable(RuntimeError):
    pass


def bank_path() -> pathlib.Path:
    if p := os.environ.get("TP_RESEARCH_BANK"):
        return pathlib.Path(p)
    root = os.environ.get("TP_AGENT_RESEARCH_DIR", "/opt/data/agent-research")
    return pathlib.Path(root) / "_bank" / "bank.sqlite"


def _connect() -> sqlite3.Connection:
    path = bank_path()
    if not path.exists():
        raise BankUnavailable(
            f"research bank not found at {path}; it is built by the agent-research-bank Airflow DAG"
        )
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    version = con.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if version is None or version[0] != SCHEMA_VERSION:
        raise BankUnavailable(
            f"research bank schema {version and version[0]!r} != {SCHEMA_VERSION!r}"
        )
    return con


def _row(row: sqlite3.Row, fields: tuple[str, ...] | None = None) -> dict:
    out = {}
    for k in fields or tuple(row.keys()):
        v = row[k]
        if k in JSON_FIELDS and isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        out[k] = v
    return out


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 1]


def _fts(text: str, mode: str = "and") -> str | None:
    toks = _tokens(text)
    if not toks:
        return None
    quoted = [f'"{t}"' for t in dict.fromkeys(toks)]
    return (" OR " if mode == "or" else " ").join(quoted)


def _item_key(key: str) -> str:
    return key[5:] if key.startswith("item:") else key


def resolve_node(con: sqlite3.Connection, ref: str) -> str:
    """Accept a node id (``instrument:EURUSD``), an item key (``run/item``) or a bare value (``EURUSD``)."""
    if con.execute("SELECT 1 FROM nodes WHERE id = ?", (ref,)).fetchone():
        return ref
    if (
        "/" in ref
        and con.execute("SELECT 1 FROM nodes WHERE id = ?", (f"item:{ref}",)).fetchone()
    ):
        return f"item:{ref}"
    prefixed = [
        f"{t}:{v}" for t in FACET_TYPES for v in (ref, ref.upper().replace("/", ""))
    ]
    rows = con.execute(
        f"SELECT id FROM nodes WHERE id IN ({','.join('?' * len(prefixed))})", prefixed
    ).fetchall()
    if len(rows) == 1:
        return str(rows[0][0])
    for cand in (ref, ref.upper().replace("/", "")):
        rows = con.execute(
            "SELECT id FROM nodes WHERE label = ? AND type != 'item'", (cand,)
        ).fetchall()
        if len(rows) == 1:
            return str(rows[0][0])
        if len(rows) > 1:
            raise ValueError(
                f"{ref!r} is ambiguous: {[r[0] for r in rows]}; pass a full node id"
            )
    rows = con.execute(
        "SELECT id FROM nodes WHERE type = 'item' AND id LIKE ?", (f"item:%/{ref}",)
    ).fetchall()
    if len(rows) == 1:
        return str(rows[0][0])
    raise ValueError(f"no node matches {ref!r}")


def _filters(
    con: sqlite3.Connection,
    theme: str | None,
    verdict: str | None,
    decision: str | None,
    since: str | None,
    about: list[str] | None,
) -> tuple[str, list[Any]]:
    where, args = [], []
    if theme:
        where.append("i.theme = ?")
        args.append(theme)
    if verdict:
        where.append("i.verdict = ?")
        args.append(verdict)
    if decision:
        where.append("i.decision = ?")
        args.append(decision)
    if since:
        where.append("i.run_date >= ?")
        args.append(since)
    for ref in about or []:
        where.append(
            "EXISTS (SELECT 1 FROM edges e WHERE e.src = 'item:' || i.key AND e.dst = ?)"
        )
        args.append(resolve_node(con, ref))
    return (" AND ".join(where) or "1"), args


# ---------------------------------------------------------------------------
# public queries
# ---------------------------------------------------------------------------


def search(
    query: str = "",
    about: list[str] | None = None,
    theme: str | None = None,
    verdict: str | None = None,
    decision: str | None = None,
    since: str | None = None,
    limit: int = 20,
) -> list[dict]:
    con = _connect()
    where, args = _filters(con, theme, verdict, decision, since, about)
    cols = ", ".join(f"i.{f}" for f in ITEM_FIELDS)
    match = _fts(query)
    if match is None:
        rows = con.execute(
            f"SELECT {cols} FROM items i WHERE {where} ORDER BY i.key DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return [_row(r, ITEM_FIELDS) for r in rows]
    out: list[dict] = []
    for m in (match, _fts(query, "or")):
        rows = con.execute(
            f"SELECT {cols}, snippet(items_fts, -1, '[', ']', '…', 16) AS snippet "
            f"FROM items_fts f JOIN items i ON i.key = f.key "
            f"WHERE items_fts MATCH ? AND {where} "
            f"ORDER BY bm25(items_fts, 0, 3.0, 1.0, 2.0, 1.5) LIMIT ?",
            (m, *args, limit),
        ).fetchall()
        out = [_row(r, ITEM_FIELDS) | {"snippet": r["snippet"]} for r in rows]
        if out:
            break
    return out


def similar(thesis: str, limit: int = 10, exclude: str | None = None) -> list[dict]:
    con = _connect()
    match = _fts(thesis, "or")
    if match is None:
        return []
    cols = ", ".join(f"i.{f}" for f in ITEM_FIELDS)
    rows = con.execute(
        f"SELECT {cols}, bm25(items_fts, 0, 4.0, 1.0, 2.0, 2.0) AS score "
        f"FROM items_fts f JOIN items i ON i.key = f.key WHERE items_fts MATCH ? "
        f"ORDER BY score LIMIT ?",
        (match, limit + 1),
    ).fetchall()
    out = [
        _row(r, ITEM_FIELDS) | {"score": round(-r["score"], 2)}
        for r in rows
        if r["key"] != exclude
    ]
    return out[:limit]


def get_item(key: str, include_findings: bool = True, max_chars: int = 20000) -> dict:
    con = _connect()
    key = _item_key(key)
    row = con.execute("SELECT * FROM items WHERE key = ?", (key,)).fetchone()
    if row is None:
        rows = con.execute(
            "SELECT key FROM items WHERE item_id = ? ORDER BY key", (key,)
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                f"no unique item {key!r}; candidates: {[r[0] for r in rows]}"
            )
        key = rows[0][0]
        row = con.execute("SELECT * FROM items WHERE key = ?", (key,)).fetchone()
    item = _row(row)
    if include_findings:
        f = con.execute(
            "SELECT findings FROM items_fts WHERE key = ?", (key,)
        ).fetchone()
        text = (f[0] if f else "") or ""
        item["findings"] = text[:max_chars] + (
            f"\n…[truncated, {len(text)} chars]" if len(text) > max_chars else ""
        )
    nid = f"item:{key}"
    item["links_out"] = [
        _row(r)
        for r in con.execute(
            "SELECT type, dst, attrs FROM edges WHERE src = ? ORDER BY type, dst",
            (nid,),
        )
    ]
    item["links_in"] = [
        _row(r)
        for r in con.execute(
            "SELECT type, src, attrs FROM edges WHERE dst = ? ORDER BY type, src",
            (nid,),
        )
    ]
    return item


def lessons(
    query: str = "",
    about: list[str] | None = None,
    type: str | None = None,
    status: str = "active",
    limit: int = 50,
) -> list[dict]:
    con = _connect()
    where, args = ["1"], []
    if status != "any":
        where.append("l.status = ?")
        args.append(status)
    if type:
        where.append("l.type = ?")
        args.append(type)
    for ref in about or []:
        node = resolve_node(con, ref)
        # About the node directly, or supported by an item linked to it.
        where.append(
            "(EXISTS (SELECT 1 FROM edges e WHERE e.src = 'lesson:' || l.id AND e.dst = ?)"
            " OR EXISTS (SELECT 1 FROM edges e JOIN edges e2 ON e2.src = e.dst"
            " WHERE e.src = 'lesson:' || l.id AND e.type IN ('supported_by', 'contradicted_by')"
            " AND e2.dst = ?))"
        )
        args += [node, node]
    sql = f"SELECT l.* FROM lessons l WHERE {' AND '.join(where)}"
    match = _fts(query, "or")
    if match:
        sql = (
            f"SELECT l.* FROM lessons_fts f JOIN lessons l ON l.id = f.id "
            f"WHERE lessons_fts MATCH ? AND {' AND '.join(where)} ORDER BY bm25(lessons_fts)"
        )
        args = [match, *args]
    else:
        sql += " ORDER BY l.n_support DESC, l.id"
    rows = con.execute(f"{sql} LIMIT ?", (*args, limit)).fetchall()
    return [_row(r) for r in rows]


def neighbours(
    node: str,
    edge_types: list[str] | None = None,
    direction: str = "both",
    depth: int = 1,
    limit: int = 200,
) -> dict:
    con = _connect()
    start = resolve_node(con, node)
    depth = max(1, min(depth, 3))
    seen = {start: 0}
    edges: list[dict] = []
    queue = deque([start])
    type_sql = ""
    targs: list[str] = []
    if edge_types:
        type_sql = f" AND type IN ({','.join('?' * len(edge_types))})"
        targs = list(edge_types)
    while queue and len(edges) < limit:
        cur = queue.popleft()
        if seen[cur] >= depth:
            continue
        found = []
        if direction in ("out", "both"):
            found += [
                (r["src"], r["dst"], r["type"], r["attrs"], r["dst"])
                for r in con.execute(
                    f"SELECT * FROM edges WHERE src = ?{type_sql}", (cur, *targs)
                )
            ]
        if direction in ("in", "both"):
            found += [
                (r["src"], r["dst"], r["type"], r["attrs"], r["src"])
                for r in con.execute(
                    f"SELECT * FROM edges WHERE dst = ?{type_sql}", (cur, *targs)
                )
            ]
        for src, dst, etype, attrs, other in found:
            edges.append(
                {"src": src, "type": etype, "dst": dst, "attrs": json.loads(attrs)}
            )
            if other not in seen:
                seen[other] = seen[cur] + 1
                queue.append(other)
            if len(edges) >= limit:
                break
    ids = list(seen)
    nodes = {
        r["id"]: {"type": r["type"], "label": r["label"]}
        for r in con.execute(
            f"SELECT * FROM nodes WHERE id IN ({','.join('?' * len(ids))})", ids
        )
    }
    item_keys = [_item_key(n) for n in ids if n.startswith("item:")]
    items = {}
    if item_keys:
        cols = ", ".join(ITEM_FIELDS)
        items = {
            r["key"]: _row(r, ITEM_FIELDS)
            for r in con.execute(
                f"SELECT {cols} FROM items WHERE key IN ({','.join('?' * len(item_keys))})",
                item_keys,
            )
        }
    return {
        "start": start,
        "nodes": nodes,
        "edges": edges,
        "items": items,
        "truncated": len(edges) >= limit,
    }


def lineage(key: str) -> dict:
    con = _connect()
    start = resolve_node(con, key)
    ancestors, cur = [], start
    while True:
        r = con.execute(
            "SELECT dst FROM edges WHERE src = ? AND type = 'continued_from'", (cur,)
        ).fetchone()
        if r is None or r[0] in ancestors:
            break
        ancestors.append(r[0])
        cur = r[0]
    descendants, frontier = [], [start]
    while frontier:
        nxt = []
        for n in frontier:
            for r in con.execute(
                "SELECT src FROM edges WHERE dst = ? AND type = 'continued_from'", (n,)
            ):
                if r[0] not in descendants:
                    descendants.append(r[0])
                    nxt.append(r[0])
        frontier = nxt
    related = [
        _row(r)
        for r in con.execute(
            "SELECT src, type, dst, attrs FROM edges WHERE (src = ? OR dst = ?) "
            "AND type IN ('variant_of', 'related_to', 'contradicts', 'pnl_corr')",
            (start, start),
        )
    ]
    cols = ", ".join(ITEM_FIELDS)
    keys = [_item_key(n) for n in [*reversed(ancestors), start, *descendants]]
    rows = {
        r["key"]: _row(r, ITEM_FIELDS)
        for r in con.execute(
            f"SELECT {cols} FROM items WHERE key IN ({','.join('?' * len(keys))})", keys
        )
    }
    return {
        "chain": [rows.get(k, {"key": k}) for k in keys],
        "start": _item_key(start),
        "related": related,
    }


def theme(slug: str | None = None) -> dict | list[dict]:
    con = _connect()
    if not slug:
        return [
            _row(r)
            for r in con.execute(
                "SELECT slug, title, description, n_items, n_trials, verdicts, decisions, "
                "sr0_ann, last_run FROM themes WHERE n_items > 0 ORDER BY last_run DESC"
            )
        ]
    row = con.execute("SELECT * FROM themes WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        raise ValueError(f"no theme {slug!r}; call research_theme() to list themes")
    out = _row(row)
    cols = ", ".join(ITEM_FIELDS)
    out["items"] = [
        _row(r, ITEM_FIELDS)
        for r in con.execute(
            f"SELECT {cols} FROM items WHERE theme = ? ORDER BY key", (slug,)
        )
    ]
    out["lessons"] = lessons(about=[f"theme:{slug}"])
    return out


def untested(target: str, dimension: str = "mechanism") -> dict:
    """Taxonomy values of ``dimension`` never tested on items linked to ``target`` (an instrument, asset class...)."""
    con = _connect()
    node = resolve_node(con, target)
    counts = {
        r[0]: r[1]
        for r in con.execute(
            "SELECT n.label, COUNT(DISTINCT e1.src) FROM edges e1 "
            "JOIN edges e2 ON e2.src = e1.src JOIN nodes n ON n.id = e2.dst "
            "WHERE e1.dst = ? AND e1.src LIKE 'item:%' AND n.type = ? GROUP BY n.label",
            (node, dimension),
        )
    }
    values = {
        r[0]: r[1]
        for r in con.execute(
            "SELECT value, description FROM taxonomy WHERE facet = ?", (dimension,)
        )
    }
    if not values:
        raise ValueError(
            f"unknown dimension {dimension!r}; use one of mechanism, trigger, horizon, "
            "asset_class, data_source, method"
        )
    n_items = con.execute(
        "SELECT COUNT(DISTINCT src) FROM edges WHERE dst = ? AND src LIKE 'item:%'",
        (node,),
    ).fetchone()[0]
    return {
        "target": node,
        "items_on_target": n_items,
        "tested": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "untested": {v: d for v, d in values.items() if v not in counts},
    }


def contradictions() -> dict:
    con = _connect()
    lesson_pairs = [
        _row(r)
        for r in con.execute(
            "SELECT e.src, e.dst, a.claim AS src_claim, b.claim AS dst_claim FROM edges e "
            "JOIN lessons a ON 'lesson:' || a.id = e.src JOIN lessons b ON 'lesson:' || b.id = e.dst "
            "WHERE e.type = 'contradicts' AND a.status = 'active'"
        )
    ]
    contested = [
        _row(r)
        for r in con.execute(
            "SELECT id, type, claim, confidence, n_support, n_contra, evidence FROM lessons "
            "WHERE n_contra > 0 AND status = 'active' ORDER BY n_contra DESC"
        )
    ]
    items = [
        _row(r)
        for r in con.execute(
            "SELECT src, dst, attrs FROM edges WHERE type = 'contradicts' AND src LIKE 'item:%'"
        )
    ]
    return {
        "lesson_vs_lesson": lesson_pairs,
        "lessons_with_counter_evidence": contested,
        "item_vs_item": items,
    }
