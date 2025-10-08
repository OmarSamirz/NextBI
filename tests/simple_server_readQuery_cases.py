"""Run documented base_readQuery MCP test cases.

This script mirrors the official Teradata MCP test cases under
``tests/cases/core_test_cases.json`` for the ``base_readQuery`` tool.

Usage:
    python -m tests.simple_server_readQuery_cases

Environment:
    DATABASE_URI must point to a valid Teradata instance.
    Optional READ_QUERY_TIMEOUT adjusts per-call timeout (seconds, default 30).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import MCP_CONFIG, ENV_PATH  # type: ignore
from mcp_use import MCPClient  # type: ignore
from mcp_use.adapters import LangChainAdapter  # type: ignore

TOOL_NAME = "base_readQuery"


@dataclass
class QueryCase:
    name: str
    sql: str


DOCUMENTED_CASES: tuple[QueryCase, ...] = (
    QueryCase(
        name="system_catalog_query",
        sql="SELECT TOP 5 DatabaseName, TableName FROM DBC.TablesV WHERE DatabaseName = 'DBC'",
    ),
    QueryCase(
        name="current_timestamp",
        sql="SELECT CURRENT_TIMESTAMP",
    ),
    QueryCase(
        name="version_info",
        sql="SELECT InfoKey, InfoData FROM DBC.DBCInfoV WHERE InfoKey = 'VERSION'",
    ),
)


def build_database_uri() -> str | None:
    if os.getenv("DATABASE_URI"):
        return os.getenv("DATABASE_URI")
    td_host = os.getenv("TD_HOST")
    td_user = os.getenv("TD_USER")
    td_pwd = os.getenv("TD_PASSWORD")
    td_port = os.getenv("TD_PORT", "1025")
    db_name = os.getenv("TD_NAME", "DBC")
    if td_host and td_user and td_pwd and db_name:
        return f"teradata://{td_user}:{td_pwd}@{td_host}:{td_port}/{db_name}"
    return None


def resolve_timeout() -> float:
    raw = os.getenv("READ_QUERY_TIMEOUT", "").strip()
    if not raw:
        return 30.0
    try:
        value = float(raw)
    except ValueError:
        print(f"[warn] invalid READ_QUERY_TIMEOUT='{raw}', using 30s")
        return 30.0
    if value <= 0:
        print(f"[warn] READ_QUERY_TIMEOUT must be positive (got {value}), using 30s")
        return 30.0
    return value


def format_summary(result: Dict[str, Any]) -> str:
    status = result.get("status")
    rows = result.get("results")
    meta = result.get("metadata")
    rows_msg = f"rows={len(rows)}" if isinstance(rows, list) else f"rows_type={type(rows).__name__}"
    meta_keys = list(meta.keys()) if isinstance(meta, dict) else None
    return f"status={status} {rows_msg} meta_keys={meta_keys}"


async def _async_main() -> int:
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    db_uri = build_database_uri()
    if not db_uri:
        print("[error] DATABASE_URI is required (or TD_* variables to assemble it)")
        return 2

    cfg = MCP_CONFIG.copy()
    cfg["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = db_uri

    timeout_s = resolve_timeout()
    print(f"[info] READ_QUERY_TIMEOUT={timeout_s:.0f}s")

    print("[info] creating MCP client (stdio)")
    client = MCPClient.from_dict(config=cfg)
    adapter = LangChainAdapter()
    tools = await adapter.create_tools(client)
    print(f"[info] discovered {len(tools)} tools")

    target = next((t for t in tools if getattr(t, "name", "") == TOOL_NAME), None)
    if not target:
        print(f"[error] tool '{TOOL_NAME}' not available")
        return 3

    async def call_sql(sql: str):
        payload = {"sql": sql}
        if hasattr(target, "ainvoke"):
            return await target.ainvoke(payload)
        if hasattr(target, "arun"):
            return await target.arun(payload)
        if hasattr(target, "invoke"):
            return target.invoke(payload)
        if hasattr(target, "run"):
            return target.run(payload)
        raise RuntimeError("Tool has no callable interface")

    passes = 0
    for case in DOCUMENTED_CASES:
        print(f"\n[case] {case.name}")
        print(f"[sql] {case.sql}")
        try:
            raw = await asyncio.wait_for(call_sql(case.sql), timeout=timeout_s)
        except asyncio.TimeoutError:
            print(f"[fail] timeout after {timeout_s:.0f}s")
            continue
        except Exception as exc:
            print(f"[fail] exception: {exc}")
            continue

        if isinstance(raw, dict):
            summary = format_summary(raw)
            if str(raw.get("status", "")).lower() == "success":
                print(f"[pass] {summary}")
                passes += 1
            else:
                print(f"[fail] unexpected status -> {summary}")
        else:
            print(f"[fail] unexpected payload type {type(raw).__name__}: {raw}")

    total = len(DOCUMENTED_CASES)
    print(f"\n[summary] passed {passes}/{total} documented cases")
    return 0 if passes == total else 4


def main() -> None:
    try:
        rc = asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("Interrupted")
        rc = 130
    if isinstance(rc, int):
        sys.exit(rc)


if __name__ == "__main__":
    main()
