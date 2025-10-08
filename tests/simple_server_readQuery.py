"""Direct read query diagnostics via MCP.

Renamed from simple_read_query for consistency with simple_server_tableList.
Executes canonical and tactical queries using the base_readQuery tool.

Run:
    python -m tests.simple_server_readQuery

Environment:
    Requires DATABASE_URI or TD_HOST/TD_USER/TD_PASSWORD (+ TD_NAME optional)

Exit codes:
    0 success path executed
    2 missing database credentials
    3 tool not found
"""
import os
import sys
import asyncio
import json
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import MCP_CONFIG, ENV_PATH  # type: ignore
from mcp_use.adapters import LangChainAdapter  # type: ignore
from mcp_use import MCPClient  # type: ignore

TOOL_NAME = 'base_readQuery'
SAMPLE_QUERIES = [
    ('list_branches', "select top 10 * from BANK_DB.branches"),
    ('version', "sel * from dbc.dbcinfo where infokey='VERSION'"),
]

def build_database_uri() -> str | None:
    if os.getenv('DATABASE_URI'):
        return os.getenv('DATABASE_URI')
    td_host = os.getenv('TD_HOST')
    td_user = os.getenv('TD_USER')
    td_pwd = os.getenv('TD_PASSWORD')
    td_port = os.getenv('TD_PORT', '1025')
    db_name = os.getenv('TD_NAME', 'BANK_DB')
    if td_host and td_user and td_pwd:
        return f"teradata://{td_user}:{td_pwd}@{td_host}:{td_port}/{db_name}"
    return None

async def _async_main():
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    db_uri = build_database_uri()
    if not db_uri:
        print('[error] Missing database credentials / URI')
        return 2

    cfg = MCP_CONFIG.copy()
    cfg['mcpServers']['teradata']['env']['DATABASE_URI'] = db_uri

    print('[info] creating MCP client (stdio)')
    client = MCPClient.from_dict(config=cfg)
    adapter = LangChainAdapter()
    tools = await adapter.create_tools(client)
    print(f'[info] loaded {len(tools)} tools')

    target = None
    for t in tools:
        if getattr(t, 'name', '') == TOOL_NAME:
            target = t
            break
    if not target:
        print(f'[error] tool {TOOL_NAME} not found')
        return 3

    timeout_env = os.getenv('READ_QUERY_TIMEOUT', '').strip()
    try:
        timeout_s = float(timeout_env) if timeout_env else 20.0
    except ValueError:
        print(f"[warn] invalid READ_QUERY_TIMEOUT value '{timeout_env}', defaulting to 20s")
        timeout_s = 20.0
    if timeout_s <= 0:
        print(f"[warn] non-positive READ_QUERY_TIMEOUT '{timeout_s}', forcing to 20s")
        timeout_s = 20.0
    print(f"[info] per-call timeout set to {timeout_s:.0f}s (READ_QUERY_TIMEOUT)")

    def build_payloads(query: str):
        return [{'sql': query}]     # canonical

    async def invoke(p):
        if hasattr(target, 'ainvoke'):
            try:
                return await target.ainvoke(p)
            except Exception as e:
                return e
        if hasattr(target, 'arun'):
            try:
                return await target.arun(p)
            except Exception as e:
                return e
        if hasattr(target, 'invoke'):
            try:
                return target.invoke(p)
            except Exception as e:
                return e
        if hasattr(target, 'run'):
            try:
                return target.run(p)
            except Exception as e:
                return e
        return RuntimeError('No callable interface on tool')

    for label, q in SAMPLE_QUERIES:
        print(f"\n[case] {label}")
        for payload in build_payloads(q):
            print(f"[info] invoking {TOOL_NAME} with keys {list(payload.keys())}")
            try:
                result = await asyncio.wait_for(invoke(payload), timeout=timeout_s)
            except asyncio.TimeoutError:
                print(f'[warn] timed out after {timeout_s:.0f}s')
                continue
            if isinstance(result, Exception):
                print(f'[warn] failed variant {payload}: {result}')
                continue
            print('[success] raw result (truncated):')
            if isinstance(result, dict):
                status = result.get('status')
                rows = result.get('results')
                meta = result.get('metadata')
                print(f'status={status} rows_type={type(rows).__name__} meta_keys={list(meta.keys()) if isinstance(meta, dict) else None}')
                print(json.dumps(result, indent=2)[:1200])
                break  # next label
            else:
                print(str(result)[:800])
                break
    return 0


def main():
    try:
        rc = asyncio.run(_async_main())
    except KeyboardInterrupt:
        print('Interrupted')
        rc = 130
    if isinstance(rc, int):
        sys.exit(rc)

if __name__ == '__main__':
    main()
