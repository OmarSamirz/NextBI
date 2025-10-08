"""Server table/list diagnostic using the table enumeration logic.

Replaces the earlier simple_server_test script. Probes the presence of the
primary table listing tool (base_tableList/tableList) and prints a concise
summary of returned tables + metadata.

Run:
    python -m tests.simple_server_tableList

Exit codes:
    0 success (at least one variant succeeded)
    2 missing DATABASE_URI / TD_* info
    3 table list tool not found
    4 all parameter variants failed
"""
import os
import sys
import asyncio
import time
import inspect
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import MCP_CONFIG, ENV_PATH  # type: ignore
from mcp_use.adapters import LangChainAdapter  # type: ignore
from mcp_use import MCPClient  # type: ignore

TOOL_CANDIDATES = ["base_tableList", "tableList"]


def build_database_uri() -> str | None:
    db_uri = os.getenv("DATABASE_URI")
    if db_uri:
        return db_uri
    td_host = os.getenv("TD_HOST")
    td_user = os.getenv("TD_USER")
    td_pwd = os.getenv("TD_PASSWORD")
    td_port = os.getenv("TD_PORT", "1025")
    db_name = os.getenv("TD_NAME")
    if td_host and td_user and td_pwd and db_name:
        return f"teradata://{td_user}:{td_pwd}@{td_host}:{td_port}/{db_name}"
    return None


async def _async_run():
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    db_uri = build_database_uri()
    if not db_uri:
        print("[error] No DATABASE_URI or TD_* components found; aborting.")
        return 2

    db_name = os.getenv("TD_NAME", "BANK_DB")

    cfg = MCP_CONFIG.copy()
    cfg["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = db_uri

    print("[info] creating MCP client (stdio)")
    client = MCPClient.from_dict(config=cfg)
    adapter = LangChainAdapter()
    tools = await adapter.create_tools(client)
    print(f"[info] loaded {len(tools)} tools")

    table_tool = None
    for t in tools:
        name = getattr(t, "name", "")
        if name in TOOL_CANDIDATES:
            table_tool = t
            break
    if not table_tool:
        print("[error] table list tool not found; candidates:", TOOL_CANDIDATES)
        return 3

    # Introspect basic schema
    schema = getattr(table_tool, 'args_schema', None)
    if schema:
        try:
            print('[debug] args_schema JSON:', schema.schema_json())
        except Exception as e:  # pragma: no cover
            print('[debug] could not serialize args_schema:', e)
    print('[debug] description:', getattr(table_tool, 'description', ''))
    for fn_name in ['ainvoke','arun','invoke','run']:
        if hasattr(table_tool, fn_name):
            try:
                print(f'[debug] signature {fn_name}:', inspect.signature(getattr(table_tool, fn_name)))
            except Exception:  # pragma: no cover
                pass

    params_variants = [
        {"database": db_name},       # empirically correct
        {"database_name": db_name},  # declared (but times out historically)
        {"db": db_name},
        {},
    ]

    async def call_variant(p):
        if hasattr(table_tool, 'ainvoke'):
            return await table_tool.ainvoke(p)
        if hasattr(table_tool, 'arun'):
            return await table_tool.arun(p)
        if hasattr(table_tool, 'invoke'):
            return table_tool.invoke(p)
        if hasattr(table_tool, 'run'):
            return table_tool.run(p)
        return RuntimeError('No callable interface found on tool')

    limit_print = int(os.getenv('TABLE_LIST_PRINT_LIMIT', 15))
    timeout = 15
    for variant in params_variants:
        print(f"[info] invoking with keys={list(variant.keys()) or 'NO PARAMS'} -> {variant}")
        start = time.time()
        try:
            result = await asyncio.wait_for(call_variant(variant), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[warn] variant timed out after {timeout}s")
            continue
        elapsed = time.time() - start
        if isinstance(result, Exception):
            print(f"[warn] variant failed: {result}")
            continue
        print(f"[success] elapsed={elapsed:.2f}s")
        if isinstance(result, dict):
            rows = result.get('results')
            meta = result.get('metadata')
            print('status:', result.get('status'))
            if isinstance(rows, list):
                print(f"tables_returned={len(rows)} (showing first {limit_print})")
                for r in rows[:limit_print]:
                    print('-', r)
            print('metadata:', meta)
        return 0

    print('[error] all parameter variants failed')
    return 4


def main():
    try:
        rc = asyncio.run(_async_run())
    except KeyboardInterrupt:
        print('Interrupted.')
        rc = 130
    if isinstance(rc, int):
        sys.exit(rc)


if __name__ == '__main__':
    main()
