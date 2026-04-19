"""MariaDB read-only tools for text-to-SQL agents."""

from __future__ import annotations

import asyncio
from typing import Any

import aiomysql
import sqlparse

from app.settings import Settings
from app.tools.base import Tool, ToolContext, ToolResult

READ_ONLY_STATEMENTS = {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "WITH"}
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "REPLACE", "TRUNCATE",
    "DROP", "CREATE", "ALTER", "RENAME", "GRANT", "REVOKE",
    "CALL", "LOCK", "UNLOCK", "SET", "LOAD",
}


class _PoolHolder:
    """Module-level pool singleton."""

    pool: aiomysql.Pool | None = None
    lock = asyncio.Lock()


async def _get_pool(settings: Settings) -> aiomysql.Pool:
    async with _PoolHolder.lock:
        if _PoolHolder.pool is None:
            _PoolHolder.pool = await aiomysql.create_pool(
                host=settings.mariadb_host,
                port=settings.mariadb_port,
                user=settings.mariadb_user,
                password=settings.mariadb_password,
                db=settings.mariadb_database,
                autocommit=True,
                minsize=1,
                maxsize=5,
                charset="utf8mb4",
            )
        return _PoolHolder.pool


def _validate_read_only(query: str) -> tuple[bool, str]:
    """Return (is_valid, error_message_or_statement_type)."""
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        return False, "Empty query."

    statements = [s for s in sqlparse.split(stripped) if s.strip()]
    if len(statements) > 1:
        return False, "Multiple statements are not allowed."

    parsed = sqlparse.parse(stripped)
    if not parsed:
        return False, "Could not parse query."

    stmt = parsed[0]
    stmt_type = (stmt.get_type() or "").upper()

    if stmt_type not in READ_ONLY_STATEMENTS:
        return False, f"Only read-only statements allowed (got {stmt_type or 'UNKNOWN'})."

    upper_tokens = {
        t.normalized.upper()
        for t in stmt.flatten()
        if t.ttype is not None and t.ttype[0] == "Keyword"
    }
    bad = upper_tokens & FORBIDDEN_KEYWORDS
    if bad:
        return False, f"Forbidden keywords detected: {', '.join(sorted(bad))}."

    return True, stmt_type


def _has_limit(query: str) -> bool:
    upper = query.upper()
    return " LIMIT " in upper or upper.endswith(" LIMIT") or "\nLIMIT" in upper


def _format_rows(columns: list[str], rows: list[tuple], truncated: bool) -> str:
    if not rows:
        return f"(0 rows)\nColumns: {', '.join(columns) if columns else '(none)'}"

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join("NULL" if v is None else str(v).replace("\n", " ").replace("|", "\\|") for v in r) + " |"
        for r in rows
    )
    suffix = f"\n\n({len(rows)} rows{' — truncated' if truncated else ''})"
    return f"{header}\n{sep}\n{body}{suffix}"


async def _execute(settings: Settings, query: str, timeout: int) -> tuple[list[str], list[tuple]]:
    pool = await _get_pool(settings)
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await asyncio.wait_for(cur.execute(query), timeout=timeout)
            rows = await cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []
            return columns, list(rows)


class MariaDBTool(Tool):
    name = "MariaDB"
    description = (
        "Execute a READ-ONLY SQL query against the configured MariaDB database and return the rows. "
        "Only SELECT / SHOW / DESCRIBE / EXPLAIN / WITH statements are permitted. "
        "Any write or DDL statement is rejected. A LIMIT is auto-applied to SELECTs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Single SQL statement to execute. Must be read-only.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows to return (applied as LIMIT if the query lacks one).",
            },
            "explain": {
                "type": "boolean",
                "description": "If true, run EXPLAIN on the provided query instead of executing it.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return bool(self.settings.mariadb_host and self.settings.mariadb_database)

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        raw_query = (input.get("query") or "").strip()
        explain = bool(input.get("explain", False))
        limit = int(input.get("limit") or self.settings.mariadb_max_rows)
        if limit <= 0:
            limit = self.settings.mariadb_max_rows

        ok, info = _validate_read_only(raw_query)
        if not ok:
            return ToolResult(output=f"Query rejected: {info}", is_error=True)
        stmt_type = info

        query = raw_query.rstrip(";").strip()
        if explain:
            query = f"EXPLAIN {query}"
        elif stmt_type == "SELECT" and not _has_limit(query):
            query = f"{query} LIMIT {limit}"

        try:
            columns, rows = await _execute(self.settings, query, self.settings.mariadb_query_timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                output=f"Query timed out after {self.settings.mariadb_query_timeout}s.",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=f"MariaDB error: {e}", is_error=True)

        truncated = len(rows) >= limit and stmt_type == "SELECT"
        output = f"Query: {query}\n\n{_format_rows(columns, rows, truncated)}"
        return ToolResult(
            output=output,
            metadata={
                "rows": len(rows),
                "columns": columns,
                "truncated": truncated,
                "statement": stmt_type,
            },
        )


class MariaDBSchemaTool(Tool):
    name = "MariaDBSchema"
    description = (
        "Inspect the schema of the configured MariaDB database. "
        "Actions: 'tables' (list all tables), 'describe' (columns of a table), 'sample' (first few rows of a table). "
        "Use this BEFORE writing SQL with the MariaDB tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tables", "describe", "sample"],
                "description": "Which schema inspection to run.",
            },
            "table": {
                "type": "string",
                "description": "Table name (required for 'describe' and 'sample').",
            },
            "sample_rows": {
                "type": "integer",
                "description": "Number of rows for 'sample' (default 3, max 10).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return bool(self.settings.mariadb_host and self.settings.mariadb_database)

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        action = (input.get("action") or "").lower()
        table = (input.get("table") or "").strip()

        if action not in {"tables", "describe", "sample"}:
            return ToolResult(output=f"Unknown action: {action}", is_error=True)

        if action in {"describe", "sample"}:
            if not table:
                return ToolResult(output=f"action='{action}' requires 'table'.", is_error=True)
            if not table.replace("_", "").isalnum():
                return ToolResult(output=f"Invalid table name: {table}", is_error=True)

        try:
            if action == "tables":
                query = "SHOW TABLES"
            elif action == "describe":
                query = f"DESCRIBE `{table}`"
            else:
                n = min(int(input.get("sample_rows") or 3), 10)
                query = f"SELECT * FROM `{table}` LIMIT {n}"

            columns, rows = await _execute(self.settings, query, self.settings.mariadb_query_timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                output=f"Schema query timed out after {self.settings.mariadb_query_timeout}s.",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=f"MariaDB error: {e}", is_error=True)

        output = f"Action: {action}" + (f" (table={table})" if table else "") + f"\n\n{_format_rows(columns, rows, False)}"
        return ToolResult(output=output, metadata={"rows": len(rows), "action": action})
