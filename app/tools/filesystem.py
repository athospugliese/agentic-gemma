"""Filesystem tools - Read, Write, Edit, Glob, Grep."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

import aiofiles

from app.tools.base import Tool, ToolContext, ToolResult
from app.utils.file_state import FileState

MAX_READ_LINES = 2000


class ReadTool(Tool):
    name = "Read"
    description = "Read a file from the filesystem. Returns contents with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "offset": {"type": "integer", "description": "Line number to start reading from (0-based)."},
            "limit": {"type": "integer", "description": "Number of lines to read."},
        },
        "required": ["file_path"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input["file_path"]
        offset = input.get("offset", 0)
        limit = input.get("limit", MAX_READ_LINES)

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()

            lines = content.splitlines(keepends=True)
            selected = lines[offset : offset + limit]
            numbered = [f"{i + offset + 1}\t{line.rstrip()}" for i, line in enumerate(selected)]
            output = "\n".join(numbered)

            if not output:
                return ToolResult(output="(empty file)")

            if len(lines) > offset + limit:
                output += f"\n\n... ({len(lines) - offset - limit} more lines)"

            # Update file state cache for edit validation
            if context.file_state_cache is not None:
                mtime = os.path.getmtime(file_path)
                is_partial = offset > 0 or limit < len(lines)
                context.file_state_cache.set(file_path, FileState(
                    content=content,
                    mtime=mtime,
                    offset=offset if is_partial else None,
                    limit=limit if is_partial else None,
                    is_partial=is_partial,
                ))

            return ToolResult(output=output)
        except FileNotFoundError:
            return ToolResult(output=f"File not found: {file_path}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error reading file: {e}", is_error=True)


class WriteTool(Tool):
    name = "Write"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["file_path", "content"],
    }

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input["file_path"]
        content = input["content"]

        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)
            return ToolResult(output=f"File written: {file_path}")
        except Exception as e:
            return ToolResult(output=f"Error writing file: {e}", is_error=True)


class EditTool(Tool):
    name = "Edit"
    description = "Replace exact string occurrences in a file."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."},
            "old_string": {"type": "string", "description": "The exact text to replace."},
            "new_string": {"type": "string", "description": "The replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input["file_path"]
        old_string = input["old_string"]
        new_string = input["new_string"]
        replace_all = input.get("replace_all", False)

        # Validate: old_string != new_string
        if old_string == new_string:
            return ToolResult(output="old_string and new_string are identical", is_error=True)

        # Enforce read-before-edit
        if context.file_state_cache is not None:
            cached = context.file_state_cache.get(file_path)
            if cached is None and os.path.exists(file_path):
                return ToolResult(
                    output=f"You must Read {file_path} before editing it.",
                    is_error=True,
                )
            if cached and cached.is_partial:
                return ToolResult(
                    output=f"File {file_path} was only partially read. Read the full file before editing.",
                    is_error=True,
                )

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()

            # Check mtime hasn't changed since last read
            if context.file_state_cache is not None:
                cached = context.file_state_cache.get(file_path)
                if cached:
                    current_mtime = os.path.getmtime(file_path)
                    if current_mtime != cached.mtime and content != cached.content:
                        return ToolResult(
                            output=f"File {file_path} was modified externally since last read. Read it again first.",
                            is_error=True,
                        )

            count = content.count(old_string)
            if count == 0:
                return ToolResult(output=f"old_string not found in {file_path}", is_error=True)
            if count > 1 and not replace_all:
                return ToolResult(
                    output=f"old_string found {count} times. Use replace_all=true or provide more context.",
                    is_error=True,
                )

            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            async with aiofiles.open(file_path, "w") as f:
                await f.write(new_content)

            # Update file state cache with new content
            if context.file_state_cache is not None:
                new_mtime = os.path.getmtime(file_path)
                context.file_state_cache.set(file_path, FileState(
                    content=new_content,
                    mtime=new_mtime,
                ))

            return ToolResult(output=f"Edited {file_path} ({count} replacement{'s' if count > 1 else ''})")
        except FileNotFoundError:
            return ToolResult(output=f"File not found: {file_path}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error editing file: {e}", is_error=True)


class GlobTool(Tool):
    name = "Glob"
    description = "Find files matching a glob pattern. Results sorted by modification time (newest first)."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": 'Glob pattern (e.g. "**/*.py").'},
            "path": {"type": "string", "description": "Directory to search in."},
        },
        "required": ["pattern"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input["pattern"]
        search_path = input.get("path", context.cwd)

        try:
            all_matches = list(Path(search_path).glob(pattern))
            if not all_matches:
                return ToolResult(output="No files matched the pattern.")

            # Sort by modification time (newest first) like TypeScript source
            all_matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            # Limit to 100 results (matching TypeScript)
            matches = all_matches[:100]
            result_lines = [str(m) for m in matches]
            output = "\n".join(result_lines)
            if len(all_matches) > 100:
                output += f"\n\n... and {len(all_matches) - 100} more files"
            return ToolResult(output=output)
        except Exception as e:
            return ToolResult(output=f"Error in glob: {e}", is_error=True)


class GrepTool(Tool):
    name = "Grep"
    description = "Search file contents using regex patterns. Uses ripgrep (rg) when available for speed."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search in."},
            "glob": {"type": "string", "description": 'File glob filter (e.g. "*.py").'},
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode (default: files_with_matches).",
            },
            "context": {"type": "integer", "description": "Lines of context before and after match."},
            "case_insensitive": {"type": "boolean", "description": "Case insensitive search."},
            "head_limit": {"type": "integer", "description": "Max results to return (default 250)."},
        },
        "required": ["pattern"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern_str = input["pattern"]
        search_path = input.get("path", context.cwd)
        glob_filter = input.get("glob")
        output_mode = input.get("output_mode", "files_with_matches")
        ctx_lines = input.get("context", 0)
        case_insensitive = input.get("case_insensitive", False)
        head_limit = input.get("head_limit", 250)

        # Try ripgrep first (10-100x faster than Python re)
        try:
            return await self._rg_search(
                pattern_str, search_path, glob_filter, output_mode, ctx_lines, case_insensitive, head_limit,
            )
        except FileNotFoundError:
            pass  # rg not found, fall back to Python

        # Python fallback
        return await self._python_search(
            pattern_str, search_path, glob_filter, output_mode, ctx_lines, case_insensitive, head_limit,
        )

    async def _rg_search(
        self, pattern: str, path: str, glob_filter: str | None,
        output_mode: str, ctx_lines: int, case_insensitive: bool, head_limit: int,
    ) -> ToolResult:
        """Search using ripgrep subprocess."""
        import asyncio

        cmd = ["rg", "--no-heading"]

        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:  # content
            cmd.append("-n")  # line numbers

        if ctx_lines > 0:
            cmd.extend(["-C", str(ctx_lines)])
        if case_insensitive:
            cmd.append("-i")
        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        cmd.extend(["-m", str(head_limit)])  # max count
        cmd.extend(["--", pattern, path])

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        output = stdout.decode(errors="replace").strip()
        if not output:
            return ToolResult(output="No matches found.")

        # Trim to head_limit lines
        lines = output.splitlines()
        if len(lines) > head_limit:
            output = "\n".join(lines[:head_limit])
            output += f"\n\n... ({len(lines) - head_limit} more results)"

        return ToolResult(output=output)

    async def _python_search(
        self, pattern_str: str, search_path: str, glob_filter: str | None,
        output_mode: str, ctx_lines: int, case_insensitive: bool, head_limit: int,
    ) -> ToolResult:
        """Fallback: Python re-based search."""
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern_str, flags)
        except re.error as e:
            return ToolResult(output=f"Invalid regex: {e}", is_error=True)

        matches: list[str] = []
        search = Path(search_path)

        try:
            if search.is_file():
                files = [search]
            else:
                files = [f for f in search.rglob("*") if f.is_file()]
                if glob_filter:
                    files = [f for f in files if fnmatch.fnmatch(f.name, glob_filter)]

            for file_path in files[:1000]:
                try:
                    text = file_path.read_text(errors="replace")
                    file_lines = text.splitlines()
                    for line_num, line in enumerate(file_lines, 1):
                        if regex.search(line):
                            if output_mode == "files_with_matches":
                                matches.append(str(file_path))
                                break
                            elif output_mode == "count":
                                matches.append(f"{file_path}:{sum(1 for l in file_lines if regex.search(l))}")
                                break
                            else:  # content
                                # Add context lines
                                start = max(0, line_num - 1 - ctx_lines)
                                end = min(len(file_lines), line_num + ctx_lines)
                                for i in range(start, end):
                                    prefix = ">" if i == line_num - 1 else " "
                                    matches.append(f"{file_path}:{i + 1}:{prefix} {file_lines[i]}")
                except (OSError, UnicodeDecodeError):
                    continue

                if len(matches) >= head_limit:
                    break

            if not matches:
                return ToolResult(output="No matches found.")

            output = "\n".join(matches[:head_limit])
            return ToolResult(output=output)
        except Exception as e:
            return ToolResult(output=f"Error in grep: {e}", is_error=True)
