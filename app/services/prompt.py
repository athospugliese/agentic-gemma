"""System prompt builder."""

from __future__ import annotations

import os
import platform
from datetime import date
from pathlib import Path
from typing import Any

from app.tools.base import Tool

if __name__ != "__main__":
    from app.agents.definitions import AgentDefinition


def build_system_prompt(
    tools: list[Tool],
    cwd: str,
    custom_prompt: str | None = None,
    claude_md_content: str | None = None,
    memory_prompt: str | None = None,
    agent_prompt: str | None = None,
    coordinator_prompt: str | None = None,
    agent_definitions: list | None = None,
) -> str:
    """Build the complete system prompt from sections.

    OpenAI: this becomes the first message with role="system".
    """
    sections: list[str] = []

    # --- STATIC (cacheable prefix) ---

    # Role
    if agent_prompt:
        sections.append(agent_prompt)
    elif custom_prompt:
        sections.append(custom_prompt)
    else:
        sections.append(_default_role_section())

    # Tool usage
    tool_section = _tool_usage_section(tools)
    if tool_section:
        sections.append(tool_section)

    # Style
    sections.append(_style_section())

    # Agent list (tells the model which agents are available)
    if agent_definitions:
        agent_section = _agent_list_section(agent_definitions, tools)
        if agent_section:
            sections.append(agent_section)

    # --- DYNAMIC (per-session) ---

    # Coordinator
    if coordinator_prompt:
        sections.append(coordinator_prompt)

    # CLAUDE.md / project instructions
    if claude_md_content:
        sections.append(f"# Project Instructions\n\n{claude_md_content}")

    # Memory
    if memory_prompt:
        sections.append(memory_prompt)

    # Environment
    sections.append(_environment_section(cwd))

    return "\n\n".join(s for s in sections if s)


def _default_role_section() -> str:
    return """# Role

You are an expert software engineering assistant. You help users with coding tasks by reading files, \
writing code, running commands, and managing projects. You have access to tools that let you interact \
with the local filesystem and execute commands.

## Key Principles
- Read and understand existing code before modifying it
- Make minimal, targeted changes - don't refactor beyond what's asked
- Prefer editing existing files over creating new ones
- Write safe, secure code - avoid common vulnerabilities
- Be concise in your responses - lead with the answer"""


def _tool_usage_section(tools: list[Tool]) -> str:
    if not tools:
        return ""
    lines = ["# Available Tools\n"]
    for tool in tools:
        lines.append(f"- **{tool.name}**: {tool.description}")
    return "\n".join(lines)


def _style_section() -> str:
    return """# Output Style
- Be concise and direct. Skip filler words and preamble.
- Use markdown formatting for code blocks.
- When referencing files, use the full path.
- Do not add features or refactor beyond what was asked."""


def _environment_section(cwd: str) -> str:
    today = date.today().isoformat()
    return f"""# Environment
- Working directory: {cwd}
- Platform: {platform.system().lower()}
- Date: {today}
- Shell: {os.environ.get("SHELL", "/bin/bash")}"""


def load_claude_md(project_dir: str) -> str | None:
    """Load project instructions from CLAUDE.md files (priority order)."""
    candidates = [
        Path.home() / ".agent" / "CLAUDE.md",
        Path(project_dir) / "CLAUDE.md",
        Path(project_dir) / ".agent" / "CLAUDE.md",
    ]

    parts: list[str] = []
    for path in candidates:
        if path.exists():
            try:
                content = path.read_text().strip()
                if content:
                    parts.append(content)
            except OSError:
                continue

    # Also load .agent/rules/*.md
    rules_dir = Path(project_dir) / ".agent" / "rules"
    if rules_dir.is_dir():
        for rule_file in sorted(rules_dir.glob("*.md")):
            try:
                content = rule_file.read_text().strip()
                if content:
                    parts.append(content)
            except OSError:
                continue

    return "\n\n---\n\n".join(parts) if parts else None


def _agent_list_section(agent_definitions: list, tools: list[Tool]) -> str:
    """Build the agent list section that tells the model which agents are available."""
    tool_names = {t.name for t in tools}

    lines = [
        "# Agent Tool",
        "",
        "Launch sub-agents to handle complex, multi-step tasks autonomously.",
        "Each agent type has specific capabilities and tools.",
        "",
        "## Available agent types",
        "",
    ]

    for agent in agent_definitions:
        tools_desc = _get_agent_tools_description(agent, tool_names)
        lines.append(f"- **{agent.agent_type}**: {agent.description} (Tools: {tools_desc})")

    lines.extend([
        "",
        "## When NOT to use the Agent tool",
        "- To read a specific file path → use Read directly",
        "- To search for a class/function definition → use Grep or Glob",
        "- To search within 2-3 known files → use Read directly",
        "",
        "## Usage notes",
        "- Always include a short description (3-5 words) summarizing the task",
        "- Launch multiple agents concurrently by making multiple tool calls in one message",
        "- Agent results are not visible to the user — summarize key findings in your response",
        "- Brief the agent with full context: explain what you're trying to accomplish and why",
        "- Never delegate understanding — include file paths, line numbers, what specifically to change",
        "- Terse command-style prompts produce shallow, generic work",
    ])

    return "\n".join(lines)


def _get_agent_tools_description(agent: Any, available_tool_names: set[str]) -> str:
    """Describe which tools an agent has access to."""
    if agent.tools is None or agent.tools == ["*"]:
        if agent.disallowed_tools:
            excluded = ", ".join(agent.disallowed_tools)
            return f"All tools except {excluded}"
        return "All tools"

    if agent.tools:
        valid = [t for t in agent.tools if t in available_tool_names]
        return ", ".join(valid) if valid else "Limited"

    return "All tools"
