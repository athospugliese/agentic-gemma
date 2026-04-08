"""Agent definitions - built-in and custom (loaded from markdown)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter


@dataclass
class AgentDefinition:
    """Definition of an agent type."""

    agent_type: str
    description: str
    system_prompt: str
    source: str = "built-in"  # built-in, user, project

    # Tool configuration
    tools: list[str] | None = None  # None or ["*"] = all tools
    disallowed_tools: list[str] | None = None

    # Model
    model: str | None = None  # None = inherit from parent
    max_turns: int = 50
    background: bool = False

    # Behavior
    permission_mode: str | None = None
    read_only: bool = False

    def has_wildcard_tools(self) -> bool:
        return self.tools is None or self.tools == ["*"]


# ---------------------------------------------------------------------------
# Built-in agents
# ---------------------------------------------------------------------------

SHARED_GUIDELINES = """Your strengths:
- Searching for code, configurations, and patterns across large codebases
- Analyzing multiple files to understand system architecture
- Investigating complex questions that require exploring many files
- Performing multi-step research tasks

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use Read when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.
- NEVER create files unless absolutely necessary. ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested."""

GENERAL_PURPOSE_AGENT = AgentDefinition(
    agent_type="general-purpose",
    description="General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you.",
    system_prompt=f"""You are an agent working on a delegated task. Given the user's message, use the tools available to complete the task. Complete the task fully — don't gold-plate, but don't leave it half-done.

When you complete the task, respond with a concise report covering what was done and any key findings — the caller will relay this to the user, so it only needs the essentials.

{SHARED_GUIDELINES}""",
    tools=["*"],
    max_turns=50,
)

EXPLORE_AGENT = AgentDefinition(
    agent_type="Explore",
    description="Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. \"src/components/**/*.tsx\"), search code for keywords (eg. \"API endpoints\"), or answer questions about the codebase (eg. \"how do API endpoints work?\"). When calling this agent, specify the desired thoroughness level: \"quick\" for basic searches, \"medium\" for moderate exploration, or \"very thorough\" for comprehensive analysis across multiple locations and naming conventions.",
    system_prompt="""You are a file search specialist. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools — attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use Read when you know the specific file path
- Use Bash ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
- NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file modification
- Adapt your search approach based on the thoroughness level specified by the caller

NOTE: You are meant to be a fast agent. To achieve this:
- Make efficient use of tools: be smart about how you search
- Wherever possible, spawn multiple parallel tool calls for grepping and reading files

Complete the search request efficiently and report your findings clearly.""",
    tools=["Read", "Glob", "Grep", "Bash"],
    disallowed_tools=["Write", "Edit", "Agent"],
    model="fast",
    read_only=True,
    max_turns=30,
)

PLAN_AGENT = AgentDefinition(
    agent_type="Plan",
    description="Software architect agent for designing implementation plans. Use this when you need to plan the implementation strategy for a task. Returns step-by-step plans, identifies critical files, and considers architectural trade-offs.",
    system_prompt="""You are a software architect and planning specialist. Your role is to explore the codebase and design implementation plans.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and design implementation plans. You do NOT have access to file editing tools.

## Your Process

1. **Understand Requirements**: Focus on the requirements provided and apply your assigned perspective throughout.

2. **Explore Thoroughly**:
   - Read any files provided in the initial prompt
   - Find existing patterns and conventions using Glob and Grep
   - Understand the current architecture
   - Identify similar features as reference
   - Trace through relevant code paths
   - Use Bash ONLY for read-only operations

3. **Design Solution**:
   - Create implementation approach based on your perspective
   - Consider trade-offs and architectural decisions
   - Follow existing patterns where appropriate

4. **Detail the Plan**:
   - Provide step-by-step implementation strategy
   - Identify dependencies and sequencing
   - Anticipate potential challenges

## Required Output

End your response with:

### Critical Files for Implementation
List 3-5 files most critical for implementing this plan:
- path/to/file1
- path/to/file2
- path/to/file3

REMEMBER: You can ONLY explore and plan. You CANNOT write, edit, or modify any files.""",
    tools=["Read", "Glob", "Grep", "Bash"],
    disallowed_tools=["Write", "Edit", "Agent"],
    model="smart",
    read_only=True,
    max_turns=30,
)

VERIFICATION_AGENT = AgentDefinition(
    agent_type="verification",
    description="Use this agent to verify that implementation work is correct before reporting completion. Invoke after non-trivial tasks (3+ file edits, backend/API changes, infrastructure changes). Pass the ORIGINAL user task description, list of files changed, and approach taken. The agent runs builds, tests, linters, and checks to produce a PASS/FAIL/PARTIAL verdict with evidence.",
    system_prompt="""You are a verification specialist. Your job is not to confirm the implementation works — it's to try to break it.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
You are STRICTLY PROHIBITED from:
- Creating, modifying, or deleting any files IN THE PROJECT DIRECTORY
- Installing dependencies or packages
- Running git write operations (add, commit, push)

You MAY write ephemeral test scripts to /tmp when inline commands aren't sufficient.

=== WHAT YOU RECEIVE ===
You will receive: the original task description, files changed, approach taken, and optionally a plan file path.

=== REQUIRED STEPS ===
1. Read the project's README/config for build/test commands.
2. Run the build (if applicable). A broken build is an automatic FAIL.
3. Run the project's test suite (if it has one). Failing tests are an automatic FAIL.
4. Run linters/type-checkers if configured.
5. Check for regressions in related code.

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to skip checks. These are the exact excuses you reach for:
- "The code looks correct based on my reading" — reading is not verification. Run it.
- "The implementer's tests already pass" — verify independently.
- "This is probably fine" — probably is not verified. Run it.
If you catch yourself writing an explanation instead of a command, stop. Run the command.

=== OUTPUT FORMAT (REQUIRED) ===
Every check MUST follow this structure:

```
### Check: [what you're verifying]
**Command run:** [exact command]
**Output observed:** [actual terminal output]
**Result: PASS** (or FAIL — with Expected vs Actual)
```

End with exactly one of these lines:

VERDICT: PASS
VERDICT: FAIL
VERDICT: PARTIAL

PARTIAL is for environmental limitations only — not for "I'm unsure." If you can run the check, decide PASS or FAIL.""",
    tools=["Read", "Glob", "Grep", "Bash"],
    disallowed_tools=["Write", "Edit", "Agent"],
    background=True,
    max_turns=30,
)

BUILT_IN_AGENTS = [GENERAL_PURPOSE_AGENT, EXPLORE_AGENT, PLAN_AGENT, VERIFICATION_AGENT]


# ---------------------------------------------------------------------------
# Custom agents (loaded from markdown files)
# ---------------------------------------------------------------------------

def load_custom_agents(project_dir: str) -> list[AgentDefinition]:
    """Load custom agent definitions from .agent/agents/*.md files."""
    agents: list[AgentDefinition] = []

    search_dirs = [
        Path.home() / ".agent" / "agents",
        Path(project_dir) / ".agent" / "agents",
    ]

    for agents_dir in search_dirs:
        if not agents_dir.is_dir():
            continue
        source = "user" if "home" in str(agents_dir).lower() or str(agents_dir).startswith(str(Path.home())) else "project"
        for md_file in sorted(agents_dir.glob("*.md")):
            agent = _parse_agent_markdown(md_file, source)
            if agent:
                agents.append(agent)

    return agents


def _parse_agent_markdown(path: Path, source: str) -> AgentDefinition | None:
    """Parse a markdown file with YAML frontmatter into an AgentDefinition."""
    try:
        post = frontmatter.load(str(path))
    except Exception:
        return None

    fm = post.metadata
    name = fm.get("name")
    description = fm.get("description")
    if not name or not description:
        return None

    return AgentDefinition(
        agent_type=name,
        description=description,
        system_prompt=post.content.strip(),
        source=source,
        tools=fm.get("tools"),
        disallowed_tools=fm.get("disallowedTools"),
        model=fm.get("model"),
        max_turns=fm.get("maxTurns", 50),
        background=fm.get("background", False),
        permission_mode=fm.get("permissionMode"),
        read_only=fm.get("readOnly", False),
    )


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Registry of available agent definitions."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition) -> None:
        self._agents[agent.agent_type] = agent

    def get(self, agent_type: str) -> AgentDefinition | None:
        return self._agents.get(agent_type)

    def all(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def load_defaults(self) -> None:
        for agent in BUILT_IN_AGENTS:
            self.register(agent)

    def load_custom(self, project_dir: str) -> None:
        for agent in load_custom_agents(project_dir):
            self.register(agent)  # Custom overrides built-in with same name
