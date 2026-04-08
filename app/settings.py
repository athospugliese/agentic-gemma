"""Settings loader with merge hierarchy (managed -> user -> project -> local)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PermissionRules(BaseModel):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class HookEntry(BaseModel):
    command: str
    matcher: str | None = None
    timeout: int = 30


class Settings(BaseSettings):
    # Provider routing: "openai" or "ollama"
    llm_provider: str = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str | None = None
    default_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"
    reasoning_model: str = "o3"
    max_tokens: int = 8192

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"

    # Paths
    data_dir: str = "~/.agent"
    project_dir: str = "."

    # Permissions
    permission_mode: str = "auto"  # auto, plan, default, bypass
    permissions: PermissionRules = Field(default_factory=PermissionRules)

    # Hooks
    hooks: dict[str, list[HookEntry]] = Field(default_factory=dict)

    # Memory
    auto_memory_enabled: bool = True
    auto_memory_directory: str | None = None

    # Agent
    max_turns: int = 200
    max_agent_turns: int = 50

    model_config = {"env_prefix": "AGENT_", "env_file": ".env", "extra": "ignore"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override wins for scalars, lists are concatenated."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def _load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file, return empty dict if not found or invalid."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def load_settings(project_dir: str | None = None) -> Settings:
    """Load settings with merge hierarchy: managed -> user -> project -> local."""
    merged: dict[str, Any] = {}

    # 1. Managed (global)
    managed_path = Path("/etc/agent/settings.json")
    merged = _deep_merge(merged, _load_json_file(managed_path))

    # 2. User
    user_path = Path.home() / ".agent" / "settings.json"
    merged = _deep_merge(merged, _load_json_file(user_path))

    # 3. Project
    if project_dir:
        project_path = Path(project_dir) / ".agent" / "settings.json"
        merged = _deep_merge(merged, _load_json_file(project_path))

    # 4. Local (gitignored)
    if project_dir:
        local_path = Path(project_dir) / ".agent" / "local.json"
        merged = _deep_merge(merged, _load_json_file(local_path))

    if project_dir:
        merged.setdefault("project_dir", project_dir)

    return Settings(**merged)
