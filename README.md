# AgenticGemma

Multi-agent system with dual LLM provider support — **OpenAI** and **Ollama** (Gemma 4 E2B).

## Setup

```bash
# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
```

## Usage

### Ollama + Gemma 4 E2B (local)

```bash
# Start Ollama and pull the model
ollama serve &
ollama pull gemma4:e2b

# Run the server
uv run uvicorn app.main:app --reload
```

### OpenAI

```bash
# Set provider and API key in .env
AGENT_LLM_PROVIDER=openai
AGENT_OPENAI_API_KEY=sk-...

# Run the server
uv run uvicorn app.main:app --reload
```

## API

```bash
# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{}'

# Send message (sync)
curl -X POST http://localhost:8000/sessions/{session_id}/messages/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'

# Send message (streaming SSE)
curl -N -X POST http://localhost:8000/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `AGENT_OPENAI_API_KEY` | | OpenAI API key |
| `AGENT_OPENAI_BASE_URL` | | Custom OpenAI base URL |
| `AGENT_DEFAULT_MODEL` | `gpt-4o` | Default OpenAI model |
| `AGENT_FAST_MODEL` | `gpt-4o-mini` | Fast OpenAI model |
| `AGENT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AGENT_OLLAMA_MODEL` | `gemma4:e2b` | Ollama model name |
| `AGENT_MAX_TOKENS` | `8192` | Max output tokens |
| `AGENT_PERMISSION_MODE` | `auto` | `auto`, `plan`, `default`, `bypass` |

## Architecture

```
User → FastAPI → QueryEngine → LLMAdapter (OpenAI or Ollama) → Model
                     ↓
                Tool calls → Bash, Read, Write, Edit, Glob, Grep, WebFetch, ...
                     ↓
                Sub-agents → Same adapter → Model
```

The provider is selected via `AGENT_LLM_PROVIDER` env var. Both providers implement the same `LLMAdapter` interface, so all features (tools, sub-agents, sessions, memory, compaction) work with either backend.
