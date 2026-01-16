# sp-solution MVP

This repo implements the Solution Architecture Contract (SAC.md) as a minimal, extensible MVP.

## What is implemented
- FastAPI transport with WebSocket chat and SSE event stream.
- Session manager with per-session event broker.
- Browser layer via MCP client (pluggable transport).
- Observation layer with size limits and dedupe.
- Policy gate for confirmation of non-whitelisted actions.
- Agent loop (LangGraph): OBSERVE -> DESCRIBE -> WAIT_USER (repeat).
- CLI REPL (Typer + prompt_toolkit + Rich) for chat and live event stream.

## Quick start
1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2) Start the server:

```bash
sp-server
```

3) Start the CLI:

```bash
sp-cli --server http://127.0.0.1:8000 --session demo
```

## Docker Compose
Build and run the API + MCP server:

```bash
docker compose up --build
```

Optional: run the CLI inside Docker in a separate terminal:

```bash
docker compose --profile cli up --build
```

Notes:
- To see the headed browser on Linux, allow X11 and pass through `DISPLAY`:
  `xhost +local:docker` before `docker compose up`.
- MCP downloads packages on first run via `npx`, so network access is required.
- If MCP fails to start, set `MCP_PACKAGE` or `MCP_CMD` in `.env` to override the startup command.
- For persistent sessions in Docker, set `MCP_ARGS` to include `--user-data-dir=/data` (default in `docker-compose.yml`).

## Host CLI (recommended for flaky Docker TTY)
Start the Docker services, then run the CLI on the host with the local TTY:

```bash
./scripts/repl.sh
```

This runs `docker compose up -d` and then executes `sp-cli` via `workon sp_solution`.
Override the defaults with `SP_SERVER` and `SP_SESSION`.

## CLI commands
- `/observe` to trigger a new observation cycle.
- `/click <eid>` to click an element.
- `/type <eid> <text>` to type into a field.
- `/scroll [direction] [amount]` to scroll (default: down 600).
- `/wait [ms]` to wait.
- `/screenshot` to request a screenshot.
- `/yes` or `/no` to confirm policy-gated actions.

## MCP configuration
The browser layer expects a Playwright MCP server endpoint. Configure it using environment variables:

- `MCP_ENDPOINT`: HTTP endpoint for the MCP server (e.g. `http://127.0.0.1:3333/mcp`).
- `MCP_TOOL_LAUNCH`, `MCP_TOOL_OBSERVE`, `MCP_TOOL_CLICK`, `MCP_TOOL_TYPE`, `MCP_TOOL_SCROLL`,
  `MCP_TOOL_WAIT`, `MCP_TOOL_SCREENSHOT`: tool names (defaults are in `sp_solution/config.py`).
- `MCP_USER_DATA_DIR`: persistent profile directory for browser sessions.

If `MCP_ENDPOINT` is not set, the browser layer will return a clear runtime error.

## OpenRouter (optional)
Set:
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`)

The MVP currently uses heuristic descriptions and only calls the LLM if configured.

## Notes
- The observation layer limits token usage and never forwards full page snapshots.
- The policy gate requires confirmation for non-whitelisted actions.
