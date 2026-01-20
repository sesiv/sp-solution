# sp-solution MVP

This repo implements the Solution Architecture Contract (SAC.md) as a minimal, extensible MVP.

## What is implemented
- FastAPI transport with WebSocket chat (`/ws/{session_id}`) and SSE events (`/events/{session_id}`).
- Session manager with per-session message queues and event broker (drop-oldest backpressure).
- MCP browser client supporting HTTP+SSE or WebSocket transport, init handshakes, and session reset.
- Observation pipeline that parses MCP snapshots (YAML or text), dedupes content, caps sizes, and detects overlays.
- Policy gate that auto-approves safe actions and asks confirmation for destructive clicks (EN/RU keywords).
- LangGraph agent loop: plan -> observe -> act with action validation, interrupts, and step limits.
- Optional OpenRouter LLM for planning/actions/state updates with structured output + screenshots.
- CLI REPL (Typer + prompt_toolkit + Rich) with live SSE logs + WebSocket chat/confirm flow.

## Implementation notes
### Agent loop
- Chat mode uses the LLM for plan/action/state updates; command mode executes single-step tool commands.
- Max tool steps: 100; invalid actions are blocked after 3 validation failures.
- Action validation ensures click/type targets exist and type actions hit editable roles only.

### Observation
- Parses MCP snapshot YAML when available; falls back to heuristic parsing from text.
- Caps interactive elements (160) and text blocks (30), trims text blocks to 200 chars, and dedupes content.
- Detects overlay-like text (cookie/privacy/subscribe/etc) and suggests candidate dismiss/accept actions.

### Policy & safety
- Allowlist covers observe/launch/scroll/wait/screenshot/stop/need_user.
- Clicks matching destructive keywords require user confirmation; other non-allowlisted actions require confirmation.

### Transport & events
- WebSocket message types: `user_message`, `user_confirm`, `control`.
- Server message types: `agent_message`, `agent_question`, `status`.
- SSE event types: `tool_call`, `tool_result`, `observation`, `policy_request`, `policy_result`, `error`.

### MCP integration
- Supports HTTP+SSE and WS endpoints; `MCP_ENDPOINT` is required.
- Tool mapping targets Playwright MCP defaults (`browser_snapshot`, `browser_click`, `browser_type`,
  `browser_press_key`, `browser_wait_for`, `browser_take_screenshot`, `browser_install`).
- Scroll adapts to `press_key`/`scroll`/`wheel` tool names to keep compatibility.

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

3) Start the CLI (use `--auto-launch` if you want to send `/launch` on startup):

```bash
sp-cli --server http://127.0.0.1:8000 --session demo
```

Then run `/launch` once if your MCP server needs the browser install/bootstrap step.

## Docker Compose
Build and run the API + MCP server:

```bash
docker compose up --build
```

Notes:
- To see the headed browser on Linux, allow X11 and pass through `DISPLAY`:
  `xhost +local:docker` before `docker compose up`.
- MCP downloads packages on first run via `npx`, so network access is required.
- If MCP fails to start, set `MCP_PACKAGE` or `MCP_CMD` in `.env` to override the startup command.
- For persistent sessions in Docker, set `MCP_ARGS` to include `--user-data-dir=/data` (default in `docker-compose.yml`).
- Chrome in Docker runs as root; `--no-sandbox` is included in the default `MCP_ARGS`.

## Host CLI (recommended for flaky Docker TTY)
Start the Docker services, then run the CLI on the host with the local TTY:

```bash
./scripts/repl.sh
```

This runs `docker compose up -d` and then executes `sp-cli` via `workon sp_solution`.
Override the defaults with `SP_SERVER` and `SP_SESSION`.

## CLI commands
- `/chat` to enter chat mode (natural language agent loop).
- `/exit` to leave chat mode and return to command mode.
- `/quit` to exit the REPL.
- `/launch` to run the MCP browser install/bootstrap tool.
- `/observe` to trigger a new observation cycle.
- `/click <eid>` to click an element.
- `/type <eid> <text>` to type into a field.
- `/scroll [direction] [amount]` to scroll (default: down 600).
- `/wait [ms]` to wait.
- `/screenshot` to request a screenshot.
- `/yes` or `/no` to confirm policy-gated actions.

Use `--auto-launch` or `SP_AUTO_LAUNCH=1` to send `/launch` automatically.

## MCP configuration
The browser layer expects a Playwright MCP server endpoint. Configure it using environment variables:

- `MCP_ENDPOINT`: HTTP(S) or WS(S) endpoint for the MCP server (e.g. `http://127.0.0.1:3333/mcp`).
- Tool names are bound to Playwright MCP defaults (`browser_snapshot`, `browser_click`, `browser_type`,
  `browser_press_key`, `browser_wait_for`, `browser_take_screenshot`, `browser_install`).
- `MCP_USER_DATA_DIR`: persistent profile directory for browser sessions (used by the MCP server).

If `MCP_ENDPOINT` is not set, the browser layer will return a clear runtime error.

## OpenRouter (optional)
Set:
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (e.g. `openai/gpt-4o-mini`)
- `OPENROUTER_PROVIDER` (optional comma-separated provider order)

Chat mode calls the LLM only if configured; command mode does not require it.

## Notes
- The observation layer limits token usage and never forwards full page snapshots.
- The policy gate requires confirmation for non-whitelisted actions.
- The agent enforces a 100-step tool limit and stops after repeated invalid actions.
