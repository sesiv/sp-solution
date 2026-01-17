#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-3333}"
HEADLESS="${HEADLESS:-false}"
MCP_ARGS="${MCP_ARGS:-}"

if [[ -d "/data" ]]; then
  if command -v pgrep >/dev/null 2>&1; then
    if ! pgrep -f '/opt/google/chrome' >/dev/null 2>&1; then
      rm -f /data/SingletonLock /data/SingletonSocket /data/SingletonCookie
    fi
  else
    rm -f /data/SingletonLock /data/SingletonSocket /data/SingletonCookie
  fi
fi

args=(--port "${PORT}")
case "${HEADLESS}" in
  true|TRUE|1|yes|YES)
    args+=(--headless)
    ;;
esac

if [[ -n "${MCP_ARGS}" ]]; then
  # shellcheck disable=SC2206
  args+=(${MCP_ARGS})
fi

if [[ -n "${MCP_CMD:-}" ]]; then
  exec ${MCP_CMD}
fi

if command -v mcp-server-playwright >/dev/null 2>&1; then
  exec mcp-server-playwright "${args[@]}"
fi

if command -v playwright-mcp >/dev/null 2>&1; then
  exec playwright-mcp "${args[@]}"
fi

if command -v mcp-playwright >/dev/null 2>&1; then
  exec mcp-playwright "${args[@]}"
fi

if [[ -n "${MCP_PACKAGE:-}" ]]; then
  exec npx --yes "${MCP_PACKAGE}" "${args[@]}"
fi

exec npx --yes github:microsoft/playwright-mcp "${args[@]}"
