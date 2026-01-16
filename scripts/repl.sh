#!/usr/bin/env bash
set -euo pipefail

docker compose up -d

if ! command -v zsh >/dev/null 2>&1; then
  echo "zsh not found. Run: workon sp_solution && sp-cli --server http://127.0.0.1:8000 --session demo" >&2
  exit 1
fi

exec zsh -ic 'workon sp_solution && exec sp-cli --server "${SP_SERVER:-http://127.0.0.1:8000}" --session "${SP_SESSION:-demo}"'
