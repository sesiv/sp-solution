from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    mcp_endpoint: str | None
    mcp_user_data_dir: str | None
    openrouter_api_key: str | None
    openrouter_model: str | None
    openrouter_provider: str | None


def load_settings() -> Settings:
    load_dotenv()
    provider = os.getenv("OPENROUTER_PROVIDER")



    return Settings(
        mcp_endpoint=os.getenv("MCP_ENDPOINT"),
        mcp_user_data_dir=os.getenv("MCP_USER_DATA_DIR"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL"),
        openrouter_provider=provider,
    )
