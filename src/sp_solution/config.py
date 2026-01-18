from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv



@dataclass(frozen=True)
class Settings:
    mcp_endpoint: str | None
    mcp_user_data_dir: str | None
    mcp_tool_launch: str
    mcp_tool_observe: str
    mcp_tool_click: str
    mcp_tool_type: str
    mcp_tool_scroll: str
    mcp_tool_wait: str
    mcp_tool_screenshot: str

    openrouter_api_key: str | None
    openrouter_model: str


DEFAULT_TOOL_LAUNCH = "browser_launch"
DEFAULT_TOOL_OBSERVE = "browser_snapshot"
DEFAULT_TOOL_CLICK = "browser_click"
DEFAULT_TOOL_TYPE = "browser_type"
DEFAULT_TOOL_SCROLL = "browser_press_key"
DEFAULT_TOOL_WAIT = "browser_wait_for"
DEFAULT_TOOL_SCREENSHOT = "browser_take_screenshot"


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        mcp_endpoint=os.getenv("MCP_ENDPOINT"),
        mcp_user_data_dir=os.getenv("MCP_USER_DATA_DIR"),
        mcp_tool_launch=os.getenv("MCP_TOOL_LAUNCH", DEFAULT_TOOL_LAUNCH),
        mcp_tool_observe=os.getenv("MCP_TOOL_OBSERVE", DEFAULT_TOOL_OBSERVE),
        mcp_tool_click=os.getenv("MCP_TOOL_CLICK", DEFAULT_TOOL_CLICK),
        mcp_tool_type=os.getenv("MCP_TOOL_TYPE", DEFAULT_TOOL_TYPE),
        mcp_tool_scroll=os.getenv("MCP_TOOL_SCROLL", DEFAULT_TOOL_SCROLL),
        mcp_tool_wait=os.getenv("MCP_TOOL_WAIT", DEFAULT_TOOL_WAIT),
        mcp_tool_screenshot=os.getenv("MCP_TOOL_SCREENSHOT", DEFAULT_TOOL_SCREENSHOT),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL"),
    )
