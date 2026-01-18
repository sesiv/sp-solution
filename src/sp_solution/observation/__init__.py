from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from ..models import InteractiveElement, Observation, PageInfo


class ObservationBuilder:
    def __init__(
        self,
        max_interactive: int = 160,
        max_text_blocks: int = 30,
        max_text_len: int = 200,
    ) -> None:
        self.max_interactive = max_interactive
        self.max_text_blocks = max_text_blocks
        self.max_text_len = max_text_len

    def build(self, raw: Dict[str, Any]) -> Observation:
        if isinstance(raw.get("content"), list):
            parsed = self._parse_mcp_text(raw.get("content", []))
            if parsed:
                page, interactive, text_blocks = parsed
                overlays = self._detect_overlays(text_blocks)
                return Observation(
                    page=page,
                    interactive=interactive,
                    text_blocks=text_blocks,
                    overlays=overlays,
                )

        page = PageInfo(
            url=raw.get("page", {}).get("url") if isinstance(raw.get("page"), dict) else raw.get("url"),
            title=raw.get("page", {}).get("title") if isinstance(raw.get("page"), dict) else raw.get("title"),
        )

        interactive = self._extract_interactive(raw)
        text_blocks = self._extract_text_blocks(raw)
        overlays = self._detect_overlays(text_blocks)

        return Observation(page=page, interactive=interactive, text_blocks=text_blocks, overlays=overlays)

    def _parse_mcp_text(
        self, content: List[Any]
    ) -> tuple[PageInfo, List[InteractiveElement], List[str]] | None:
        text = self._content_to_text(content)
        if not text:
            return None
        lines = text.splitlines()
        sections = self._split_sections(lines)
        candidates: List[tuple[PageInfo, List[InteractiveElement], List[str]]] = []
        for section in sections:
            page = PageInfo(
                url=self._find_prefixed(section, "- Page URL:"),
                title=self._find_prefixed(section, "- Page Title:"),
            )
            snapshot_lines = self._extract_snapshot_block(section)
            if not snapshot_lines:
                continue
            interactive = self._parse_snapshot_interactive(snapshot_lines)
            text_blocks = self._parse_snapshot_text_blocks(snapshot_lines)
            candidates.append((page, interactive, text_blocks))
        if not candidates:
            page = PageInfo(
                url=self._find_prefixed(lines, "- Page URL:"),
                title=self._find_prefixed(lines, "- Page Title:"),
            )
            snapshot_lines = self._extract_snapshot_block(lines)
            interactive = self._parse_snapshot_interactive(snapshot_lines)
            text_blocks = self._parse_snapshot_text_blocks(snapshot_lines)
            return page, interactive, text_blocks
        best = max(candidates, key=lambda item: (len(item[1]), len(item[2])))
        return best

    def _content_to_text(self, content: List[Any]) -> str:
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    def _find_prefixed(self, lines: List[str], prefix: str) -> str | None:
        for line in lines:
            if line.strip().startswith(prefix):
                return line.split(prefix, 1)[1].strip() or None
        return None

    def _extract_snapshot_block(self, lines: List[str]) -> List[str]:
        start = None
        end = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```") and "yaml" in stripped and start is None:
                start = index + 1
                continue
            if stripped.startswith("```") and start is not None:
                end = index
                break
        if start is not None and end is not None and start < end:
            return lines[start:end]
        for index, line in enumerate(lines):
            if line.strip().startswith("- Page Snapshot:"):
                return lines[index + 1 :]
        return []

    def _split_sections(self, lines: List[str]) -> List[List[str]]:
        indices: List[int] = []
        for index, line in enumerate(lines):
            if line.strip().startswith("- Page URL:"):
                indices.append(index)
        if not indices:
            return [lines]
        sections: List[List[str]] = []
        for i, start in enumerate(indices):
            end = indices[i + 1] if i + 1 < len(indices) else len(lines)
            sections.append(lines[start:end])
        return sections

    def _parse_snapshot_interactive(self, lines: List[str]) -> List[InteractiveElement]:
        ref_re = re.compile(r"ref=([A-Za-z0-9_-]+)")
        role_re = re.compile(r"-\s*([A-Za-z0-9_]+)")
        name_re = re.compile(r"\"([^\"]+)\"")
        result: List[InteractiveElement] = []
        seen: set[str] = set()
        for line in lines:
            if "ref=" not in line:
                continue
            ref_match = ref_re.search(line)
            if not ref_match:
                continue
            lowered = line.lower()
            disabled = "disabled" in lowered or "aria-disabled=true" in lowered
            eid = ref_match.group(1)
            if eid in seen:
                continue
            seen.add(eid)
            role_match = role_re.search(line)
            name_match = name_re.search(line)
            result.append(
                InteractiveElement(
                    eid=eid,
                    role=role_match.group(1) if role_match else None,
                    name=name_match.group(1) if name_match else None,
                    visible=True,
                    disabled=disabled,
                )
            )
            if len(result) >= self.max_interactive:
                break
        return result

    def _parse_snapshot_text_blocks(self, lines: List[str]) -> List[str]:
        name_re = re.compile(r"\"([^\"]+)\"")
        blocks: List[str] = []
        seen: set[str] = set()
        for line in lines:
            if "ref=" in line:
                continue
            cleaned = line.strip().lstrip("-").strip()
            if not cleaned:
                continue
            match = name_re.search(cleaned)
            text = match.group(1) if match else cleaned
            text = text[: self.max_text_len]
            if text in seen:
                continue
            seen.add(text)
            blocks.append(text)
            if len(blocks) >= self.max_text_blocks:
                break
        return blocks

    def _extract_interactive(self, raw: Dict[str, Any]) -> List[InteractiveElement]:
        candidates: Iterable[Dict[str, Any]] = []
        for key in ("interactive", "elements", "nodes", "items"):
            if isinstance(raw.get(key), list):
                candidates = raw[key]
                break

        dedupe: set[Tuple[str | None, str | None, str | None, str | None]] = set()
        result: List[InteractiveElement] = []

        for item in candidates:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("eid") or item.get("id") or item.get("node_id") or "")
            if not eid:
                continue
            role = self._safe_str(item.get("role"))
            name = self._safe_str(item.get("name") or item.get("label"))
            value = self._safe_str(item.get("value"))
            placeholder = self._safe_str(item.get("placeholder"))
            disabled = item.get("disabled")
            visible = item.get("visible")
            bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None

            signature = (role, name, value, placeholder)
            if signature in dedupe:
                continue
            dedupe.add(signature)

            result.append(
                InteractiveElement(
                    eid=eid,
                    role=role,
                    name=name,
                    value=value,
                    placeholder=placeholder,
                    disabled=disabled,
                    visible=visible,
                    bbox=bbox,
                )
            )
            if len(result) >= self.max_interactive:
                break

        return result

    def _extract_text_blocks(self, raw: Dict[str, Any]) -> List[str]:
        blocks: List[str] = []
        candidates: Iterable[Any] = []
        for key in ("text_blocks", "text", "visible_text", "content"):
            if isinstance(raw.get(key), list):
                candidates = raw[key]
                break

        for item in candidates:
            text = self._safe_str(item)
            if not text:
                continue
            trimmed = text[: self.max_text_len]
            blocks.append(trimmed)
            if len(blocks) >= self.max_text_blocks:
                break

        return blocks

    def _detect_overlays(self, text_blocks: List[str]) -> List[str]:
        overlays: List[str] = []
        keywords = [
            "cookie",
            "privacy",
            "subscribe",
            "sign up",
            "newsletter",
            "geo",
            "location",
            "permission",
        ]
        for text in text_blocks:
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                overlays.append(text)
        return overlays[:5]

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
