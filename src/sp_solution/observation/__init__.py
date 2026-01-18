from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for snapshot parsing
    yaml = None

from ..models import InteractiveElement, Observation, OverlayAction, PageInfo

logger = logging.getLogger(__name__)


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
                overlays, overlay_actions = self._detect_overlays(text_blocks, interactive)
                return Observation(
                    page=page,
                    interactive=interactive,
                    text_blocks=text_blocks,
                    overlays=overlays,
                    overlay_actions=overlay_actions,
                )

        page = PageInfo(
            url=raw.get("page", {}).get("url") if isinstance(raw.get("page"), dict) else raw.get("url"),
            title=raw.get("page", {}).get("title") if isinstance(raw.get("page"), dict) else raw.get("title"),
        )

        interactive = self._extract_interactive(raw)
        text_blocks = self._extract_text_blocks(raw)
        overlays, overlay_actions = self._detect_overlays(text_blocks, interactive)

        return Observation(
            page=page,
            interactive=interactive,
            text_blocks=text_blocks,
            overlays=overlays,
            overlay_actions=overlay_actions,
        )

    def _parse_mcp_text(
        self, content: List[Any]
    ) -> tuple[PageInfo, List[InteractiveElement], List[str]] | None:
        text = self._content_to_text(content)
        if not text:
            return None
        lines = text.splitlines()
        sections = self._split_sections(lines)
        candidates: List[tuple[PageInfo, List[InteractiveElement], List[str]]] = []
        all_interactive: List[InteractiveElement] = []
        for section in sections:
            page = PageInfo(
                url=self._find_prefixed(section, "- Page URL:"),
                title=self._find_prefixed(section, "- Page Title:"),
            )
            snapshot_lines = self._extract_snapshot_block(section)
            if not snapshot_lines:
                continue
            snapshot_data = self._load_snapshot_structured(snapshot_lines)
            interactive = self._parse_snapshot_interactive(snapshot_lines, snapshot_data, limit=None)
            text_blocks = self._parse_snapshot_text_blocks(snapshot_lines, snapshot_data)
            candidates.append((page, interactive, text_blocks))
            all_interactive.extend(interactive)
        if not candidates:
            page = PageInfo(
                url=self._find_prefixed(lines, "- Page URL:"),
                title=self._find_prefixed(lines, "- Page Title:"),
            )
            snapshot_lines = self._extract_snapshot_block(lines)
            snapshot_data = self._load_snapshot_structured(snapshot_lines)
            interactive = self._parse_snapshot_interactive(snapshot_lines, snapshot_data, limit=None)
            text_blocks = self._parse_snapshot_text_blocks(snapshot_lines, snapshot_data)
            return page, interactive, text_blocks
        filtered = [item for item in candidates if self._has_content(item)] or candidates
        non_ads = [item for item in filtered if not self._is_ad_frame(item[0])] or filtered
        with_url = [item for item in non_ads if item[0].url]
        ranked = with_url or non_ads
        best = max(ranked, key=self._score_candidate)
        merged_interactive = self._merge_interactive_elements(all_interactive)
        return best[0], merged_interactive, best[2]

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
        blocks: List[List[str]] = []
        start = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```"):
                lang = stripped[3:].strip().lower()
                if start is None and lang in ("", "yaml", "yml"):
                    start = index + 1
                    continue
                if start is not None:
                    if start < index:
                        blocks.append(lines[start:index])
                    start = None
        if blocks:
            return max(blocks, key=len)
        for index, line in enumerate(lines):
            if line.strip().startswith("- Page Snapshot:"):
                return lines[index + 1 :]
        logger.warning("Snapshot block not found in MCP content.")
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

    def _is_ad_frame(self, page: PageInfo) -> bool:
        if not page.url:
            return False
        url = page.url.lower()
        return "safeframe" in url or "doubleclick" in url or "adservice" in url

    def _has_content(self, item: tuple[PageInfo, List[InteractiveElement], List[str]]) -> bool:
        return bool(item[1] or item[2])

    def _score_candidate(self, item: tuple[PageInfo, List[InteractiveElement], List[str]]) -> float:
        page, interactive, text_blocks = item
        text_count = len(text_blocks)
        interactive_count = len(interactive)
        score = interactive_count + 0.4 * text_count
        if page.url:
            score += 5.0
        else:
            score -= 2.0
        if page.title:
            score += 1.5
        if interactive_count == 0 and text_count > 0:
            score -= 1.0
        if text_count and interactive_count / max(text_count, 1) < 0.05:
            score -= 1.5
        return score

    def _parse_snapshot_interactive(
        self,
        lines: List[str],
        snapshot_data: Any | None = None,
        limit: int | None = None,
    ) -> List[InteractiveElement]:
        if snapshot_data is None:
            snapshot_data = self._load_snapshot_structured(lines)
        if snapshot_data is not None:
            return self._parse_snapshot_interactive_structured(snapshot_data, limit=limit)
        return self._parse_snapshot_interactive_heuristic(lines, limit=limit)

    def _parse_snapshot_text_blocks(
        self, lines: List[str], snapshot_data: Any | None = None
    ) -> List[str]:
        if snapshot_data is None:
            snapshot_data = self._load_snapshot_structured(lines)
        if snapshot_data is not None:
            return self._parse_snapshot_text_blocks_structured(snapshot_data)
        return self._parse_snapshot_text_blocks_heuristic(lines)

    def _load_snapshot_structured(self, lines: List[str]) -> Any | None:
        if yaml is None:
            return None
        text = "\n".join(lines).strip()
        if not text:
            return None
        try:
            return yaml.safe_load(text)
        except Exception as exc:
            logger.warning("Failed to parse snapshot YAML: %s", exc)
            return None

    def _parse_snapshot_interactive_structured(
        self, snapshot_data: Any, limit: int | None = None
    ) -> List[InteractiveElement]:
        result: List[InteractiveElement] = []
        seen: set[str] = set()
        for node in self._iter_snapshot_dicts(snapshot_data):
            eid = self._extract_eid(node)
            if not eid or eid in seen:
                continue
            attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else None
            role = self._extract_node_value(node, ("role",), attrs)
            name = self._extract_accessible_name(node, attrs)
            value = self._extract_node_value(node, ("value",), attrs)
            placeholder = self._extract_node_value(node, ("placeholder",), attrs)
            disabled = self._extract_disabled_from_node(node, attrs)
            visible = self._extract_visible_from_node(node, attrs)
            bbox = node.get("bbox") if isinstance(node.get("bbox"), dict) else None
            priority = self._priority_for_element(role, name, value, placeholder)
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
                    priority=priority,
                )
            )
            seen.add(eid)
            if limit is not None and len(result) >= limit:
                break
        return result

    def _parse_snapshot_interactive_heuristic(
        self, lines: List[str], limit: int | None = None
    ) -> List[InteractiveElement]:
        ref_re = re.compile(r"ref=([A-Za-z0-9_-]+)")
        role_re = re.compile(r"\brole:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
        name_re = re.compile(r"\bname:\s*\"([^\"]+)\"", re.IGNORECASE)
        quoted_re = re.compile(r"\"([^\"]+)\"")
        result: List[InteractiveElement] = []
        seen: set[str] = set()
        for line in lines:
            if "ref=" not in line:
                continue
            ref_match = ref_re.search(line)
            if not ref_match:
                continue
            eid = ref_match.group(1)
            if eid in seen:
                continue
            seen.add(eid)
            role_match = role_re.search(line)
            name_match = name_re.search(line) or quoted_re.search(line)
            disabled = self._bool_from_text(line, "disabled")
            aria_disabled = self._bool_from_text(line, "aria-disabled")
            disabled = self._first_non_none(disabled, aria_disabled)
            visible = self._bool_from_text(line, "visible")
            hidden = self._bool_from_text(line, "hidden")
            if visible is None and hidden is not None:
                visible = not hidden
            role = role_match.group(1) if role_match else None
            name = name_match.group(1) if name_match else None
            priority = self._priority_for_element(role, name, None, None)
            result.append(
                InteractiveElement(
                    eid=eid,
                    role=role,
                    name=name,
                    visible=visible,
                    disabled=disabled,
                    priority=priority,
                )
            )
            if limit is not None and len(result) >= limit:
                break
        return result

    def _parse_snapshot_text_blocks_structured(self, snapshot_data: Any) -> List[str]:
        blocks: List[str] = []
        seen: set[str] = set()
        for node in self._iter_snapshot_dicts(snapshot_data):
            attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else None
            for key in ("text", "name", "label", "aria-label", "title", "value"):
                value = node.get(key)
                self._maybe_add_text_block(value, blocks, seen)
                if attrs:
                    self._maybe_add_text_block(attrs.get(key), blocks, seen)
                if len(blocks) >= self.max_text_blocks:
                    return blocks
        return blocks

    def _parse_snapshot_text_blocks_heuristic(self, lines: List[str]) -> List[str]:
        name_re = re.compile(r"\"([^\"]+)\"")
        blocks: List[str] = []
        seen: set[str] = set()
        for line in lines:
            if "ref=" in line:
                continue
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = cleaned.lstrip("-").strip()
            if not cleaned or self._looks_like_yaml_key(cleaned):
                continue
            match = name_re.search(cleaned)
            text = match.group(1) if match else cleaned
            self._maybe_add_text_block(text, blocks, seen)
            if len(blocks) >= self.max_text_blocks:
                break
        return blocks

    def _iter_snapshot_dicts(self, snapshot_data: Any) -> Iterable[Dict[str, Any]]:
        stack: List[Any] = [snapshot_data]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                yield current
                for value in current.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)

    def _extract_eid(self, node: Dict[str, Any]) -> str | None:
        for key in ("ref", "eid", "id"):
            value = node.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return None

    def _extract_accessible_name(
        self, node: Dict[str, Any], attrs: Dict[str, Any] | None
    ) -> str | None:
        return self._first_non_none(
            self._extract_node_value(node, ("name",), attrs),
            self._extract_node_value(node, ("aria-label", "ariaLabel"), attrs),
            self._extract_node_value(node, ("label",), attrs),
            self._extract_node_value(node, ("title",), attrs),
            self._extract_node_value(node, ("text",), attrs),
        )

    def _extract_node_value(
        self,
        node: Dict[str, Any],
        keys: Tuple[str, ...],
        attrs: Dict[str, Any] | None,
    ) -> str | None:
        for key in keys:
            value = self._coerce_text(node.get(key))
            if value:
                return value
            if attrs:
                value = self._coerce_text(attrs.get(key))
                if value:
                    return value
        return None

    def _extract_visible_from_node(
        self, node: Dict[str, Any], attrs: Dict[str, Any] | None
    ) -> bool | None:
        visible = self._first_non_none(
            self._coerce_bool(node.get("visible")),
            self._coerce_bool(attrs.get("visible")) if attrs else None,
        )
        hidden = self._first_non_none(
            self._coerce_bool(node.get("hidden")),
            self._coerce_bool(attrs.get("hidden")) if attrs else None,
        )
        aria_hidden = self._first_non_none(
            self._coerce_bool(node.get("aria-hidden")),
            self._coerce_bool(attrs.get("aria-hidden")) if attrs else None,
        )
        if visible is None:
            if hidden is not None:
                return not hidden
            if aria_hidden is not None:
                return not aria_hidden
        return visible

    def _extract_disabled_from_node(
        self, node: Dict[str, Any], attrs: Dict[str, Any] | None
    ) -> bool | None:
        return self._first_non_none(
            self._coerce_bool(node.get("disabled")),
            self._coerce_bool(attrs.get("disabled")) if attrs else None,
            self._coerce_bool(node.get("aria-disabled")),
            self._coerce_bool(attrs.get("aria-disabled")) if attrs else None,
            self._coerce_bool(node.get("ariaDisabled")),
            self._coerce_bool(attrs.get("ariaDisabled")) if attrs else None,
        )

    def _first_non_none(self, *values: Any) -> Any | None:
        for value in values:
            if value is not None:
                return value
        return None

    def _coerce_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
        return None

    def _coerce_text(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return None
        if isinstance(value, list):
            parts = [self._safe_str(item) for item in value]
            parts = [part for part in parts if part]
            return " ".join(parts) if parts else None
        return self._safe_str(value)

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split())

    def _looks_like_yaml_key(self, text: str) -> bool:
        if re.match(r"^[A-Za-z0-9_-]+:\s*$", text):
            return True
        if re.match(r"^[A-Za-z0-9_-]+:\s*[\[{]\s*$", text):
            return True
        return False

    def _should_include_text(self, text: str) -> bool:
        if len(text) < 2:
            return False
        if self._looks_like_yaml_key(text):
            return False
        if not any(char.isalpha() for char in text) and " " not in text and len(text) < 6:
            return False
        return True

    def _maybe_add_text_block(self, value: Any, blocks: List[str], seen: set[str]) -> None:
        text = self._coerce_text(value)
        if not text:
            return
        normalized = self._normalize_text(text)
        if not normalized or not self._should_include_text(normalized):
            return
        trimmed = normalized[: self.max_text_len]
        key = trimmed.lower()
        if key in seen:
            return
        seen.add(key)
        blocks.append(trimmed)

    def _priority_for_element(
        self,
        role: str | None,
        name: str | None,
        value: str | None,
        placeholder: str | None,
    ) -> int | None:
        role_text = (role or "").lower()
        label = " ".join(part for part in (name, value, placeholder) if part).lower()
        if role_text in {"textbox", "searchbox", "combobox", "textarea", "input"}:
            return 3
        if any(
            keyword in label
            for keyword in (
                "sign in",
                "log in",
                "login",
                "sign up",
                "register",
                "continue",
                "next",
                "submit",
                "checkout",
                "pay",
                "buy",
                "confirm",
            )
        ):
            return 3
        if role_text in {"button", "link"} and label:
            return 2
        if role_text:
            return 1
        return None

    def _bool_from_text(self, line: str, key: str) -> bool | None:
        match = re.search(rf"{re.escape(key)}\s*[:=]\s*(true|false)", line, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return None

    def _element_label(self, element: InteractiveElement) -> str:
        parts = [element.name, element.value, element.placeholder, element.role]
        return " ".join(part for part in parts if part)

    def _merge_interactive_entry(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if key == "eid":
                continue
            if base.get(key) is None and value is not None:
                base[key] = value

    def _merge_interactive_elements(
        self, elements: Iterable[InteractiveElement]
    ) -> List[InteractiveElement]:
        entries: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for element in elements:
            data = element.model_dump()
            eid = data.get("eid")
            if not eid:
                continue
            if eid not in entries:
                entries[eid] = data
                order.append(eid)
            else:
                self._merge_interactive_entry(entries[eid], data)
        merged: List[InteractiveElement] = []
        for eid in order:
            entry = entries[eid]
            if entry.get("priority") is None:
                entry["priority"] = self._priority_for_element(
                    entry.get("role"),
                    entry.get("name"),
                    entry.get("value"),
                    entry.get("placeholder"),
                )
            merged.append(InteractiveElement(**entry))
        return merged

    def _extract_interactive(self, raw: Dict[str, Any]) -> List[InteractiveElement]:
        candidates: Iterable[Dict[str, Any]] = []
        for key in ("interactive", "elements", "nodes", "items"):
            if isinstance(raw.get(key), list):
                candidates = raw[key]
                break

        entries: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

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
            disabled = self._coerce_bool(item.get("disabled"))
            visible = self._coerce_bool(item.get("visible"))
            bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
            entry = {
                "eid": eid,
                "role": role,
                "name": name,
                "value": value,
                "placeholder": placeholder,
                "disabled": disabled,
                "visible": visible,
                "bbox": bbox,
            }
            if eid not in entries:
                entries[eid] = entry
                order.append(eid)
            else:
                self._merge_interactive_entry(entries[eid], entry)

        result: List[InteractiveElement] = []
        for eid in order:
            entry = entries[eid]
            priority = self._priority_for_element(
                entry.get("role"),
                entry.get("name"),
                entry.get("value"),
                entry.get("placeholder"),
            )
            result.append(
                InteractiveElement(
                    eid=eid,
                    role=entry.get("role"),
                    name=entry.get("name"),
                    value=entry.get("value"),
                    placeholder=entry.get("placeholder"),
                    disabled=entry.get("disabled"),
                    visible=entry.get("visible"),
                    bbox=entry.get("bbox"),
                    priority=priority,
                )
            )
            if len(result) >= self.max_interactive:
                break

        return result

    def _extract_text_blocks(self, raw: Dict[str, Any]) -> List[str]:
        blocks: List[str] = []
        seen: set[str] = set()
        candidates: Iterable[Any] = []
        for key in ("text_blocks", "text", "visible_text", "content"):
            if isinstance(raw.get(key), list):
                candidates = raw[key]
                break

        for item in candidates:
            value: Any = item
            if isinstance(item, dict):
                for key in ("text", "value", "name", "label", "title", "aria-label"):
                    if key in item:
                        value = item.get(key)
                        break
            self._maybe_add_text_block(value, blocks, seen)
            if len(blocks) >= self.max_text_blocks:
                break

        return blocks

    def _detect_overlays(
        self, text_blocks: List[str], interactive: List[InteractiveElement]
    ) -> tuple[List[str], List[OverlayAction]]:
        overlays: List[str] = []
        overlay_actions: List[OverlayAction] = []
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
        action_keywords = [
            "accept",
            "agree",
            "allow",
            "close",
            "dismiss",
            "reject",
            "decline",
            "deny",
            "continue",
            "ok",
            "got it",
        ]
        for text in text_blocks:
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                overlays.append(text)
        if overlays:
            candidate_eids: List[str] = []
            for element in interactive:
                label = self._element_label(element).lower()
                if not label:
                    continue
                if any(keyword in label for keyword in action_keywords):
                    if element.disabled is True:
                        continue
                    candidate_eids.append(element.eid)
            if candidate_eids:
                deduped = list(dict.fromkeys(candidate_eids))[:8]
                for text in overlays[:5]:
                    overlay_actions.append(OverlayAction(text=text, eids=deduped))
        return overlays[:5], overlay_actions

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
