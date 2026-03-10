# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for vllm-mlx.

Handles Qwen's tool calling formats:
- XML style: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket style: [Calling tool: func_name({"arg": "value"})]
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["qwen", "qwen3"])
class QwenToolParser(ToolParser):
    """
    Tool call parser for Qwen models.

    Supports multiple Qwen tool call formats:
    - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Bracket: [Calling tool: func_name({"arg": "value"})]

    Used when --enable-auto-tool-choice --tool-call-parser qwen are set.
    """

    # Pattern for XML-style: <tool_call>{"json"}</tool_call>
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # Pattern for bracket-style: [Calling tool: func_name({...})]
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    # Possible tool call marker prefixes to buffer before confirming
    _TOOL_MARKERS = ["<tool_call>", "[Calling tool:"]

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._pending_content = ""

    def reset(self) -> None:
        super().reset()
        self._pending_content = ""

    @staticmethod
    def _ends_with_marker_prefix(text: str, markers: list[str]) -> bool:
        """Check if text ends with a prefix of any tool call marker."""
        for marker in markers:
            for length in range(1, len(marker)):
                if text.endswith(marker[:length]):
                    return True
        return False

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Qwen model response.
        """
        tool_calls = []

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(model_output)

        # Try bracket pattern first (Qwen3 style)
        bracket_matches = self.BRACKET_PATTERN.findall(cleaned_text)
        for name, args_str in bracket_matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                continue

        if bracket_matches:
            cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text).strip()

        # Try XML pattern (traditional Qwen style)
        xml_matches = self.XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if xml_matches:
            cleaned_text = self.XML_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming Qwen model output.
        """
        # Check for tool call markers
        has_tool_marker = (
            "<tool_call>" in current_text or "[Calling tool:" in current_text
        )

        if not has_tool_marker:
            # Check if trailing text could be the start of a tool marker
            # (e.g., "<", "<tool", "<tool_call"). Buffer until we know.
            if self._ends_with_marker_prefix(current_text, self._TOOL_MARKERS):
                self._pending_content += delta_text
                return None
            # Not a prefix — flush any pending content with this delta
            if self._pending_content:
                flushed = self._pending_content + delta_text
                self._pending_content = ""
                return {"content": flushed}
            return {"content": delta_text}

        # Clear pending buffer — it was a real tool call prefix
        self._pending_content = ""

        # If we're in a tool call, accumulate and parse at the end.
        # Check current_text (not delta_text) because closing tags like
        # </tool_call> are often split across multiple token deltas.
        # Only trigger when the closing tag is NEW (not already in previous_text).
        closing_is_new = (
            ("</tool_call>" in current_text and "</tool_call>" not in previous_text)
            or (")]" in current_text and ")]" not in previous_text)
        )
        if closing_is_new:
            # Tool call complete, parse the whole thing
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        return None
