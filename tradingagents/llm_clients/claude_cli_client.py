"""Claude CLI LangChain integration.

Wraps the `claude` CLI tool as a LangChain BaseChatModel, following the
subprocess pattern used by Nexio's adapter-claude-local. Supports tool
calling (bind_tools) required by TradingAgents analyst nodes.
"""

import json
import subprocess
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from .base_client import BaseLLMClient


def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    """Convert LangChain messages into a single text prompt for Claude CLI."""
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"[System]\n{msg.content}")
        elif isinstance(msg, HumanMessage):
            parts.append(f"[User]\n{msg.content}")
        elif isinstance(msg, AIMessage):
            text = msg.content or ""
            if msg.tool_calls:
                calls = json.dumps(msg.tool_calls, indent=2)
                text += f"\n[Tool Calls]\n{calls}"
            if text.strip():
                parts.append(f"[Assistant]\n{text}")
        elif isinstance(msg, ToolMessage):
            parts.append(f"[Tool Result: {msg.name}]\n{msg.content}")
        else:
            parts.append(str(msg.content))
    return "\n\n".join(parts)


def _build_tool_description(tools: Sequence[BaseTool]) -> str:
    """Build a text description of available tools for the prompt."""
    if not tools:
        return ""
    lines = [
        "You have the following tools available. To call a tool, respond with "
        "a JSON block in this exact format:\n"
        '```tool_call\n{"name": "<tool_name>", "arguments": {<args>}}\n```\n'
        "You may call multiple tools by including multiple such blocks.\n"
        "Available tools:"
    ]
    for tool in tools:
        schema = tool.args_schema.schema() if tool.args_schema else {}
        props = schema.get("properties", {})
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in props.items()
        )
        lines.append(f"- {tool.name}({params}): {tool.description}")
    return "\n".join(lines)


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool_call JSON blocks from Claude CLI response text."""
    tool_calls = []
    marker = "```tool_call"
    idx = 0
    while True:
        start = text.find(marker, idx)
        if start == -1:
            break
        json_start = text.find("\n", start) + 1
        end = text.find("```", json_start)
        if end == -1:
            break
        raw = text[json_start:end].strip()
        try:
            parsed = json.loads(raw)
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "name": parsed["name"],
                    "args": parsed.get("arguments", parsed.get("args", {})),
                }
            )
        except (json.JSONDecodeError, KeyError):
            pass
        idx = end + 3
    return tool_calls


def _parse_stream_json(stdout: str) -> str:
    """Parse Claude CLI --output-format stream-json output.

    Extracts assistant text blocks from the JSON stream, falling back to
    the final result summary.
    """
    assistant_texts = []
    result_text = ""

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            assistant_texts.append(text)

        elif etype == "result":
            result_text = event.get("result", "")

    return "\n\n".join(assistant_texts) if assistant_texts else result_text


class ClaudeCLIChatModel(BaseChatModel):
    """LangChain ChatModel that delegates to the `claude` CLI subprocess."""

    model: str = "claude-sonnet-4-6"
    claude_command: str = "claude"
    timeout_sec: int = 300
    _tools: List[BaseTool] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def bind_tools(self, tools: Sequence[Any], **kwargs) -> "ClaudeCLIChatModel":
        """Return a copy with tools bound for tool-calling analysts."""
        new = self.model_copy()
        new._tools = list(tools)
        return new

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        prompt = _messages_to_prompt(messages)

        if self._tools:
            tool_desc = _build_tool_description(self._tools)
            prompt = f"{tool_desc}\n\n{prompt}"

        prompt += (
            "\n\nIMPORTANT: Respond directly with your analysis. "
            "Do NOT use any Claude Code tools (Read, Write, Edit, Bash, etc). "
            "Only use the tool_call format described above if you need to call "
            "the financial data tools listed."
        )

        args = [
            self.claude_command,
            "--print", "-",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--max-turns", "1",
        ]

        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Claude CLI timed out after {self.timeout_sec}s"
            )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0 and not stdout:
            raise RuntimeError(
                f"Claude CLI failed (exit {result.returncode}): {stderr}"
            )

        response_text = _parse_stream_json(stdout)
        if not response_text:
            response_text = stdout

        tool_calls = _parse_tool_calls(response_text) if self._tools else []

        if tool_calls:
            clean_text = response_text
            for marker in ("```tool_call",):
                while marker in clean_text:
                    start = clean_text.find(marker)
                    end = clean_text.find("```", start + len(marker))
                    if end != -1:
                        clean_text = clean_text[:start] + clean_text[end + 3:]
                    else:
                        break
            clean_text = clean_text.strip()
        else:
            clean_text = response_text

        message = AIMessage(
            content=clean_text,
            tool_calls=tool_calls,
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "claude_command": self.claude_command}


def _clean_env():
    """Return a clean env dict stripping Claude Code nesting vars."""
    import os
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("CLAUDE_CODE_") or key == "CLAUDE_INNER":
            del env[key]
    return env


class ClaudeCLIClient(BaseLLMClient):
    """LLM client adapter for Claude CLI."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = "claude-cli"

    def get_llm(self) -> Any:
        return ClaudeCLIChatModel(
            model=self.model,
            timeout_sec=self.kwargs.get("timeout", 300),
        )

    def validate_model(self) -> bool:
        return "claude" in self.model.lower()
