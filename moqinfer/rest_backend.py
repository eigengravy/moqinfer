"""REST/SSE backend — same interface as AgentBackend, HTTP/SSE transport.

Talks to moqinfer.rest_server (which shares the exact same inference core
as the MoQ server), so the ONLY variable in benchmarks is transport.
"""

import asyncio
import json
import uuid
from typing import AsyncIterator, Callable, Optional

import httpx

from moqinfer.backend import ChatResult, GenerateResult


class RestBackend:
    """Inference backend over HTTP/SSE via moqinfer.rest_server.

    Drop-in replacement for AgentBackend — same ``connect / chat / generate / close``
    interface, but each request is an HTTP/SSE call instead of a MoQ broadcast.

    Usage:
        backend = await RestBackend.connect("http://localhost:8000")
        result = await backend.chat(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[...],
            tool_executor=my_executor,
        )
        await backend.close()
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._url: str = ""

    @classmethod
    async def connect(cls, url: str = "http://localhost:8000") -> "RestBackend":
        """Create a RestBackend pointing at a moqinfer.rest_server instance."""
        self = cls()
        self._url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        return self

    async def _request_stream(self, request_data: dict) -> AsyncIterator[dict]:
        """POST to /v1/inference, parse SSE frames."""
        async with self._client.stream(
            "POST", f"{self._url}/v1/inference", json=request_data
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield json.loads(data)

    # ── Simple text generation ──────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        sampling_params: Optional[dict] = None,
    ) -> GenerateResult:
        """Non-streaming text completion."""
        result = GenerateResult(text="", model="", finish_reason="")
        async for frame in self.generate_stream(
            prompt, max_tokens=max_tokens, sampling_params=sampling_params
        ):
            if frame["type"] == "start":
                result.model = frame.get("model", "unknown")
            elif frame["type"] == "token":
                result.text += frame["text"]
            elif frame["type"] == "done":
                result.finish_reason = frame.get("finish_reason", "unknown")
        return result

    async def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        sampling_params: Optional[dict] = None,
    ) -> AsyncIterator[dict]:
        """Streaming text completion — yields the same frame format as AgentBackend."""
        params = dict(sampling_params or {})
        params.setdefault("max_tokens", max_tokens)
        request_data = {
            "request_id": str(uuid.uuid4()),
            "prompt": prompt,
            "sampling_params": params,
        }
        async for frame in self._request_stream(request_data):
            yield frame

    # ── Chat with tool calling ──────────────────────────────────────────

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: int = 512,
        sampling_params: Optional[dict] = None,
    ) -> AsyncIterator[dict]:
        """Single-turn streaming chat — yields the same frame format as AgentBackend."""
        params = dict(sampling_params or {})
        params.setdefault("max_tokens", max_tokens)
        request_data = {
            "request_id": str(uuid.uuid4()),
            "messages": messages,
            "sampling_params": params,
        }
        if tools:
            request_data["tools"] = tools
        async for frame in self._request_stream(request_data):
            yield frame

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        tool_executor: Optional[Callable] = None,
        max_tokens: int = 512,
        max_rounds: int = 10,
        sampling_params: Optional[dict] = None,
    ) -> ChatResult:
        """Multi-turn chat with automatic tool execution — same API as AgentBackend.chat()."""
        messages = list(messages)  # don't mutate caller's list

        for _round in range(max_rounds):
            result = ChatResult(
                text="", model="", finish_reason="", tool_calls=[], messages=messages
            )
            async for frame in self.chat_stream(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                sampling_params=sampling_params,
            ):
                if frame["type"] == "start":
                    result.model = frame.get("model", "unknown")
                elif frame["type"] == "token":
                    result.text += frame["text"]
                elif frame["type"] == "tool_calls":
                    result.tool_calls = frame.get("tool_calls", [])
                elif frame["type"] == "done":
                    result.finish_reason = frame.get("finish_reason", "unknown")

            # No tool calls or no executor — return
            if result.finish_reason != "tool_calls" or not tool_executor:
                result.messages = messages
                return result

            # Append assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in result.tool_calls
                    ],
                }
            )

            # Execute each tool
            for tc in result.tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments", "")
                arguments = json.loads(raw_args) if raw_args else {}
                if asyncio.iscoroutinefunction(tool_executor):
                    tool_result = await tool_executor(name, arguments)
                else:
                    tool_result = tool_executor(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(tool_result),
                    }
                )

        # Exhausted max_rounds
        result.messages = messages
        return result

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
