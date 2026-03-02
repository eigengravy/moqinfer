"""MoQ agents backend — multiplexes concurrent user sessions over a single MoQ connection."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import moq_py

BROADCAST_NAME = "inference"


@dataclass
class GenerateResult:
    """Accumulated result from a non-streaming generate call."""

    text: str
    model: str
    finish_reason: str


@dataclass
class ChatResult:
    """Result from a chat call, potentially with tool calls."""

    text: str
    model: str
    finish_reason: str
    tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


class AgentBackend:
    """Multi-user inference backend over a single persistent MoQ connection.

    All requests and responses flow over a single MoQ connection through
    the relay. Requests are sent as broadcasts (one per request), and all
    responses come back as groups on a single "data" track on the server's
    "inference" broadcast — demuxed by request_id.

    Usage:
        backend = await AgentBackend.connect("https://localhost:4443/")

        # Simple text generation
        result = await backend.generate("What is life?")
        print(result.text)

        # Chat with tool calling
        result = await backend.chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[{"type": "function", "function": {"name": "calc", ...}}],
            tool_executor=my_tool_executor,
        )
        print(result.text)

        await backend.close()
    """

    def __init__(self):
        self._session = None
        self._req_origin = None
        self._resp_origin = None
        self._server_bc = None
        self._resp_track = None
        self._announce_task = None
        self._dispatch_task = None
        self._server_found = None
        self._response_waiters = {}  # request_id -> Future[(first_frame, group)]
        self._closed = False

    @classmethod
    async def connect(
        cls,
        url: str = "https://localhost:4443/",
        *,
        tls_disable_verify: bool = True,
    ) -> "AgentBackend":
        """Create and connect an AgentBackend to the relay."""
        self = cls()
        self._server_found = asyncio.Event()

        # Create origins
        self._req_origin = moq_py.MoqOrigin.produce()
        self._resp_origin = moq_py.MoqOrigin.produce()

        # Connect — single MoQ connection for all requests
        config = moq_py.MoqClientConfig(tls_disable_verify=tls_disable_verify)
        client = moq_py.MoqClient.create(config)
        client.with_publish(self._req_origin.consume())
        client.with_consume(self._resp_origin)
        self._session = await client.connect(url)

        # Discover server broadcast
        self._announce_task = asyncio.create_task(self._announce_loop())
        await self._server_found.wait()

        # Subscribe to server's response track and start dispatching
        self._resp_track = self._server_bc.subscribe_track("data")
        self._dispatch_task = asyncio.create_task(self._dispatch_responses())

        return self

    async def _announce_loop(self):
        """Discover the server's 'inference' broadcast."""
        resp_consumer = self._resp_origin.consume()
        while not self._closed:
            announce = await resp_consumer.announced()
            if announce is None:
                break
            path, bc = announce
            if bc is None:
                continue
            if path == BROADCAST_NAME:
                self._server_bc = bc
                self._server_found.set()
                return

    async def _dispatch_responses(self):
        """Read response groups from the server's data track and dispatch by request_id."""
        while not self._closed:
            group = await self._resp_track.next_group()
            if group is None:
                break
            # First frame contains the start message with request_id
            first_frame = await group.read_frame()
            if first_frame is None:
                continue
            parsed = json.loads(first_frame)
            request_id = parsed.get("request_id")
            if request_id and request_id in self._response_waiters:
                self._response_waiters.pop(request_id).set_result(
                    (first_frame, group)
                )

    async def _request_stream(self, request_data: dict) -> AsyncIterator[dict]:
        """Send a request over MoQ and yield response frames.

        Low-level helper used by both generate_stream() and chat_stream().
        """
        if self._closed:
            raise RuntimeError("AgentBackend is closed")

        request_id = request_data["request_id"]

        # Register waiter for response BEFORE sending request
        future = asyncio.get_event_loop().create_future()
        self._response_waiters[request_id] = future

        # Create a broadcast for this request (one broadcast per request)
        req_bc = self._req_origin.create_broadcast(request_id)
        req_bc.dynamic()
        req_track = req_bc.create_track("data")
        req_group = req_track.append_group()
        req_group.write_frame(json.dumps(request_data).encode())
        req_group.finish()

        # Wait for our response group (dispatched by _dispatch_responses)
        first_frame, resp_group = await future

        # Yield the first frame (start message) then remaining frames
        yield json.loads(first_frame)
        while True:
            frame_data = await resp_group.read_frame()
            if frame_data is None:
                break
            yield json.loads(frame_data)

    # ── Simple text generation ──────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        sampling_params: Optional[dict] = None,
    ) -> GenerateResult:
        """Send an inference request and return the complete result."""
        result = GenerateResult(text="", model="", finish_reason="")
        async for frame in self.generate_stream(
            prompt,
            max_tokens=max_tokens,
            sampling_params=sampling_params,
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
        """Send an inference request and yield response frames as they arrive."""
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
        """Single-turn chat request. Sends messages+tools, yields response frames."""
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
        """Multi-turn chat with automatic tool execution loop.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI format.
            tool_executor: Async callable (name, arguments_dict) -> str.
                Called when the model invokes a tool. If None and the model
                returns tool_calls, returns immediately with finish_reason="tool_calls".
            max_tokens: Maximum tokens per generation turn.
            max_rounds: Maximum tool-calling rounds before stopping.
            sampling_params: Additional sampling parameters.

        Returns:
            ChatResult with final text, tool calls, and full conversation history.
        """
        messages = list(messages)  # don't mutate caller's list

        for _round in range(max_rounds):
            # Single-turn request
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

            # No tool calls or no executor — return as-is
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

            # Execute each tool and append results
            for tc in result.tool_calls:
                name = tc["function"]["name"]
                arguments = json.loads(tc["function"]["arguments"])
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
        """Close the backend connection."""
        self._closed = True
        for task in [self._announce_task, self._dispatch_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._session is not None:
            del self._session
            self._session = None
