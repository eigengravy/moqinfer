"""Shared inference core — engine creation, prompt building, token streaming.

Both the MoQ server (server.py) and REST/SSE server (rest_server.py) use
this module so the ONLY variable in benchmarks is the transport layer.
"""

import json
import re
import uuid

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract <tool_call>...</tool_call> blocks from model output.

    Returns (content_without_tool_calls, list_of_tool_calls).
    Handles Qwen3's JSON-in-XML format:
        <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    """
    tool_calls = []
    content = text
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments", {})),
                    },
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
        content = content.replace(match.group(0), "")
    # Strip <think>...</think> reasoning blocks from content
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content, tool_calls


def create_engine(seed: int = 42) -> tuple[AsyncLLM, object]:
    """Create vLLM AsyncLLM engine and tokenizer.

    Returns (engine, tokenizer).
    """
    engine_args = AsyncEngineArgs(model=MODEL, max_model_len=4096, seed=seed)
    engine = AsyncLLM.from_engine_args(engine_args)
    return engine, engine.tokenizer


async def run_inference(engine, tokenizer, req: dict):
    """Async generator yielding frames: start, token*, tool_calls?, done.

    Consumes the same request dict format as the MoQ protocol:
      - {"prompt": "...", "sampling_params": {...}}
      - {"messages": [...], "tools": [...], "sampling_params": {...}}

    Yields dicts with the standard frame protocol:
      - {"type": "start", "request_id": ..., "model": ...}
      - {"type": "token", "text": delta}
      - {"type": "tool_calls", "tool_calls": [...]}  (if tool calls found)
      - {"type": "done", "finish_reason": "stop"|"tool_calls"}
    """
    request_id = req.get("request_id", str(uuid.uuid4()))

    # Start frame
    yield {"type": "start", "request_id": request_id, "model": MODEL}

    # Build prompt: chat messages+tools or raw prompt
    messages = req.get("messages")
    tools = req.get("tools")
    if messages:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = req.get("prompt", "")

    # Stream tokens from vLLM
    params = SamplingParams(**(req.get("sampling_params") or {"max_tokens": 512}))
    prev_text = ""
    full_text = ""

    async for output in engine.generate(prompt, params, request_id=request_id):
        for o in output.outputs:
            if len(o.text) > len(prev_text):
                delta = o.text[len(prev_text):]
                prev_text = o.text
                full_text = o.text
                yield {"type": "token", "text": delta}

    # Check for tool calls in the output
    if tools:
        _content, tool_calls = parse_tool_calls(full_text)
    else:
        tool_calls = []

    if tool_calls:
        yield {"type": "tool_calls", "tool_calls": tool_calls}

    finish_reason = "tool_calls" if tool_calls else "stop"
    yield {"type": "done", "finish_reason": finish_reason}
