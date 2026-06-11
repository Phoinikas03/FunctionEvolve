"""
LLM API client factory + usage statistics logging.

Supports OpenAI-compatible interfaces (including vLLM local services and
remote OpenAI/compatible services), as well as a custom Requests direct mode.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import requests


JSON_ONLY_OUTPUT_INSTRUCTION = (
    "Return only a JSON array/object. No markdown, no explanations."
)
_GUIDED_JSON_CLIENT_IDS: set[int] = set()


def _env_flag(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_enable_guided_json(mode: str, base_url: Optional[str]) -> bool:
    override = _env_flag("LLM_GUIDED_JSON")
    if override is not None:
        return override
    if mode == "vllm":
        return True
    if not base_url:
        return False
    lower_url = base_url.lower()
    return (
        "127.0.0.1" in lower_url
        or "localhost" in lower_url
        or "vllm" in lower_url
    )


def add_guided_json_schema(
    create_kwargs: Dict[str, Any],
    schema: Optional[Dict[str, Any]],
    api_client: Any,
) -> Dict[str, Any]:
    """Attach a vLLM guided_json schema via OpenAI SDK extra_body when enabled."""
    guided_enabled = (
        getattr(api_client, "_guided_json_enabled", False)
        or id(api_client) in _GUIDED_JSON_CLIENT_IDS
    )
    if not schema or not guided_enabled:
        return create_kwargs
    extra_body = create_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        extra_body = dict(extra_body)
    else:
        extra_body = {}
    extra_body["guided_json"] = schema
    create_kwargs["extra_body"] = extra_body
    return create_kwargs


def with_json_only_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of chat messages with a strict JSON-only output instruction."""
    updated = [dict(msg) for msg in messages]
    for msg in updated:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            content = msg["content"].rstrip()
            if JSON_ONLY_OUTPUT_INSTRUCTION not in content:
                msg["content"] = f"{content}\n\n{JSON_ONLY_OUTPUT_INSTRUCTION}"
            return updated
    return [{"role": "system", "content": JSON_ONLY_OUTPUT_INSTRUCTION}] + updated


def _mark_guided_json_support(client: Any, enabled: bool) -> None:
    if enabled:
        _GUIDED_JSON_CLIENT_IDS.add(id(client))
    try:
        client._guided_json_enabled = enabled
    except Exception:
        pass


class LLMUsageLogger:
    """Thread-safe LLM call usage CSV logger.

    CSV columns: timestamp, component, prompt_tokens, completion_tokens,
                 total_tokens, duration_s
    """

    _FIELDNAMES = [
        "timestamp", "component",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "duration_s",
    ]

    def __init__(self, csv_path: str):
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        self._lock = threading.Lock()
        self._closed = False

    def log(
        self,
        component: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        duration_s: float,
    ) -> None:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "component": component,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": f"{duration_s:.2f}",
        }
        with self._lock:
            if self._closed or self._file.closed:
                return
            try:
                self._writer.writerow(row)
                self._file.flush()
            except (OSError, ValueError):
                self._closed = True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._file.close()
            except OSError:
                pass


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


@lru_cache(maxsize=32)
def _tiktoken_encoding(model: str) -> Any:
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


# On networks that cannot reach OpenAI's blob storage, the first cl100k_base load
# hangs forever inside an SSL handshake (tiktoken downloads the BPE vocab with no
# timeout), stalling the whole run *before* any LLM request is sent. Bound that
# load and disable tiktoken process-wide on failure.
_TIKTOKEN_STATE = {"disabled": _env_flag("LLM_DISABLE_TIKTOKEN") is True}


def _try_tiktoken_encoding(model: str, timeout: Optional[float] = None) -> Optional[Any]:
    """Return a tiktoken encoding, or None if it cannot load quickly.

    Set LLM_DISABLE_TIKTOKEN=1 to skip tiktoken entirely. Otherwise the first load
    is bounded by ``timeout`` (default 6s, override with LLM_TIKTOKEN_TIMEOUT) so an
    offline BPE download cannot hang the process; the cl100k_base vocab is ~1.6 MB,
    a few seconds on any working link, and is cached on disk after the first
    success. On timeout/failure tiktoken is disabled for the rest of the run and
    callers fall back to a local tokenizer or a character heuristic.
    """
    if _TIKTOKEN_STATE["disabled"]:
        return None
    if timeout is None:
        raw = os.getenv("LLM_TIKTOKEN_TIMEOUT")
        try:
            timeout = float(raw) if raw else 6.0
        except ValueError:
            timeout = 6.0
        if timeout <= 0:
            timeout = 6.0
    box: Dict[str, Any] = {}

    def _work() -> None:
        try:
            box["enc"] = _tiktoken_encoding(model)
        except BaseException as exc:  # noqa: BLE001 - report and fall back
            box["err"] = exc

    th = threading.Thread(target=_work, name="tiktoken-load", daemon=True)
    th.start()
    th.join(timeout)
    if "enc" in box:
        return box["enc"]
    _TIKTOKEN_STATE["disabled"] = True
    reason = (
        "timed out (no network to openaipublic blob?)"
        if "err" not in box
        else f"{type(box['err']).__name__}: {box['err']}"
    )
    print(
        f"[LLMClient] tiktoken cl100k_base unavailable ({reason}); using local "
        f"tokenizer / char heuristic for token counting from now on.",
        flush=True,
    )
    return None


@lru_cache(maxsize=32)
def _transformers_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def count_prompt_tokens(model: str, messages: list[dict[str, Any]]) -> int:
    """Estimate chat prompt tokens for pre-request completion budgeting.

    tiktoken is used first. For non-OpenAI model names, it falls back to the
    cl100k_base encoding. If tiktoken is unavailable, a locally cached
    transformers tokenizer is used when possible; otherwise this falls back to
    a conservative character estimate.
    """
    enc = _try_tiktoken_encoding(model)
    if enc is not None:
        try:
            tokens = 3
            for msg in messages:
                tokens += 3
                for key in ("role", "name"):
                    value = msg.get(key)
                    if value:
                        tokens += len(enc.encode(str(value)))
                tokens += len(enc.encode(_content_to_text(msg.get("content"))))
            return tokens
        except Exception:
            pass

    prompt_text = "\n".join(
        f"{msg.get('role', '')}: {_content_to_text(msg.get('content'))}"
        for msg in messages
    )
    # Only use the transformers tokenizer if it is already imported (or explicitly
    # opted in via LLM_USE_TRANSFORMERS_TOKENIZER=1). Importing transformers solely
    # to count tokens can stall for a long time on slow/networked filesystems (the
    # import stats every installed package), and for vLLM-served model names the
    # local tokenizer almost never resolves anyway. Otherwise use a char heuristic.
    if "transformers" in sys.modules or _env_flag("LLM_USE_TRANSFORMERS_TOKENIZER"):
        try:
            tokenizer = _transformers_tokenizer(model)
            return len(tokenizer.encode(prompt_text, add_special_tokens=True))
        except Exception:
            pass
    return max(1, (len(prompt_text) + 3) // 4)


def resolve_completion_max_tokens(
    model: str,
    messages: list[dict[str, Any]],
    total_max_tokens: Optional[int],
) -> Optional[int]:
    """Convert a total token budget into the API completion-token limit."""
    if total_max_tokens is None:
        return None
    prompt_tokens = count_prompt_tokens(model, messages)
    completion_tokens = int(total_max_tokens) - prompt_tokens
    if completion_tokens <= 0:
        raise ValueError(
            f"max_tokens={total_max_tokens} is not enough for prompt_tokens={prompt_tokens}; "
            "increase max_tokens in llm_config.yaml."
        )
    return completion_tokens


# ================================================================
# Requests-mode lightweight wrapper
# ================================================================

class _RequestsChatCompletions:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def create(self, model: str, messages: list, **kwargs) -> Any:
        # Pull the request timeout out of kwargs so it is applied to the HTTP
        # call instead of being forwarded into the JSON payload.
        request_timeout = kwargs.pop("timeout", None)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # ✅ Handle Anthropic Vertex API requirement: extract "system" role from messages
        # Anthropic requires a top-level `system` parameter, not "system" as a message role
        payload_messages = messages
        system_prompt = None
        
        if "anthropic" in self.base_url:
            # Extract system messages and move them to top-level parameter
            new_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content")
                else:
                    new_messages.append(msg)
            payload_messages = new_messages
        
        payload: Dict[str, Any] = {"messages": payload_messages}
        payload.update(kwargs)
        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict) and "anthropic" not in self.base_url:
            payload.update(extra_body)
        
        # Add system parameter if extracted
        if system_prompt:
            payload["system"] = system_prompt

        # ✅ Handle GPT-5.2 parameter name change: max_tokens -> max_completion_tokens
        if "gpt-5" in model.lower():
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")

        # ✅ Handle reasoning_effort parameter for advanced reasoning models
        # Note: For o1-like models with reasoning_effort:
        # - temperature MUST be 1.0 (default only, no customization allowed)
        # - Cannot use top_p or frequency_penalty
        has_reasoning_effort = "reasoning_effort" in payload
        if has_reasoning_effort and "gpt-5" in model.lower():
            # GPT-5.2 with reasoning_effort REQUIRES temperature=1.0
            payload["temperature"] = 1.0  # ← MUST be 1.0, no other values allowed
            # Remove incompatible parameters
            payload.pop("top_p", None)
            payload.pop("top_k", None)
            payload.pop("frequency_penalty", None)
            payload.pop("presence_penalty", None)
            print(f"[Reasoning Mode] GPT-5.2 reasoning_effort={payload['reasoning_effort']}, forced temperature=1.0")

        # Anthropic Vertex rawPredict endpoints don't accept a model field
        if "anthropic" not in self.base_url:
            payload["model"] = model

        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload,
                timeout=request_timeout)
        except requests.RequestException as exc:
            print(
                f"[API Error] request to {self.base_url} failed after "
                f"timeout={request_timeout}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        if response.status_code != 200:
            print(f"[API Error] HTTP {response.status_code} - {response.text}", flush=True)
        response.raise_for_status()

        data = response.json()
        
        # ✅ Debug: print response structure if content seems empty
        if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
            print(f"[API Warning] Empty response received. Full response: {data}")

        class _Msg:
            def __init__(self, content: str):
                self.content = content

        class _Choice:
            def __init__(self, message: str):
                self.message = _Msg(message)

        class _Usage:
            def __init__(self, p: int, c: int, t: int):
                self.prompt_tokens = p
                self.completion_tokens = c
                self.total_tokens = t

        class _Response:
            def __init__(self, d: dict):
                content = ""
                if "choices" in d and d["choices"]:
                    content = d["choices"][0]["message"]["content"]
                elif "content" in d and isinstance(d["content"], list):
                    content = d["content"][0].get("text", "")
                self.choices = [_Choice(content)]

                usage = d.get("usage", {})
                p = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                c = usage.get("completion_tokens",
                              usage.get("output_tokens", 0))
                t = usage.get("total_tokens", p + c)
                self.usage = _Usage(p, c, t)

        return _Response(data)


class _RequestsChat:
    def __init__(self, base_url: str, api_key: str):
        self.completions = _RequestsChatCompletions(base_url, api_key)


class RequestsLLMClient:
    """Lightweight requests-based client mimicking the OpenAI client interface."""

    def __init__(self, base_url: str, api_key: str):
        self.chat = _RequestsChat(base_url, api_key)


# ================================================================
# Client factory
# ================================================================

def build_openai_client(
    model: str,
    base_url: Optional[str] = None,
    mode: str = "openai",
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Build an LLM client (OpenAI-compatible, vLLM, or Requests direct).

    Parameters
    ----------
    model    : model name (for logging / payload)
    base_url : API endpoint URL
    mode     : 'openai' | 'vllm' | 'requests'
    api_key  : explicit API key; falls back to OPENAI_API_KEY

    API keys may be provided directly or as ${ENV_VAR}; if omitted, the
    OPENAI_API_KEY environment variable is used. Request timeout can be
    provided directly or via LLM_REQUEST_TIMEOUT.
    """
    def _resolve_key(value: Optional[str]) -> str:
        if not value:
            return os.getenv("OPENAI_API_KEY", "")
        if value.startswith("${") and value.endswith("}"):
            return os.getenv(value[2:-1], "")
        return value

    resolved_key = _resolve_key(api_key)

    def _normalize_vllm_base_url(url: Optional[str]) -> str:
        resolved = (url or "http://localhost:8000").rstrip("/")
        if resolved.endswith("/v1"):
            return resolved
        return f"{resolved}/v1"

    # ---- Requests mode ----
    if mode == "requests":
        if not base_url:
            raise ValueError("Requests mode requires an explicit base_url.")
        if not resolved_key:
            raise ValueError(
                "Requests mode requires api_key or OPENAI_API_KEY environment variable.")
        print(f"[LLMClient] Requests mode, URL: {base_url}, model: {model}")
        client = RequestsLLMClient(base_url=base_url, api_key=resolved_key)
        _mark_guided_json_support(
            client,
            _should_enable_guided_json(mode, base_url),
        )
        return client

    # ---- OpenAI SDK mode (openai / vllm) ----
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "Please install the openai library: pip install openai>=1.14")

    if mode == "vllm":
        resolved_url = _normalize_vllm_base_url(base_url)
        resolved_key = resolved_key or "EMPTY"
        print(f"[LLMClient] vLLM service: {resolved_url}, model: {model}")
    elif mode == "openai":
        resolved_url = base_url
        resolved_key = resolved_key or "EMPTY"
        url_info = f" @ {resolved_url}" if resolved_url else ""
        print(f"[LLMClient] OpenAI mode, model: {model}{url_info}")
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Choose from 'openai', 'vllm', 'requests'")

    resolved_timeout = timeout
    if resolved_timeout is None:
        raw_timeout = os.getenv("LLM_REQUEST_TIMEOUT")
        if raw_timeout:
            try:
                resolved_timeout = float(raw_timeout)
            except ValueError:
                print(f"[LLMClient] Ignoring invalid LLM_REQUEST_TIMEOUT={raw_timeout!r}")

    kwargs: Dict[str, Any] = {"api_key": resolved_key}
    if resolved_url:
        kwargs["base_url"] = resolved_url
    if resolved_timeout is not None:
        kwargs["timeout"] = resolved_timeout
    client = OpenAI(**kwargs)
    _mark_guided_json_support(
        client,
        _should_enable_guided_json(mode, resolved_url),
    )
    return client


# ================================================================
# Bounded + logged chat-completion call
# ================================================================

# Finite default so a stalled request fails loudly instead of hanging forever.
DEFAULT_LLM_REQUEST_TIMEOUT = 300.0


def _resolve_request_timeout(explicit: Optional[float] = None) -> Optional[float]:
    """Resolve the per-request timeout in seconds.

    Priority: explicit arg > LLM_REQUEST_TIMEOUT env > DEFAULT_LLM_REQUEST_TIMEOUT.
    Set LLM_REQUEST_TIMEOUT=0 (or negative) to disable the timeout entirely.
    """
    if explicit is not None:
        return explicit
    raw = os.getenv("LLM_REQUEST_TIMEOUT")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            print(f"[LLMClient] Ignoring invalid LLM_REQUEST_TIMEOUT={raw!r}", flush=True)
        else:
            return value if value > 0 else None
    return DEFAULT_LLM_REQUEST_TIMEOUT


def _client_base_url(api_client: Any) -> str:
    """Best-effort endpoint string for logging (OpenAI SDK or Requests client)."""
    try:
        url = getattr(api_client, "base_url", None)
        if not url:
            url = api_client.chat.completions.base_url
        return str(url)
    except Exception:
        return "?"


def chat_completion(
    api_client: Any,
    create_kwargs: Dict[str, Any],
    *,
    component: str = "llm",
    timeout: Optional[float] = None,
) -> Any:
    """Invoke ``chat.completions.create`` with a bounded timeout and debug logging.

    A finite timeout (``LLM_REQUEST_TIMEOUT`` env, default 300s) is injected so a
    stalled request raises ``APITimeoutError`` / ``requests`` timeout instead of
    hanging forever. On failure the endpoint, model, elapsed time and exception are
    always printed; set ``LLM_DEBUG=1`` to also log every request start and finish.
    """
    resolved_timeout = _resolve_request_timeout(timeout)
    call_kwargs = dict(create_kwargs)
    if resolved_timeout is not None and "timeout" not in call_kwargs:
        call_kwargs["timeout"] = resolved_timeout

    debug = bool(_env_flag("LLM_DEBUG"))
    model = create_kwargs.get("model", "?")
    base_url = _client_base_url(api_client)
    t0 = time.monotonic()
    if debug:
        print(
            f"[LLM→] {component} model={model} url={base_url} "
            f"timeout={resolved_timeout}s ...",
            flush=True,
        )
    try:
        response = api_client.chat.completions.create(**call_kwargs)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(
            f"[LLM✗] {component} model={model} url={base_url} "
            f"FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    if debug:
        elapsed = time.monotonic() - t0
        print(f"[LLM←] {component} model={model} ok in {elapsed:.1f}s", flush=True)
    return response
