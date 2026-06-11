#!/usr/bin/env python3
"""
本地 OpenAI 兼容 HTTP 服务：监听端口，将 /v1/chat/completions 请求通过
src.llm_client.build_openai_client 转发到 llm_config.yaml 中配置的上游 API。

用法（在仓库根目录）:
  export OPENAI_API_KEY=...
  python baseline/local_server.py --llm-config llm_config.yaml --port 8765

客户端将 base_url 设为 http://127.0.0.1:8765/v1 即可走本地代理。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

# 仓库根目录
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _completion_to_dict(resp: Any, model: str) -> Dict[str, Any]:
    """将 chat.completions.create 的返回值转为 OpenAI REST JSON 结构。"""
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    content = ""
    p, c, t = 0, 0, 0
    if hasattr(resp, "choices") and resp.choices:
        msg = resp.choices[0].message
        content = getattr(msg, "content", None) or ""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        p = getattr(usage, "prompt_tokens", 0) or 0
        c = getattr(usage, "completion_tokens", 0) or 0
        t = getattr(usage, "total_tokens", 0) or (p + c)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    upstream_gen: Dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/health", "/v1/models"):
            self._send_json(
                200,
                {"status": "ok", "service": "symregression-local-openai-proxy"},
            )
            return
        self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(
                404, {"error": {"message": "Not found", "type": "not_found"}}
            )
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(
                400, {"error": {"message": "Invalid JSON body", "type": "invalid_json"}}
            )
            return

        if body.get("stream"):
            self._send_json(
                400,
                {
                    "error": {
                        "message": "Streaming not supported; set stream=false",
                        "type": "unsupported",
                    }
                },
            )
            return

        gen = _Handler.upstream_gen
        model = body.get("model") or gen.get("model", "")
        messages = body.get("messages")
        if not model or not isinstance(messages, list):
            self._send_json(
                400,
                {"error": {"message": "Missing model or messages", "type": "invalid_request"}},
            )
            return

        from src.llm_client import build_openai_client, resolve_completion_max_tokens

        base_url = gen.get("base_url")
        mode = gen.get("mode", "openai")
        client = build_openai_client(model, base_url, mode=mode)

        temperature = body.get("temperature", gen.get("temperature", 0.7))
        if "max_tokens" in body:
            max_tokens = body["max_tokens"]
        else:
            max_tokens = resolve_completion_max_tokens(
                model, messages, gen.get("max_tokens", 8192))

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        av = body.get("anthropic_version") or gen.get("anthropic_version")
        if av:
            kwargs["anthropic_version"] = av

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            self._send_json(
                502,
                {
                    "error": {
                        "message": str(e),
                        "type": "upstream_error",
                    }
                },
            )
            return

        self._send_json(200, _completion_to_dict(resp, model))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible local proxy using src.llm_client"
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=str(_REPO_ROOT / "llm_config.yaml"),
        help="YAML with generator section (upstream base_url, mode, model, ...)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    cfg_path = Path(args.llm_config)
    if not cfg_path.is_file():
        print(f"Error: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    gen = full.get("generator")
    if not isinstance(gen, dict) or not gen.get("base_url"):
        print(
            "Error: llm_config must contain generator.base_url for upstream forwarding.",
            file=sys.stderr,
        )
        sys.exit(1)

    _Handler.upstream_gen = gen

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"[local_server] Listening http://{args.host}:{args.port} "
        f"→ upstream mode={gen.get('mode')} model={gen.get('model')}"
    )
    print(f"[local_server] Set client base_url to http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[local_server] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
