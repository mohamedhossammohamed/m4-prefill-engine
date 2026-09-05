"""
OpenAI-Compatible Chat Completion Server & Web UI for Bonsai 27B.

Endpoints:
- GET  /v1/models: lists active models (prism-ml/Ternary-Bonsai-27B)
- POST /v1/chat/completions: OpenAI-compatible chat endpoint
- GET  /: Web Chat UI interface
- GET  /health: health check endpoint
"""

import http.server
from http.server import ThreadingHTTPServer
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

BONSAI_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BONSAI_DIR / "runtime"
PROJECT_ROOT = BONSAI_DIR.parent.parent
for p in (BONSAI_DIR, RUNTIME_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PORT = int(os.environ.get("PORT", 8000))
API_KEY_FILE = BONSAI_DIR / "api_key.txt"

# Ensure API Key is generated and persisted locally
if not API_KEY_FILE.exists():
    fallback_key_file = PROJECT_ROOT / "models" / "bonsai-27b-ternary" / "api_key.txt"
    if fallback_key_file.exists():
        API_KEY_FILE.write_text(fallback_key_file.read_text().strip())
    else:
        API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        generated_key = f"sk-bonsai-{secrets.token_hex(16)}"
        API_KEY_FILE.write_text(generated_key)
        print(f"[*] Generated new API Key saved to {API_KEY_FILE}: {generated_key}")

generated_key = API_KEY_FILE.read_text().strip()
print(f"[*] Loaded API Key from {API_KEY_FILE}")

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bonsai 1.7B Chat</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --panel-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --accent-color: #2ea043;
            --user-msg: #1f6feb;
            --bot-msg: #21262d;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        header {
            background-color: var(--panel-bg);
            padding: 12px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 {
            margin: 0;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge {
            background: #238636;
            color: white;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 12px;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 8px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .message.user {
            align-self: flex-end;
            background-color: var(--user-msg);
            color: white;
        }
        .message.assistant {
            align-self: flex-start;
            background-color: var(--bot-msg);
            border: 1px solid var(--border-color);
        }
        #input-panel {
            background-color: var(--panel-bg);
            border-top: 1px solid var(--border-color);
            padding: 16px 20px;
            display: flex;
            gap: 12px;
        }
        textarea {
            flex: 1;
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 10px 14px;
            border-radius: 6px;
            resize: none;
            font-family: inherit;
            font-size: 0.95rem;
            height: 44px;
            box-sizing: border-box;
        }
        textarea:focus {
            outline: none;
            border-color: #58a6ff;
        }
        button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.2s;
            height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            user-select: none;
        }
        button:hover {
            background-color: #3fb950;
        }
        button:disabled {
            background-color: #484f58;
            cursor: not-allowed;
        }
        .meta-bar {
            font-size: 0.8rem;
            color: #8b949e;
            padding: 4px 20px;
            background: #090d13;
            border-top: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <header>
        <h1>🌿 Bonsai Ternary Engine <span class="badge">1.7B Metal GPU (~95 tok/s)</span></h1>
        <div id="status" style="font-size: 0.85rem; color: #58a6ff;">Ready</div>
    </header>
    <div id="chat-container">
        <div class="message assistant">Hello! I am Bonsai 1.7B running with hardware-native Metal GPU acceleration on Apple Silicon unified memory. How can I assist you?</div>
    </div>
    <div id="input-panel">
        <textarea id="prompt-input" placeholder="Type your message here... (Enter to send, Shift+Enter for newline)"></textarea>
        <div style="display: flex; flex-direction: column; gap: 8px; justify-content: center;">
            <div style="display: flex; gap: 6px; align-items: center;">
                <label style="font-size: 0.75rem; color: #8b949e;">Tokens:</label>
                <select id="max-tokens-select" style="background: var(--bg-color); color: var(--text-color); border: 1px solid var(--border-color); border-radius: 4px; padding: 2px 6px; font-size: 0.8rem;">
                    <option value="64">64 tokens (~0.6s)</option>
                    <option value="128" selected>128 tokens (~1.3s)</option>
                    <option value="256">256 tokens (~2.6s)</option>
                    <option value="512">512 tokens (~5.2s)</option>
                </select>
            </div>
            <div style="display: flex; gap: 6px;">
                <button id="send-btn" onclick="sendMessage()">Send</button>
                <button id="stop-btn" onclick="stopGeneration()" style="background-color: #da3633; display: none;">Stop</button>
            </div>
        </div>
    </div>
    <div class="meta-bar">
        <span id="stat-model">Model: prism-ml/Ternary-Bonsai-1.7B-gguf (PQ2_0)</span>
        <span id="stat-spec">Backend: Apple Silicon Metal GPU (~95-120 tok/s)</span>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const stopBtn = document.getElementById('stop-btn');
        const statusEl = document.getElementById('status');

        let messages = [];
        let abortController = null;

        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function stopGeneration() {
            if (abortController) {
                abortController.abort();
                abortController = null;
            }
        }

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text) return;

            messages.push({"role": "user", "content": text});
            appendMessage('user', text);
            promptInput.value = '';
            promptInput.disabled = true;
            sendBtn.disabled = true;
            stopBtn.style.display = 'inline-block';
            statusEl.innerText = 'Generating response...';

            const assistantMsgDiv = appendMessage('assistant', '💭 Thinking...');
            assistantMsgDiv.style.fontStyle = 'italic';
            assistantMsgDiv.style.color = '#8b949e';
            const maxTokens = parseInt(document.getElementById('max-tokens-select').value) || 32;

            abortController = new AbortController();

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + window.API_KEY
                    },
                    body: JSON.stringify({
                        model: 'prism-ml/Ternary-Bonsai-1.7B',
                        messages: messages,
                        temperature: 0.7,
                        max_tokens: maxTokens,
                        stream: true
                    }),
                    signal: abortController.signal
                });

                if (!response.ok) {
                    throw new Error('API Error: ' + response.statusText);
                }

                statusEl.innerText = 'Streaming response...';
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let accumulatedReply = '';
                let buffer = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split(/\r?\n/);
                    buffer = lines.pop();

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || trimmed === 'data: [DONE]') continue;
                        if (trimmed.startsWith('data: ')) {
                            try {
                                const parsed = JSON.parse(trimmed.slice(6));
                                const delta = parsed.choices[0]?.delta?.content || '';
                                if (delta) {
                                    if (!accumulatedReply) {
                                        assistantMsgDiv.style.fontStyle = 'normal';
                                        assistantMsgDiv.style.color = 'var(--text-color)';
                                    }
                                    accumulatedReply += delta;
                                    assistantMsgDiv.innerHTML = renderMessageContent(accumulatedReply);
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            } catch (e) {}
                        }
                    }
                }

                if (!accumulatedReply) {
                    assistantMsgDiv.innerText = '(Finished)';
                }
                messages.push({"role": "assistant", "content": accumulatedReply});
                statusEl.innerText = 'Ready';
            } catch (err) {
                if (err.name === 'AbortError') {
                    statusEl.innerText = 'Stopped';
                } else {
                    assistantMsgDiv.innerText = 'Error: ' + err.message;
                    statusEl.innerText = 'Error';
                }
            } finally {
                promptInput.disabled = false;
                sendBtn.disabled = false;
                stopBtn.style.display = 'none';
                abortController = null;
                promptInput.focus();
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        function escapeHtml(unsafe) {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function renderMessageContent(raw) {
            if (raw.includes('</think>')) {
                const parts = raw.split('</think>');
                const thought = parts[0].replace('<think>', '').trim();
                const reply = parts.slice(1).join('</think>').trim();
                return `<details style="margin-bottom: 8px; font-size: 0.85rem; color: #8b949e; border-left: 2px solid #30363d; padding-left: 8px;"><summary style="cursor: pointer; user-select: none; color: #58a6ff;">💭 Thought process</summary><div style="margin-top: 4px; white-space: pre-wrap;">${escapeHtml(thought)}</div></details><div style="white-space: pre-wrap;">${escapeHtml(reply || '(Empty reply)')}</div>`;
            } else if (raw.startsWith('<think>') || raw.toLowerCase().startsWith("here's a thinking process") || raw.toLowerCase().startsWith("thinking process")) {
                const thought = raw.replace('<think>', '').trim();
                return `<details style="margin-bottom: 8px; font-size: 0.85rem; color: #8b949e; border-left: 2px solid #30363d; padding-left: 8px;" open><summary style="cursor: pointer; user-select: none; color: #e3b341;">💭 Thinking...</summary><div style="margin-top: 4px; white-space: pre-wrap;">${escapeHtml(thought)}</div></details>`;
            }
            return `<div style="white-space: pre-wrap;">${escapeHtml(raw)}</div>`;
        }

        function appendMessage(role, content) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerText = content;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return div;
        }

        // Embed local API key directly from server
        window.API_KEY = "GENERATED_KEY_PLACEHOLDER";
    </script>
</body>
</html>
"""

class BonsaiServerHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = CHAT_HTML.replace("GENERATED_KEY_PLACEHOLDER", generated_key).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/health":
            self._send_json(200, {"status": "ok", "time": time.time()})
            return

        if parsed.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": "prism-ml/Ternary-Bonsai-1.7B",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "prism-ml",
                        "architecture": "qwen3-ternary-gqa"
                    },
                    {
                        "id": "prism-ml/Ternary-Bonsai-27B",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "prism-ml",
                        "architecture": "qwen35-hybrid-linear-attention"
                    }
                ]
            })
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            # Verify API Key authorization header
            auth_header = self.headers.get("Authorization", "")
            expected_auth = f"Bearer {generated_key}"
            if auth_header != expected_auth:
                self._send_json(401, {
                    "error": {
                        "message": "Invalid API key provided. Include 'Authorization: Bearer <key>'.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key"
                    }
                })
                return

            content_len = int(self.headers.get("Content-Length", 0))
            post_bytes = self.rfile.read(content_len)
            try:
                req_data = json.loads(post_bytes.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Malformed JSON: {e}"}})
                return

            messages = req_data.get("messages", [])
            max_tokens = min(req_data.get("max_tokens", 128), 1024)
            temp = float(req_data.get("temperature", 0.7))
            stream = bool(req_data.get("stream", False))

            # Fast path: proxy directly to local Metal GPU backend (port 8080)
            backend_url = getattr(self.server, "backend_url", "http://127.0.0.1:8080")
            try:
                proxy_req = urllib.request.Request(
                    f"{backend_url}/v1/chat/completions",
                    data=post_bytes,
                    headers={"Content-Type": "application/json"}
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(proxy_req, timeout=120) as b_resp:
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        for raw_line in b_resp:
                            self.wfile.write(raw_line)
                            self.wfile.flush()
                            if raw_line.strip() == b"data: [DONE]":
                                break
                        self.close_connection = True
                        return
                    else:
                        b_body = b_resp.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(b_body)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(b_body)
                        return
            except Exception as e:
                print(f"[!] Backend proxy error: {e}, falling back to local engine", flush=True)

            # Local fallback engine
            chat_id = secrets.token_hex(12)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                if self.server.engine is not None:
                    for delta in self.server.engine.generate_stream(messages, max_tokens=max_tokens, temperature=temp):
                        chunk_obj = {
                            "id": f"chatcmpl-{chat_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "prism-ml/Ternary-Bonsai-1.7B",
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
                        }
                        try:
                            self.wfile.write(f"data: {json.dumps(chunk_obj)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                else:
                    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "Hello")
                    chunk_obj = {
                        "id": f"chatcmpl-{chat_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "prism-ml/Ternary-Bonsai-1.7B",
                        "choices": [{"index": 0, "delta": {"content": f"Bonsai: {last_user}"}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_obj)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                done_obj = {
                    "id": f"chatcmpl-{chat_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "prism-ml/Ternary-Bonsai-1.7B",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                try:
                    self.wfile.write(f"data: {json.dumps(done_obj)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return

            reply_text = self.server.generate_response(messages, max_tokens, temp)
            resp_obj = {
                "id": f"chatcmpl-{chat_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "prism-ml/Ternary-Bonsai-1.7B",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": reply_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(reply_text.split()),
                    "total_tokens": len(str(messages)) + len(reply_text.split())
                }
            }
            self._send_json(200, resp_obj)
            return

        self.send_error(404, "Not Found")


class BonsaiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, engine=None, backend_url="http://127.0.0.1:8080"):
        super().__init__(server_address, handler_class)
        self.engine = engine
        self.backend_url = backend_url

    def generate_response(self, messages: List[dict], max_tokens: int, temp: float) -> str:
        if self.engine is not None:
            text, stats = self.engine.generate(messages, max_tokens=max_tokens, temperature=temp, top_k=20, top_p=0.95, n=3, k=4)
            return text
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "Hello")
        return f"Bonsai: Processed '{last_user}'."


def run_server(port: int = PORT):
    server = BonsaiHTTPServer(("0.0.0.0", port), BonsaiServerHandler, engine=None, backend_url="http://127.0.0.1:8080")
    print(f"[*] Bonsai Server listening on http://localhost:{port}", flush=True)
    print(f"[*] Serving Bonsai 1.7B via Metal GPU (~95 tok/s) and Web Chat UI", flush=True)
    print(f"[*] Web Chat UI available at http://localhost:{port}/", flush=True)
    print(f"[*] API Key: {generated_key}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()


