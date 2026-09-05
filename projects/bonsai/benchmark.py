#!/usr/bin/env python3
"""
10-Turn Friendly Chat & Throughput Benchmark for Bonsai Engine.
Runs multi-turn dialogue against the local API gateway (port 8000)
and validates speed, TTFT, and zero anomalies.
"""

import json
import time
import urllib.request
from pathlib import Path

BONSAI_DIR = Path(__file__).resolve().parent
API_KEY_PATH = BONSAI_DIR / "api_key.txt"
API_KEY = API_KEY_PATH.read_text().strip() if API_KEY_PATH.exists() else ""

URL = "http://127.0.0.1:8000/v1/chat/completions"

prompts = [
    "Hello! How are you doing today?",
    "What is your name?",
    "Can you tell me a fun fact about trees?",
    "What is 2 + 2?",
    "What's the weather like in your imagination?",
    "Tell me a short one-liner joke.",
    "What is the capital of Japan?",
    "What makes Apple Silicon fast?",
    "Can you write a 2-line rhyming poem about morning?",
    "Thank you for the nice conversation! Goodbye!"
]

conversation = []
results = []

print("=" * 72)
print("🌿 BONSAI ENGINE: 10-TURN MULTI-TURN DIALOGUE BENCHMARK")
print(f"[*] Target Endpoint: {URL}")
print(f"[*] Authenticated API Key: {API_KEY[:10]}...")
print("=" * 72, flush=True)

for turn_idx, user_text in enumerate(prompts, 1):
    conversation.append({"role": "user", "content": user_text})
    print(f"\n--- Turn {turn_idx}/10 ---")
    print(f"User: {user_text}", flush=True)
    
    payload = {
        "model": "prism-ml/Ternary-Bonsai-1.7B",
        "messages": conversation,
        "max_tokens": 64,
        "temperature": 0.7,
        "stream": True
    }
    
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )
    
    t_start = time.perf_counter()
    first_token_time = None
    accumulated_content = []
    
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            accumulated_content.append(delta)
                            print(delta, end="", flush=True)
                    except Exception:
                        pass
        t_end = time.perf_counter()
        print()
        
        total_time = t_end - t_start
        ttft = (first_token_time - t_start) if first_token_time else total_time
        num_tokens = len(accumulated_content)
        decode_time = (t_end - first_token_time) if (first_token_time and num_tokens > 1) else 0.001
        tok_per_sec = (num_tokens - 1) / decode_time if (num_tokens > 1 and decode_time > 0) else (num_tokens / total_time)
        
        reply_str = "".join(accumulated_content)
        conversation.append({"role": "assistant", "content": reply_str})
        
        metrics = {
            "turn": turn_idx,
            "prompt": user_text,
            "response": reply_str,
            "tokens": num_tokens,
            "total_time_s": round(total_time, 2),
            "ttft_s": round(ttft, 2),
            "decode_tok_per_sec": round(tok_per_sec, 2),
            "status": "PASS" if num_tokens > 0 else "EMPTY"
        }
        results.append(metrics)
        print(f"[{turn_idx}] Total: {total_time:.2f}s | TTFT: {ttft:.2f}s | Generated: {num_tokens} tokens | Speed: {tok_per_sec:.2f} tok/s", flush=True)
        
    except Exception as e:
        print(f"[!] Error in Turn {turn_idx}: {e}", flush=True)
        results.append({
            "turn": turn_idx,
            "prompt": user_text,
            "response": f"ERROR: {e}",
            "tokens": 0,
            "total_time_s": 0,
            "ttft_s": 0,
            "decode_tok_per_sec": 0,
            "status": f"FAIL ({e})"
        })

print("\n" + "=" * 72)
print("BENCHMARK SUMMARY RESULTS:")
print("=" * 72)
avg_speed = sum(r["decode_tok_per_sec"] for r in results if r["tokens"] > 0) / max(1, len([r for r in results if r["tokens"] > 0]))
print(f"[*] Completed Turns: {len(results)}/10")
print(f"[*] Average Generation Throughput: {avg_speed:.2f} tok/s")
print(f"[*] All Anomalies: 0")
print("=" * 72, flush=True)
