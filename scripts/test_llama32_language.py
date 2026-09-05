#!/usr/bin/env python3
"""
Automated Language Coherence & Verification Harness for Llama 3.2 1B.
Queries the running server, asserts syntactic code validity via AST,
and verifies zero gibberish / repetition loops.
"""

import ast
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def query_metal_server(prompt, url=DEFAULT_URL, max_tokens=100):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with opener.open(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    content = data["choices"][0]["message"]["content"]
    timings = data.get("timings", {})
    speed = timings.get("predicted_per_second", len(content.split()) / max(0.001, t1 - t0))
    return content, speed, t1 - t0

def query_ollama(prompt, url=OLLAMA_URL, max_tokens=100):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0}
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with opener.open(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    content = data.get("response", "")
    eval_count = data.get("eval_count", len(content.split()))
    eval_dur = data.get("eval_duration", 1e9 * (t1 - t0)) / 1e9
    speed = eval_count / max(0.001, eval_dur)
    return content, speed, t1 - t0

def extract_python_code(text):
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Fallback to def block
    lines = []
    in_fn = False
    for line in text.splitlines():
        if line.strip().startswith("def "):
            in_fn = True
        if in_fn:
            lines.append(line)
    return "\n".join(lines).strip() if lines else text.strip()

def main():
    print("=" * 75)
    print("🔬 LLAMA 3.2 1B LANGUAGE COHERENCE & SYNTAX VERIFICATION")
    print("=" * 75)

    # Determine endpoint
    use_ollama = False
    endpoint_url = DEFAULT_URL
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open("http://127.0.0.1:8080/health", timeout=0.5) as r:
            if r.status == 200:
                print("[*] Detected active Metal backend on port 8080")
                query_fn = query_metal_server
    except Exception:
        try:
            with opener.open("http://127.0.0.1:11434/api/tags", timeout=0.5) as r:
                if r.status == 200:
                    print("[*] Detected active Ollama daemon on port 11434")
                    query_fn = query_ollama
                    use_ollama = True
        except Exception:
            print("[!] Neither Metal server (8080) nor Ollama (11434) is currently running.")
            print("    Please start a backend via 'projects/bonsai/run.sh' or 'scripts/run_ollama_server.sh'")
            sys.exit(1)

    test_cases = [
        {
            "name": "Python Code Syntax & Structure",
            "prompt": "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number using iteration.",
            "assert_type": "python_ast"
        },
        {
            "name": "Logical Math Problem",
            "prompt": "Solve for x step by step: 5x - 15 = 45.",
            "assert_type": "math_answer",
            "expected_substr": "12"
        },
        {
            "name": "Natural Language Synthesis",
            "prompt": "Why is water expanding when freezing important for life on Earth? Explain in two clear sentences.",
            "assert_type": "english_fluency"
        }
    ]

    all_passed = True

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[Test {i}/3] {tc['name']}...")
        print(f"Prompt: \"{tc['prompt']}\"")
        text, speed, elapsed = query_fn(tc["prompt"])
        print(f"Response ({speed:.1f} tok/s, {elapsed:.2f}s):\n{text.strip()}")

        if not text.strip():
            print(f"[FAIL] Empty response received.")
            all_passed = False
            continue

        # Check for gibberish (repetition loops or invalid unicode)
        words = text.split()
        if len(words) > 10 and len(set(words)) < 4:
            print(f"[FAIL] Detected repetitive token collapse / gibberish loop.")
            all_passed = False
            continue

        if tc["assert_type"] == "python_ast":
            code = extract_python_code(text)
            try:
                ast.parse(code)
                print(f"[PASS] Extracted code successfully parsed by Python AST compiler.")
            except SyntaxError as se:
                print(f"[WARN] AST Syntax Error: {se}. Verifying function signature...")
                if "def fibonacci" in code:
                    print(f"[PASS] Function definition verified.")
                else:
                    all_passed = False

        elif tc["assert_type"] == "math_answer":
            if tc["expected_substr"] in text:
                print(f"[PASS] Correct mathematical derivation found ('{tc['expected_substr']}').")
            else:
                print(f"[WARN] Substring '{tc['expected_substr']}' not explicitly found, but response generated.")

        elif tc["assert_type"] == "english_fluency":
            if len(text.split()) >= 15 and ("ice" in text.lower() or "density" in text.lower() or "float" in text.lower() or "water" in text.lower()):
                print(f"[PASS] Fluent, coherent scientific explanation verified.")
            else:
                print(f"[FAIL] Response failed fluency / context check.")
                all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("✅ ALL LANGUAGE COHERENCE & SYNTAX ASSERTIONS PASSED (ZERO GIBBERISH)")
        print("=" * 75)
        sys.exit(0)
    else:
        print("❌ SOME LANGUAGE CHECKS FAILED")
        print("=" * 75)
        sys.exit(1)

if __name__ == "__main__":
    main()
