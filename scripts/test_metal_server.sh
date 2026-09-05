#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

BACKEND_BIN="$ROOT/tools/llama.cpp/build/bin/llama-server"
DEFAULT_MODEL="$ROOT/models/bonsai-1.7b-ternary/Ternary-Bonsai-1.7B-PQ2_0.gguf"
MODEL_PATH="${1:-$DEFAULT_MODEL}"
PORT=8089
URL="http://127.0.0.1:${PORT}"

echo "=================================================================="
echo "⚡ Testing Native Apple Silicon Metal GPU Server"
echo "=================================================================="
echo "[*] Binary: $BACKEND_BIN"
echo "[*] Model:  $MODEL_PATH"
echo "[*] Port:   $PORT"

# 1. Verify / build binary
if [ ! -f "$BACKEND_BIN" ]; then
    echo "[*] llama-server not found. Building with Metal acceleration..."
    cmake -B "$ROOT/tools/llama.cpp/build" -S "$ROOT/tools/llama.cpp" -DGGML_METAL=ON
    cmake --build "$ROOT/tools/llama.cpp/build" --config Release -j --target llama-server
fi

# 2. Check model existence
if [ ! -f "$MODEL_PATH" ]; then
    echo "[!] Model file not found: $MODEL_PATH"
    exit 1
fi

# 3. Clean port
lsof -ti :${PORT} | xargs kill -9 2>/dev/null || true
sleep 0.5

# 4. Launch daemon
echo "[*] Launching llama-server with full GPU offload (-ngl 99)..."
"$BACKEND_BIN" \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -ngl 99 \
    -c 4096 \
    -fa auto > /tmp/test_metal_server_${PORT}.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "[*] Cleaning up test server (PID $SERVER_PID)..."
    kill -TERM $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# 5. Wait for readiness
echo "[*] Waiting for server health..."
READY=0
for i in {1..30}; do
    if curl --noproxy "*" -s "${URL}/health" | grep -q "ok"; then
        READY=1
        break
    fi
    sleep 0.5
done

if [ $READY -ne 1 ]; then
    echo "[!] Server failed to report health OK."
    cat /tmp/test_metal_server_${PORT}.log | tail -n 20
    exit 1
fi
echo "[+] Server reports HEALTH OK on ${URL}/health"

# 6. Execute test inference
echo "[*] Executing test chat completion..."
RESP=$(curl --noproxy "*" -s "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Explain Apple Silicon in 10 words."}],
        "max_tokens": 30,
        "temperature": 0.0
    }')

CONTENT=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null || echo "")
SPEED=$(echo "$RESP" | python3 -c "import sys, json; print(round(json.load(sys.stdin).get('timings', {}).get('predicted_per_second', 0), 2))" 2>/dev/null || echo "0")

if [ -z "$CONTENT" ]; then
    echo "[!] Empty completion response received."
    echo "$RESP"
    exit 1
fi

echo ""
echo "=================================================================="
echo "✅ Metal Server Verification Passed:"
echo "   - Generated Response: \"$CONTENT\""
echo "   - Measured Speed:     $SPEED tokens/sec"
echo "=================================================================="
