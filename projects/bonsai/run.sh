#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="${VENV}/bin/python3"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

BACKEND_BIN="$ROOT/tools/llama.cpp/build/bin/llama-server"
MODEL_1_7B="$DIR/models/bonsai-1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf"
API_KEY_FILE="$DIR/api_key.txt"

echo "=================================================================="
echo "🌿 Starting Bonsai Ternary Engine (Apple Silicon Metal GPU)"
echo "=================================================================="

# 1. Start or verify Metal GPU backend on port 8080
if ! curl --noproxy "*" -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "[*] Launching Metal GPU accelerated backend (port 8080)..."
    if [ ! -f "$BACKEND_BIN" ]; then
        echo "[!] llama-server binary not found at $BACKEND_BIN"
        echo "[*] Building llama.cpp Metal binary..."
        cmake -B "$ROOT/tools/llama.cpp/build" -S "$ROOT/tools/llama.cpp" -DGGML_METAL=ON
        cmake --build "$ROOT/tools/llama.cpp/build" --config Release -j --target llama-server
    fi
    nohup "$BACKEND_BIN" -m "$MODEL_1_7B" --port 8080 --host 127.0.0.1 -ngl 99 -c 4096 -fa auto > /tmp/bonsai_backend_8080.log 2>&1 &
    
    echo "[*] Waiting for backend to initialize..."
    for i in {1..30}; do
        if curl --noproxy "*" -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
fi
echo "[+] Backend operational on http://127.0.0.1:8080"

# 2. Start or verify Gateway Server on port 8000
if ! curl --noproxy "*" -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[*] Launching Bonsai Gateway & Web Chat UI (port 8000)..."
    PYTHONUNBUFFERED=1 PYTHONPATH="$DIR/runtime:$ROOT:$PYTHONPATH" nohup "$PYTHON" "$DIR/server.py" > /tmp/bonsai_server_8000.log 2>&1 &
    
    for i in {1..20}; do
        if curl --noproxy "*" -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
            break
        fi
        sleep 0.3
    done
fi
echo "[+] Gateway operational on http://127.0.0.1:8000"

API_KEY="$(cat "$API_KEY_FILE" 2>/dev/null || echo "")"
echo ""
echo "=================================================================="
echo "🚀 Bonsai Engine is LIVE and ready:"
echo "   - Web Chat UI:  http://localhost:8000/"
echo "   - API Endpoint: http://localhost:8000/v1/chat/completions"
echo "   - API Key:      $API_KEY"
echo "   - Model:        prism-ml/Ternary-Bonsai-1.7B (Metal GPU ~95 tok/s)"
echo "=================================================================="

if [ "$1" == "--test" ] || [ "$1" == "--benchmark" ]; then
    echo ""
    echo "[*] Running multi-turn dialogue benchmark..."
    "$PYTHON" "$DIR/benchmark.py"
fi
