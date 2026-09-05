#!/usr/bin/env bash
set -e

OLLAMA_URL="http://127.0.0.1:11434"
MODEL_NAME="llama3.2:1b"

echo "=================================================================="
echo "🦙 Ollama Server & Model Manager"
echo "=================================================================="

# 1. Check ollama binary
if ! command -v ollama >/dev/null 2>&1; then
    echo "[!] Ollama CLI not found in PATH."
    echo "    Please install Ollama from https://ollama.com/"
    exit 1
fi

# 2. Check if Ollama daemon is running
if ! curl --noproxy "*" -s "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo "[*] Ollama daemon not responding. Launching 'ollama serve' in background..."
    nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
    
    echo "[*] Waiting for Ollama daemon to initialize..."
    for i in {1..20}; do
        if curl --noproxy "*" -s "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
fi

if ! curl --noproxy "*" -s "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo "[!] Failed to connect to Ollama daemon on ${OLLAMA_URL}."
    exit 1
fi
echo "[+] Ollama daemon operational on ${OLLAMA_URL}"

# 3. Check model availability
if ! ollama list | grep -q "${MODEL_NAME}"; then
    echo "[*] Model '${MODEL_NAME}' not found. Pulling..."
    ollama pull "${MODEL_NAME}"
fi
echo "[+] Model '${MODEL_NAME}' is ready."

echo ""
echo "=================================================================="
echo "🚀 Ollama is ready for serving and comparative benchmarking:"
echo "   - API Endpoint:   ${OLLAMA_URL}/api/generate"
echo "   - Active Model:   ${MODEL_NAME}"
echo "=================================================================="
