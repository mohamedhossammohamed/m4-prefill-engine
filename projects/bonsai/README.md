# 🌿 Bonsai Ternary Engine (Apple Silicon Deployment)

This directory contains the self-contained, modular project for running **PrismML Bonsai** ternary models on Apple Silicon with hardware-accelerated Metal GPU execution and out-of-core weight streaming.

---

## ⚡ Quickstart (1-Command Launch)

Start both the Metal GPU compute backend and the Web Chat UI / OpenAI REST API:

```bash
# Launch server and open UI at http://localhost:8000/
./projects/bonsai/run.sh

# Or launch and immediately run the 10-turn dialogue benchmark:
./projects/bonsai/run.sh --benchmark
```

### Active Endpoints
* **Web Chat UI:** [http://localhost:8000/](http://localhost:8000/)
* **OpenAI Chat Endpoint:** `POST http://localhost:8000/v1/chat/completions`
* **Models Endpoint:** `GET http://localhost:8000/v1/models`
* **Health Check:** `GET http://localhost:8000/health`
* **API Key:** Stored locally in `projects/bonsai/api_key.txt` (passed via `Authorization: Bearer <key>`)

---

## 🏗️ Models & Hardware Offload

| Model | Architecture | Size | Hardware Path | Measured Speed |
| :--- | :--- | :---: | :--- | :---: |
| **Bonsai 1.7B** | Qwen3-1.7B Ternary (PQ2_0 g128) | 441.8 MB | **Metal GPU** (All 28 layers unified memory) | **~86 – 95 tok/s** |
| **Bonsai 27B** | Qwen3.5 64-Layer Hybrid (48 Linear + 16 GQA) | 6.67 GB | **Out-of-Core Streaming** (Peak resident < 1.1 GB) | Speculative n-gram |

---

## 📁 Directory Layout

```
projects/bonsai/
├── README.md             # This guide
├── run.sh                # Single executable launcher script
├── server.py             # Port 8000 gateway server & Web Chat UI
├── benchmark.py          # 10-turn multi-turn dialogue test suite
├── api_key.txt           # Authentication bearer token
├── runtime/              # Model-specific hybrid linear-attention runtime
│   ├── bonsai_engine.py          # Bonsai 27B hybrid prefill & streaming engine
│   ├── bonsai_layer.h/.mm        # Dynamic layer router (48 linear attn vs 16 GQA)
│   ├── bonsai_paged_kv.h         # Bounded recurrent state & paged KV manager
│   ├── bonsai_weight_streamer.py # Out-of-core layer weight streamer
│   ├── bonsai_tokenizer.py       # Self-contained 248k vocabulary tokenizer
│   ├── dequant.c                 # ARM NEON vector dequantizer
│   └── Makefile                  # Runtime build target for libdequant.dylib
└── models/               # Symlinks to models directory
    ├── bonsai-1.7b/      # Ternary-Bonsai-1.7B-PQ2_0.gguf
    └── bonsai-27b/       # Ternary-Bonsai-27B-Q2_0.gguf
```

---

## 📡 REST API Examples

### Streaming Chat Completion
```bash
API_KEY=$(cat projects/bonsai/api_key.txt)

curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "prism-ml/Ternary-Bonsai-1.7B",
    "messages": [
      {"role": "user", "content": "Explain how Apple Silicon unified memory works in one sentence."}
    ],
    "max_tokens": 64,
    "stream": true
  }'
```

### Non-Streaming Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "prism-ml/Ternary-Bonsai-1.7B",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 30
  }'
```
