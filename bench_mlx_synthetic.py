import time
import math
import os
import sys

def run_mlx_benchmark():
    import mlx.core as mx
    import mlx.nn as nn

    os.makedirs("benchmarks/logs", exist_ok=True)

    print("=" * 115)
    print("        MLX SYNTHETIC PREFILL BENCHMARK (APPLE M4 / MLX METAL)               ")
    print("        Timing Method: Wall-Clock around mx.eval (Shared Parity Method)       ")
    print("        Iterations:    10 Warmup / 20 Measured Runs                          ")
    print("=" * 115)
    print(f"[+] MLX Version: {mx.__version__}")
    print(f"[+] Device:      {mx.default_device()}")

    class TransformerLayerMLX(nn.Module):
        def __init__(self, K, H, D, N_mlp, bits=4, group_size=32):
            super().__init__()
            self.K = K
            self.H = H
            self.D = D
            self.N_mlp = N_mlp
            self.attn_dim = H * D
            self.scale = 1.0 / math.sqrt(D)

            # Projections with quantized weights (Q4)
            # Create linear layers and quantize them to 4-bit
            self.wq = nn.Linear(K, self.attn_dim, bias=False)
            self.wk = nn.Linear(K, self.attn_dim, bias=False)
            self.wv = nn.Linear(K, self.attn_dim, bias=False)
            self.wo = nn.Linear(self.attn_dim, K, bias=False)

            self.w_gate = nn.Linear(K, N_mlp, bias=False)
            self.w_up = nn.Linear(K, N_mlp, bias=False)
            self.w_down = nn.Linear(N_mlp, K, bias=False)

            # Quantize all linear layers to 4-bit (group_size=32, standard Q4)
            nn.quantize(self.wq, group_size=group_size, bits=bits)
            nn.quantize(self.wk, group_size=group_size, bits=bits)
            nn.quantize(self.wv, group_size=group_size, bits=bits)
            nn.quantize(self.wo, group_size=group_size, bits=bits)
            nn.quantize(self.w_gate, group_size=group_size, bits=bits)
            nn.quantize(self.w_up, group_size=group_size, bits=bits)
            nn.quantize(self.w_down, group_size=group_size, bits=bits)

        def __call__(self, x):
            # x: [1, M, K]
            M = x.shape[1]
            
            # 1. QKV Projections
            q = self.wq(x).reshape(1, M, self.H, self.D).transpose(0, 2, 1, 3) # [1, H, M, D]
            k = self.wk(x).reshape(1, M, self.H, self.D).transpose(0, 2, 1, 3) # [1, H, M, D]
            v = self.wv(x).reshape(1, M, self.H, self.D).transpose(0, 2, 1, 3) # [1, H, M, D]

            # 2. Causal FlashAttention
            attn_out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal"
            ) # [1, H, M, D]
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(1, M, self.attn_dim) # [1, M, H*D]

            # 3. Output Projection & Residual
            x_mid = x + self.wo(attn_out)

            # 4. SwiGLU MLP
            gate = nn.silu(self.w_gate(x_mid))
            up = self.w_up(x_mid)
            mlp_down = self.w_down(gate * up)

            # 5. Final Residual
            out = x_mid + mlp_down
            return out

    # Model specifications
    models = {
        "1B": {
            "name": "1B Transformer (LLaMA-3.2 1B shape)",
            "K": 2048, "H": 32, "D": 64, "N_mlp": 5632, "layers": 16,
            "weight_mb": 27.56
        },
        "8B": {
            "name": "8B Transformer (LLaMA-3.1 8B shape)",
            "K": 4096, "H": 32, "D": 128, "N_mlp": 14336, "layers": 32,
            "weight_mb": 130.50
        }
    }

    seq_lengths = [33, 127, 128, 129, 512, 1023, 1024, 2047, 2048]
    WARMUP = 10
    MEASURE = 20

    summary_log_path = "benchmarks/logs/bench_mlx_summary.txt"
    with open(summary_log_path, "w") as sum_log:
        sum_log.write("========================================================================================\n")
        sum_log.write(" MLX SYNTHETIC PREFILL BENCHMARK REPORT (APPLE M4 / MLX METAL)\n")
        sum_log.write(" Timing Method: Wall-Clock around mx.eval (Shared Parity Method)\n")
        sum_log.write(" Iterations:    10 Warmup / 20 Measured Runs\n")
        sum_log.write("========================================================================================\n\n")
        sum_log.write("[DISCLOSURE BLOCK]\n")
        sum_log.write("All cross-engine numbers use synthetic in-UMA weights with exact model shapes, no disk I/O, no tokenizer (M is the token count), prefill-only (single forward pass, no generation). This measures kernel execution on identical workloads, not end-to-end product latency.\n\n")

    for model_key, cfg in models.items():
        print(f"\n{'='*115}")
        print(f">>> BENCHMARKING MLX: {cfg['name']}")
        print(f"    - Dimensions: K={cfg['K']}, H={cfg['H']}, D={cfg['D']}, N_mlp={cfg['N_mlp']}, Layers={cfg['layers']}")
        print(f"{'='*115}")

        layer = TransformerLayerMLX(cfg['K'], cfg['H'], cfg['D'], cfg['N_mlp'])
        mx.eval(layer.parameters())

        print(f"{'M (Tokens)':<12} | {'MLX Median (ms)':<16} | {'MLX [Min-Max] (ms)':<22} | {'Throughput (tok/s)':<20} | {'Full-Model Estimate (measured layer x L)':<35}")
        print("-" * 115)

        for M in seq_lengths:
            # Deterministic synthetic prompt input in FP16 directly with MLX
            mx.random.seed(1337 + M)
            x = (mx.random.normal((1, M, cfg['K'])) * 0.35).astype(mx.float16)

            # 10 Warmup passes
            for _ in range(WARMUP):
                out = layer(x)
                mx.eval(out)

            # 20 Measured iterations
            times = []
            for _ in range(MEASURE):
                t0 = time.perf_counter()
                out = layer(x)
                mx.eval(out)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0) # ms

            times.sort()
            med_ms = 0.5 * (times[9] + times[10]) # 20 items median
            min_ms = times[0]
            max_ms = times[-1]
            tok_s = (M / (med_ms / 1000.0))
            full_model_s = (med_ms * cfg['layers']) / 1000.0

            print(f"{M:<12} | {med_ms:14.2f} ms | [{min_ms:6.2f} - {max_ms:6.2f}] ms | {tok_s:18.0f} tok/s | {full_model_s:14.2f} s")

            # Save individual log file
            log_filename = f"benchmarks/logs/bench_mlx_{model_key}_{M}.txt"
            with open(log_filename, "w") as f:
                f.write("========================================================================================\n")
                f.write(f" CONFIG:\n")
                f.write(f"   - Engine:             MLX ({mx.__version__})\n")
                f.write(f"   - Model Tier:         {model_key} Transformer\n")
                f.write(f"   - Hidden Dimension K: {cfg['K']}\n")
                f.write(f"   - Attention Heads H:  {cfg['H']}\n")
                f.write(f"   - Head Dimension D:   {cfg['D']}\n")
                f.write(f"   - Intermediate MLP N: {cfg['N_mlp']}\n")
                f.write(f"   - Layer Count L:      {cfg['layers']} layers\n")
                f.write(f"   - Sequence Length M:  {M} tokens\n")
                f.write(f"   - Quantization:       Q4_0 weights ({cfg['weight_mb']:.2f} MB/layer)\n")
                f.write("========================================================================================\n\n")
                f.write("[DISCLOSURE BLOCK]\n")
                f.write("All cross-engine numbers use synthetic in-UMA weights with exact model shapes, no disk I/O, no tokenizer (M is the token count), prefill-only (single forward pass, no generation). This measures kernel execution on identical workloads, not end-to-end product latency.\n\n")
                f.write(f"[1] TIMING & PERFORMANCE METRICS (Median [Min - Max] over {MEASURE} iterations):\n")
                f.write(f"    - 1-Layer Latency (Wall):     {med_ms:.2f} ms [{min_ms:.2f}-{max_ms:.2f}] ms\n")
                f.write(f"    - 1-Layer Throughput:         {tok_s:.0f} tok/s\n")
                f.write(f"    - Full-Model Estimate ({cfg['layers']}L): {full_model_s:.2f} s\n")
                f.write(f"    - Full-Model Estimate Tput:   {tok_s / cfg['layers']:.0f} tok/s\n")

            with open(summary_log_path, "a") as sum_log:
                sum_log.write(f"[{model_key}] M={M:4d}: {med_ms:8.2f} ms [{min_ms:6.2f}-{max_ms:6.2f}] | {tok_s:8.0f} tok/s | Full ({cfg['layers']}L): {full_model_s:6.2f} s\n")

if __name__ == "__main__":
    run_mlx_benchmark()
