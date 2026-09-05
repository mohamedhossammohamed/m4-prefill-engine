# Project: m4-prefill-engine Modularization & Decoupling

## Architecture
The `m4-prefill-engine` modularization decouples quantization formats, attention kernels, and MLP activation epilogues into pluggable modules while guaranteeing bit-exact mathematical parity and zero latency regression on Apple Silicon (M4).

### Subsystem Boundaries & Data Flow
1. **Core Metrology & Memory Layer (`core/`)**:
   - `core/memory/`: 16KB system page-aligned allocation (`page_allocator.h/.mm` with `AlignedBuffer<T>`), UMA physical memory tracking (`uma_tracker.h/.mm` querying `task_vm_info.phys_footprint`), cold cache / Unified Buffer Cache eviction (`cache_flush.h/.mm`), and unified quantization block type definitions (`quant_types.h`).
   - `core/metrology/`: High-precision LCG PRNG (`prng.h/.mm`), non-finite assertion tripwires (`tripwires.h/.mm`), timer & variance calculation (`telemetry.h/.mm`), standard output formatting (`telemetry_format.h/.mm`), and benchmark execution standards (`bench_standards.h`).
2. **Pluggable Header-Only Metal Shading Architecture (`include/metal/`)**:
   - `include/metal/common/`: 128-bit LSU loaders (`float4`/`uint4`), SIMD butterfly reduction trees (`simd_shuffle_down`), and bank-conflict-free padded SRAM tiles (`shmem[64][36]`).
   - `include/metal/quant/`: Self-contained stateless quantization unpacker headers (`q4_0.metal`, `mlx_4bit.metal`, `q4_k.metal`, `ternary_1_58.metal`, `var_rate_affine.metal`, `exl3.metal`).
   - `include/metal/ops/`: Codec-parameterized 2D BlockMMA GEMM core (`gemm_mma.metal`), branchless vector ALU ternary GEMM (`gemm_ternary_vec.metal`), Dual-SIMD cooperative SwiGLU (`swiglu_dual_simd.metal`), and barrier-free FlashAttention (`flash_attn.metal`).
   - Entry points: Thin instantiation shaders (`quant_router_kernels.metal`, `unified_kernels.metal`, etc.) that include ops headers and instantiate target kernels with zero duplicated logic.
3. **Host Dynamic Codec Registry & Layer Composition (`src/router/` & `models/`)**:
   - `src/router/quant_registry.h`: Declarative registry (`QuantCodecDescriptor`, `QuantRegistry`, `REGISTER_QUANT_CODEC`) decoupling format metadata, block dimensions, and kernel symbol names. Adding a new format requires 1 line in the registry.
   - `models/transformer_layer.h`: Composable layer coordinator (`TransformerLayerCoordinator`) managing RMSNorm, QKV projection GEMM, FlashAttention, and SwiGLU MLP via compile-time traits or configuration objects.
4. **Compatibility & Benchmark Harness**:
   - Thin backward-compatibility wrappers retaining all 15 Makefile targets and CLI interfaces (`bench_m4`, `bench_universal_router`, `unified_prefill_engine`, `flash_attn_bench`, `pipelined_bench`, `thermal_stress_test`, etc.).
   - Mandatory enforcement of all 11 metrological invariants from `agents/RED_TEAM_AUDITS.md`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 16KB Direct I/O Alignment | Runtime-enforced 16KB system page alignment (`posix_memalign` + `F_NOCACHE`) with programmatic assertions in `core/memory/page_allocator.h` | M1 (R1) | ORIGINAL_REQUEST §R1, RED_TEAM_AUDITS Invariant 1 |
| 2 | Cache Purge & Cold Eviction | 32MB pre-benchmark SLC cache flush and UBC dummy file purge in `core/memory/cache_flush.h` | M1 (R1) | ORIGINAL_REQUEST §R1, RED_TEAM_AUDITS Invariant 2 |
| 3 | UMA Physical Footprint Tracking | Hardware-accurate memory accounting via `task_vm_info.phys_footprint` in `core/memory/uma_tracker.h` | M1 (R1) | ORIGINAL_REQUEST §R1, RED_TEAM_AUDITS Invariant 4 |
| 4 | Non-Finite Tripwires | Systematic NaN/Inf detection on all CPU and GPU outputs in `core/metrology/tripwires.h` | M1 (R1) | ORIGINAL_REQUEST §R1, RED_TEAM_AUDITS Invariant 6 |
| 5 | Deduplicated PRNG & Telemetry | Authoritative PRNG and metric calculation in `core/metrology/prng.h` and `core/metrology/telemetry.h` | M1 (R1) | ORIGINAL_REQUEST §R1, Survey |
| 6 | Common Metal Primitives | 128-bit LSU vector loaders, SIMD butterfly reductions, and stride-36 padded SRAM buffers in `include/metal/common/` | M2 (R2) | ORIGINAL_REQUEST §R2, RED_TEAM_AUDITS Invariant 8 |
| 7 | Pluggable Metal Unpackers | Modular unpackers for `q4_0`, `mlx_4bit`, `q4_k`, `ternary_1_58`, `var_rate_affine`, `exl3` in `include/metal/quant/` | M2 (R2) | ORIGINAL_REQUEST §R2, RED_TEAM_AUDITS Invariant 7 |
| 8 | Parameterized BlockMMA GEMM | Decoupled 2D BlockMMA GEMM core parameterized by quantization unpacker traits in `include/metal/ops/gemm_mma.metal` | M2 (R2) | ORIGINAL_REQUEST §R2 |
| 9 | Dual-SIMD SwiGLU & FlashAttention | Modular Dual-SIMD cooperative SwiGLU and barrier-free FlashAttention in `include/metal/ops/` | M2 (R2) | ORIGINAL_REQUEST §R2 |
| 10 | Thin Kernel Entrypoints | Thin `.metal` wrappers instantiating modular ops for runtime JIT compilation | M2 (R2) | ORIGINAL_REQUEST §R2 |
| 11 | Dynamic Codec Registry | Declarative registry `src/router/quant_registry.h` with single-line format addition macro | M3 (R3) | ORIGINAL_REQUEST §R3 |
| 12 | Composable Transformer Layer | Unified layer coordinator `models/transformer_layer.h` connecting RMSNorm, QKV, Attn, SwiGLU | M3 (R3) | ORIGINAL_REQUEST §R3 |
| 13 | Makefile & Target Preservation | Preserve all 15 build targets with identical flags and dependencies | M4 (R4) | ORIGINAL_REQUEST §R4 |
| 14 | CLI Interface Compatibility | Preserve CLI behavior (`bench_all_scales <model> <M>`, all others 0 args) | M4 (R4) | ORIGINAL_REQUEST §R4 |
| 15 | Honest Telemetry Reporting | Eliminate fake `[LOCKED]` strings and `N/A (GPU-Only)` diffs at scale | M4 (R4) | RED_TEAM_AUDITS Invariant 3 & 9 |
| 16 | E2E Requirement-Driven Testing | Comprehensive test suite covering boundary tokens (33, 127, 128, 129, 2048), NaN tripwires, performance envelope | M0 (Test Track) | ORIGINAL_REQUEST Acceptance Criteria |
| 17 | Final Invariants Audit & Parity Verification | Verification of all 11 metrological invariants and MaxDiff <= 0.0078 | M5 (Final) | ORIGINAL_REQUEST Acceptance Criteria |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Track | Requirement-driven test harness, runner, and test cases covering all 4 tiers; produces `TEST_READY.md` | None | IN_PROGRESS |
| M1 | R1: Metrology & Memory Invariant Isolation | Implement `core/memory/` and `core/metrology/`, unit test invariants, eliminate duplication | None | PLANNED |
| M2 | R2: Header-Only Metal Shading Architecture | Modularize MSL into `include/metal/` (common, quant, ops, entry points), decouple GEMM from unpackers | M1 | PLANNED |
| M3 | R3: Host Dynamic Codec Registry & Layer Composition | Implement `src/router/quant_registry.h` and `models/transformer_layer.h`, eliminate monoliths | M1, M2 | PLANNED |
| M4 | R4: Blast Radius Containment & Compatibility Harness | Adapt all 15 Makefile targets and benchmark binaries via thin wrappers, enforce honest reporting | M1, M2, M3 | PLANNED |
| M5 | Final Verification & Adversarial Hardening | Verify 100% E2E test pass, 1.5% variance envelope, <= 64 registers, all 11 invariants, forensic audit | M0, M4 | COMPLETED |
| M6 | Metal vs. Ollama Metrology & Ternary Deployment (v0.3.1) | 100-prompt comparative benchmark on identical weights, language coherence AST tests, self-contained projects/bonsai packaging | M5 | COMPLETED |

---

## Interface Contracts

### 1. `core/memory/` ↔ Host & Shaders
- `core/memory/page_allocator.h`:
  ```cpp
  template <typename T>
  class AlignedBuffer {
  public:
      AlignedBuffer(size_t count, size_t alignment = 16384);
      ~AlignedBuffer();
      T* data();
      const T* data() const;
      size_t size() const;
      size_t bytes() const;
  };
  void* allocate_16kb_aligned(size_t bytes);
  void free_16kb_aligned(void* ptr);
  void assert_16kb_aligned(const void* ptr);
  ```
- `core/memory/uma_tracker.h`:
  ```cpp
  double get_uma_phys_footprint_mb();
  ```
- `core/memory/cache_flush.h`:
  ```cpp
  void purge_unified_buffer_cache(const char* dummy_path = "/tmp/ubc_purge_dummy");
  void cold_cache_evict_cpu(void* buffer, size_t bytes = 32 * 1024 * 1024);
  ```
- `core/memory/quant_types.h`:
  Authoritative host C++ definitions of `block_q4_0`, `block_mlx_4bit`, `block_q4_K`, `block_ternary_1_58`, `block_var_rate_affine`, `block_exl3`.

### 2. `core/metrology/` ↔ Host Benchmarks
- `core/metrology/prng.h`:
  ```cpp
  void prng_seed(uint32_t seed);
  uint32_t prng_next_u32();
  float prng_rand_uniform(float min_val = -1.0f, float max_val = 1.0f);
  ```
- `core/metrology/tripwires.h`:
  ```cpp
  bool verify_finite(const float* data, size_t count, const char* tensor_name);
  bool verify_finite(const __fp16* data, size_t count, const char* tensor_name);
  float compute_max_diff(const float* cpu_gold, const __fp16* gpu_out, size_t count);
  ```
- `core/metrology/telemetry.h`:
  ```cpp
  double compute_median(std::vector<double>& samples);
  double compute_variance_percentage(const std::vector<double>& samples, double baseline_median);
  ```

### 3. `include/metal/` ↔ Host JIT & Kernels
- Unpacker Concept:
  ```metal
  struct UnpackerQ4_0 {
      static constant uint BLOCK_SIZE = 32;
      static inline void unpack_column(
          constant const void* weights,
          uint col, uint kb, uint K,
          threadgroup half sh_B[32][64],
          uint tid
      );
  };
  ```
- BlockMMA GEMM Signature:
  ```metal
  template <typename TCodec, bool DIRECT_HEAD_ROUTING = false>
  kernel void block_mma_gemm(
      constant const half* A [[buffer(0)]],
      constant const void* B [[buffer(1)]],
      device half* C [[buffer(2)]],
      constant const uint& M [[buffer(3)]],
      constant const uint& N [[buffer(4)]],
      constant const uint& K [[buffer(5)]],
      uint3 tg_pos [[threadgroup_position_in_grid]],
      uint tid [[thread_index_in_threadgroup]]
  );
  ```

### 4. `src/router/quant_registry.h` ↔ Router & Models
- `QuantCodecDescriptor`:
  ```cpp
  struct QuantCodecDescriptor {
      int format_id;
      std::string name;
      size_t block_bytes;
      size_t block_weights;
      std::string kernel_symbol;
      std::string direct_head_kernel_symbol;
      void (*weight_generator)(void* dst, size_t elements);
      void (*cpu_gold_ref)(const float* A, const void* B, float* C, int M, int N, int K);
  };
  class QuantRegistry {
  public:
      static QuantRegistry& instance();
      void register_codec(const QuantCodecDescriptor& desc);
      const QuantCodecDescriptor* get(int format_id) const;
      const QuantCodecDescriptor* get(const std::string& name) const;
      std::vector<const QuantCodecDescriptor*> all() const;
  };
  ```

### 5. `models/transformer_layer.h` ↔ Engine Entries
- `TransformerLayerCoordinator`:
  ```cpp
  struct TransformerLayerConfig {
      int M, D, H, HD, intermediate_dim;
      int quant_format;
      float eps;
  };
  class TransformerLayerCoordinator {
  public:
      TransformerLayerCoordinator(id<MTLDevice> device, id<MTLCommandQueue> queue, const TransformerLayerConfig& config);
      void forward_gpu(id<MTLCommandBuffer> cb, id<MTLBuffer> in, id<MTLBuffer> out);
      void forward_cpu_gold(const float* in, float* out);
  };
  ```

---

## Code Layout
```
/Users/mohammedhossam/Documents/antigravity/wonderful-darwin/
├── core/
│   ├── memory/
│   │   ├── page_allocator.h / page_allocator.mm
│   │   ├── uma_tracker.h / uma_tracker.mm
│   │   ├── cache_flush.h / cache_flush.mm
│   │   └── quant_types.h
│   ├── metal/
│   │   └── shader_loader.h / shader_loader.mm
│   └── metrology/
│       ├── prng.h / prng.mm
│       ├── tripwires.h / tripwires.mm
│       ├── telemetry.h / telemetry.mm
│       ├── telemetry_format.h / telemetry_format.mm
│       └── bench_standards.h
├── include/
│   └── metal/
│       ├── common/
│       │   ├── lsu.metal
│       │   ├── math.metal
│       │   ├── simd_reduce.metal
│       │   ├── sram_tile.metal
│       │   └── types.metal
│       ├── quant/
│       │   ├── codec_traits.metal
│       │   ├── q4_0.metal
│       │   ├── mlx_4bit.metal
│       │   ├── q4_k.metal
│       │   ├── ternary_1_58.metal
│       │   ├── var_rate_affine.metal
│       │   └── exl3.metal
│       ├── ops/
│       │   ├── gemm_mma.metal
│       │   ├── gemm_ternary_vec.metal
│       │   ├── swiglu_dual_simd.metal
│       │   └── flash_attention.metal
│       └── prefill_kernels.metal
├── src/
│   └── router/
│       └── quant_registry.h / quant_registry.mm
├── models/
│   └── transformer_layer.h / transformer_layer.mm
├── tests/
│   ├── test_core_invariants.mm
│   ├── test_metal_headers.mm
│   ├── test_quant_registry.mm
│   ├── test_kernel_parity.mm
│   ├── test_transformer_layer.mm
│   ├── test_modular_kernels.metal
│   └── e2e/
│       ├── test_tier1_feature_coverage.mm
│       ├── test_common.h
│       └── cpu_gold_reference.h
├── Makefile (updated with search paths, preserving all 15 targets)
└── <bench_*.mm> (thin compatibility wrappers)
```
