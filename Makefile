CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -I. -Icore -Iinclude
FRAMEWORKS = -framework Metal -framework Foundation -framework MetalPerformanceShaders

CORE_SRCS = core/memory/page_allocator.mm \
            core/memory/uma_tracker.mm \
            core/memory/cache_flush.mm \
            core/metrology/prng.mm \
            core/metrology/tripwires.mm \
            core/metrology/telemetry.mm \
            core/metrology/telemetry_format.mm \
            core/metal/shader_loader.mm \
            core/weights/gguf_loader.mm \
            src/router/quant_registry.mm

TARGETS = bench_m4 micro_bench pipelined_bench flash_attn_bench unified_prefill_engine thermal_stress_test bench_8b_engine bench_all_scales brick1_micro_bench brick2_fused_bench brick3_swiglu_bench brick4_attn_bench bench_universal_router bench_streaming_kv bench_streaming_1m libm4_bridge.dylib

all: $(TARGETS)

bench_streaming_1m: bench_streaming_1m.mm streaming_1m_engine.mm streaming_1m_engine.h streaming_1m_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o bench_streaming_1m bench_streaming_1m.mm streaming_1m_engine.mm

bench_streaming_kv: bench_streaming_kv.mm AsyncKVRingBuffer.mm AsyncKVRingBuffer.h streaming_kv_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o bench_streaming_kv bench_streaming_kv.mm AsyncKVRingBuffer.mm

bench_universal_router: bench_universal_router.mm quant_router_kernels.metal quant_router.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o bench_universal_router bench_universal_router.mm

brick4_attn_bench: brick4_attn_bench.mm brick4_attn_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o brick4_attn_bench brick4_attn_bench.mm

brick3_swiglu_bench: brick3_swiglu_bench.mm brick3_swiglu_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o brick3_swiglu_bench brick3_swiglu_bench.mm

brick2_fused_bench: brick2_fused_bench.mm brick2_fused_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o brick2_fused_bench brick2_fused_bench.mm

brick1_micro_bench: brick1_micro_bench.mm brick1_micro_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o brick1_micro_bench brick1_micro_bench.mm

bench_m4: bench_engine.mm kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o bench_m4 bench_engine.mm

micro_bench: micro_bench.mm micro_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o micro_bench micro_bench.mm

pipelined_bench: pipelined_bench.mm pipelined_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o pipelined_bench pipelined_bench.mm

flash_attn_bench: flash_attn_bench.mm flash_attn_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o flash_attn_bench flash_attn_bench.mm

unified_prefill_engine: unified_prefill_engine.mm unified_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o unified_prefill_engine unified_prefill_engine.mm

bench_8b_engine: bench_8b_engine.mm unified_8b_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o bench_8b_engine bench_8b_engine.mm

thermal_stress_test: thermal_stress_test.mm unified_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o thermal_stress_test thermal_stress_test.mm

bench_all_scales: bench_all_scales.mm unified_multi_scale_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o bench_all_scales bench_all_scales.mm

libm4_bridge.dylib: core/bridge/m4_mlx_bridge.mm core/bridge/m4_mlx_bridge.h $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) -fobjc-arc -shared -fPIC $(FRAMEWORKS) -o libm4_bridge.dylib core/bridge/m4_mlx_bridge.mm $(CORE_SRCS)

test_core_invariants: tests/test_core_invariants.mm $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/test_core_invariants tests/test_core_invariants.mm $(CORE_SRCS)

test_metal_headers: tests/test_metal_headers.mm $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/test_metal_headers tests/test_metal_headers.mm $(CORE_SRCS)

test_quant_registry: tests/test_quant_registry.mm src/router/quant_registry.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/test_quant_registry tests/test_quant_registry.mm src/router/quant_registry.mm

test_kernel_parity: tests/test_kernel_parity.mm $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/test_kernel_parity tests/test_kernel_parity.mm $(CORE_SRCS)

test_transformer_layer: tests/test_transformer_layer.mm models/transformer_layer.mm $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/test_transformer_layer tests/test_transformer_layer.mm models/transformer_layer.mm $(CORE_SRCS)

test_tier1_feature_coverage: tests/e2e/test_tier1_feature_coverage.mm $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/e2e/test_tier1_feature_coverage tests/e2e/test_tier1_feature_coverage.mm $(CORE_SRCS)

test_gguf_loader: tests/test_gguf_loader.mm $(CORE_SRCS)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o tests/test_gguf_loader tests/test_gguf_loader.mm $(CORE_SRCS)

test: test_core_invariants test_metal_headers test_quant_registry test_kernel_parity test_transformer_layer test_tier1_feature_coverage test_gguf_loader
	./tests/test_core_invariants
	./tests/test_metal_headers
	./tests/test_quant_registry
	./tests/test_kernel_parity
	./tests/test_transformer_layer
	./tests/e2e/test_tier1_feature_coverage
	./tests/test_gguf_loader

clean:
	rm -f $(TARGETS) tests/test_core_invariants tests/test_metal_headers tests/test_quant_registry tests/test_kernel_parity tests/test_transformer_layer tests/e2e/test_tier1_feature_coverage tests/test_gguf_loader

