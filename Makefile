CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall
FRAMEWORKS = -framework Metal -framework Foundation -framework MetalPerformanceShaders

TARGETS = bench_m4 micro_bench pipelined_bench flash_attn_bench unified_prefill_engine thermal_stress_test bench_8b_engine bench_all_scales brick1_micro_bench brick2_fused_bench brick3_swiglu_bench brick4_attn_bench bench_universal_router bench_streaming_kv

all: $(TARGETS)

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

clean:
	rm -f $(TARGETS)

