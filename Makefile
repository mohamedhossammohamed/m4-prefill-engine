CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall
FRAMEWORKS = -framework Metal -framework Foundation -framework MetalPerformanceShaders

TARGETS = bench_m4 micro_bench pipelined_bench flash_attn_bench unified_prefill_engine thermal_stress_test

all: $(TARGETS)

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

thermal_stress_test: thermal_stress_test.mm unified_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o thermal_stress_test thermal_stress_test.mm

clean:
	rm -f $(TARGETS)

