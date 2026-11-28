SHELL      := /bin/bash
.SHELLFLAGS := -euo pipefail -c
CXX        ?= g++
NVCC       ?= $(shell command -v nvcc 2>/dev/null || echo /opt/cuda/bin/nvcc)
ARCH       ?= sm_86
BUILD_DIR  := /tmp/siligemma_build

CXXFLAGS    := -O3 -std=c++20 -Wall -Wextra -Werror
NVCCFLAGS   := -arch=$(ARCH) -O3 -std=c++20 --ptxas-options=-v --Werror all-warnings -Xcompiler -Wall,-Wextra,-Werror -ldl -lpthread
SO_NVCCFLAGS := -arch=$(ARCH) -O3 -std=c++20 -shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror --ptxas-options=-v --Werror all-warnings -DSILIGEMMA_BUILD_SHARED_SO

.PHONY: all compile-cuda audit-sass test-cpu clean

all: test-cpu

$(BUILD_DIR):
	@mkdir -p "$(BUILD_DIR)"

compile-cuda: $(BUILD_DIR)/siligemma_kernels_sm86.o $(BUILD_DIR)/siligemma_bench_sm86 $(BUILD_DIR)/libsiligemma_kernels.so

$(BUILD_DIR)/siligemma_kernels_sm86.o: kernels/siligemma_kernels_sm86.cu kernels/siligemma_arena.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -Ikernels -c kernels/siligemma_kernels_sm86.cu -o $@ 2>&1 | tee "$(BUILD_DIR)/ptxas_kernels.log"

$(BUILD_DIR)/libsiligemma_kernels.so: kernels/siligemma_kernels_sm86.cu kernels/siligemma_arena.hpp | $(BUILD_DIR)
	$(NVCC) $(SO_NVCCFLAGS) -I. -Ikernels kernels/siligemma_kernels_sm86.cu -o $@ 2>&1 | tee "$(BUILD_DIR)/ptxas_so.log"

$(BUILD_DIR)/siligemma_bench_sm86: kernels/siligemma_bench_sm86.cu kernels/siligemma_kernels_sm86.cu kernels/siligemma_arena.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -Ikernels kernels/siligemma_bench_sm86.cu -o $@ 2>&1 | tee "$(BUILD_DIR)/ptxas_bench.log"

audit-sass: compile-cuda
	cuobjdump --dump-sass "$(BUILD_DIR)/siligemma_kernels_sm86.o" > "$(BUILD_DIR)/kernels.sass"
	cuobjdump --dump-sass "$(BUILD_DIR)/siligemma_bench_sm86" > "$(BUILD_DIR)/bench.sass"
	cuobjdump --dump-sass "$(BUILD_DIR)/libsiligemma_kernels.so" > "$(BUILD_DIR)/so.sass"
	@if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$(BUILD_DIR)/ptxas_bench.log" "$(BUILD_DIR)/ptxas_kernels.log" "$(BUILD_DIR)/ptxas_so.log"; then \
		echo "ERROR: non-zero stack or register spill detected" >&2; \
		exit 1; \
	fi
	@if grep -E '\b(LDL|STL)(\.[A-Z0-9]+)?\b' "$(BUILD_DIR)/kernels.sass" "$(BUILD_DIR)/bench.sass" "$(BUILD_DIR)/so.sass"; then \
		echo "ERROR: local memory instruction LDL/STL detected in SASS" >&2; \
		exit 1; \
	fi
	@python3 -I -B -c 'import re, sys; files = sys.argv[1:]; regs = [int(m.group(1)) for p in files for m in re.finditer(r"Used\s+(\d+)\s+registers", open(p, encoding="utf-8").read())]; bad = [r for r in regs if r > 48]; (print(f"PTXAS_REGS_CHECK_PASS count={len(regs)} max={max(regs)}") if regs and not bad else sys.exit(f"ERROR: registers > 48: {bad}"))' "$(BUILD_DIR)/ptxas_kernels.log" "$(BUILD_DIR)/ptxas_bench.log" "$(BUILD_DIR)/ptxas_so.log"
	@echo "ZERO_SPILL_SASS_AUDIT_PASS=1"

test-cpu: $(BUILD_DIR)
	@if [ ! -f tests/test_siligemma_cpu.py ]; then \
		echo "ERROR: tests/test_siligemma_cpu.py lives in the weight-free source repo (Moissanite-Labs/siligemma); run '\''make compile-cuda && make audit-sass'\'' in the weight bundle." >&2; \
		exit 2; \
	fi
	printf '#include "siligemma_arena.hpp"\nint main() { return 0; }\n' | CUDA_VISIBLE_DEVICES="" $(CXX) $(CXXFLAGS) -I. -Ikernels -x c++ - -lm -ldl -lpthread -o "$(BUILD_DIR)/cpu_arena_ref"
	CUDA_VISIBLE_DEVICES="" "$(BUILD_DIR)/cpu_arena_ref"
	CUDA_VISIBLE_DEVICES="" $(CXX) $(CXXFLAGS) -I. -Ikernels tools/slgb_cpu_worker.cpp -lm -ldl -lpthread -o "$(BUILD_DIR)/slgb_cpu_worker"
	CUDA_VISIBLE_DEVICES="" python3 -I -B tests/test_siligemma_cpu.py

clean:
	rm -f /tmp/siligemma_build/cpu_arena_ref /tmp/siligemma_build/slgb_cpu_worker /tmp/siligemma_build/siligemma_kernels_sm86.o /tmp/siligemma_build/libsiligemma_kernels.so /tmp/siligemma_build/siligemma_bench_sm86 /tmp/siligemma_build/ptxas_kernels.log /tmp/siligemma_build/ptxas_so.log /tmp/siligemma_build/ptxas_bench.log /tmp/siligemma_build/kernels.sass /tmp/siligemma_build/so.sass /tmp/siligemma_build/bench.sass
	@rmdir /tmp/siligemma_build 2>/dev/null || true
