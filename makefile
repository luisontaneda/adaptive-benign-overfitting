# Top-level Makefile

include make/common.mk
include make/libs.mk
include make/baselines.mk
include make/experiments.mk
include make/benchmarks.mk

TEST_SRCS := tests/test_abo_smoke.cpp tests/test_abo_debug.cpp tests/test_abo_paper.cpp tests/test_sorf.cpp tests/test_rff_vs_sorf.cpp
TEST_BINS := $(patsubst tests/%.cpp,$(BIN_DIR)/%,$(TEST_SRCS))

$(OBJ_DIR)/%.o: tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/test_%: $(OBJ_DIR)/test_%.o libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: all experiments benchmarks libs baselines tests clean debug help check

all: experiments benchmarks baselines

tests: $(TEST_BINS)

check: tests
	@echo "Running unit tests..."
	@for t in $(TEST_BINS); do \
		printf "==> %s\n" "$$t"; \
		"$$t" || exit 1; \
	done

help:
	@echo "Available make targets:"
	@echo "  make all         - Build experiments, benchmarks, and baselines"
	@echo "  make experiments - Build experiment binaries"
	@echo "  make benchmarks  - Build benchmark binaries"
	@echo "  make baselines   - Build baseline binaries"
	@echo "  make tests       - Build test binaries"
	@echo "  make check       - Build and run unit tests"
	@echo "  make clean       - Remove generated artifacts"
	@echo "  make debug       - Build all with DEBUG=1"
	@echo
	@echo "Common makefile options:"
	@echo "  MODE=debug      - Enable debug compilation mode"
	@echo "  targetName      - Optional target name for build"
	@echo

# Convenience umbrella targets (also nice for `make experiments`)
experiments: $(EXPERIMENT_PROGS)

benchmarks: $(BENCHMARK_PROGS)

libs: libcore.a
baselines: $(BASELINE_PROGS)

debug:
	$(MAKE) DEBUG=1 all
	

clean: clean-experiments clean-benchmarks clean-libs clean-baselines
	$(RM) -r $(OBJ_DIR)
