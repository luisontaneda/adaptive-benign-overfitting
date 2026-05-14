include make/common.mk

dd_test_MAIN     := $(EXP_DIR)/double_descent/dd_test_non_linear.cpp
EURUSD_test_MAIN := $(EXP_DIR)/EURUSD/test_EURUSD.cpp
elect_test_MAIN  := $(EXP_DIR)/electricity/test_elect.cpp
gridsearch_test_MAIN := $(EXP_DIR)/gridsearch/electricity/test_elect.cpp
gridsearch_test_best_MAIN := $(EXP_DIR)/gridsearch/electricity/test_elect_best_hyperpar.cpp
gridsearch_eurusd_test_MAIN := $(EXP_DIR)/gridsearch/EURUSD/test_eurusd.cpp
gridsearch_eurusd_test_best_MAIN := $(EXP_DIR)/gridsearch/EURUSD/test_eurusd_best_hyperpar.cpp
frontier_test_MAIN := $(EXP_DIR)/gridsearch/EURUSD/frontier_test_eurusd.cpp
gridsearch_test_best_bench_MAIN := $(EXP_DIR)/gridsearch/electricity/test_elect_best_hyperpar_bench.cpp
gridsearch_eurusd_test_best_bench_MAIN := $(EXP_DIR)/gridsearch/EURUSD/test_eurusd_best_hyperpar_bench.cpp

dd_test_OBJS     := $(call make-objs,$(dd_test_MAIN))
EURUSD_test_OBJS := $(call make-objs,$(EURUSD_test_MAIN))
elect_test_OBJS  := $(call make-objs,$(elect_test_MAIN))
gridsearch_test_OBJS := $(call make-objs,$(gridsearch_test_MAIN))
gridsearch_test_best_OBJS := $(call make-objs,$(gridsearch_test_best_MAIN))
gridsearch_eurusd_test_OBJS := $(call make-objs,$(gridsearch_eurusd_test_MAIN))
gridsearch_eurusd_test_best_OBJS := $(call make-objs,$(gridsearch_eurusd_test_best_MAIN))
frontier_test_OBJS := $(call make-objs,$(frontier_test_MAIN))
gridsearch_test_best_bench_OBJS := $(call make-objs,$(gridsearch_test_best_bench_MAIN))
gridsearch_eurusd_test_best_bench_OBJS := $(call make-objs,$(gridsearch_eurusd_test_best_bench_MAIN))

# --- Real binaries go to bin/ -----------------------------------------------

$(BIN_DIR)/dd_test: $(dd_test_OBJS) libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/EURUSD_test: $(EURUSD_test_OBJS) libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/elect_test: $(elect_test_OBJS) libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_test: $(gridsearch_test_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_test_best: $(gridsearch_test_best_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_eurusd_test: $(gridsearch_eurusd_test_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_eurusd_test_best: $(gridsearch_eurusd_test_best_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/frontier_test: $(frontier_test_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# ---- Benchmark variants (require Google Benchmark) ---------------------------

BENCH_HEADER := libs/benchmark/include/benchmark/benchmark.h

ifneq ($(wildcard $(BENCH_HEADER)),)

BENCH_CPPFLAGS := -Ilibs/benchmark/include -DFMT_HEADER_ONLY
BENCH_LDFLAGS  := -Llibs/benchmark/build/src
BENCH_LDLIBS   := -lbenchmark -lpthread

# Apply benchmark includes to benchmark objects
$(gridsearch_test_best_bench_OBJS): CPPFLAGS += $(BENCH_CPPFLAGS)
$(gridsearch_eurusd_test_best_bench_OBJS): CPPFLAGS += $(BENCH_CPPFLAGS)

$(BIN_DIR)/gridsearch_test_best_bench: $(gridsearch_test_best_bench_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $(BENCH_LDFLAGS) $^ $(LDLIBS) $(BENCH_LDLIBS) -o $@

$(BIN_DIR)/gridsearch_eurusd_test_best_bench: $(gridsearch_eurusd_test_best_bench_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $(BENCH_LDFLAGS) $^ $(LDLIBS) $(BENCH_LDLIBS) -o $@

else

$(BIN_DIR)/gridsearch_test_best_bench:
	@echo "Skipping gridsearch_test_best_bench: Google Benchmark not found (libs/benchmark/)"

$(BIN_DIR)/gridsearch_eurusd_test_best_bench:
	@echo "Skipping gridsearch_eurusd_test_best_bench: Google Benchmark not found (libs/benchmark/)"

endif

# --- SORF variants (same sources compiled with -DUSE_SORF) ------------------

$(OBJ_DIR)/double_descent/dd_test_non_linear_sorf.o: \
    $(EXP_DIR)/double_descent/dd_test_non_linear.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_SORF -c $< -o $@

$(OBJ_DIR)/EURUSD/test_EURUSD_sorf.o: \
    $(EXP_DIR)/EURUSD/test_EURUSD.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_SORF -c $< -o $@

$(OBJ_DIR)/electricity/test_elect_sorf.o: \
    $(EXP_DIR)/electricity/test_elect.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_SORF -c $< -o $@

$(BIN_DIR)/dd_test_sorf: $(OBJ_DIR)/double_descent/dd_test_non_linear_sorf.o libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/EURUSD_test_sorf: $(OBJ_DIR)/EURUSD/test_EURUSD_sorf.o libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/elect_test_sorf: $(OBJ_DIR)/electricity/test_elect_sorf.o libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# --- Convenience aliases ----------------------------------------------------

.PHONY: dd_test EURUSD_test elect_test gridsearch_test gridsearch_test_best gridsearch_eurusd_test gridsearch_eurusd_test_best frontier_test
.PHONY: dd_test_sorf EURUSD_test_sorf elect_test_sorf
.PHONY: gridsearch_test_best_bench gridsearch_eurusd_test_best_bench

dd_test: $(BIN_DIR)/dd_test
EURUSD_test: $(BIN_DIR)/EURUSD_test
elect_test: $(BIN_DIR)/elect_test
gridsearch_test: $(BIN_DIR)/gridsearch_test
gridsearch_test_best: $(BIN_DIR)/gridsearch_test_best
gridsearch_eurusd_test: $(BIN_DIR)/gridsearch_eurusd_test
gridsearch_eurusd_test_best: $(BIN_DIR)/gridsearch_eurusd_test_best
frontier_test: $(BIN_DIR)/frontier_test
dd_test_sorf:     $(BIN_DIR)/dd_test_sorf
EURUSD_test_sorf: $(BIN_DIR)/EURUSD_test_sorf
elect_test_sorf:  $(BIN_DIR)/elect_test_sorf
gridsearch_test_best_bench: $(BIN_DIR)/gridsearch_test_best_bench
gridsearch_eurusd_test_best_bench: $(BIN_DIR)/gridsearch_eurusd_test_best_bench

EXPERIMENT_PROGS := $(BIN_DIR)/dd_test $(BIN_DIR)/EURUSD_test $(BIN_DIR)/elect_test \
                    $(BIN_DIR)/gridsearch_test $(BIN_DIR)/gridsearch_test_best \
                    $(BIN_DIR)/gridsearch_eurusd_test $(BIN_DIR)/gridsearch_eurusd_test_best \
                    $(BIN_DIR)/frontier_test \
                    $(BIN_DIR)/dd_test_sorf $(BIN_DIR)/EURUSD_test_sorf $(BIN_DIR)/elect_test_sorf \
                    $(BIN_DIR)/gridsearch_test_best_bench $(BIN_DIR)/gridsearch_eurusd_test_best_bench

SORF_OBJS := $(OBJ_DIR)/double_descent/dd_test_non_linear_sorf.o \
             $(OBJ_DIR)/EURUSD/test_EURUSD_sorf.o \
             $(OBJ_DIR)/electricity/test_elect_sorf.o

.PHONY: clean-experiments
clean-experiments:
	$(RM) $(EXPERIMENT_PROGS) $(dd_test_OBJS) $(EURUSD_test_OBJS) $(elect_test_OBJS) $(gridsearch_test_OBJS) \
	$(gridsearch_test_best_OBJS) $(gridsearch_eurusd_test_OBJS) $(gridsearch_eurusd_test_best_OBJS) $(frontier_test_OBJS) \
	$(SORF_OBJS)
