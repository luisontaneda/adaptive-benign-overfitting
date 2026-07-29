include make/common.mk

# Fallback: If OBJEXT is not defined in common.mk, default to .o
OBJEXT ?= .o

# ==============================================================================
# Define all experiments (format: target_name:main_source)
# ==============================================================================
EXPERIMENTS := \
  dd_test:$(EXP_DIR)/double_descent/dd_test_non_linear.cpp \
  EURUSD_test:$(EXP_DIR)/EURUSD/test_EURUSD.cpp \
  elect_test:$(EXP_DIR)/electricity/test_elect.cpp \
  gridsearch_test:$(EXP_DIR)/gridsearch/electricity/test_elect.cpp \
  gridsearch_test_best:$(EXP_DIR)/gridsearch/electricity/test_elect_best_hyperpar.cpp \
  gridsearch_eurusd_test:$(EXP_DIR)/gridsearch/EURUSD/test_eurusd.cpp \
  gridsearch_eurusd_test_best:$(EXP_DIR)/gridsearch/EURUSD/test_eurusd_best_hyperpar.cpp \
  frontier_test:$(EXP_DIR)/gridsearch/EURUSD/frontier_test_eurusd.cpp \
  gridsearch_test_best_bench:$(EXP_DIR)/gridsearch/electricity/test_elect_best_hyperpar_bench.cpp \
  gridsearch_eurusd_test_best_bench:$(EXP_DIR)/gridsearch/EURUSD/test_eurusd_best_hyperpar_bench.cpp

# Helper macros to split the "target:source" pairs safely
split-target = $(word 1,$(subst :, ,$1))
split-source = $(word 2,$(subst :, ,$1))

# Generate variables from EXPERIMENTS list
$(foreach e,$(EXPERIMENTS),\
  $(eval $(call split-target,$e)_MAIN := $(call split-source,$e)))

$(foreach e,$(EXPERIMENTS),\
  $(eval $(call split-target,$e)_OBJS := $(OBJ_DIR)/$(call split-target,$e)$(OBJEXT)))

# ==============================================================================
# Extract names and generate targets
# ==============================================================================
PROG_NAMES := $(foreach e,$(EXPERIMENTS),$(call split-target,$e))
EXPERIMENT_PROGS := $(addprefix $(BIN_DIR)/,$(PROG_NAMES))
SORF_NAMES := dd_test EURUSD_test elect_test
SORF_OBJS := $(addprefix $(OBJ_DIR)/,$(addsuffix _sorf.o,$(SORF_NAMES)))

# ==============================================================================
# Dynamic Compilation Rules (Tells Make how to build the .o files)
# ==============================================================================
# This template generates a specific compilation rule for every single experiment
# Dynamic Compilation Rules (Tells Make how to build the .o files)
# Bench source files need the benchmark include path from libs/benchmark
# NOTE: bench targets use flattened object names, so explicit bench rules are required.
define BENCH_COMPILE_RULE
$$(OBJ_DIR)/$(1)$$(OBJEXT): $(2) | $$(OBJ_DIR)
	@mkdir -p $$(@D)
	$$(CXX) $$(CPPFLAGS) $$(BENCH_CPPFLAGS) $$(CXXFLAGS) -c $$< -o $$@
endef

define COMPILE_RULE
$$(OBJ_DIR)/$(call split-target,$1)$$(OBJEXT): $(call split-source,$1) | $$(OBJ_DIR)
	@mkdir -p $$(@D)
	$$(CXX) $$(CPPFLAGS) $$(CXXFLAGS) -c $$< -o $$@
endef

# Evaluate the compilation rule template for every non-bench experiment
$(foreach e,$(EXPERIMENTS),$(eval $(if $(findstring _bench,$(call split-target,$e)),,$(call COMPILE_RULE,$e))))

# Evaluate explicit bench compile rules for bench experiments
$(foreach e,$(EXPERIMENTS),$(eval $(if $(findstring _bench,$(call split-target,$e)),$(call BENCH_COMPILE_RULE,$(call split-target,$e),$(call split-source,$e)))))

# Create object directory if missing
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# ==============================================================================
# Linking rules with variable dependencies
# ==============================================================================
$(BIN_DIR)/dd_test $(BIN_DIR)/EURUSD_test $(BIN_DIR)/elect_test: $(BIN_DIR)/%: $(OBJ_DIR)/%$(OBJEXT) libcore.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_test $(BIN_DIR)/gridsearch_test_best $(BIN_DIR)/gridsearch_eurusd_test \
$(BIN_DIR)/gridsearch_eurusd_test_best $(BIN_DIR)/frontier_test: $(BIN_DIR)/%: $(OBJ_DIR)/%$(OBJEXT) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Benchmark variants
$(BIN_DIR)/gridsearch_test_best_bench $(BIN_DIR)/gridsearch_eurusd_test_best_bench: CPPFLAGS += -DFMT_HEADER_ONLY

$(BIN_DIR)/%_bench: $(OBJ_DIR)/%_bench.o libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(BENCH_LDFLAGS) $^ $(LDLIBS) -lbenchmark -lpthread -o $@

# ==============================================================================
# SORF Compilation Rule
# ==============================================================================
$(OBJ_DIR)/%_sorf.o: $(EXP_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_SORF -c $< -o $@

$(OBJ_DIR)/double_descent/%_sorf.o: $(EXP_DIR)/double_descent/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_SORF -c $< -o $@

$(BIN_DIR)/%_sorf: $(OBJ_DIR)/%_sorf.o libcore.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# ==============================================================================
# Convenience phony targets
# ==============================================================================
.PHONY: all $(PROG_NAMES) $(addsuffix _sorf,$(SORF_NAMES)) gridsearch_test_best_bench gridsearch_eurusd_test_best_bench

all: $(EXPERIMENT_PROGS)

# Cleaned up short-hand rules so they don't loop endlessly or throw warnings
$(PROG_NAMES): %: $(BIN_DIR)/%
$(addsuffix _sorf,$(SORF_NAMES)): %: $(BIN_DIR)/%

# ==============================================================================
# Cleanup
# ==============================================================================
.PHONY: clean-experiments
clean-experiments:
	$(RM) $(EXPERIMENT_PROGS) $(addprefix $(BIN_DIR)/,$(addsuffix _sorf,$(SORF_NAMES))) \
	      $(addprefix $(BIN_DIR)/,$(addsuffix _bench,gridsearch_test_best gridsearch_eurusd_test_best))