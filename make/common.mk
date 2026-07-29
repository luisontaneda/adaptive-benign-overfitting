# --- Toolchain ---------------------------------------------------------------
CXX       ?= g++
CPPFLAGS  ?= -Iinclude -MMD -MP

INCLUDES = -Iinclude -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/include/eigen3 -Ilibs/eigen -I/usr/lib/lapack

# Per-machine overrides: create make/local.mk to override INCLUDES, LDFLAGS, LDLIBS
-include make/local.mk

# Base Warnings and Configuration flags (FIXED: -std=c++17)
BASE_CXXFLAGS = -std=c++17 $(INCLUDES) -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_STRUCTURE \
			-Wall -Wno-shadow \
			-Wno-unused-parameter -Wno-sign-compare -Wno-unused-variable \
			-Wno-reorder -Wno-comment -Wno-deprecated-declarations

LDFLAGS   ?= -Llibs/lib
LDLIBS    ?= -lopenblas -llapacke -lm -lfmt

# ==============================================================================
# GLOBAL BUILD CONFIGURATION 
# Setting 'MODE=release' or 'MODE=debug' propagates down to static libraries too!
# ==============================================================================
MODE ?= release

ifeq ($(MODE),debug)
    # Traceable debug configuration
    CXXFLAGS := $(BASE_CXXFLAGS) -g -DEIGEN_INITIALIZE_MATRICES_BY_ZERO -DLOG_LEVEL=4 -O0 -U_FORTIFY_SOURCE
else
    # Highly optimized production environment
    CXXFLAGS := $(BASE_CXXFLAGS) -O3 -march=native -DLOG_LEVEL=3 -DNDEBUG
endif

# --- Layout ------------------------------------------------------------------
SRC_DIR := src
OBJ_DIR := obj
EXP_DIR := experiments
BENCH_DIR := benchmarks
BIN_DIR := bin

# --- Object rule helper ------------------------------------------------------
make-objs = \
  $(patsubst %.cpp,%.o, \
    $(patsubst %.cc,%.o, \
      $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%, \
        $(patsubst $(EXP_DIR)/%,$(OBJ_DIR)/%, \
          $(patsubst $(BENCH_DIR)/%,$(OBJ_DIR)/%,$(1)) \
        ) \
      ) \
    ) \
  )

# --- Compile rules -----------------------------------------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(EXP_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
	
$(OBJ_DIR)/%.o: $(BENCH_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Guard prevents target overriding warnings across included mk fragments
ifndef BIN_DIR_RULE
BIN_DIR_RULE := 1
$(BIN_DIR):
	@mkdir -p $@
endif