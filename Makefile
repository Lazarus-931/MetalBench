KERNEL_DIR := metal/kernels
METAL_DIR  := metal/scripts
BUILD_DIR  := build

# Kernels live flat at metal/kernels/<set>/<name>.metal where <set> is one of
# common / standard / full. Each compiles to build/<name>.metallib — names
# are globally unique across sets (c1, c2, ..., s1, s2, ..., f1, f2, ...).
# metal/kernels/utils/ holds shared headers (utils.metal) — never built directly.
SET_DIRS    := common standard full
# Flat:  metal/kernels/<set>/<name>.metal             →  build/<name>.metallib
# Variant: metal/kernels/<set>/<name>/<chip>.metal    →  build/<name>__<chip>.metallib
# where <chip> is 'default' or a generation tag like 'm4', 'm5'.
KERNEL_SRCS := $(foreach d,$(SET_DIRS),$(wildcard $(KERNEL_DIR)/$(d)/*.metal))
VARIANT_SRCS := $(foreach d,$(SET_DIRS),$(wildcard $(KERNEL_DIR)/$(d)/*/*.metal))
UTILS_DIR   := $(KERNEL_DIR)/utils

HOST_SRCS  := $(METAL_DIR)/main.mm $(METAL_DIR)/timing.mm $(METAL_DIR)/setup.cpp
HOST_HDRS  := $(METAL_DIR)/timing.h $(METAL_DIR)/setup.h
HOST_BIN   := $(BUILD_DIR)/metalbench_host

METAL      := xcrun -sdk macosx metal
METALLIB   := xcrun -sdk macosx metallib
CXX        := clang++
CXXFLAGS   := -std=c++17 -fobjc-arc -O2 -Wall
FRAMEWORKS := -framework Metal -framework Foundation
METAL_INCS := -I$(UTILS_DIR)

.PHONY: all kernels host clean
all: kernels host
host: $(HOST_BIN)

$(BUILD_DIR):
	@mkdir -p $@

# One generated rule per kernel source so make tracks per-file dependencies
# even though sources live under different <set> subdirectories.
define KERNEL_RULE
$(BUILD_DIR)/$(notdir $(basename $(1))).metallib: $(1) $(wildcard $(UTILS_DIR)/*.metal) | $(BUILD_DIR)
	$(METAL) $(METAL_INCS) -gline-tables-only -frecord-sources -c $$< -o $$(@:.metallib=.air)
	$(METALLIB) $$(@:.metallib=.air) -o $$@
KERNEL_TARGETS += $(BUILD_DIR)/$(notdir $(basename $(1))).metallib
endef
$(foreach src,$(KERNEL_SRCS),$(eval $(call KERNEL_RULE,$(src))))

# Variant rule: metal/kernels/<set>/<name>/<chip>.metal → build/<name>__<chip>.metallib
define VARIANT_RULE
$(BUILD_DIR)/$(notdir $(patsubst %/,%,$(dir $(1))))__$(notdir $(basename $(1))).metallib: $(1) $(wildcard $(UTILS_DIR)/*.metal) | $(BUILD_DIR)
	$(METAL) $(METAL_INCS) -gline-tables-only -frecord-sources -c $$< -o $$(@:.metallib=.air)
	$(METALLIB) $$(@:.metallib=.air) -o $$@
KERNEL_TARGETS += $(BUILD_DIR)/$(notdir $(patsubst %/,%,$(dir $(1))))__$(notdir $(basename $(1))).metallib
endef
$(foreach src,$(VARIANT_SRCS),$(eval $(call VARIANT_RULE,$(src))))

kernels: $(KERNEL_TARGETS)

$(HOST_BIN): $(HOST_SRCS) $(HOST_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -I$(METAL_DIR) $(HOST_SRCS) -o $@

clean:
	rm -rf $(BUILD_DIR)

# Regenerate derived markdown (best_times.md, LINK.md) from session.json.
# Run after kernel changes that touched session.json; also wired into the pre-commit hook.
refresh:
	python3 scripts/render_best_times.py
	python3 scripts/render_link_md.py

# Install the pre-commit hook that auto-refreshes derived markdown when session.json is staged.
install-hooks:
	@git config core.hooksPath scripts/git-hooks
	@chmod +x scripts/git-hooks/pre-commit
	@echo "[hooks] installed — scripts/git-hooks/pre-commit will run on every commit"

.PHONY: kernels clean refresh install-hooks
