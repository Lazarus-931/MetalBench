KERNEL_DIR := src/kernels
HOST_DIR   := src/host
BUILD_DIR  := build

KERNELS    := $(wildcard $(KERNEL_DIR)/*.metal)
AIRS       := $(patsubst $(KERNEL_DIR)/%.metal,$(BUILD_DIR)/%.air,$(KERNELS))
METALLIBS  := $(patsubst $(KERNEL_DIR)/%.metal,$(BUILD_DIR)/%.metallib,$(KERNELS))

HOST_SRC   := $(HOST_DIR)/main.mm
HOST_BIN   := $(BUILD_DIR)/metalbench_host

METAL      := xcrun -sdk macosx metal
METALLIB   := xcrun -sdk macosx metallib
CXX        := clang++
CXXFLAGS   := -std=c++17 -fobjc-arc -O2 -Wall
FRAMEWORKS := -framework Metal -framework Foundation

.PHONY: all kernels host clean
all: kernels host

kernels: $(METALLIBS)
host:    $(HOST_BIN)

$(BUILD_DIR):
	@mkdir -p $@

$(BUILD_DIR)/%.air: $(KERNEL_DIR)/%.metal | $(BUILD_DIR)
	$(METAL) -gline-tables-only -frecord-sources -c $< -o $@

$(BUILD_DIR)/%.metallib: $(BUILD_DIR)/%.air
	$(METALLIB) $< -o $@

$(HOST_BIN): $(HOST_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
