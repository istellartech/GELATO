# =============================================================================
# GELATO Build System (CMake wrapper)
# =============================================================================
# Usage:
#   make              Build all pybind11 modules (Release)
#   make clean        Remove build artifacts and compiled modules
#   make rebuild      Clean + build
#   make info         Print CMake cache variables
# =============================================================================

BUILD_DIR := build

all:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j$(nproc)

clean:
	rm -rf $(BUILD_DIR)
	rm -f lib/*_c*.so

rebuild: clean all

info:
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -N -LA 2>/dev/null | head -40

.PHONY: all clean rebuild info
