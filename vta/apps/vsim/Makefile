export PYTHONPATH:=$(PWD)/python:$(PYTHONPATH)

BUILD_DIR = $(shell python python/vsim/config.py --get-build-name)

TVM_DIR = $(abspath ../../../)

VSIM_TARGET = verilog
VSIM_TOP_NAME = TestAccel
VSIM_BUILD_NAME = build

# optional
VSIM_TRACE_NAME = trace.vcd

.PHONY: cmake

cmake: | $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

$(BUILD_DIR):
	mkdir -p $@

run:
	python3 tests/python/test_vsim.py | grep PASS

clean:
	-rm -rf $(BUILD_DIR)
