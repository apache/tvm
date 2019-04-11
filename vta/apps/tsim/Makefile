export PYTHONPATH:=$(PWD)/python:$(PYTHONPATH)

BUILD_DIR = $(shell python python/tsim/config.py --get-build-name)

TVM_DIR = $(abspath ../../../)

TSIM_TARGET = verilog
TSIM_TOP_NAME = TestAccel
TSIM_BUILD_NAME = build

# optional
TSIM_TRACE_NAME = trace.vcd

.PHONY: cmake

cmake: | $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

$(BUILD_DIR):
	mkdir -p $@

run:
	python3 tests/python/test_tsim.py | grep PASS

clean:
	-rm -rf $(BUILD_DIR)
