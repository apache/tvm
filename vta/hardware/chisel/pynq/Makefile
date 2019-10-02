TOP = VTA
CONFIG = DefaultPynqConfig
VTA_TARGET = hw_emu
TEST_NAME = simple_load_store

root_dir = $(abspath ../../../)
script_dir = $(abspath .)/scripts
sim_dir = $(abspath .)/sim
out_dir = $(abspath .)/out
chisel = $(abspath ../)

num_threads = 12

verilog_dir = $(root_dir)/build/chisel
generated_verilog = $(verilog_dir)/$(TOP).$(CONFIG).v

testbench_file = $(sim_dir)/$(TEST_NAME)/tb.v

default: $(out_dir)/system.bit

$(out_dir)/system.bit: $(out_dir)/ip/component.xml
	vivado -mode batch -source $(script_dir)/vivado.tcl \
	-tclargs $(out_dir) $(num_threads) $(VTA_TARGET) $(testbench_file) \
	$(sim_dir) $(TEST_NAME)

$(out_dir)/ip/component.xml: $(generated_verilog)
	vivado -mode batch -source $(script_dir)/package_rtl.tcl \
	-tclargs $(verilog_dir) $(out_dir)

$(generated_verilog): | $(out_dir)
	make -C $(chisel) verilog

$(out_dir):
	mkdir -p $@

clean:
	-rm -rf $(out_dir) *.log *.jou *.hdf NA
	make -C $(chisel) cleanall
