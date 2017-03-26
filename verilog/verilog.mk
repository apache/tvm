VPI_CFLAGS=`iverilog-vpi --cflags`
VPI_LDFLAGS=`iverilog-vpi --ldflags`

VER_SRCS = $(wildcard verilog/*.v)

VER_LIBS=lib/tvm_vpi.vpi

lib/tvm_vpi.vpi: verilog/tvm_vpi.cc verilog/tvm_vpi.h
	@mkdir -p $(@D)
	$(CXX) $(VPI_CFLAGS) $(CFLAGS) -o $@ $(filter %.cc, $^) $(LDFLAGS) $(VPI_LDFLAGS)
