// TVM mmap maps virtual DRAM into interface of SRAM.
// This allows create testcases that directly access DRAM.

// Read only memory map, one cycle read.
// Usage: create and pass instance to additional arguments of $tvm_session.
module tvm_vpi_read_mmap
  #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 8,
    parameter BASE_ADDR_WIDTH = 32
    )
   (
    input                       clk,
    input                       rst,
    // Read Ports
    input [ADDR_WIDTH-1:0]      addr, // Local offset in terms of number of units
    output [DATA_WIDTH-1:0]     data_out, // The data port for read
    // Configure port
    input [BASE_ADDR_WIDTH-1:0] mmap_addr // The base address of memory map.
    );
   reg [DATA_WIDTH-1:0]         reg_data;
   assign data_out = reg_data;
endmodule

// Write only memory map, one cycle write.
// Usage: create and pass instance to additional arguments of $tvm_session.
module tvm_vpi_write_mmap
  #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 8,
    parameter BASE_ADDR_WIDTH = 32
    )
   (
    input                       clk,
    input                       rst,
    // Write Ports
    input [ADDR_WIDTH-1:0]      addr, // Local offset in terms of number of units
    input [DATA_WIDTH-1:0]      data_in, // The data port for write
    input                       en, // The enable port for write
    // Configure port
    input [BASE_ADDR_WIDTH-1:0] mmap_addr // The base address of memap
    );
endmodule
