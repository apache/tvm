// Module to access TVM VPI simulated RAM.
//
// You only see the wires and registers but no logics here.
// The real computation is implemented via TVM VPI
//
// Usage: create and pass instance to additional arguments of $tvm_session.
// Then  it will be automatically hook up the RAM logic.
//
module tvm_vpi_ram
  # ( parameter READ_WIDTH = 8,
      parameter WRITE_WIDTH = 8
      )
   ( clk,
     rst,
     in_read_dequeue,
     in_write_enable,
     in_write_data,
     ctrl_read_req,
     ctrl_read_addr,
     ctrl_read_size,
     ctrl_write_req,
     ctrl_write_addr,
     ctrl_write_size,
     out_read_data,
     out_read_valid,
     out_write_full
     );
   input clk;
   input rst;
   input in_read_dequeue;
   input in_write_enable;
   input [WRITE_WIDTH-1:0] in_write_data;
   input                   ctrl_read_req;
   input [31:0]            ctrl_read_addr;
   input [31:0]            ctrl_read_size;
   input                   ctrl_write_req;
   input [31:0]            ctrl_write_addr;
   input [31:0]            ctrl_write_size;
   output [READ_WIDTH-1:0] out_read_data;
   output                  out_read_valid;
   output                  out_write_full;
   reg [READ_WIDTH-1:0]    out_reg_read_data;
   reg                     out_reg_read_valid;
   reg                     out_reg_write_full;
   // The wires up.
   assign out_read_data = out_reg_read_data;
   assign out_read_valid = out_reg_read_valid;
   assign out_write_full = out_reg_write_full;
endmodule
