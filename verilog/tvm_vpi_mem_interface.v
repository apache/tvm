// Memory controller to access TVM VPI simulated RAM.
//
// You only see the wires and registers but no logics here.
// The real computation is implemented via TVM VPI
//
// Usage: create and pass instance to additional arguments of $tvm_session.
// Then  it will be automatically hook up the RAM logic.
//
module tvm_vpi_mem_interface
  #(
    parameter READ_WIDTH = 8,
    parameter WRITE_WIDTH = 8,
    parameter ADDR_WIDTH = 32,
    parameter SIZE_WIDTH = 32
    )
   (
    input                   clk,
    input                   rst,
    // Read Ports
    input                   read_en, // Read buffer enable
    output [READ_WIDTH-1:0] read_data_out, // The data port for read
    output                  read_data_valid, // Read is valid.
    // Write ports
    input                   write_en, // Write buffer enable
    input [WRITE_WIDTH-1:0] write_data_in, // Input data to write.
    output                  write_data_ready, // There are still pending write
    // Status port
    // Control signal ports to issue tasks
    input                   host_read_req,   // Read request
    input [ADDR_WIDTH-1:0]  host_read_addr,  // The address to issue a read task
    input [SIZE_WIDTH-1:0]  host_read_size,  // The size of a read
    input                   host_write_req,  // Write request.
    input [ADDR_WIDTH-1:0]  host_write_addr, // The write address
    input [SIZE_WIDTH-1:0]  host_write_size  // The write size
    );
   reg [READ_WIDTH-1:0]    reg_read_data;
   reg                     reg_read_valid;
   reg                     reg_write_ready;
   // The wires up.
   assign read_data_out = reg_read_data;
   assign read_data_valid = reg_read_valid;
   assign write_data_ready = reg_write_ready;
endmodule
