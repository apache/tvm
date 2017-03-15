`include "tvm_marcos.v"

module main();
   parameter PER = 10;
   parameter WIDTH = 8;
   reg clk;
   reg rst;
   reg read_dequeue;
   reg write_enable;
   reg [WIDTH-1:0] write_data;
   reg             ctrl_read_req;
   reg [31:0]      ctrl_read_addr;
   reg [31:0]      ctrl_read_size;
   reg             ctrl_write_req;
   reg [31:0]      ctrl_write_addr;
   reg [31:0]      ctrl_write_size;
   wire [WIDTH-1:0] read_data;
   wire             read_valid;
   wire             write_full;


   always begin
      #(PER/2) clk =~ clk;
   end

   tvm_vpi_ram #
     (
      .READ_WIDTH(WIDTH),
      .WRITE_WIDTH(WIDTH))
   myram
     (
      .clk(clk),
      .rst(rst),
      .in_read_dequeue(read_dequeue),
      .in_write_enable(write_enable),
      .in_write_data(write_data),
      .ctrl_read_req(ctrl_read_req),
      .ctrl_read_addr(ctrl_read_addr),
      .ctrl_read_size(ctrl_read_size),
      .ctrl_write_req(ctrl_write_req),
      .ctrl_write_addr(ctrl_write_addr),
      .ctrl_write_size(ctrl_write_size),
      .out_read_data(read_data),
      .out_read_valid(read_valid),
      .out_write_full(write_full)
      );

   initial begin
      // pass myram to session to hook it up with simulation
      $tvm_session(clk, myram);
   end
endmodule
