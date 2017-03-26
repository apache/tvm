`include "tvm_marcos.v"

module main();
   `TVM_DEFINE_TEST_SIGNAL(clk, rst)

   reg[31:0] in_data;
   wire[31:0] out_data;
   wire in_ready;
   reg in_valid;
   reg out_ready;
   wire out_valid;

   `CACHE_REG(32, in_data, in_valid, in_ready,
              out_data, out_valid, out_ready)

   initial begin
      // This will allow tvm session to be called every cycle.
      $tvm_session(clk);
   end
endmodule
