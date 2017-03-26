`include "tvm_marcos.v"

module main();
   `TVM_DEFINE_TEST_SIGNAL(clk, rst)

   wire[3:0] counter;
   counter counter_unit1(.clk(clk), .rst(rst), .out(counter));

   initial begin
      // This will allow tvm session to be called every cycle.
      $tvm_session(clk);
   end
endmodule
