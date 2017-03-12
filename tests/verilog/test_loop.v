`include "tvm_marcos.v"

module main();
   parameter PER = 10;
   reg clk;
   reg rst;
   wire init;
   wire done;
   wire enable;

   `NORMAL_LOOP_LEAF(iter0, 4, init0, enable, iter0_done, 0, 4, 1)
   `NORMAL_LOOP_NEST(iter1, 4, init, iter0_done, iter1_done, 0, 3, 1, init0)

   assign done = iter0_done;

   always begin
      #(PER/2) clk =~ clk;
   end

   initial begin
      // This will allow tvm session to be called every cycle.
      $tvm_session(clk);
   end
endmodule
