`include "tvm_marcos.v"

module main();
   `TVM_DEFINE_TEST_SIGNAL(clk, rst)

   reg ready;
   wire lp_ready;

   `NONSTOP_LOOP(iter0, 4, 0, lp_ready, iter0_finish, 0, 4)
   `NONSTOP_LOOP(iter1, 4, 0, iter0_finish, iter1_finish, 0, 3)
   `WRAP_LOOP_ONCE(0, valid, ready, iter1_finish, loop_ready)
   assign lp_ready = loop_ready;


   initial begin
      // This will allow tvm session to be called every cycle.
      $tvm_session(clk);
   end
endmodule
