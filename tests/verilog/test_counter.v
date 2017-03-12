module main();
   parameter PER = 10;
   reg clk;
   reg rst;
   wire [3:0] counter;

   counter counter_unit1(.clk(clk), .rst(rst), .out(counter));
   always begin
      #(PER/2) clk =~ clk;
   end
   initial begin
      // This will allow tvm session to be called every cycle.
      $tvm_session(clk);
   end
endmodule
