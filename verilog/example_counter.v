// a counter that counts up
// Use as example of testcaase
module counter(clk, rst, out);
   input clk;
   input rst;
   output [3:0] out;
   reg [3:0] counter;
   assign out =  counter;

   always @(posedge clk) begin
      if (rst) begin
         counter <= 0;
      end else begin
         counter <= counter +1;
      end
   end
endmodule
