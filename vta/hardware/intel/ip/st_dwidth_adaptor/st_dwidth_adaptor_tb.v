`timescale 1ns/1ps

module st_dwidth_adaptor_tb;

   reg          reset = 0;
   reg          clk = 0;
   reg          snk_valid;
   reg [127:0]  snk_data;
   reg         src_ready;
   wire        snk_ready;
   wire        src_valid;
   wire [511:0] src_data;

   initial begin
      #0   reset <= 1;
      #1   reset <= 0;

      snk_valid <= 0;
      snk_data <= 0;
      src_ready <= 0;
      @(posedge clk);
      @(posedge clk);
      
      snk_valid <= 1;
      snk_data <= 16'hffff;
      src_ready <= 0;
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      snk_valid <= 0;
      @(posedge clk);
      src_ready <= 1;
      @(posedge clk);
      src_ready <= 0;
      @(posedge clk);
      @(posedge clk);
      
      snk_valid <= 1;
      snk_data <= 16'haa55;
      src_ready <= 0;
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      @(posedge clk);
      snk_valid <= 0;
      @(posedge clk);
      src_ready <= 1;
      @(posedge clk);
      src_ready <= 0;
      @(posedge clk);
      @(posedge clk);
      
      $finish;
   end
   
   always #5 clk=!clk;
   
   st_dwidth_adaptor st_dwidth_adaptor_0(clk, reset, snk_data, snk_valid, snk_ready, src_data, src_valid, src_ready);

   initial begin
      $monitor("time=%2d: clk=%1b, reset=%1b, snk_ready=%1b, snk_valid=%1b, snk_data=%x, src_ready=%1b, src_valid=%1b, src_data=%x", 
               $time, clk, reset, snk_ready, snk_valid, snk_data, src_ready, src_valid, src_data);
   end
   
endmodule

