//Legal Notice: (C)2013 Altera Corporation. All rights reserved.  Your
//use of Altera Corporation's design tools, logic functions and other
//software and tools, and its AMPP partner logic functions, and any
//output files any of the foregoing (including device programming or
//simulation files), and any associated documentation or information are
//expressly subject to the terms and conditions of the Altera Program
//License Subscription Agreement or other applicable license agreement,
//including, without limitation, that your use is for the sole purpose
//of programming logic devices manufactured by Altera and sold by Altera
//or its authorized distributors.  Please refer to the applicable
//agreement for further details.

module debounce (
  clk,
  reset_n,
  data_in,
  data_out
);

  parameter WIDTH = 32;           // set to be the width of the bus being debounced
  parameter POLARITY = "HIGH";    // set to be "HIGH" for active high debounce or "LOW" for active low debounce
  parameter TIMEOUT = 50000;      // number of input clock cycles the input signal needs to be in the active state
  parameter TIMEOUT_WIDTH = 16;   // set to be ceil(log2(TIMEOUT))
  
  input wire clk;
  input wire reset_n;
  
  input wire [WIDTH-1:0] data_in;
  output wire [WIDTH-1:0] data_out;
  
  reg [TIMEOUT_WIDTH-1:0] counter [0:WIDTH-1];
  wire counter_reset [0:WIDTH-1];
  wire counter_enable [0:WIDTH-1];
  
  // need one counter per input to debounce
  genvar i;
  generate for (i = 0; i < WIDTH; i = i+1)
  begin:  debounce_counter_loop
    always @ (posedge clk or negedge reset_n)
    begin
      if (reset_n == 0)
      begin
        counter[i] <= 0;
      end
      else
      begin
        if (counter_reset[i] == 1)  // resetting the counter needs to win
        begin
          counter[i] <= 0;
        end
        else if (counter_enable[i] == 1)
        begin
          counter[i] <= counter[i] + 1'b1;
        end
      end
    end
 
    if (POLARITY == "HIGH")
    begin
      assign counter_reset[i] = (data_in[i] == 0);
      assign counter_enable[i] = (data_in[i] == 1) & (counter[i] < TIMEOUT);
      assign data_out[i] = (counter[i] == TIMEOUT) ? 1'b1 : 1'b0;
    end
    else
    begin
      assign counter_reset[i] = (data_in[i] == 1);
      assign counter_enable[i] = (data_in[i] == 0) & (counter[i] < TIMEOUT);
      assign data_out[i] = (counter[i] == TIMEOUT) ? 1'b0 : 1'b1;    
    end
    
  end  
  endgenerate
  
endmodule
