module st_dwidth_adaptor #(
	parameter INPUT_WIDTH=128,
    parameter OUTPUT_WIDTH=512
) (
    input clk,
    input reset,
    input [INPUT_WIDTH-1:0] snk_data,
    input snk_valid,
    output reg snk_ready,
    output reg [OUTPUT_WIDTH-1:0] src_data,
    output reg src_valid,
    input src_ready
);

reg [3:0] state = 0;
reg [7:0] counter = 0;
   
   initial begin
      $dumpfile("test.vcd");
      $dumpvars;
   end

always @(posedge clk or posedge reset) begin
  if (reset) begin
    state <= 0;
  end
  else if (snk_valid && counter < (OUTPUT_WIDTH/INPUT_WIDTH)) begin
    state <= 1; // start buffering
  end
  else if ((!src_ready) && counter == (OUTPUT_WIDTH/INPUT_WIDTH)) begin
    if (state==1)
      state <= state+1; // provide data
	 else if (state==2)
	   state <= state;
  end
  else if ((state==2) && src_ready && counter == (OUTPUT_WIDTH/INPUT_WIDTH)) begin
	   state <= 3; // idle
  end
end


always @(posedge clk or posedge reset) begin
  if (reset) begin
    snk_ready <= 0;
	 src_data <= 0;
	 src_valid <= 0;
	 counter <= 0;
  end
  else if (state==1) begin 
     if (snk_valid && counter<(OUTPUT_WIDTH/INPUT_WIDTH)-1) begin
		    counter <= counter + 1;
        snk_ready <= 1;
		    src_valid <= 0;
     end
     else if (snk_valid && counter==(OUTPUT_WIDTH/INPUT_WIDTH)-1) begin
		    counter <= counter + 1;
        snk_ready <= 1;
		    src_valid <= 1;
     end
     else begin
        counter <= counter;
		    src_valid <= 1;
        snk_ready <= 0;
     end
	end
  else if (state==2) begin 
     if (src_ready && counter == (OUTPUT_WIDTH/INPUT_WIDTH)) begin
        src_valid <= 0;
        snk_ready <= 0;
		    counter <= 0;
     end
     else begin
        src_valid <= 1;
        snk_ready <= 0;
		    counter <= counter;
     end
	end
end

always @(posedge clk or posedge reset) begin
    if (reset) begin
       src_data <= 0;
    end
    else if (state==1) begin
    	 case (counter)
    		 0: src_data[INPUT_WIDTH-1:0] <= snk_data; 
    		 1: src_data[INPUT_WIDTH*2-1:INPUT_WIDTH] <= snk_data;
    		 2: src_data[INPUT_WIDTH*3-1:INPUT_WIDTH*2] <= snk_data; 
    		 3: src_data[INPUT_WIDTH*4-1:INPUT_WIDTH*3] <= snk_data;
    	 endcase // case (counter)
    end   
end
   
endmodule

