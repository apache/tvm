module vta_adaptor (input                clock_clk,
                    input                reset_reset,
						  input                fetch_insn_count_avs_address,
						  output               fetch_insn_count_avs_waitrequest,
						  input                fetch_insn_count_avs_write,
						  input        [31:0]  fetch_insn_count_avs_writedata,
						  output  reg  [31:0]  fetch_insn_count_data,
						  output  reg  [31:0]  fetch_insn_count_2_data,
						  input                fetch_call_avs_address,
						  output               fetch_call_avs_waitrequest,
						  input                fetch_call_avs_write,
						  input        [7:0]   fetch_call_avs_writedata,
						  output               fetch_call_valid,
						  input                fetch_call_stall,
						  input                fetch_insns_avs_address,
						  output               fetch_insns_avs_waitrequest,
						  input                fetch_insns_avs_write,
						  input        [31:0]  fetch_insns_avs_writedata,
						  output       [31:0]  fetch_insns_data,
						  // outputs_cfg slave
						  input                outputs_cfg_address,
						  output  reg          outputs_cfg_waitrequest,
						  input                outputs_cfg_write,
						  input        [31:0]  outputs_cfg_writedata,
						  input                outputs_cfg_read,
						  output  reg  [31:0]  outputs_cfg_readdata,
						  // f2h_axi_master
						  output wire  [31:0]  f2h_axi_master_address,
						  input                f2h_axi_master_waitrequest,
						  output wire          f2h_axi_master_write,
						  output wire [127:0]  f2h_axi_master_writedata,
						  output wire          f2h_axi_master_read,
						  input       [127:0]  f2h_axi_master_readdata,
						  // avalon_csr_master
						  output  reg          avalon_csr_master_address,
						  input                avalon_csr_master_waitrequest,
						  output  reg          avalon_csr_master_write,
						  output  reg  [31:0]  avalon_csr_master_writedata,
						  output  reg          avalon_csr_master_read,
						  input        [31:0]  avalon_csr_master_readdata,
						  // outputs slave
						  input        [31:0]  outputs_address,
						  output wire          outputs_waitrequest,
						  input                outputs_write,
						  input       [127:0]  outputs_writedata,
						  input                outputs_read,
						  output wire [127:0]  outputs_readdata
						  );

reg   [3:0]    awcache = 4'b1111;
reg   [4:0]    awuser = 5'b00001;
reg   [3:0]    arcache = 4'b1111;
reg   [4:0]    aruser = 5'b00001;
reg   [2:0]    awprot = 3'b100;
reg   [2:0]    arprot = 3'b100;
reg   [7:0]    cfg_state;
reg   [31:0]   base_address = 0;
						  
assign fetch_insn_count_avs_waitrequest = 1'b0;
assign fetch_call_avs_waitrequest = 1'b0;
assign fetch_insns_avs_waitrequest = 1'b0;

assign fetch_call_valid = fetch_call_avs_writedata[0];
assign fetch_insns_data[31:0] = fetch_insns_avs_writedata[31:0];

assign f2h_axi_master_address[31:0] = outputs_address[31:0] + base_address[31:0];
assign outputs_waitrequest = f2h_axi_master_waitrequest;
assign f2h_axi_master_write = outputs_write;
assign f2h_axi_master_writedata[127:0] = outputs_writedata[127:0];
assign f2h_axi_master_read = outputs_read;
assign outputs_readdata[127:0] = f2h_axi_master_readdata[127:0];

always @(posedge clock_clk or posedge reset_reset)
begin
  if (reset_reset) begin
    fetch_insn_count_data[31:0] <= 0;
	 fetch_insn_count_2_data[31:0] <= 0;
  end
  else if (fetch_insn_count_avs_write)
  begin
    fetch_insn_count_data[31:0] <= fetch_insn_count_avs_writedata[31:0];
	 fetch_insn_count_2_data[31:0] <= fetch_insn_count_avs_writedata[31:0];
  end
end
  
///////////////////////////////
// state controller
///////////////////////////////

always @(posedge clock_clk or posedge reset_reset) begin
	if (reset_reset) begin
	  cfg_state <= 7;
	end
	else begin
	  if (outputs_cfg_write) begin
		 cfg_state <= 0;
	  end
	  if (!avalon_csr_master_waitrequest) begin
	    if (cfg_state == 7) begin
	      cfg_state <= cfg_state;
	    end
	    else begin
	      cfg_state <= cfg_state+1;
	    end
	  end
   end
end

///////////////////////////////
// axi_slave
///////////////////////////////

always @(posedge clock_clk or posedge reset_reset) begin
	if (reset_reset) begin
	  base_address <= 0;
	end
	else begin
     if (outputs_cfg_write) begin
		 base_address <= outputs_cfg_writedata;
     end
	end
end

///////////////////////////////
// avalon_csr_master
///////////////////////////////

always @(posedge clock_clk or posedge reset_reset) begin
	if (reset_reset) begin
	  avalon_csr_master_writedata <= 0;
	  avalon_csr_master_address <= 0;
	  avalon_csr_master_write <= 0;
	  avalon_csr_master_read <= 0; 
	end
	else begin
	  case (cfg_state) 
	    8'h00: begin avalon_csr_master_address <= 8'h00; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= awcache; end
	    8'h01: begin avalon_csr_master_address <= 8'h04; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= awprot ; end
	    8'h02: begin avalon_csr_master_address <= 8'h08; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= awuser ; end
	    8'h03: begin avalon_csr_master_address <= 8'h10; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= arcache; end
	    8'h04: begin avalon_csr_master_address <= 8'h14; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= arprot ; end
	    8'h05: begin avalon_csr_master_address <= 8'h18; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= aruser ; end
	    8'h06: begin avalon_csr_master_address <= 8'h1C; avalon_csr_master_write <= 1; avalon_csr_master_read <= 0; avalon_csr_master_writedata <= 0      ; end
		 8'h07: begin                                     avalon_csr_master_write <= 0; avalon_csr_master_read <= 0; end
	  endcase
	end
end

endmodule
