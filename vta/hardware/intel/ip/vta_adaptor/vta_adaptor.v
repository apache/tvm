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
						  input                compute_call_avs_address,
						  output               compute_call_avs_waitrequest,
						  input                compute_call_avs_write,
						  input        [7:0]   compute_call_avs_writedata,
						  output               compute_call_valid,
						  input                compute_call_stall,
						  input                compute_biases_avs_address,
						  output               compute_biases_avs_waitrequest,
						  input                compute_biases_avs_write,
						  input        [31:0]  compute_biases_avs_writedata,
						  output       [31:0]  compute_biases_data,
						  input                compute_upos_avs_address,
						  output               compute_upos_avs_waitrequest,
						  input                compute_upos_avs_write,
						  input        [31:0]  compute_upos_avs_writedata,
						  output       [31:0]  compute_upos_data,
						  input                compute_done_avs_address,
						  output               compute_done_avs_waitrequest,
						  output               compute_done_avs_read,
						  output       [31:0]  compute_done_avs_readdata,
						  output       [31:0]  compute_done_data,
						  input                store_call_avs_address,
						  output               store_call_avs_waitrequest,
						  input                store_call_avs_write,
						  input        [7:0]   store_call_avs_writedata,
						  output               store_call_valid,
						  input                store_call_stall,
						  input                store_outputs_avs_address,
						  output               store_outputs_avs_waitrequest,
						  input                store_outputs_avs_write,
						  input        [31:0]  store_outputs_avs_writedata,
						  output       [31:0]  store_outputs_data
						  );

assign fetch_insn_count_avs_waitrequest = 1'b0;
assign fetch_call_avs_waitrequest = 1'b0;
assign fetch_insns_avs_waitrequest = 1'b0;
assign compute_call_avs_waitrequest = 1'b0;
assign compute_biases_avs_waitrequest = 1'b0;
assign compute_uops_avs_waitrequest = 1'b0;
assign store_call_avs_waitrequest = 1'b0;
assign store_outputs_avs_waitrequest = 1'b0;
						  
assign fetch_call_valid = fetch_call_avs_writedata[0];
assign fetch_insns_data[31:0] = fetch_insns_avs_writedata[31:0];

always @(posedge clock_clk)
begin
  if (fetch_insn_count_avs_write)
  begin
    fetch_insn_count_data[31:0] <= fetch_insn_count_avs_writedata[31:0];
	 fetch_insn_count_2_data[31:0] <= fetch_insn_count_avs_writedata[31:0];
  end
end
  
assign compute_call_valid = compute_call_avs_writedata[0];
assign compute_biases_data[31:0] = compute_biases_avs_writedata[31:0];
assign compute_upos_data[31:0] = compute_upos_avs_writedata[31:0];

assign store_call_valid = store_call_avs_writedata[0];
assign store_outputs_data[31:0] = store_outputs_avs_writedata[31:0];

endmodule
