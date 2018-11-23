module msgdma_to_hls_avst (input clock_clk,
                           input          reset_reset,
                           input [127:0]  avst_sink_data,
                           input          avst_sink_startofpacket,
                           input          avst_sink_endofpacket,
                           input          avst_sink_channel,
                           input [2:0]    avst_sink_empty,
                           output         avst_sink_ready,
                           input          avst_sink_valid,
                           output [127:0] avst_source_data,
                           output         avst_source_startofpacket,
                           output         avst_source_endofpacket,
                           output         avst_source_channel,
                           input          avst_source_ready,
                           output         avst_source_valid
                           );

// assign avst_source_data[63:0] = {avst_sink_data[7:0], avst_sink_data[15:8], avst_sink_data[23:16], avst_sink_data[31:24],
//                                  avst_sink_data[39:32], avst_sink_data[47:40], avst_sink_data[55:48], avst_sink_data[63:56]};
assign avst_source_data[127:0] = {avst_sink_data[7:0], avst_sink_data[15:8], avst_sink_data[23:16], avst_sink_data[31:24],
                                  avst_sink_data[39:32], avst_sink_data[47:40], avst_sink_data[55:48], avst_sink_data[63:56],
                                  avst_sink_data[71:64], avst_sink_data[79:72], avst_sink_data[87:80], avst_sink_data[95:88],
                                  avst_sink_data[103:96], avst_sink_data[111:104], avst_sink_data[119:112], avst_sink_data[127:120]};
assign avst_source_startofpacket = avst_sink_startofpacket;
assign avst_source_endofpacket = avst_sink_endofpacket;
assign avst_source_valid = avst_sink_valid;
assign avst_sink_ready = avst_source_ready;

//always @(posedge clock_clk)
//begin
//  if (avst_sink_valid == 1'b1)
//    avst_sink_ready <= 1'b1;
//  else
//    avst_sink_ready <= 1'b0;
//end

endmodule

