module msgdma_to_hls_avst 
#(
									parameter LENGTH = 128
)
(input clock_clk,
                           input          reset_reset,
                           input [LENGTH-1:0]  avst_sink_data,
                           input          avst_sink_startofpacket,
                           input          avst_sink_endofpacket,
                           input          avst_sink_channel,
                           input [2:0]    avst_sink_empty,
                           output         avst_sink_ready,
                           input          avst_sink_valid,
                           output [LENGTH-1:0] avst_source_data,
                           output         avst_source_startofpacket,
                           output         avst_source_endofpacket,
                           output         avst_source_channel,
                           input          avst_source_ready,
                           output         avst_source_valid
                           );

// assign avst_source_data[63:0] = {avst_sink_data[7:0], avst_sink_data[15:8], avst_sink_data[23:16], avst_sink_data[31:24],
//                                  avst_sink_data[39:32], avst_sink_data[47:40], avst_sink_data[55:48], avst_sink_data[63:56]};
//assign avst_source_data[127:0] = {avst_sink_data[7:0], avst_sink_data[15:8], avst_sink_data[23:16], avst_sink_data[31:24],
//                                  avst_sink_data[39:32], avst_sink_data[47:40], avst_sink_data[55:48], avst_sink_data[63:56],
//                                  avst_sink_data[71:64], avst_sink_data[79:72], avst_sink_data[87:80], avst_sink_data[95:88],
//                                  avst_sink_data[103:96], avst_sink_data[111:104], avst_sink_data[119:112], avst_sink_data[127:120]};

// for i in range(512/8): print 'avst_sink_data[%d:%d], ' % (i*8+7,i*8,),

assign avst_source_data[LENGTH-1:0] = (LENGTH==512)?{avst_sink_data[7:0],  avst_sink_data[15:8],  avst_sink_data[23:16],  avst_sink_data[31:24],  
avst_sink_data[39:32],  avst_sink_data[47:40],  avst_sink_data[55:48],  avst_sink_data[63:56],  
avst_sink_data[71:64],  avst_sink_data[79:72],  avst_sink_data[87:80],  avst_sink_data[95:88],  
avst_sink_data[103:96],  avst_sink_data[111:104],  avst_sink_data[119:112],  avst_sink_data[127:120],  
avst_sink_data[135:128],  avst_sink_data[143:136],  avst_sink_data[151:144],  avst_sink_data[159:152],  
avst_sink_data[167:160],  avst_sink_data[175:168],  avst_sink_data[183:176],  avst_sink_data[191:184],  
avst_sink_data[199:192],  avst_sink_data[207:200],  avst_sink_data[215:208],  avst_sink_data[223:216],  
avst_sink_data[231:224],  avst_sink_data[239:232],  avst_sink_data[247:240],  avst_sink_data[255:248],  
avst_sink_data[263:256],  avst_sink_data[271:264],  avst_sink_data[279:272],  avst_sink_data[287:280],  
avst_sink_data[295:288],  avst_sink_data[303:296],  avst_sink_data[311:304],  avst_sink_data[319:312],  
avst_sink_data[327:320],  avst_sink_data[335:328],  avst_sink_data[343:336],  avst_sink_data[351:344],  
avst_sink_data[359:352],  avst_sink_data[367:360],  avst_sink_data[375:368],  avst_sink_data[383:376],  
avst_sink_data[391:384],  avst_sink_data[399:392],  avst_sink_data[407:400],  avst_sink_data[415:408],  
avst_sink_data[423:416],  avst_sink_data[431:424],  avst_sink_data[439:432],  avst_sink_data[447:440],  
avst_sink_data[455:448],  avst_sink_data[463:456],  avst_sink_data[471:464],  avst_sink_data[479:472],  
avst_sink_data[487:480],  avst_sink_data[495:488],  avst_sink_data[503:496],  avst_sink_data[511:504]}:
											 (LENGTH==128)?{avst_sink_data[7:0], avst_sink_data[15:8], avst_sink_data[23:16], avst_sink_data[31:24],
                                  avst_sink_data[39:32], avst_sink_data[47:40], avst_sink_data[55:48], avst_sink_data[63:56],
                                  avst_sink_data[71:64], avst_sink_data[79:72], avst_sink_data[87:80], avst_sink_data[95:88],
                                  avst_sink_data[103:96], avst_sink_data[111:104], avst_sink_data[119:112], avst_sink_data[127:120]}:
											 (LENGTH==32)?{avst_sink_data[7:0], avst_sink_data[15:8], avst_sink_data[23:16], avst_sink_data[31:24]}:
											 {avst_sink_data[7:0]};
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

