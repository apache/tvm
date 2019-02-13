module Compute(
  input          clock,
  input          reset,
  output         io_done_waitrequest,
  input          io_done_address,
  input          io_done_read,
  output         io_done_readdata,
  input          io_done_write,
  input          io_done_writedata,
  input          io_uops_waitrequest,
  output [31:0]  io_uops_address,
  output         io_uops_read,
  input  [31:0]  io_uops_readdata,
  output         io_uops_write,
  output [31:0]  io_uops_writedata,
  input          io_biases_waitrequest,
  output [31:0]  io_biases_address,
  output         io_biases_read,
  input  [127:0] io_biases_readdata,
  output         io_biases_write,
  output [127:0] io_biases_writedata,
  output         io_gemm_queue_ready,
  input          io_gemm_queue_valid,
  input  [127:0] io_gemm_queue_data,
  output         io_l2g_dep_queue_ready,
  input          io_l2g_dep_queue_valid,
  input          io_l2g_dep_queue_data,
  output         io_s2g_dep_queue_ready,
  input          io_s2g_dep_queue_valid,
  input          io_s2g_dep_queue_data,
  input          io_g2l_dep_queue_ready,
  output         io_g2l_dep_queue_valid,
  output         io_g2l_dep_queue_data,
  input          io_g2s_dep_queue_ready,
  output         io_g2s_dep_queue_valid,
  output         io_g2s_dep_queue_data,
  input          io_inp_mem_waitrequest,
  output [14:0]  io_inp_mem_address,
  output         io_inp_mem_read,
  input  [63:0]  io_inp_mem_readdata,
  output         io_inp_mem_write,
  output [63:0]  io_inp_mem_writedata,
  input          io_wgt_mem_waitrequest,
  output [17:0]  io_wgt_mem_address,
  output         io_wgt_mem_read,
  input  [63:0]  io_wgt_mem_readdata,
  output         io_wgt_mem_write,
  output [63:0]  io_wgt_mem_writedata,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_read,
  input  [127:0] io_out_mem_readdata,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata
);
  reg [511:0] acc_mem [0:255] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 33:20]
  reg [511:0] _RAND_0;
  wire [511:0] acc_mem__T_422_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_422_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_424_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_424_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_400_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_400_addr; // @[Compute.scala 33:20]
  wire  acc_mem__T_400_mask; // @[Compute.scala 33:20]
  wire  acc_mem__T_400_en; // @[Compute.scala 33:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem_uop_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem_uop_addr; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_358_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_358_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_358_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_358_en; // @[Compute.scala 34:20]
  wire  started; // @[Compute.scala 31:17]
  reg [127:0] insn; // @[Compute.scala 36:28]
  reg [127:0] _RAND_2;
  wire  _T_201; // @[Compute.scala 37:31]
  wire  insn_valid; // @[Compute.scala 37:40]
  wire [2:0] opcode; // @[Compute.scala 39:29]
  wire  pop_prev_dep; // @[Compute.scala 40:29]
  wire  pop_next_dep; // @[Compute.scala 41:29]
  wire  push_prev_dep; // @[Compute.scala 42:29]
  wire  push_next_dep; // @[Compute.scala 43:29]
  wire [1:0] memory_type; // @[Compute.scala 45:25]
  wire [15:0] sram_base; // @[Compute.scala 46:25]
  wire [31:0] dram_base; // @[Compute.scala 47:25]
  wire [15:0] uop_cntr_max; // @[Compute.scala 49:25]
  wire [3:0] y_pad_0; // @[Compute.scala 51:25]
  wire [3:0] x_pad_0; // @[Compute.scala 53:25]
  wire [3:0] x_pad_1; // @[Compute.scala 54:25]
  wire [15:0] _GEN_290; // @[Compute.scala 58:30]
  wire [15:0] _GEN_292; // @[Compute.scala 59:30]
  wire [16:0] _T_205; // @[Compute.scala 59:30]
  wire [15:0] _T_206; // @[Compute.scala 59:30]
  wire [15:0] _GEN_293; // @[Compute.scala 59:39]
  wire [16:0] _T_207; // @[Compute.scala 59:39]
  wire [15:0] x_size_total; // @[Compute.scala 59:39]
  wire [19:0] y_offset; // @[Compute.scala 60:31]
  wire  opcode_finish_en; // @[Compute.scala 63:34]
  wire  _T_210; // @[Compute.scala 64:32]
  wire  _T_212; // @[Compute.scala 64:60]
  wire  opcode_load_en; // @[Compute.scala 64:50]
  wire  opcode_gemm_en; // @[Compute.scala 65:32]
  wire  opcode_alu_en; // @[Compute.scala 66:31]
  wire  memory_type_uop_en; // @[Compute.scala 68:40]
  wire  memory_type_acc_en; // @[Compute.scala 69:40]
  reg [2:0] state; // @[Compute.scala 72:22]
  reg [31:0] _RAND_3;
  wire  idle; // @[Compute.scala 74:20]
  wire  dump; // @[Compute.scala 75:20]
  wire  busy; // @[Compute.scala 76:20]
  wire  push; // @[Compute.scala 77:20]
  wire  done; // @[Compute.scala 78:20]
  reg  uops_read; // @[Compute.scala 81:24]
  reg [31:0] _RAND_4;
  reg [31:0] uops_data; // @[Compute.scala 82:24]
  reg [31:0] _RAND_5;
  reg  biases_read; // @[Compute.scala 84:24]
  reg [31:0] _RAND_6;
  reg [127:0] biases_data_0; // @[Compute.scala 87:24]
  reg [127:0] _RAND_7;
  reg [127:0] biases_data_1; // @[Compute.scala 87:24]
  reg [127:0] _RAND_8;
  reg [127:0] biases_data_2; // @[Compute.scala 87:24]
  reg [127:0] _RAND_9;
  reg [127:0] biases_data_3; // @[Compute.scala 87:24]
  reg [127:0] _RAND_10;
  reg  out_mem_write; // @[Compute.scala 89:31]
  reg [31:0] _RAND_11;
  wire  _T_234; // @[Compute.scala 93:37]
  wire  uop_cntr_en; // @[Compute.scala 93:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 95:25]
  reg [31:0] _RAND_12;
  wire  _T_236; // @[Compute.scala 96:38]
  wire  _T_237; // @[Compute.scala 96:56]
  wire  uop_cntr_wrap; // @[Compute.scala 96:71]
  wire [18:0] _T_239; // @[Compute.scala 98:29]
  wire [19:0] _T_241; // @[Compute.scala 98:46]
  wire [18:0] acc_cntr_max; // @[Compute.scala 98:46]
  wire  _T_242; // @[Compute.scala 99:37]
  wire  acc_cntr_en; // @[Compute.scala 99:59]
  reg [15:0] acc_cntr_val; // @[Compute.scala 101:25]
  reg [31:0] _RAND_13;
  wire [18:0] _GEN_295; // @[Compute.scala 102:38]
  wire  _T_244; // @[Compute.scala 102:38]
  wire  _T_245; // @[Compute.scala 102:56]
  wire  acc_cntr_wrap; // @[Compute.scala 102:71]
  wire  _T_249; // @[Compute.scala 105:37]
  wire  out_cntr_en; // @[Compute.scala 105:56]
  reg [15:0] dst_offset_in; // @[Compute.scala 107:25]
  reg [31:0] _RAND_14;
  wire  _T_251; // @[Compute.scala 108:38]
  wire  _T_252; // @[Compute.scala 108:56]
  wire  out_cntr_wrap; // @[Compute.scala 108:71]
  reg  pop_prev_dep_ready; // @[Compute.scala 111:35]
  reg [31:0] _RAND_15;
  reg  pop_next_dep_ready; // @[Compute.scala 112:35]
  reg [31:0] _RAND_16;
  wire  push_prev_dep_valid; // @[Compute.scala 113:43]
  wire  push_next_dep_valid; // @[Compute.scala 114:43]
  reg  push_prev_dep_ready; // @[Compute.scala 115:36]
  reg [31:0] _RAND_17;
  reg  push_next_dep_ready; // @[Compute.scala 116:36]
  reg [31:0] _RAND_18;
  reg  gemm_queue_ready; // @[Compute.scala 118:33]
  reg [31:0] _RAND_19;
  wire  _T_263; // @[Compute.scala 121:23]
  wire  _T_264; // @[Compute.scala 121:40]
  wire  _T_265; // @[Compute.scala 122:25]
  wire [2:0] _GEN_0; // @[Compute.scala 122:43]
  wire [2:0] _GEN_1; // @[Compute.scala 121:58]
  wire  _T_267; // @[Compute.scala 130:18]
  wire  _T_269; // @[Compute.scala 130:41]
  wire  _T_270; // @[Compute.scala 130:38]
  wire  _T_271; // @[Compute.scala 130:14]
  wire  _T_272; // @[Compute.scala 130:79]
  wire  _T_273; // @[Compute.scala 130:62]
  wire [2:0] _GEN_2; // @[Compute.scala 130:97]
  wire  _T_274; // @[Compute.scala 131:38]
  wire  _T_275; // @[Compute.scala 131:14]
  wire [2:0] _GEN_3; // @[Compute.scala 131:63]
  wire  _T_276; // @[Compute.scala 132:38]
  wire  _T_277; // @[Compute.scala 132:14]
  wire [2:0] _GEN_4; // @[Compute.scala 132:63]
  wire  _T_280; // @[Compute.scala 139:22]
  wire  _T_281; // @[Compute.scala 139:30]
  wire  _GEN_5; // @[Compute.scala 139:57]
  wire  _T_283; // @[Compute.scala 142:22]
  wire  _T_284; // @[Compute.scala 142:30]
  wire  _GEN_6; // @[Compute.scala 142:57]
  wire  _T_288; // @[Compute.scala 149:29]
  wire  _T_289; // @[Compute.scala 149:55]
  wire  _GEN_7; // @[Compute.scala 149:64]
  wire  _T_291; // @[Compute.scala 152:29]
  wire  _T_292; // @[Compute.scala 152:55]
  wire  _GEN_8; // @[Compute.scala 152:64]
  wire  _T_295; // @[Compute.scala 157:22]
  wire  _T_296; // @[Compute.scala 157:19]
  wire  _T_297; // @[Compute.scala 157:37]
  wire  _T_298; // @[Compute.scala 157:61]
  wire  _T_299; // @[Compute.scala 157:45]
  wire [16:0] _T_301; // @[Compute.scala 158:34]
  wire [15:0] _T_302; // @[Compute.scala 158:34]
  wire [15:0] _GEN_9; // @[Compute.scala 157:77]
  wire  _T_304; // @[Compute.scala 160:24]
  wire  _T_305; // @[Compute.scala 160:21]
  wire  _T_306; // @[Compute.scala 160:39]
  wire  _T_307; // @[Compute.scala 160:63]
  wire  _T_308; // @[Compute.scala 160:47]
  wire [16:0] _T_310; // @[Compute.scala 161:34]
  wire [15:0] _T_311; // @[Compute.scala 161:34]
  wire [15:0] _GEN_10; // @[Compute.scala 160:79]
  wire  _T_313; // @[Compute.scala 163:26]
  wire  _T_314; // @[Compute.scala 163:23]
  wire  _T_315; // @[Compute.scala 163:41]
  wire  _T_316; // @[Compute.scala 163:65]
  wire  _T_317; // @[Compute.scala 163:49]
  wire [16:0] _T_319; // @[Compute.scala 164:34]
  wire [15:0] _T_320; // @[Compute.scala 164:34]
  wire [15:0] _GEN_11; // @[Compute.scala 163:81]
  wire  _GEN_16; // @[Compute.scala 168:27]
  wire  _GEN_17; // @[Compute.scala 168:27]
  wire  _GEN_18; // @[Compute.scala 168:27]
  wire  _GEN_19; // @[Compute.scala 168:27]
  wire [2:0] _GEN_20; // @[Compute.scala 168:27]
  wire  _T_328; // @[Compute.scala 181:52]
  wire  _T_329; // @[Compute.scala 181:43]
  wire  _GEN_21; // @[Compute.scala 183:27]
  reg  _T_337; // @[Compute.scala 187:30]
  reg [31:0] _RAND_20;
  wire [31:0] _GEN_297; // @[Compute.scala 191:33]
  wire [32:0] _T_340; // @[Compute.scala 191:33]
  wire [31:0] _T_341; // @[Compute.scala 191:33]
  wire [34:0] _GEN_298; // @[Compute.scala 191:49]
  wire [34:0] uop_dram_addr; // @[Compute.scala 191:49]
  wire [16:0] _T_343; // @[Compute.scala 192:33]
  wire [15:0] uop_sram_addr; // @[Compute.scala 192:33]
  wire  _T_345; // @[Compute.scala 193:31]
  wire  _T_346; // @[Compute.scala 193:28]
  wire  _T_347; // @[Compute.scala 193:46]
  wire [16:0] _T_352; // @[Compute.scala 200:42]
  wire [16:0] _T_353; // @[Compute.scala 200:42]
  wire [15:0] _T_354; // @[Compute.scala 200:42]
  wire  _T_355; // @[Compute.scala 200:24]
  wire  _GEN_22; // @[Compute.scala 200:50]
  wire [31:0] _GEN_299; // @[Compute.scala 205:36]
  wire [32:0] _T_359; // @[Compute.scala 205:36]
  wire [31:0] _T_360; // @[Compute.scala 205:36]
  wire [31:0] _GEN_300; // @[Compute.scala 205:47]
  wire [32:0] _T_361; // @[Compute.scala 205:47]
  wire [31:0] _T_362; // @[Compute.scala 205:47]
  wire [34:0] _GEN_301; // @[Compute.scala 205:58]
  wire [34:0] _T_364; // @[Compute.scala 205:58]
  wire [35:0] _T_366; // @[Compute.scala 205:66]
  wire [35:0] _GEN_302; // @[Compute.scala 205:76]
  wire [36:0] _T_367; // @[Compute.scala 205:76]
  wire [35:0] _T_368; // @[Compute.scala 205:76]
  wire [42:0] _GEN_303; // @[Compute.scala 205:92]
  wire [42:0] acc_dram_addr; // @[Compute.scala 205:92]
  wire [19:0] _GEN_304; // @[Compute.scala 206:36]
  wire [20:0] _T_370; // @[Compute.scala 206:36]
  wire [19:0] _T_371; // @[Compute.scala 206:36]
  wire [19:0] _GEN_305; // @[Compute.scala 206:47]
  wire [20:0] _T_372; // @[Compute.scala 206:47]
  wire [19:0] _T_373; // @[Compute.scala 206:47]
  wire [22:0] _GEN_306; // @[Compute.scala 206:58]
  wire [22:0] _T_375; // @[Compute.scala 206:58]
  wire [23:0] _T_377; // @[Compute.scala 206:66]
  wire [23:0] _GEN_307; // @[Compute.scala 206:76]
  wire [24:0] _T_378; // @[Compute.scala 206:76]
  wire [23:0] _T_379; // @[Compute.scala 206:76]
  wire [23:0] _T_381; // @[Compute.scala 206:92]
  wire [24:0] _T_383; // @[Compute.scala 206:100]
  wire [24:0] _T_384; // @[Compute.scala 206:100]
  wire [23:0] acc_sram_addr; // @[Compute.scala 206:100]
  wire  _T_386; // @[Compute.scala 207:33]
  wire [15:0] _GEN_12; // @[Compute.scala 213:30]
  wire [2:0] _T_392; // @[Compute.scala 213:30]
  wire [127:0] _GEN_25; // @[Compute.scala 213:48]
  wire [127:0] _GEN_26; // @[Compute.scala 213:48]
  wire [127:0] _GEN_27; // @[Compute.scala 213:48]
  wire [127:0] _GEN_28; // @[Compute.scala 213:48]
  wire  _T_398; // @[Compute.scala 214:43]
  wire [255:0] _T_401; // @[Cat.scala 30:58]
  wire [255:0] _T_402; // @[Cat.scala 30:58]
  wire [1:0] alu_opcode; // @[Compute.scala 224:24]
  wire  use_imm; // @[Compute.scala 225:21]
  wire [15:0] imm_raw; // @[Compute.scala 226:21]
  wire [15:0] _T_404; // @[Compute.scala 227:25]
  wire  _T_406; // @[Compute.scala 227:32]
  wire [31:0] _T_408; // @[Cat.scala 30:58]
  wire [16:0] _T_410; // @[Cat.scala 30:58]
  wire [31:0] _T_411; // @[Compute.scala 227:16]
  wire [31:0] imm; // @[Compute.scala 227:89]
  wire [10:0] _T_412; // @[Compute.scala 235:20]
  wire [15:0] _GEN_308; // @[Compute.scala 235:47]
  wire [16:0] _T_413; // @[Compute.scala 235:47]
  wire [15:0] dst_idx; // @[Compute.scala 235:47]
  wire [10:0] _T_414; // @[Compute.scala 236:20]
  wire [15:0] _GEN_309; // @[Compute.scala 236:47]
  wire [16:0] _T_415; // @[Compute.scala 236:47]
  wire [15:0] src_idx; // @[Compute.scala 236:47]
  reg [511:0] dst_vector; // @[Compute.scala 239:23]
  reg [511:0] _RAND_21;
  reg [511:0] src_vector; // @[Compute.scala 240:23]
  reg [511:0] _RAND_22;
  wire  alu_opcode_min_en; // @[Compute.scala 258:38]
  wire  alu_opcode_max_en; // @[Compute.scala 259:38]
  wire  _T_856; // @[Compute.scala 278:20]
  wire [31:0] _T_857; // @[Compute.scala 281:31]
  wire [31:0] _T_858; // @[Compute.scala 281:72]
  wire [31:0] _T_859; // @[Compute.scala 282:31]
  wire [31:0] _T_860; // @[Compute.scala 282:72]
  wire [31:0] _T_861; // @[Compute.scala 281:31]
  wire [31:0] _T_862; // @[Compute.scala 281:72]
  wire [31:0] _T_863; // @[Compute.scala 282:31]
  wire [31:0] _T_864; // @[Compute.scala 282:72]
  wire [31:0] _T_865; // @[Compute.scala 281:31]
  wire [31:0] _T_866; // @[Compute.scala 281:72]
  wire [31:0] _T_867; // @[Compute.scala 282:31]
  wire [31:0] _T_868; // @[Compute.scala 282:72]
  wire [31:0] _T_869; // @[Compute.scala 281:31]
  wire [31:0] _T_870; // @[Compute.scala 281:72]
  wire [31:0] _T_871; // @[Compute.scala 282:31]
  wire [31:0] _T_872; // @[Compute.scala 282:72]
  wire [31:0] _T_873; // @[Compute.scala 281:31]
  wire [31:0] _T_874; // @[Compute.scala 281:72]
  wire [31:0] _T_875; // @[Compute.scala 282:31]
  wire [31:0] _T_876; // @[Compute.scala 282:72]
  wire [31:0] _T_877; // @[Compute.scala 281:31]
  wire [31:0] _T_878; // @[Compute.scala 281:72]
  wire [31:0] _T_879; // @[Compute.scala 282:31]
  wire [31:0] _T_880; // @[Compute.scala 282:72]
  wire [31:0] _T_881; // @[Compute.scala 281:31]
  wire [31:0] _T_882; // @[Compute.scala 281:72]
  wire [31:0] _T_883; // @[Compute.scala 282:31]
  wire [31:0] _T_884; // @[Compute.scala 282:72]
  wire [31:0] _T_885; // @[Compute.scala 281:31]
  wire [31:0] _T_886; // @[Compute.scala 281:72]
  wire [31:0] _T_887; // @[Compute.scala 282:31]
  wire [31:0] _T_888; // @[Compute.scala 282:72]
  wire [31:0] _T_889; // @[Compute.scala 281:31]
  wire [31:0] _T_890; // @[Compute.scala 281:72]
  wire [31:0] _T_891; // @[Compute.scala 282:31]
  wire [31:0] _T_892; // @[Compute.scala 282:72]
  wire [31:0] _T_893; // @[Compute.scala 281:31]
  wire [31:0] _T_894; // @[Compute.scala 281:72]
  wire [31:0] _T_895; // @[Compute.scala 282:31]
  wire [31:0] _T_896; // @[Compute.scala 282:72]
  wire [31:0] _T_897; // @[Compute.scala 281:31]
  wire [31:0] _T_898; // @[Compute.scala 281:72]
  wire [31:0] _T_899; // @[Compute.scala 282:31]
  wire [31:0] _T_900; // @[Compute.scala 282:72]
  wire [31:0] _T_901; // @[Compute.scala 281:31]
  wire [31:0] _T_902; // @[Compute.scala 281:72]
  wire [31:0] _T_903; // @[Compute.scala 282:31]
  wire [31:0] _T_904; // @[Compute.scala 282:72]
  wire [31:0] _T_905; // @[Compute.scala 281:31]
  wire [31:0] _T_906; // @[Compute.scala 281:72]
  wire [31:0] _T_907; // @[Compute.scala 282:31]
  wire [31:0] _T_908; // @[Compute.scala 282:72]
  wire [31:0] _T_909; // @[Compute.scala 281:31]
  wire [31:0] _T_910; // @[Compute.scala 281:72]
  wire [31:0] _T_911; // @[Compute.scala 282:31]
  wire [31:0] _T_912; // @[Compute.scala 282:72]
  wire [31:0] _T_913; // @[Compute.scala 281:31]
  wire [31:0] _T_914; // @[Compute.scala 281:72]
  wire [31:0] _T_915; // @[Compute.scala 282:31]
  wire [31:0] _T_916; // @[Compute.scala 282:72]
  wire [31:0] _T_917; // @[Compute.scala 281:31]
  wire [31:0] _T_918; // @[Compute.scala 281:72]
  wire [31:0] _T_919; // @[Compute.scala 282:31]
  wire [31:0] _T_920; // @[Compute.scala 282:72]
  wire [31:0] _GEN_51; // @[Compute.scala 279:30]
  wire [31:0] _GEN_52; // @[Compute.scala 279:30]
  wire [31:0] _GEN_53; // @[Compute.scala 279:30]
  wire [31:0] _GEN_54; // @[Compute.scala 279:30]
  wire [31:0] _GEN_55; // @[Compute.scala 279:30]
  wire [31:0] _GEN_56; // @[Compute.scala 279:30]
  wire [31:0] _GEN_57; // @[Compute.scala 279:30]
  wire [31:0] _GEN_58; // @[Compute.scala 279:30]
  wire [31:0] _GEN_59; // @[Compute.scala 279:30]
  wire [31:0] _GEN_60; // @[Compute.scala 279:30]
  wire [31:0] _GEN_61; // @[Compute.scala 279:30]
  wire [31:0] _GEN_62; // @[Compute.scala 279:30]
  wire [31:0] _GEN_63; // @[Compute.scala 279:30]
  wire [31:0] _GEN_64; // @[Compute.scala 279:30]
  wire [31:0] _GEN_65; // @[Compute.scala 279:30]
  wire [31:0] _GEN_66; // @[Compute.scala 279:30]
  wire [31:0] _GEN_67; // @[Compute.scala 279:30]
  wire [31:0] _GEN_68; // @[Compute.scala 279:30]
  wire [31:0] _GEN_69; // @[Compute.scala 279:30]
  wire [31:0] _GEN_70; // @[Compute.scala 279:30]
  wire [31:0] _GEN_71; // @[Compute.scala 279:30]
  wire [31:0] _GEN_72; // @[Compute.scala 279:30]
  wire [31:0] _GEN_73; // @[Compute.scala 279:30]
  wire [31:0] _GEN_74; // @[Compute.scala 279:30]
  wire [31:0] _GEN_75; // @[Compute.scala 279:30]
  wire [31:0] _GEN_76; // @[Compute.scala 279:30]
  wire [31:0] _GEN_77; // @[Compute.scala 279:30]
  wire [31:0] _GEN_78; // @[Compute.scala 279:30]
  wire [31:0] _GEN_79; // @[Compute.scala 279:30]
  wire [31:0] _GEN_80; // @[Compute.scala 279:30]
  wire [31:0] _GEN_81; // @[Compute.scala 279:30]
  wire [31:0] _GEN_82; // @[Compute.scala 279:30]
  wire [31:0] _GEN_83; // @[Compute.scala 290:20]
  wire [31:0] _GEN_84; // @[Compute.scala 290:20]
  wire [31:0] _GEN_85; // @[Compute.scala 290:20]
  wire [31:0] _GEN_86; // @[Compute.scala 290:20]
  wire [31:0] _GEN_87; // @[Compute.scala 290:20]
  wire [31:0] _GEN_88; // @[Compute.scala 290:20]
  wire [31:0] _GEN_89; // @[Compute.scala 290:20]
  wire [31:0] _GEN_90; // @[Compute.scala 290:20]
  wire [31:0] _GEN_91; // @[Compute.scala 290:20]
  wire [31:0] _GEN_92; // @[Compute.scala 290:20]
  wire [31:0] _GEN_93; // @[Compute.scala 290:20]
  wire [31:0] _GEN_94; // @[Compute.scala 290:20]
  wire [31:0] _GEN_95; // @[Compute.scala 290:20]
  wire [31:0] _GEN_96; // @[Compute.scala 290:20]
  wire [31:0] _GEN_97; // @[Compute.scala 290:20]
  wire [31:0] _GEN_98; // @[Compute.scala 290:20]
  wire [31:0] src_0_0; // @[Compute.scala 278:36]
  wire [31:0] src_1_0; // @[Compute.scala 278:36]
  wire  _T_985; // @[Compute.scala 295:34]
  wire [31:0] _T_986; // @[Compute.scala 295:24]
  wire [31:0] mix_val_0; // @[Compute.scala 278:36]
  wire [7:0] _T_987; // @[Compute.scala 297:37]
  wire [31:0] _T_988; // @[Compute.scala 298:30]
  wire [31:0] _T_989; // @[Compute.scala 298:59]
  wire [32:0] _T_990; // @[Compute.scala 298:49]
  wire [31:0] _T_991; // @[Compute.scala 298:49]
  wire [31:0] _T_992; // @[Compute.scala 298:79]
  wire [31:0] add_val_0; // @[Compute.scala 278:36]
  wire [31:0] add_res_0; // @[Compute.scala 278:36]
  wire [7:0] _T_993; // @[Compute.scala 300:37]
  wire [4:0] _T_995; // @[Compute.scala 301:60]
  wire [31:0] _T_996; // @[Compute.scala 301:49]
  wire [31:0] _T_997; // @[Compute.scala 301:84]
  wire [31:0] shr_val_0; // @[Compute.scala 278:36]
  wire [31:0] shr_res_0; // @[Compute.scala 278:36]
  wire [7:0] _T_998; // @[Compute.scala 303:37]
  wire [31:0] src_0_1; // @[Compute.scala 278:36]
  wire [31:0] src_1_1; // @[Compute.scala 278:36]
  wire  _T_999; // @[Compute.scala 295:34]
  wire [31:0] _T_1000; // @[Compute.scala 295:24]
  wire [31:0] mix_val_1; // @[Compute.scala 278:36]
  wire [7:0] _T_1001; // @[Compute.scala 297:37]
  wire [31:0] _T_1002; // @[Compute.scala 298:30]
  wire [31:0] _T_1003; // @[Compute.scala 298:59]
  wire [32:0] _T_1004; // @[Compute.scala 298:49]
  wire [31:0] _T_1005; // @[Compute.scala 298:49]
  wire [31:0] _T_1006; // @[Compute.scala 298:79]
  wire [31:0] add_val_1; // @[Compute.scala 278:36]
  wire [31:0] add_res_1; // @[Compute.scala 278:36]
  wire [7:0] _T_1007; // @[Compute.scala 300:37]
  wire [4:0] _T_1009; // @[Compute.scala 301:60]
  wire [31:0] _T_1010; // @[Compute.scala 301:49]
  wire [31:0] _T_1011; // @[Compute.scala 301:84]
  wire [31:0] shr_val_1; // @[Compute.scala 278:36]
  wire [31:0] shr_res_1; // @[Compute.scala 278:36]
  wire [7:0] _T_1012; // @[Compute.scala 303:37]
  wire [31:0] src_0_2; // @[Compute.scala 278:36]
  wire [31:0] src_1_2; // @[Compute.scala 278:36]
  wire  _T_1013; // @[Compute.scala 295:34]
  wire [31:0] _T_1014; // @[Compute.scala 295:24]
  wire [31:0] mix_val_2; // @[Compute.scala 278:36]
  wire [7:0] _T_1015; // @[Compute.scala 297:37]
  wire [31:0] _T_1016; // @[Compute.scala 298:30]
  wire [31:0] _T_1017; // @[Compute.scala 298:59]
  wire [32:0] _T_1018; // @[Compute.scala 298:49]
  wire [31:0] _T_1019; // @[Compute.scala 298:49]
  wire [31:0] _T_1020; // @[Compute.scala 298:79]
  wire [31:0] add_val_2; // @[Compute.scala 278:36]
  wire [31:0] add_res_2; // @[Compute.scala 278:36]
  wire [7:0] _T_1021; // @[Compute.scala 300:37]
  wire [4:0] _T_1023; // @[Compute.scala 301:60]
  wire [31:0] _T_1024; // @[Compute.scala 301:49]
  wire [31:0] _T_1025; // @[Compute.scala 301:84]
  wire [31:0] shr_val_2; // @[Compute.scala 278:36]
  wire [31:0] shr_res_2; // @[Compute.scala 278:36]
  wire [7:0] _T_1026; // @[Compute.scala 303:37]
  wire [31:0] src_0_3; // @[Compute.scala 278:36]
  wire [31:0] src_1_3; // @[Compute.scala 278:36]
  wire  _T_1027; // @[Compute.scala 295:34]
  wire [31:0] _T_1028; // @[Compute.scala 295:24]
  wire [31:0] mix_val_3; // @[Compute.scala 278:36]
  wire [7:0] _T_1029; // @[Compute.scala 297:37]
  wire [31:0] _T_1030; // @[Compute.scala 298:30]
  wire [31:0] _T_1031; // @[Compute.scala 298:59]
  wire [32:0] _T_1032; // @[Compute.scala 298:49]
  wire [31:0] _T_1033; // @[Compute.scala 298:49]
  wire [31:0] _T_1034; // @[Compute.scala 298:79]
  wire [31:0] add_val_3; // @[Compute.scala 278:36]
  wire [31:0] add_res_3; // @[Compute.scala 278:36]
  wire [7:0] _T_1035; // @[Compute.scala 300:37]
  wire [4:0] _T_1037; // @[Compute.scala 301:60]
  wire [31:0] _T_1038; // @[Compute.scala 301:49]
  wire [31:0] _T_1039; // @[Compute.scala 301:84]
  wire [31:0] shr_val_3; // @[Compute.scala 278:36]
  wire [31:0] shr_res_3; // @[Compute.scala 278:36]
  wire [7:0] _T_1040; // @[Compute.scala 303:37]
  wire [31:0] src_0_4; // @[Compute.scala 278:36]
  wire [31:0] src_1_4; // @[Compute.scala 278:36]
  wire  _T_1041; // @[Compute.scala 295:34]
  wire [31:0] _T_1042; // @[Compute.scala 295:24]
  wire [31:0] mix_val_4; // @[Compute.scala 278:36]
  wire [7:0] _T_1043; // @[Compute.scala 297:37]
  wire [31:0] _T_1044; // @[Compute.scala 298:30]
  wire [31:0] _T_1045; // @[Compute.scala 298:59]
  wire [32:0] _T_1046; // @[Compute.scala 298:49]
  wire [31:0] _T_1047; // @[Compute.scala 298:49]
  wire [31:0] _T_1048; // @[Compute.scala 298:79]
  wire [31:0] add_val_4; // @[Compute.scala 278:36]
  wire [31:0] add_res_4; // @[Compute.scala 278:36]
  wire [7:0] _T_1049; // @[Compute.scala 300:37]
  wire [4:0] _T_1051; // @[Compute.scala 301:60]
  wire [31:0] _T_1052; // @[Compute.scala 301:49]
  wire [31:0] _T_1053; // @[Compute.scala 301:84]
  wire [31:0] shr_val_4; // @[Compute.scala 278:36]
  wire [31:0] shr_res_4; // @[Compute.scala 278:36]
  wire [7:0] _T_1054; // @[Compute.scala 303:37]
  wire [31:0] src_0_5; // @[Compute.scala 278:36]
  wire [31:0] src_1_5; // @[Compute.scala 278:36]
  wire  _T_1055; // @[Compute.scala 295:34]
  wire [31:0] _T_1056; // @[Compute.scala 295:24]
  wire [31:0] mix_val_5; // @[Compute.scala 278:36]
  wire [7:0] _T_1057; // @[Compute.scala 297:37]
  wire [31:0] _T_1058; // @[Compute.scala 298:30]
  wire [31:0] _T_1059; // @[Compute.scala 298:59]
  wire [32:0] _T_1060; // @[Compute.scala 298:49]
  wire [31:0] _T_1061; // @[Compute.scala 298:49]
  wire [31:0] _T_1062; // @[Compute.scala 298:79]
  wire [31:0] add_val_5; // @[Compute.scala 278:36]
  wire [31:0] add_res_5; // @[Compute.scala 278:36]
  wire [7:0] _T_1063; // @[Compute.scala 300:37]
  wire [4:0] _T_1065; // @[Compute.scala 301:60]
  wire [31:0] _T_1066; // @[Compute.scala 301:49]
  wire [31:0] _T_1067; // @[Compute.scala 301:84]
  wire [31:0] shr_val_5; // @[Compute.scala 278:36]
  wire [31:0] shr_res_5; // @[Compute.scala 278:36]
  wire [7:0] _T_1068; // @[Compute.scala 303:37]
  wire [31:0] src_0_6; // @[Compute.scala 278:36]
  wire [31:0] src_1_6; // @[Compute.scala 278:36]
  wire  _T_1069; // @[Compute.scala 295:34]
  wire [31:0] _T_1070; // @[Compute.scala 295:24]
  wire [31:0] mix_val_6; // @[Compute.scala 278:36]
  wire [7:0] _T_1071; // @[Compute.scala 297:37]
  wire [31:0] _T_1072; // @[Compute.scala 298:30]
  wire [31:0] _T_1073; // @[Compute.scala 298:59]
  wire [32:0] _T_1074; // @[Compute.scala 298:49]
  wire [31:0] _T_1075; // @[Compute.scala 298:49]
  wire [31:0] _T_1076; // @[Compute.scala 298:79]
  wire [31:0] add_val_6; // @[Compute.scala 278:36]
  wire [31:0] add_res_6; // @[Compute.scala 278:36]
  wire [7:0] _T_1077; // @[Compute.scala 300:37]
  wire [4:0] _T_1079; // @[Compute.scala 301:60]
  wire [31:0] _T_1080; // @[Compute.scala 301:49]
  wire [31:0] _T_1081; // @[Compute.scala 301:84]
  wire [31:0] shr_val_6; // @[Compute.scala 278:36]
  wire [31:0] shr_res_6; // @[Compute.scala 278:36]
  wire [7:0] _T_1082; // @[Compute.scala 303:37]
  wire [31:0] src_0_7; // @[Compute.scala 278:36]
  wire [31:0] src_1_7; // @[Compute.scala 278:36]
  wire  _T_1083; // @[Compute.scala 295:34]
  wire [31:0] _T_1084; // @[Compute.scala 295:24]
  wire [31:0] mix_val_7; // @[Compute.scala 278:36]
  wire [7:0] _T_1085; // @[Compute.scala 297:37]
  wire [31:0] _T_1086; // @[Compute.scala 298:30]
  wire [31:0] _T_1087; // @[Compute.scala 298:59]
  wire [32:0] _T_1088; // @[Compute.scala 298:49]
  wire [31:0] _T_1089; // @[Compute.scala 298:49]
  wire [31:0] _T_1090; // @[Compute.scala 298:79]
  wire [31:0] add_val_7; // @[Compute.scala 278:36]
  wire [31:0] add_res_7; // @[Compute.scala 278:36]
  wire [7:0] _T_1091; // @[Compute.scala 300:37]
  wire [4:0] _T_1093; // @[Compute.scala 301:60]
  wire [31:0] _T_1094; // @[Compute.scala 301:49]
  wire [31:0] _T_1095; // @[Compute.scala 301:84]
  wire [31:0] shr_val_7; // @[Compute.scala 278:36]
  wire [31:0] shr_res_7; // @[Compute.scala 278:36]
  wire [7:0] _T_1096; // @[Compute.scala 303:37]
  wire [31:0] src_0_8; // @[Compute.scala 278:36]
  wire [31:0] src_1_8; // @[Compute.scala 278:36]
  wire  _T_1097; // @[Compute.scala 295:34]
  wire [31:0] _T_1098; // @[Compute.scala 295:24]
  wire [31:0] mix_val_8; // @[Compute.scala 278:36]
  wire [7:0] _T_1099; // @[Compute.scala 297:37]
  wire [31:0] _T_1100; // @[Compute.scala 298:30]
  wire [31:0] _T_1101; // @[Compute.scala 298:59]
  wire [32:0] _T_1102; // @[Compute.scala 298:49]
  wire [31:0] _T_1103; // @[Compute.scala 298:49]
  wire [31:0] _T_1104; // @[Compute.scala 298:79]
  wire [31:0] add_val_8; // @[Compute.scala 278:36]
  wire [31:0] add_res_8; // @[Compute.scala 278:36]
  wire [7:0] _T_1105; // @[Compute.scala 300:37]
  wire [4:0] _T_1107; // @[Compute.scala 301:60]
  wire [31:0] _T_1108; // @[Compute.scala 301:49]
  wire [31:0] _T_1109; // @[Compute.scala 301:84]
  wire [31:0] shr_val_8; // @[Compute.scala 278:36]
  wire [31:0] shr_res_8; // @[Compute.scala 278:36]
  wire [7:0] _T_1110; // @[Compute.scala 303:37]
  wire [31:0] src_0_9; // @[Compute.scala 278:36]
  wire [31:0] src_1_9; // @[Compute.scala 278:36]
  wire  _T_1111; // @[Compute.scala 295:34]
  wire [31:0] _T_1112; // @[Compute.scala 295:24]
  wire [31:0] mix_val_9; // @[Compute.scala 278:36]
  wire [7:0] _T_1113; // @[Compute.scala 297:37]
  wire [31:0] _T_1114; // @[Compute.scala 298:30]
  wire [31:0] _T_1115; // @[Compute.scala 298:59]
  wire [32:0] _T_1116; // @[Compute.scala 298:49]
  wire [31:0] _T_1117; // @[Compute.scala 298:49]
  wire [31:0] _T_1118; // @[Compute.scala 298:79]
  wire [31:0] add_val_9; // @[Compute.scala 278:36]
  wire [31:0] add_res_9; // @[Compute.scala 278:36]
  wire [7:0] _T_1119; // @[Compute.scala 300:37]
  wire [4:0] _T_1121; // @[Compute.scala 301:60]
  wire [31:0] _T_1122; // @[Compute.scala 301:49]
  wire [31:0] _T_1123; // @[Compute.scala 301:84]
  wire [31:0] shr_val_9; // @[Compute.scala 278:36]
  wire [31:0] shr_res_9; // @[Compute.scala 278:36]
  wire [7:0] _T_1124; // @[Compute.scala 303:37]
  wire [31:0] src_0_10; // @[Compute.scala 278:36]
  wire [31:0] src_1_10; // @[Compute.scala 278:36]
  wire  _T_1125; // @[Compute.scala 295:34]
  wire [31:0] _T_1126; // @[Compute.scala 295:24]
  wire [31:0] mix_val_10; // @[Compute.scala 278:36]
  wire [7:0] _T_1127; // @[Compute.scala 297:37]
  wire [31:0] _T_1128; // @[Compute.scala 298:30]
  wire [31:0] _T_1129; // @[Compute.scala 298:59]
  wire [32:0] _T_1130; // @[Compute.scala 298:49]
  wire [31:0] _T_1131; // @[Compute.scala 298:49]
  wire [31:0] _T_1132; // @[Compute.scala 298:79]
  wire [31:0] add_val_10; // @[Compute.scala 278:36]
  wire [31:0] add_res_10; // @[Compute.scala 278:36]
  wire [7:0] _T_1133; // @[Compute.scala 300:37]
  wire [4:0] _T_1135; // @[Compute.scala 301:60]
  wire [31:0] _T_1136; // @[Compute.scala 301:49]
  wire [31:0] _T_1137; // @[Compute.scala 301:84]
  wire [31:0] shr_val_10; // @[Compute.scala 278:36]
  wire [31:0] shr_res_10; // @[Compute.scala 278:36]
  wire [7:0] _T_1138; // @[Compute.scala 303:37]
  wire [31:0] src_0_11; // @[Compute.scala 278:36]
  wire [31:0] src_1_11; // @[Compute.scala 278:36]
  wire  _T_1139; // @[Compute.scala 295:34]
  wire [31:0] _T_1140; // @[Compute.scala 295:24]
  wire [31:0] mix_val_11; // @[Compute.scala 278:36]
  wire [7:0] _T_1141; // @[Compute.scala 297:37]
  wire [31:0] _T_1142; // @[Compute.scala 298:30]
  wire [31:0] _T_1143; // @[Compute.scala 298:59]
  wire [32:0] _T_1144; // @[Compute.scala 298:49]
  wire [31:0] _T_1145; // @[Compute.scala 298:49]
  wire [31:0] _T_1146; // @[Compute.scala 298:79]
  wire [31:0] add_val_11; // @[Compute.scala 278:36]
  wire [31:0] add_res_11; // @[Compute.scala 278:36]
  wire [7:0] _T_1147; // @[Compute.scala 300:37]
  wire [4:0] _T_1149; // @[Compute.scala 301:60]
  wire [31:0] _T_1150; // @[Compute.scala 301:49]
  wire [31:0] _T_1151; // @[Compute.scala 301:84]
  wire [31:0] shr_val_11; // @[Compute.scala 278:36]
  wire [31:0] shr_res_11; // @[Compute.scala 278:36]
  wire [7:0] _T_1152; // @[Compute.scala 303:37]
  wire [31:0] src_0_12; // @[Compute.scala 278:36]
  wire [31:0] src_1_12; // @[Compute.scala 278:36]
  wire  _T_1153; // @[Compute.scala 295:34]
  wire [31:0] _T_1154; // @[Compute.scala 295:24]
  wire [31:0] mix_val_12; // @[Compute.scala 278:36]
  wire [7:0] _T_1155; // @[Compute.scala 297:37]
  wire [31:0] _T_1156; // @[Compute.scala 298:30]
  wire [31:0] _T_1157; // @[Compute.scala 298:59]
  wire [32:0] _T_1158; // @[Compute.scala 298:49]
  wire [31:0] _T_1159; // @[Compute.scala 298:49]
  wire [31:0] _T_1160; // @[Compute.scala 298:79]
  wire [31:0] add_val_12; // @[Compute.scala 278:36]
  wire [31:0] add_res_12; // @[Compute.scala 278:36]
  wire [7:0] _T_1161; // @[Compute.scala 300:37]
  wire [4:0] _T_1163; // @[Compute.scala 301:60]
  wire [31:0] _T_1164; // @[Compute.scala 301:49]
  wire [31:0] _T_1165; // @[Compute.scala 301:84]
  wire [31:0] shr_val_12; // @[Compute.scala 278:36]
  wire [31:0] shr_res_12; // @[Compute.scala 278:36]
  wire [7:0] _T_1166; // @[Compute.scala 303:37]
  wire [31:0] src_0_13; // @[Compute.scala 278:36]
  wire [31:0] src_1_13; // @[Compute.scala 278:36]
  wire  _T_1167; // @[Compute.scala 295:34]
  wire [31:0] _T_1168; // @[Compute.scala 295:24]
  wire [31:0] mix_val_13; // @[Compute.scala 278:36]
  wire [7:0] _T_1169; // @[Compute.scala 297:37]
  wire [31:0] _T_1170; // @[Compute.scala 298:30]
  wire [31:0] _T_1171; // @[Compute.scala 298:59]
  wire [32:0] _T_1172; // @[Compute.scala 298:49]
  wire [31:0] _T_1173; // @[Compute.scala 298:49]
  wire [31:0] _T_1174; // @[Compute.scala 298:79]
  wire [31:0] add_val_13; // @[Compute.scala 278:36]
  wire [31:0] add_res_13; // @[Compute.scala 278:36]
  wire [7:0] _T_1175; // @[Compute.scala 300:37]
  wire [4:0] _T_1177; // @[Compute.scala 301:60]
  wire [31:0] _T_1178; // @[Compute.scala 301:49]
  wire [31:0] _T_1179; // @[Compute.scala 301:84]
  wire [31:0] shr_val_13; // @[Compute.scala 278:36]
  wire [31:0] shr_res_13; // @[Compute.scala 278:36]
  wire [7:0] _T_1180; // @[Compute.scala 303:37]
  wire [31:0] src_0_14; // @[Compute.scala 278:36]
  wire [31:0] src_1_14; // @[Compute.scala 278:36]
  wire  _T_1181; // @[Compute.scala 295:34]
  wire [31:0] _T_1182; // @[Compute.scala 295:24]
  wire [31:0] mix_val_14; // @[Compute.scala 278:36]
  wire [7:0] _T_1183; // @[Compute.scala 297:37]
  wire [31:0] _T_1184; // @[Compute.scala 298:30]
  wire [31:0] _T_1185; // @[Compute.scala 298:59]
  wire [32:0] _T_1186; // @[Compute.scala 298:49]
  wire [31:0] _T_1187; // @[Compute.scala 298:49]
  wire [31:0] _T_1188; // @[Compute.scala 298:79]
  wire [31:0] add_val_14; // @[Compute.scala 278:36]
  wire [31:0] add_res_14; // @[Compute.scala 278:36]
  wire [7:0] _T_1189; // @[Compute.scala 300:37]
  wire [4:0] _T_1191; // @[Compute.scala 301:60]
  wire [31:0] _T_1192; // @[Compute.scala 301:49]
  wire [31:0] _T_1193; // @[Compute.scala 301:84]
  wire [31:0] shr_val_14; // @[Compute.scala 278:36]
  wire [31:0] shr_res_14; // @[Compute.scala 278:36]
  wire [7:0] _T_1194; // @[Compute.scala 303:37]
  wire [31:0] src_0_15; // @[Compute.scala 278:36]
  wire [31:0] src_1_15; // @[Compute.scala 278:36]
  wire  _T_1195; // @[Compute.scala 295:34]
  wire [31:0] _T_1196; // @[Compute.scala 295:24]
  wire [31:0] mix_val_15; // @[Compute.scala 278:36]
  wire [7:0] _T_1197; // @[Compute.scala 297:37]
  wire [31:0] _T_1198; // @[Compute.scala 298:30]
  wire [31:0] _T_1199; // @[Compute.scala 298:59]
  wire [32:0] _T_1200; // @[Compute.scala 298:49]
  wire [31:0] _T_1201; // @[Compute.scala 298:49]
  wire [31:0] _T_1202; // @[Compute.scala 298:79]
  wire [31:0] add_val_15; // @[Compute.scala 278:36]
  wire [31:0] add_res_15; // @[Compute.scala 278:36]
  wire [7:0] _T_1203; // @[Compute.scala 300:37]
  wire [4:0] _T_1205; // @[Compute.scala 301:60]
  wire [31:0] _T_1206; // @[Compute.scala 301:49]
  wire [31:0] _T_1207; // @[Compute.scala 301:84]
  wire [31:0] shr_val_15; // @[Compute.scala 278:36]
  wire [31:0] shr_res_15; // @[Compute.scala 278:36]
  wire [7:0] _T_1208; // @[Compute.scala 303:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 278:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 278:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 278:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 278:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 308:48]
  wire  alu_opcode_add_en; // @[Compute.scala 309:39]
  wire  _T_1211; // @[Compute.scala 310:37]
  wire  _T_1212; // @[Compute.scala 310:34]
  wire  _T_1213; // @[Compute.scala 310:52]
  wire [4:0] _T_1215; // @[Compute.scala 311:58]
  wire [4:0] _T_1216; // @[Compute.scala 311:58]
  wire [3:0] _T_1217; // @[Compute.scala 311:58]
  wire [15:0] _GEN_310; // @[Compute.scala 311:40]
  wire  _T_1218; // @[Compute.scala 311:40]
  wire  _T_1219; // @[Compute.scala 311:23]
  wire  _T_1222; // @[Compute.scala 311:66]
  wire  _GEN_275; // @[Compute.scala 311:85]
  wire [16:0] _T_1225; // @[Compute.scala 314:34]
  wire [16:0] _T_1226; // @[Compute.scala 314:34]
  wire [15:0] _T_1227; // @[Compute.scala 314:34]
  wire [22:0] _GEN_311; // @[Compute.scala 314:41]
  wire [22:0] _T_1229; // @[Compute.scala 314:41]
  wire [63:0] _T_1236; // @[Cat.scala 30:58]
  wire [127:0] _T_1244; // @[Cat.scala 30:58]
  wire [63:0] _T_1251; // @[Cat.scala 30:58]
  wire [127:0] _T_1259; // @[Cat.scala 30:58]
  wire [63:0] _T_1266; // @[Cat.scala 30:58]
  wire [127:0] _T_1274; // @[Cat.scala 30:58]
  wire [127:0] _T_1275; // @[Compute.scala 318:8]
  assign acc_mem__T_422_addr = dst_idx[7:0];
  assign acc_mem__T_422_data = acc_mem[acc_mem__T_422_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_424_addr = src_idx[7:0];
  assign acc_mem__T_424_data = acc_mem[acc_mem__T_424_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_400_data = {_T_402,_T_401};
  assign acc_mem__T_400_addr = acc_sram_addr[7:0];
  assign acc_mem__T_400_mask = 1'h1;
  assign acc_mem__T_400_en = _T_305 ? _T_398 : 1'h0;
  assign uop_mem_uop_addr = 10'h0;
  assign uop_mem_uop_data = uop_mem[uop_mem_uop_addr]; // @[Compute.scala 34:20]
  assign uop_mem__T_358_data = uops_data;
  assign uop_mem__T_358_addr = uop_sram_addr[9:0];
  assign uop_mem__T_358_mask = 1'h1;
  assign uop_mem__T_358_en = 1'h1;
  assign started = reset == 1'h0; // @[Compute.scala 31:17]
  assign _T_201 = insn != 128'h0; // @[Compute.scala 37:31]
  assign insn_valid = _T_201 & started; // @[Compute.scala 37:40]
  assign opcode = insn[2:0]; // @[Compute.scala 39:29]
  assign pop_prev_dep = insn[3]; // @[Compute.scala 40:29]
  assign pop_next_dep = insn[4]; // @[Compute.scala 41:29]
  assign push_prev_dep = insn[5]; // @[Compute.scala 42:29]
  assign push_next_dep = insn[6]; // @[Compute.scala 43:29]
  assign memory_type = insn[8:7]; // @[Compute.scala 45:25]
  assign sram_base = insn[24:9]; // @[Compute.scala 46:25]
  assign dram_base = insn[56:25]; // @[Compute.scala 47:25]
  assign uop_cntr_max = insn[95:80]; // @[Compute.scala 49:25]
  assign y_pad_0 = insn[115:112]; // @[Compute.scala 51:25]
  assign x_pad_0 = insn[123:120]; // @[Compute.scala 53:25]
  assign x_pad_1 = insn[127:124]; // @[Compute.scala 54:25]
  assign _GEN_290 = {{12'd0}, y_pad_0}; // @[Compute.scala 58:30]
  assign _GEN_292 = {{12'd0}, x_pad_0}; // @[Compute.scala 59:30]
  assign _T_205 = _GEN_292 + uop_cntr_max; // @[Compute.scala 59:30]
  assign _T_206 = _GEN_292 + uop_cntr_max; // @[Compute.scala 59:30]
  assign _GEN_293 = {{12'd0}, x_pad_1}; // @[Compute.scala 59:39]
  assign _T_207 = _T_206 + _GEN_293; // @[Compute.scala 59:39]
  assign x_size_total = _T_206 + _GEN_293; // @[Compute.scala 59:39]
  assign y_offset = x_size_total * _GEN_290; // @[Compute.scala 60:31]
  assign opcode_finish_en = opcode == 3'h3; // @[Compute.scala 63:34]
  assign _T_210 = opcode == 3'h0; // @[Compute.scala 64:32]
  assign _T_212 = opcode == 3'h1; // @[Compute.scala 64:60]
  assign opcode_load_en = _T_210 | _T_212; // @[Compute.scala 64:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Compute.scala 65:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Compute.scala 66:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Compute.scala 68:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Compute.scala 69:40]
  assign idle = state == 3'h0; // @[Compute.scala 74:20]
  assign dump = state == 3'h1; // @[Compute.scala 75:20]
  assign busy = state == 3'h2; // @[Compute.scala 76:20]
  assign push = state == 3'h3; // @[Compute.scala 77:20]
  assign done = state == 3'h4; // @[Compute.scala 78:20]
  assign _T_234 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 93:37]
  assign uop_cntr_en = _T_234 & insn_valid; // @[Compute.scala 93:59]
  assign _T_236 = uop_cntr_val == uop_cntr_max; // @[Compute.scala 96:38]
  assign _T_237 = _T_236 & uop_cntr_en; // @[Compute.scala 96:56]
  assign uop_cntr_wrap = _T_237 & busy; // @[Compute.scala 96:71]
  assign _T_239 = uop_cntr_max * 16'h4; // @[Compute.scala 98:29]
  assign _T_241 = _T_239 + 19'h1; // @[Compute.scala 98:46]
  assign acc_cntr_max = _T_239 + 19'h1; // @[Compute.scala 98:46]
  assign _T_242 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 99:37]
  assign acc_cntr_en = _T_242 & insn_valid; // @[Compute.scala 99:59]
  assign _GEN_295 = {{3'd0}, acc_cntr_val}; // @[Compute.scala 102:38]
  assign _T_244 = _GEN_295 == acc_cntr_max; // @[Compute.scala 102:38]
  assign _T_245 = _T_244 & acc_cntr_en; // @[Compute.scala 102:56]
  assign acc_cntr_wrap = _T_245 & busy; // @[Compute.scala 102:71]
  assign _T_249 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 105:37]
  assign out_cntr_en = _T_249 & insn_valid; // @[Compute.scala 105:56]
  assign _T_251 = dst_offset_in == 16'h9; // @[Compute.scala 108:38]
  assign _T_252 = _T_251 & out_cntr_en; // @[Compute.scala 108:56]
  assign out_cntr_wrap = _T_252 & busy; // @[Compute.scala 108:71]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Compute.scala 113:43]
  assign push_next_dep_valid = push_next_dep & push; // @[Compute.scala 114:43]
  assign _T_263 = uop_cntr_wrap | acc_cntr_wrap; // @[Compute.scala 121:23]
  assign _T_264 = _T_263 | out_cntr_wrap; // @[Compute.scala 121:40]
  assign _T_265 = push_prev_dep | push_next_dep; // @[Compute.scala 122:25]
  assign _GEN_0 = _T_265 ? 3'h3 : 3'h4; // @[Compute.scala 122:43]
  assign _GEN_1 = _T_264 ? _GEN_0 : state; // @[Compute.scala 121:58]
  assign _T_267 = pop_prev_dep_ready == 1'h0; // @[Compute.scala 130:18]
  assign _T_269 = pop_next_dep_ready == 1'h0; // @[Compute.scala 130:41]
  assign _T_270 = _T_267 & _T_269; // @[Compute.scala 130:38]
  assign _T_271 = busy & _T_270; // @[Compute.scala 130:14]
  assign _T_272 = pop_prev_dep | pop_next_dep; // @[Compute.scala 130:79]
  assign _T_273 = _T_271 & _T_272; // @[Compute.scala 130:62]
  assign _GEN_2 = _T_273 ? 3'h1 : _GEN_1; // @[Compute.scala 130:97]
  assign _T_274 = pop_prev_dep_ready | pop_next_dep_ready; // @[Compute.scala 131:38]
  assign _T_275 = dump & _T_274; // @[Compute.scala 131:14]
  assign _GEN_3 = _T_275 ? 3'h2 : _GEN_2; // @[Compute.scala 131:63]
  assign _T_276 = push_prev_dep_ready | push_next_dep_ready; // @[Compute.scala 132:38]
  assign _T_277 = push & _T_276; // @[Compute.scala 132:14]
  assign _GEN_4 = _T_277 ? 3'h4 : _GEN_3; // @[Compute.scala 132:63]
  assign _T_280 = pop_prev_dep & dump; // @[Compute.scala 139:22]
  assign _T_281 = _T_280 & io_l2g_dep_queue_valid; // @[Compute.scala 139:30]
  assign _GEN_5 = _T_281 ? 1'h1 : pop_prev_dep_ready; // @[Compute.scala 139:57]
  assign _T_283 = pop_next_dep & dump; // @[Compute.scala 142:22]
  assign _T_284 = _T_283 & io_s2g_dep_queue_valid; // @[Compute.scala 142:30]
  assign _GEN_6 = _T_284 ? 1'h1 : pop_next_dep_ready; // @[Compute.scala 142:57]
  assign _T_288 = push_prev_dep_valid & io_g2l_dep_queue_ready; // @[Compute.scala 149:29]
  assign _T_289 = _T_288 & push; // @[Compute.scala 149:55]
  assign _GEN_7 = _T_289 ? 1'h1 : push_prev_dep_ready; // @[Compute.scala 149:64]
  assign _T_291 = push_next_dep_valid & io_g2s_dep_queue_ready; // @[Compute.scala 152:29]
  assign _T_292 = _T_291 & push; // @[Compute.scala 152:55]
  assign _GEN_8 = _T_292 ? 1'h1 : push_next_dep_ready; // @[Compute.scala 152:64]
  assign _T_295 = io_uops_waitrequest == 1'h0; // @[Compute.scala 157:22]
  assign _T_296 = uops_read & _T_295; // @[Compute.scala 157:19]
  assign _T_297 = _T_296 & busy; // @[Compute.scala 157:37]
  assign _T_298 = uop_cntr_val < uop_cntr_max; // @[Compute.scala 157:61]
  assign _T_299 = _T_297 & _T_298; // @[Compute.scala 157:45]
  assign _T_301 = uop_cntr_val + 16'h1; // @[Compute.scala 158:34]
  assign _T_302 = uop_cntr_val + 16'h1; // @[Compute.scala 158:34]
  assign _GEN_9 = _T_299 ? _T_302 : uop_cntr_val; // @[Compute.scala 157:77]
  assign _T_304 = io_biases_waitrequest == 1'h0; // @[Compute.scala 160:24]
  assign _T_305 = biases_read & _T_304; // @[Compute.scala 160:21]
  assign _T_306 = _T_305 & busy; // @[Compute.scala 160:39]
  assign _T_307 = _GEN_295 < acc_cntr_max; // @[Compute.scala 160:63]
  assign _T_308 = _T_306 & _T_307; // @[Compute.scala 160:47]
  assign _T_310 = acc_cntr_val + 16'h1; // @[Compute.scala 161:34]
  assign _T_311 = acc_cntr_val + 16'h1; // @[Compute.scala 161:34]
  assign _GEN_10 = _T_308 ? _T_311 : acc_cntr_val; // @[Compute.scala 160:79]
  assign _T_313 = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 163:26]
  assign _T_314 = out_mem_write & _T_313; // @[Compute.scala 163:23]
  assign _T_315 = _T_314 & busy; // @[Compute.scala 163:41]
  assign _T_316 = dst_offset_in < 16'h9; // @[Compute.scala 163:65]
  assign _T_317 = _T_315 & _T_316; // @[Compute.scala 163:49]
  assign _T_319 = dst_offset_in + 16'h1; // @[Compute.scala 164:34]
  assign _T_320 = dst_offset_in + 16'h1; // @[Compute.scala 164:34]
  assign _GEN_11 = _T_317 ? _T_320 : dst_offset_in; // @[Compute.scala 163:81]
  assign _GEN_16 = gemm_queue_ready ? 1'h0 : _GEN_5; // @[Compute.scala 168:27]
  assign _GEN_17 = gemm_queue_ready ? 1'h0 : _GEN_6; // @[Compute.scala 168:27]
  assign _GEN_18 = gemm_queue_ready ? 1'h0 : _GEN_7; // @[Compute.scala 168:27]
  assign _GEN_19 = gemm_queue_ready ? 1'h0 : _GEN_8; // @[Compute.scala 168:27]
  assign _GEN_20 = gemm_queue_ready ? 3'h2 : _GEN_4; // @[Compute.scala 168:27]
  assign _T_328 = idle | done; // @[Compute.scala 181:52]
  assign _T_329 = io_gemm_queue_valid & _T_328; // @[Compute.scala 181:43]
  assign _GEN_21 = gemm_queue_ready ? 1'h0 : _T_329; // @[Compute.scala 183:27]
  assign _GEN_297 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 191:33]
  assign _T_340 = dram_base + _GEN_297; // @[Compute.scala 191:33]
  assign _T_341 = dram_base + _GEN_297; // @[Compute.scala 191:33]
  assign _GEN_298 = {{3'd0}, _T_341}; // @[Compute.scala 191:49]
  assign uop_dram_addr = _GEN_298 << 2'h2; // @[Compute.scala 191:49]
  assign _T_343 = sram_base + uop_cntr_val; // @[Compute.scala 192:33]
  assign uop_sram_addr = sram_base + uop_cntr_val; // @[Compute.scala 192:33]
  assign _T_345 = uop_cntr_wrap == 1'h0; // @[Compute.scala 193:31]
  assign _T_346 = uop_cntr_en & _T_345; // @[Compute.scala 193:28]
  assign _T_347 = _T_346 & busy; // @[Compute.scala 193:46]
  assign _T_352 = uop_cntr_max - 16'h1; // @[Compute.scala 200:42]
  assign _T_353 = $unsigned(_T_352); // @[Compute.scala 200:42]
  assign _T_354 = _T_353[15:0]; // @[Compute.scala 200:42]
  assign _T_355 = uop_cntr_val == _T_354; // @[Compute.scala 200:24]
  assign _GEN_22 = _T_355 ? 1'h0 : _T_347; // @[Compute.scala 200:50]
  assign _GEN_299 = {{12'd0}, y_offset}; // @[Compute.scala 205:36]
  assign _T_359 = dram_base + _GEN_299; // @[Compute.scala 205:36]
  assign _T_360 = dram_base + _GEN_299; // @[Compute.scala 205:36]
  assign _GEN_300 = {{28'd0}, x_pad_0}; // @[Compute.scala 205:47]
  assign _T_361 = _T_360 + _GEN_300; // @[Compute.scala 205:47]
  assign _T_362 = _T_360 + _GEN_300; // @[Compute.scala 205:47]
  assign _GEN_301 = {{3'd0}, _T_362}; // @[Compute.scala 205:58]
  assign _T_364 = _GEN_301 << 2'h2; // @[Compute.scala 205:58]
  assign _T_366 = _T_364 * 35'h1; // @[Compute.scala 205:66]
  assign _GEN_302 = {{20'd0}, acc_cntr_val}; // @[Compute.scala 205:76]
  assign _T_367 = _T_366 + _GEN_302; // @[Compute.scala 205:76]
  assign _T_368 = _T_366 + _GEN_302; // @[Compute.scala 205:76]
  assign _GEN_303 = {{7'd0}, _T_368}; // @[Compute.scala 205:92]
  assign acc_dram_addr = _GEN_303 << 3'h4; // @[Compute.scala 205:92]
  assign _GEN_304 = {{4'd0}, sram_base}; // @[Compute.scala 206:36]
  assign _T_370 = _GEN_304 + y_offset; // @[Compute.scala 206:36]
  assign _T_371 = _GEN_304 + y_offset; // @[Compute.scala 206:36]
  assign _GEN_305 = {{16'd0}, x_pad_0}; // @[Compute.scala 206:47]
  assign _T_372 = _T_371 + _GEN_305; // @[Compute.scala 206:47]
  assign _T_373 = _T_371 + _GEN_305; // @[Compute.scala 206:47]
  assign _GEN_306 = {{3'd0}, _T_373}; // @[Compute.scala 206:58]
  assign _T_375 = _GEN_306 << 2'h2; // @[Compute.scala 206:58]
  assign _T_377 = _T_375 * 23'h1; // @[Compute.scala 206:66]
  assign _GEN_307 = {{8'd0}, acc_cntr_val}; // @[Compute.scala 206:76]
  assign _T_378 = _T_377 + _GEN_307; // @[Compute.scala 206:76]
  assign _T_379 = _T_377 + _GEN_307; // @[Compute.scala 206:76]
  assign _T_381 = _T_379 >> 2'h2; // @[Compute.scala 206:92]
  assign _T_383 = _T_381 - 24'h1; // @[Compute.scala 206:100]
  assign _T_384 = $unsigned(_T_383); // @[Compute.scala 206:100]
  assign acc_sram_addr = _T_384[23:0]; // @[Compute.scala 206:100]
  assign _T_386 = done == 1'h0; // @[Compute.scala 207:33]
  assign _GEN_12 = acc_cntr_val % 16'h4; // @[Compute.scala 213:30]
  assign _T_392 = _GEN_12[2:0]; // @[Compute.scala 213:30]
  assign _GEN_25 = 3'h0 == _T_392 ? io_biases_readdata : biases_data_0; // @[Compute.scala 213:48]
  assign _GEN_26 = 3'h1 == _T_392 ? io_biases_readdata : biases_data_1; // @[Compute.scala 213:48]
  assign _GEN_27 = 3'h2 == _T_392 ? io_biases_readdata : biases_data_2; // @[Compute.scala 213:48]
  assign _GEN_28 = 3'h3 == _T_392 ? io_biases_readdata : biases_data_3; // @[Compute.scala 213:48]
  assign _T_398 = _T_392 == 3'h0; // @[Compute.scala 214:43]
  assign _T_401 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_402 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 224:24]
  assign use_imm = insn[110]; // @[Compute.scala 225:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 226:21]
  assign _T_404 = $signed(imm_raw); // @[Compute.scala 227:25]
  assign _T_406 = $signed(_T_404) < $signed(16'sh0); // @[Compute.scala 227:32]
  assign _T_408 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_410 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_411 = _T_406 ? _T_408 : {{15'd0}, _T_410}; // @[Compute.scala 227:16]
  assign imm = $signed(_T_411); // @[Compute.scala 227:89]
  assign _T_412 = uop_mem_uop_data[10:0]; // @[Compute.scala 235:20]
  assign _GEN_308 = {{5'd0}, _T_412}; // @[Compute.scala 235:47]
  assign _T_413 = _GEN_308 + dst_offset_in; // @[Compute.scala 235:47]
  assign dst_idx = _GEN_308 + dst_offset_in; // @[Compute.scala 235:47]
  assign _T_414 = uop_mem_uop_data[21:11]; // @[Compute.scala 236:20]
  assign _GEN_309 = {{5'd0}, _T_414}; // @[Compute.scala 236:47]
  assign _T_415 = _GEN_309 + dst_offset_in; // @[Compute.scala 236:47]
  assign src_idx = _GEN_309 + dst_offset_in; // @[Compute.scala 236:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 258:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 259:38]
  assign _T_856 = insn_valid & out_cntr_en; // @[Compute.scala 278:20]
  assign _T_857 = src_vector[31:0]; // @[Compute.scala 281:31]
  assign _T_858 = $signed(_T_857); // @[Compute.scala 281:72]
  assign _T_859 = dst_vector[31:0]; // @[Compute.scala 282:31]
  assign _T_860 = $signed(_T_859); // @[Compute.scala 282:72]
  assign _T_861 = src_vector[63:32]; // @[Compute.scala 281:31]
  assign _T_862 = $signed(_T_861); // @[Compute.scala 281:72]
  assign _T_863 = dst_vector[63:32]; // @[Compute.scala 282:31]
  assign _T_864 = $signed(_T_863); // @[Compute.scala 282:72]
  assign _T_865 = src_vector[95:64]; // @[Compute.scala 281:31]
  assign _T_866 = $signed(_T_865); // @[Compute.scala 281:72]
  assign _T_867 = dst_vector[95:64]; // @[Compute.scala 282:31]
  assign _T_868 = $signed(_T_867); // @[Compute.scala 282:72]
  assign _T_869 = src_vector[127:96]; // @[Compute.scala 281:31]
  assign _T_870 = $signed(_T_869); // @[Compute.scala 281:72]
  assign _T_871 = dst_vector[127:96]; // @[Compute.scala 282:31]
  assign _T_872 = $signed(_T_871); // @[Compute.scala 282:72]
  assign _T_873 = src_vector[159:128]; // @[Compute.scala 281:31]
  assign _T_874 = $signed(_T_873); // @[Compute.scala 281:72]
  assign _T_875 = dst_vector[159:128]; // @[Compute.scala 282:31]
  assign _T_876 = $signed(_T_875); // @[Compute.scala 282:72]
  assign _T_877 = src_vector[191:160]; // @[Compute.scala 281:31]
  assign _T_878 = $signed(_T_877); // @[Compute.scala 281:72]
  assign _T_879 = dst_vector[191:160]; // @[Compute.scala 282:31]
  assign _T_880 = $signed(_T_879); // @[Compute.scala 282:72]
  assign _T_881 = src_vector[223:192]; // @[Compute.scala 281:31]
  assign _T_882 = $signed(_T_881); // @[Compute.scala 281:72]
  assign _T_883 = dst_vector[223:192]; // @[Compute.scala 282:31]
  assign _T_884 = $signed(_T_883); // @[Compute.scala 282:72]
  assign _T_885 = src_vector[255:224]; // @[Compute.scala 281:31]
  assign _T_886 = $signed(_T_885); // @[Compute.scala 281:72]
  assign _T_887 = dst_vector[255:224]; // @[Compute.scala 282:31]
  assign _T_888 = $signed(_T_887); // @[Compute.scala 282:72]
  assign _T_889 = src_vector[287:256]; // @[Compute.scala 281:31]
  assign _T_890 = $signed(_T_889); // @[Compute.scala 281:72]
  assign _T_891 = dst_vector[287:256]; // @[Compute.scala 282:31]
  assign _T_892 = $signed(_T_891); // @[Compute.scala 282:72]
  assign _T_893 = src_vector[319:288]; // @[Compute.scala 281:31]
  assign _T_894 = $signed(_T_893); // @[Compute.scala 281:72]
  assign _T_895 = dst_vector[319:288]; // @[Compute.scala 282:31]
  assign _T_896 = $signed(_T_895); // @[Compute.scala 282:72]
  assign _T_897 = src_vector[351:320]; // @[Compute.scala 281:31]
  assign _T_898 = $signed(_T_897); // @[Compute.scala 281:72]
  assign _T_899 = dst_vector[351:320]; // @[Compute.scala 282:31]
  assign _T_900 = $signed(_T_899); // @[Compute.scala 282:72]
  assign _T_901 = src_vector[383:352]; // @[Compute.scala 281:31]
  assign _T_902 = $signed(_T_901); // @[Compute.scala 281:72]
  assign _T_903 = dst_vector[383:352]; // @[Compute.scala 282:31]
  assign _T_904 = $signed(_T_903); // @[Compute.scala 282:72]
  assign _T_905 = src_vector[415:384]; // @[Compute.scala 281:31]
  assign _T_906 = $signed(_T_905); // @[Compute.scala 281:72]
  assign _T_907 = dst_vector[415:384]; // @[Compute.scala 282:31]
  assign _T_908 = $signed(_T_907); // @[Compute.scala 282:72]
  assign _T_909 = src_vector[447:416]; // @[Compute.scala 281:31]
  assign _T_910 = $signed(_T_909); // @[Compute.scala 281:72]
  assign _T_911 = dst_vector[447:416]; // @[Compute.scala 282:31]
  assign _T_912 = $signed(_T_911); // @[Compute.scala 282:72]
  assign _T_913 = src_vector[479:448]; // @[Compute.scala 281:31]
  assign _T_914 = $signed(_T_913); // @[Compute.scala 281:72]
  assign _T_915 = dst_vector[479:448]; // @[Compute.scala 282:31]
  assign _T_916 = $signed(_T_915); // @[Compute.scala 282:72]
  assign _T_917 = src_vector[511:480]; // @[Compute.scala 281:31]
  assign _T_918 = $signed(_T_917); // @[Compute.scala 281:72]
  assign _T_919 = dst_vector[511:480]; // @[Compute.scala 282:31]
  assign _T_920 = $signed(_T_919); // @[Compute.scala 282:72]
  assign _GEN_51 = alu_opcode_max_en ? $signed(_T_858) : $signed(_T_860); // @[Compute.scala 279:30]
  assign _GEN_52 = alu_opcode_max_en ? $signed(_T_860) : $signed(_T_858); // @[Compute.scala 279:30]
  assign _GEN_53 = alu_opcode_max_en ? $signed(_T_862) : $signed(_T_864); // @[Compute.scala 279:30]
  assign _GEN_54 = alu_opcode_max_en ? $signed(_T_864) : $signed(_T_862); // @[Compute.scala 279:30]
  assign _GEN_55 = alu_opcode_max_en ? $signed(_T_866) : $signed(_T_868); // @[Compute.scala 279:30]
  assign _GEN_56 = alu_opcode_max_en ? $signed(_T_868) : $signed(_T_866); // @[Compute.scala 279:30]
  assign _GEN_57 = alu_opcode_max_en ? $signed(_T_870) : $signed(_T_872); // @[Compute.scala 279:30]
  assign _GEN_58 = alu_opcode_max_en ? $signed(_T_872) : $signed(_T_870); // @[Compute.scala 279:30]
  assign _GEN_59 = alu_opcode_max_en ? $signed(_T_874) : $signed(_T_876); // @[Compute.scala 279:30]
  assign _GEN_60 = alu_opcode_max_en ? $signed(_T_876) : $signed(_T_874); // @[Compute.scala 279:30]
  assign _GEN_61 = alu_opcode_max_en ? $signed(_T_878) : $signed(_T_880); // @[Compute.scala 279:30]
  assign _GEN_62 = alu_opcode_max_en ? $signed(_T_880) : $signed(_T_878); // @[Compute.scala 279:30]
  assign _GEN_63 = alu_opcode_max_en ? $signed(_T_882) : $signed(_T_884); // @[Compute.scala 279:30]
  assign _GEN_64 = alu_opcode_max_en ? $signed(_T_884) : $signed(_T_882); // @[Compute.scala 279:30]
  assign _GEN_65 = alu_opcode_max_en ? $signed(_T_886) : $signed(_T_888); // @[Compute.scala 279:30]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_888) : $signed(_T_886); // @[Compute.scala 279:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_890) : $signed(_T_892); // @[Compute.scala 279:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_892) : $signed(_T_890); // @[Compute.scala 279:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_894) : $signed(_T_896); // @[Compute.scala 279:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_896) : $signed(_T_894); // @[Compute.scala 279:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_898) : $signed(_T_900); // @[Compute.scala 279:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_900) : $signed(_T_898); // @[Compute.scala 279:30]
  assign _GEN_73 = alu_opcode_max_en ? $signed(_T_902) : $signed(_T_904); // @[Compute.scala 279:30]
  assign _GEN_74 = alu_opcode_max_en ? $signed(_T_904) : $signed(_T_902); // @[Compute.scala 279:30]
  assign _GEN_75 = alu_opcode_max_en ? $signed(_T_906) : $signed(_T_908); // @[Compute.scala 279:30]
  assign _GEN_76 = alu_opcode_max_en ? $signed(_T_908) : $signed(_T_906); // @[Compute.scala 279:30]
  assign _GEN_77 = alu_opcode_max_en ? $signed(_T_910) : $signed(_T_912); // @[Compute.scala 279:30]
  assign _GEN_78 = alu_opcode_max_en ? $signed(_T_912) : $signed(_T_910); // @[Compute.scala 279:30]
  assign _GEN_79 = alu_opcode_max_en ? $signed(_T_914) : $signed(_T_916); // @[Compute.scala 279:30]
  assign _GEN_80 = alu_opcode_max_en ? $signed(_T_916) : $signed(_T_914); // @[Compute.scala 279:30]
  assign _GEN_81 = alu_opcode_max_en ? $signed(_T_918) : $signed(_T_920); // @[Compute.scala 279:30]
  assign _GEN_82 = alu_opcode_max_en ? $signed(_T_920) : $signed(_T_918); // @[Compute.scala 279:30]
  assign _GEN_83 = use_imm ? $signed(imm) : $signed(_GEN_52); // @[Compute.scala 290:20]
  assign _GEN_84 = use_imm ? $signed(imm) : $signed(_GEN_54); // @[Compute.scala 290:20]
  assign _GEN_85 = use_imm ? $signed(imm) : $signed(_GEN_56); // @[Compute.scala 290:20]
  assign _GEN_86 = use_imm ? $signed(imm) : $signed(_GEN_58); // @[Compute.scala 290:20]
  assign _GEN_87 = use_imm ? $signed(imm) : $signed(_GEN_60); // @[Compute.scala 290:20]
  assign _GEN_88 = use_imm ? $signed(imm) : $signed(_GEN_62); // @[Compute.scala 290:20]
  assign _GEN_89 = use_imm ? $signed(imm) : $signed(_GEN_64); // @[Compute.scala 290:20]
  assign _GEN_90 = use_imm ? $signed(imm) : $signed(_GEN_66); // @[Compute.scala 290:20]
  assign _GEN_91 = use_imm ? $signed(imm) : $signed(_GEN_68); // @[Compute.scala 290:20]
  assign _GEN_92 = use_imm ? $signed(imm) : $signed(_GEN_70); // @[Compute.scala 290:20]
  assign _GEN_93 = use_imm ? $signed(imm) : $signed(_GEN_72); // @[Compute.scala 290:20]
  assign _GEN_94 = use_imm ? $signed(imm) : $signed(_GEN_74); // @[Compute.scala 290:20]
  assign _GEN_95 = use_imm ? $signed(imm) : $signed(_GEN_76); // @[Compute.scala 290:20]
  assign _GEN_96 = use_imm ? $signed(imm) : $signed(_GEN_78); // @[Compute.scala 290:20]
  assign _GEN_97 = use_imm ? $signed(imm) : $signed(_GEN_80); // @[Compute.scala 290:20]
  assign _GEN_98 = use_imm ? $signed(imm) : $signed(_GEN_82); // @[Compute.scala 290:20]
  assign src_0_0 = _T_856 ? $signed(_GEN_51) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_0 = _T_856 ? $signed(_GEN_83) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_985 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 295:34]
  assign _T_986 = _T_985 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 295:24]
  assign mix_val_0 = _T_856 ? $signed(_T_986) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_987 = mix_val_0[7:0]; // @[Compute.scala 297:37]
  assign _T_988 = $unsigned(src_0_0); // @[Compute.scala 298:30]
  assign _T_989 = $unsigned(src_1_0); // @[Compute.scala 298:59]
  assign _T_990 = _T_988 + _T_989; // @[Compute.scala 298:49]
  assign _T_991 = _T_988 + _T_989; // @[Compute.scala 298:49]
  assign _T_992 = $signed(_T_991); // @[Compute.scala 298:79]
  assign add_val_0 = _T_856 ? $signed(_T_992) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_0 = _T_856 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_993 = add_res_0[7:0]; // @[Compute.scala 300:37]
  assign _T_995 = src_1_0[4:0]; // @[Compute.scala 301:60]
  assign _T_996 = _T_988 >> _T_995; // @[Compute.scala 301:49]
  assign _T_997 = $signed(_T_996); // @[Compute.scala 301:84]
  assign shr_val_0 = _T_856 ? $signed(_T_997) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_0 = _T_856 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_998 = shr_res_0[7:0]; // @[Compute.scala 303:37]
  assign src_0_1 = _T_856 ? $signed(_GEN_53) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_1 = _T_856 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_999 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 295:34]
  assign _T_1000 = _T_999 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 295:24]
  assign mix_val_1 = _T_856 ? $signed(_T_1000) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1001 = mix_val_1[7:0]; // @[Compute.scala 297:37]
  assign _T_1002 = $unsigned(src_0_1); // @[Compute.scala 298:30]
  assign _T_1003 = $unsigned(src_1_1); // @[Compute.scala 298:59]
  assign _T_1004 = _T_1002 + _T_1003; // @[Compute.scala 298:49]
  assign _T_1005 = _T_1002 + _T_1003; // @[Compute.scala 298:49]
  assign _T_1006 = $signed(_T_1005); // @[Compute.scala 298:79]
  assign add_val_1 = _T_856 ? $signed(_T_1006) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_1 = _T_856 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1007 = add_res_1[7:0]; // @[Compute.scala 300:37]
  assign _T_1009 = src_1_1[4:0]; // @[Compute.scala 301:60]
  assign _T_1010 = _T_1002 >> _T_1009; // @[Compute.scala 301:49]
  assign _T_1011 = $signed(_T_1010); // @[Compute.scala 301:84]
  assign shr_val_1 = _T_856 ? $signed(_T_1011) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_1 = _T_856 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1012 = shr_res_1[7:0]; // @[Compute.scala 303:37]
  assign src_0_2 = _T_856 ? $signed(_GEN_55) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_2 = _T_856 ? $signed(_GEN_85) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1013 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 295:34]
  assign _T_1014 = _T_1013 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 295:24]
  assign mix_val_2 = _T_856 ? $signed(_T_1014) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1015 = mix_val_2[7:0]; // @[Compute.scala 297:37]
  assign _T_1016 = $unsigned(src_0_2); // @[Compute.scala 298:30]
  assign _T_1017 = $unsigned(src_1_2); // @[Compute.scala 298:59]
  assign _T_1018 = _T_1016 + _T_1017; // @[Compute.scala 298:49]
  assign _T_1019 = _T_1016 + _T_1017; // @[Compute.scala 298:49]
  assign _T_1020 = $signed(_T_1019); // @[Compute.scala 298:79]
  assign add_val_2 = _T_856 ? $signed(_T_1020) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_2 = _T_856 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1021 = add_res_2[7:0]; // @[Compute.scala 300:37]
  assign _T_1023 = src_1_2[4:0]; // @[Compute.scala 301:60]
  assign _T_1024 = _T_1016 >> _T_1023; // @[Compute.scala 301:49]
  assign _T_1025 = $signed(_T_1024); // @[Compute.scala 301:84]
  assign shr_val_2 = _T_856 ? $signed(_T_1025) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_2 = _T_856 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1026 = shr_res_2[7:0]; // @[Compute.scala 303:37]
  assign src_0_3 = _T_856 ? $signed(_GEN_57) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_3 = _T_856 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1027 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 295:34]
  assign _T_1028 = _T_1027 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 295:24]
  assign mix_val_3 = _T_856 ? $signed(_T_1028) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1029 = mix_val_3[7:0]; // @[Compute.scala 297:37]
  assign _T_1030 = $unsigned(src_0_3); // @[Compute.scala 298:30]
  assign _T_1031 = $unsigned(src_1_3); // @[Compute.scala 298:59]
  assign _T_1032 = _T_1030 + _T_1031; // @[Compute.scala 298:49]
  assign _T_1033 = _T_1030 + _T_1031; // @[Compute.scala 298:49]
  assign _T_1034 = $signed(_T_1033); // @[Compute.scala 298:79]
  assign add_val_3 = _T_856 ? $signed(_T_1034) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_3 = _T_856 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1035 = add_res_3[7:0]; // @[Compute.scala 300:37]
  assign _T_1037 = src_1_3[4:0]; // @[Compute.scala 301:60]
  assign _T_1038 = _T_1030 >> _T_1037; // @[Compute.scala 301:49]
  assign _T_1039 = $signed(_T_1038); // @[Compute.scala 301:84]
  assign shr_val_3 = _T_856 ? $signed(_T_1039) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_3 = _T_856 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1040 = shr_res_3[7:0]; // @[Compute.scala 303:37]
  assign src_0_4 = _T_856 ? $signed(_GEN_59) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_4 = _T_856 ? $signed(_GEN_87) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1041 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 295:34]
  assign _T_1042 = _T_1041 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 295:24]
  assign mix_val_4 = _T_856 ? $signed(_T_1042) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1043 = mix_val_4[7:0]; // @[Compute.scala 297:37]
  assign _T_1044 = $unsigned(src_0_4); // @[Compute.scala 298:30]
  assign _T_1045 = $unsigned(src_1_4); // @[Compute.scala 298:59]
  assign _T_1046 = _T_1044 + _T_1045; // @[Compute.scala 298:49]
  assign _T_1047 = _T_1044 + _T_1045; // @[Compute.scala 298:49]
  assign _T_1048 = $signed(_T_1047); // @[Compute.scala 298:79]
  assign add_val_4 = _T_856 ? $signed(_T_1048) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_4 = _T_856 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1049 = add_res_4[7:0]; // @[Compute.scala 300:37]
  assign _T_1051 = src_1_4[4:0]; // @[Compute.scala 301:60]
  assign _T_1052 = _T_1044 >> _T_1051; // @[Compute.scala 301:49]
  assign _T_1053 = $signed(_T_1052); // @[Compute.scala 301:84]
  assign shr_val_4 = _T_856 ? $signed(_T_1053) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_4 = _T_856 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1054 = shr_res_4[7:0]; // @[Compute.scala 303:37]
  assign src_0_5 = _T_856 ? $signed(_GEN_61) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_5 = _T_856 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1055 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 295:34]
  assign _T_1056 = _T_1055 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 295:24]
  assign mix_val_5 = _T_856 ? $signed(_T_1056) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1057 = mix_val_5[7:0]; // @[Compute.scala 297:37]
  assign _T_1058 = $unsigned(src_0_5); // @[Compute.scala 298:30]
  assign _T_1059 = $unsigned(src_1_5); // @[Compute.scala 298:59]
  assign _T_1060 = _T_1058 + _T_1059; // @[Compute.scala 298:49]
  assign _T_1061 = _T_1058 + _T_1059; // @[Compute.scala 298:49]
  assign _T_1062 = $signed(_T_1061); // @[Compute.scala 298:79]
  assign add_val_5 = _T_856 ? $signed(_T_1062) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_5 = _T_856 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1063 = add_res_5[7:0]; // @[Compute.scala 300:37]
  assign _T_1065 = src_1_5[4:0]; // @[Compute.scala 301:60]
  assign _T_1066 = _T_1058 >> _T_1065; // @[Compute.scala 301:49]
  assign _T_1067 = $signed(_T_1066); // @[Compute.scala 301:84]
  assign shr_val_5 = _T_856 ? $signed(_T_1067) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_5 = _T_856 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1068 = shr_res_5[7:0]; // @[Compute.scala 303:37]
  assign src_0_6 = _T_856 ? $signed(_GEN_63) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_6 = _T_856 ? $signed(_GEN_89) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1069 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 295:34]
  assign _T_1070 = _T_1069 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 295:24]
  assign mix_val_6 = _T_856 ? $signed(_T_1070) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1071 = mix_val_6[7:0]; // @[Compute.scala 297:37]
  assign _T_1072 = $unsigned(src_0_6); // @[Compute.scala 298:30]
  assign _T_1073 = $unsigned(src_1_6); // @[Compute.scala 298:59]
  assign _T_1074 = _T_1072 + _T_1073; // @[Compute.scala 298:49]
  assign _T_1075 = _T_1072 + _T_1073; // @[Compute.scala 298:49]
  assign _T_1076 = $signed(_T_1075); // @[Compute.scala 298:79]
  assign add_val_6 = _T_856 ? $signed(_T_1076) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_6 = _T_856 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1077 = add_res_6[7:0]; // @[Compute.scala 300:37]
  assign _T_1079 = src_1_6[4:0]; // @[Compute.scala 301:60]
  assign _T_1080 = _T_1072 >> _T_1079; // @[Compute.scala 301:49]
  assign _T_1081 = $signed(_T_1080); // @[Compute.scala 301:84]
  assign shr_val_6 = _T_856 ? $signed(_T_1081) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_6 = _T_856 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1082 = shr_res_6[7:0]; // @[Compute.scala 303:37]
  assign src_0_7 = _T_856 ? $signed(_GEN_65) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_7 = _T_856 ? $signed(_GEN_90) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1083 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 295:34]
  assign _T_1084 = _T_1083 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 295:24]
  assign mix_val_7 = _T_856 ? $signed(_T_1084) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1085 = mix_val_7[7:0]; // @[Compute.scala 297:37]
  assign _T_1086 = $unsigned(src_0_7); // @[Compute.scala 298:30]
  assign _T_1087 = $unsigned(src_1_7); // @[Compute.scala 298:59]
  assign _T_1088 = _T_1086 + _T_1087; // @[Compute.scala 298:49]
  assign _T_1089 = _T_1086 + _T_1087; // @[Compute.scala 298:49]
  assign _T_1090 = $signed(_T_1089); // @[Compute.scala 298:79]
  assign add_val_7 = _T_856 ? $signed(_T_1090) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_7 = _T_856 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1091 = add_res_7[7:0]; // @[Compute.scala 300:37]
  assign _T_1093 = src_1_7[4:0]; // @[Compute.scala 301:60]
  assign _T_1094 = _T_1086 >> _T_1093; // @[Compute.scala 301:49]
  assign _T_1095 = $signed(_T_1094); // @[Compute.scala 301:84]
  assign shr_val_7 = _T_856 ? $signed(_T_1095) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_7 = _T_856 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1096 = shr_res_7[7:0]; // @[Compute.scala 303:37]
  assign src_0_8 = _T_856 ? $signed(_GEN_67) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_8 = _T_856 ? $signed(_GEN_91) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1097 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 295:34]
  assign _T_1098 = _T_1097 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 295:24]
  assign mix_val_8 = _T_856 ? $signed(_T_1098) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1099 = mix_val_8[7:0]; // @[Compute.scala 297:37]
  assign _T_1100 = $unsigned(src_0_8); // @[Compute.scala 298:30]
  assign _T_1101 = $unsigned(src_1_8); // @[Compute.scala 298:59]
  assign _T_1102 = _T_1100 + _T_1101; // @[Compute.scala 298:49]
  assign _T_1103 = _T_1100 + _T_1101; // @[Compute.scala 298:49]
  assign _T_1104 = $signed(_T_1103); // @[Compute.scala 298:79]
  assign add_val_8 = _T_856 ? $signed(_T_1104) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_8 = _T_856 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1105 = add_res_8[7:0]; // @[Compute.scala 300:37]
  assign _T_1107 = src_1_8[4:0]; // @[Compute.scala 301:60]
  assign _T_1108 = _T_1100 >> _T_1107; // @[Compute.scala 301:49]
  assign _T_1109 = $signed(_T_1108); // @[Compute.scala 301:84]
  assign shr_val_8 = _T_856 ? $signed(_T_1109) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_8 = _T_856 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1110 = shr_res_8[7:0]; // @[Compute.scala 303:37]
  assign src_0_9 = _T_856 ? $signed(_GEN_69) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_9 = _T_856 ? $signed(_GEN_92) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1111 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 295:34]
  assign _T_1112 = _T_1111 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 295:24]
  assign mix_val_9 = _T_856 ? $signed(_T_1112) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1113 = mix_val_9[7:0]; // @[Compute.scala 297:37]
  assign _T_1114 = $unsigned(src_0_9); // @[Compute.scala 298:30]
  assign _T_1115 = $unsigned(src_1_9); // @[Compute.scala 298:59]
  assign _T_1116 = _T_1114 + _T_1115; // @[Compute.scala 298:49]
  assign _T_1117 = _T_1114 + _T_1115; // @[Compute.scala 298:49]
  assign _T_1118 = $signed(_T_1117); // @[Compute.scala 298:79]
  assign add_val_9 = _T_856 ? $signed(_T_1118) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_9 = _T_856 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1119 = add_res_9[7:0]; // @[Compute.scala 300:37]
  assign _T_1121 = src_1_9[4:0]; // @[Compute.scala 301:60]
  assign _T_1122 = _T_1114 >> _T_1121; // @[Compute.scala 301:49]
  assign _T_1123 = $signed(_T_1122); // @[Compute.scala 301:84]
  assign shr_val_9 = _T_856 ? $signed(_T_1123) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_9 = _T_856 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1124 = shr_res_9[7:0]; // @[Compute.scala 303:37]
  assign src_0_10 = _T_856 ? $signed(_GEN_71) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_10 = _T_856 ? $signed(_GEN_93) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1125 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 295:34]
  assign _T_1126 = _T_1125 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 295:24]
  assign mix_val_10 = _T_856 ? $signed(_T_1126) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1127 = mix_val_10[7:0]; // @[Compute.scala 297:37]
  assign _T_1128 = $unsigned(src_0_10); // @[Compute.scala 298:30]
  assign _T_1129 = $unsigned(src_1_10); // @[Compute.scala 298:59]
  assign _T_1130 = _T_1128 + _T_1129; // @[Compute.scala 298:49]
  assign _T_1131 = _T_1128 + _T_1129; // @[Compute.scala 298:49]
  assign _T_1132 = $signed(_T_1131); // @[Compute.scala 298:79]
  assign add_val_10 = _T_856 ? $signed(_T_1132) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_10 = _T_856 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1133 = add_res_10[7:0]; // @[Compute.scala 300:37]
  assign _T_1135 = src_1_10[4:0]; // @[Compute.scala 301:60]
  assign _T_1136 = _T_1128 >> _T_1135; // @[Compute.scala 301:49]
  assign _T_1137 = $signed(_T_1136); // @[Compute.scala 301:84]
  assign shr_val_10 = _T_856 ? $signed(_T_1137) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_10 = _T_856 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1138 = shr_res_10[7:0]; // @[Compute.scala 303:37]
  assign src_0_11 = _T_856 ? $signed(_GEN_73) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_11 = _T_856 ? $signed(_GEN_94) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1139 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 295:34]
  assign _T_1140 = _T_1139 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 295:24]
  assign mix_val_11 = _T_856 ? $signed(_T_1140) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1141 = mix_val_11[7:0]; // @[Compute.scala 297:37]
  assign _T_1142 = $unsigned(src_0_11); // @[Compute.scala 298:30]
  assign _T_1143 = $unsigned(src_1_11); // @[Compute.scala 298:59]
  assign _T_1144 = _T_1142 + _T_1143; // @[Compute.scala 298:49]
  assign _T_1145 = _T_1142 + _T_1143; // @[Compute.scala 298:49]
  assign _T_1146 = $signed(_T_1145); // @[Compute.scala 298:79]
  assign add_val_11 = _T_856 ? $signed(_T_1146) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_11 = _T_856 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1147 = add_res_11[7:0]; // @[Compute.scala 300:37]
  assign _T_1149 = src_1_11[4:0]; // @[Compute.scala 301:60]
  assign _T_1150 = _T_1142 >> _T_1149; // @[Compute.scala 301:49]
  assign _T_1151 = $signed(_T_1150); // @[Compute.scala 301:84]
  assign shr_val_11 = _T_856 ? $signed(_T_1151) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_11 = _T_856 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1152 = shr_res_11[7:0]; // @[Compute.scala 303:37]
  assign src_0_12 = _T_856 ? $signed(_GEN_75) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_12 = _T_856 ? $signed(_GEN_95) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1153 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 295:34]
  assign _T_1154 = _T_1153 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 295:24]
  assign mix_val_12 = _T_856 ? $signed(_T_1154) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1155 = mix_val_12[7:0]; // @[Compute.scala 297:37]
  assign _T_1156 = $unsigned(src_0_12); // @[Compute.scala 298:30]
  assign _T_1157 = $unsigned(src_1_12); // @[Compute.scala 298:59]
  assign _T_1158 = _T_1156 + _T_1157; // @[Compute.scala 298:49]
  assign _T_1159 = _T_1156 + _T_1157; // @[Compute.scala 298:49]
  assign _T_1160 = $signed(_T_1159); // @[Compute.scala 298:79]
  assign add_val_12 = _T_856 ? $signed(_T_1160) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_12 = _T_856 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1161 = add_res_12[7:0]; // @[Compute.scala 300:37]
  assign _T_1163 = src_1_12[4:0]; // @[Compute.scala 301:60]
  assign _T_1164 = _T_1156 >> _T_1163; // @[Compute.scala 301:49]
  assign _T_1165 = $signed(_T_1164); // @[Compute.scala 301:84]
  assign shr_val_12 = _T_856 ? $signed(_T_1165) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_12 = _T_856 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1166 = shr_res_12[7:0]; // @[Compute.scala 303:37]
  assign src_0_13 = _T_856 ? $signed(_GEN_77) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_13 = _T_856 ? $signed(_GEN_96) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1167 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 295:34]
  assign _T_1168 = _T_1167 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 295:24]
  assign mix_val_13 = _T_856 ? $signed(_T_1168) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1169 = mix_val_13[7:0]; // @[Compute.scala 297:37]
  assign _T_1170 = $unsigned(src_0_13); // @[Compute.scala 298:30]
  assign _T_1171 = $unsigned(src_1_13); // @[Compute.scala 298:59]
  assign _T_1172 = _T_1170 + _T_1171; // @[Compute.scala 298:49]
  assign _T_1173 = _T_1170 + _T_1171; // @[Compute.scala 298:49]
  assign _T_1174 = $signed(_T_1173); // @[Compute.scala 298:79]
  assign add_val_13 = _T_856 ? $signed(_T_1174) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_13 = _T_856 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1175 = add_res_13[7:0]; // @[Compute.scala 300:37]
  assign _T_1177 = src_1_13[4:0]; // @[Compute.scala 301:60]
  assign _T_1178 = _T_1170 >> _T_1177; // @[Compute.scala 301:49]
  assign _T_1179 = $signed(_T_1178); // @[Compute.scala 301:84]
  assign shr_val_13 = _T_856 ? $signed(_T_1179) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_13 = _T_856 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1180 = shr_res_13[7:0]; // @[Compute.scala 303:37]
  assign src_0_14 = _T_856 ? $signed(_GEN_79) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_14 = _T_856 ? $signed(_GEN_97) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1181 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 295:34]
  assign _T_1182 = _T_1181 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 295:24]
  assign mix_val_14 = _T_856 ? $signed(_T_1182) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1183 = mix_val_14[7:0]; // @[Compute.scala 297:37]
  assign _T_1184 = $unsigned(src_0_14); // @[Compute.scala 298:30]
  assign _T_1185 = $unsigned(src_1_14); // @[Compute.scala 298:59]
  assign _T_1186 = _T_1184 + _T_1185; // @[Compute.scala 298:49]
  assign _T_1187 = _T_1184 + _T_1185; // @[Compute.scala 298:49]
  assign _T_1188 = $signed(_T_1187); // @[Compute.scala 298:79]
  assign add_val_14 = _T_856 ? $signed(_T_1188) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_14 = _T_856 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1189 = add_res_14[7:0]; // @[Compute.scala 300:37]
  assign _T_1191 = src_1_14[4:0]; // @[Compute.scala 301:60]
  assign _T_1192 = _T_1184 >> _T_1191; // @[Compute.scala 301:49]
  assign _T_1193 = $signed(_T_1192); // @[Compute.scala 301:84]
  assign shr_val_14 = _T_856 ? $signed(_T_1193) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_14 = _T_856 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1194 = shr_res_14[7:0]; // @[Compute.scala 303:37]
  assign src_0_15 = _T_856 ? $signed(_GEN_81) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign src_1_15 = _T_856 ? $signed(_GEN_98) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1195 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 295:34]
  assign _T_1196 = _T_1195 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 295:24]
  assign mix_val_15 = _T_856 ? $signed(_T_1196) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1197 = mix_val_15[7:0]; // @[Compute.scala 297:37]
  assign _T_1198 = $unsigned(src_0_15); // @[Compute.scala 298:30]
  assign _T_1199 = $unsigned(src_1_15); // @[Compute.scala 298:59]
  assign _T_1200 = _T_1198 + _T_1199; // @[Compute.scala 298:49]
  assign _T_1201 = _T_1198 + _T_1199; // @[Compute.scala 298:49]
  assign _T_1202 = $signed(_T_1201); // @[Compute.scala 298:79]
  assign add_val_15 = _T_856 ? $signed(_T_1202) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign add_res_15 = _T_856 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1203 = add_res_15[7:0]; // @[Compute.scala 300:37]
  assign _T_1205 = src_1_15[4:0]; // @[Compute.scala 301:60]
  assign _T_1206 = _T_1198 >> _T_1205; // @[Compute.scala 301:49]
  assign _T_1207 = $signed(_T_1206); // @[Compute.scala 301:84]
  assign shr_val_15 = _T_856 ? $signed(_T_1207) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign shr_res_15 = _T_856 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 278:36]
  assign _T_1208 = shr_res_15[7:0]; // @[Compute.scala 303:37]
  assign short_cmp_res_0 = _T_856 ? _T_987 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_0 = _T_856 ? _T_993 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_0 = _T_856 ? _T_998 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_1 = _T_856 ? _T_1001 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_1 = _T_856 ? _T_1007 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_1 = _T_856 ? _T_1012 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_2 = _T_856 ? _T_1015 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_2 = _T_856 ? _T_1021 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_2 = _T_856 ? _T_1026 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_3 = _T_856 ? _T_1029 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_3 = _T_856 ? _T_1035 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_3 = _T_856 ? _T_1040 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_4 = _T_856 ? _T_1043 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_4 = _T_856 ? _T_1049 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_4 = _T_856 ? _T_1054 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_5 = _T_856 ? _T_1057 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_5 = _T_856 ? _T_1063 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_5 = _T_856 ? _T_1068 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_6 = _T_856 ? _T_1071 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_6 = _T_856 ? _T_1077 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_6 = _T_856 ? _T_1082 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_7 = _T_856 ? _T_1085 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_7 = _T_856 ? _T_1091 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_7 = _T_856 ? _T_1096 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_8 = _T_856 ? _T_1099 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_8 = _T_856 ? _T_1105 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_8 = _T_856 ? _T_1110 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_9 = _T_856 ? _T_1113 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_9 = _T_856 ? _T_1119 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_9 = _T_856 ? _T_1124 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_10 = _T_856 ? _T_1127 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_10 = _T_856 ? _T_1133 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_10 = _T_856 ? _T_1138 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_11 = _T_856 ? _T_1141 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_11 = _T_856 ? _T_1147 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_11 = _T_856 ? _T_1152 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_12 = _T_856 ? _T_1155 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_12 = _T_856 ? _T_1161 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_12 = _T_856 ? _T_1166 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_13 = _T_856 ? _T_1169 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_13 = _T_856 ? _T_1175 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_13 = _T_856 ? _T_1180 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_14 = _T_856 ? _T_1183 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_14 = _T_856 ? _T_1189 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_14 = _T_856 ? _T_1194 : 8'h0; // @[Compute.scala 278:36]
  assign short_cmp_res_15 = _T_856 ? _T_1197 : 8'h0; // @[Compute.scala 278:36]
  assign short_add_res_15 = _T_856 ? _T_1203 : 8'h0; // @[Compute.scala 278:36]
  assign short_shr_res_15 = _T_856 ? _T_1208 : 8'h0; // @[Compute.scala 278:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 308:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 309:39]
  assign _T_1211 = out_cntr_wrap == 1'h0; // @[Compute.scala 310:37]
  assign _T_1212 = opcode_alu_en & _T_1211; // @[Compute.scala 310:34]
  assign _T_1213 = _T_1212 & busy; // @[Compute.scala 310:52]
  assign _T_1215 = 4'h9 - 4'h1; // @[Compute.scala 311:58]
  assign _T_1216 = $unsigned(_T_1215); // @[Compute.scala 311:58]
  assign _T_1217 = _T_1216[3:0]; // @[Compute.scala 311:58]
  assign _GEN_310 = {{12'd0}, _T_1217}; // @[Compute.scala 311:40]
  assign _T_1218 = dst_offset_in == _GEN_310; // @[Compute.scala 311:40]
  assign _T_1219 = out_mem_write & _T_1218; // @[Compute.scala 311:23]
  assign _T_1222 = _T_1219 & _T_313; // @[Compute.scala 311:66]
  assign _GEN_275 = _T_1222 ? 1'h0 : _T_1213; // @[Compute.scala 311:85]
  assign _T_1225 = dst_idx - 16'h1; // @[Compute.scala 314:34]
  assign _T_1226 = $unsigned(_T_1225); // @[Compute.scala 314:34]
  assign _T_1227 = _T_1226[15:0]; // @[Compute.scala 314:34]
  assign _GEN_311 = {{7'd0}, _T_1227}; // @[Compute.scala 314:41]
  assign _T_1229 = _GEN_311 << 3'h4; // @[Compute.scala 314:41]
  assign _T_1236 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1244 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1236}; // @[Cat.scala 30:58]
  assign _T_1251 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1259 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1251}; // @[Cat.scala 30:58]
  assign _T_1266 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1274 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1266}; // @[Cat.scala 30:58]
  assign _T_1275 = alu_opcode_add_en ? _T_1259 : _T_1274; // @[Compute.scala 318:8]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 186:23]
  assign io_done_readdata = _T_337; // @[Compute.scala 187:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 195:19]
  assign io_uops_read = uops_read; // @[Compute.scala 194:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 32'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 208:21]
  assign io_biases_read = biases_read; // @[Compute.scala 209:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = gemm_queue_ready; // @[Compute.scala 182:23]
  assign io_l2g_dep_queue_ready = pop_prev_dep_ready & dump; // @[Compute.scala 135:26]
  assign io_s2g_dep_queue_ready = pop_next_dep_ready & dump; // @[Compute.scala 136:26]
  assign io_g2l_dep_queue_valid = push_prev_dep & push; // @[Compute.scala 147:26]
  assign io_g2l_dep_queue_data = 1'h1; // @[Compute.scala 145:25]
  assign io_g2s_dep_queue_valid = push_next_dep & push; // @[Compute.scala 148:26]
  assign io_g2s_dep_queue_data = 1'h1; // @[Compute.scala 146:25]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = _T_1229[16:0]; // @[Compute.scala 314:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_write; // @[Compute.scala 315:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1244 : _T_1275; // @[Compute.scala 317:24]
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE
  integer initvar;
  initial begin
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      #0.002 begin end
    `endif
  _RAND_0 = {16{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 256; initvar = initvar+1)
    acc_mem[initvar] = _RAND_0[511:0];
  `endif // RANDOMIZE_MEM_INIT
  _RAND_1 = {1{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 1024; initvar = initvar+1)
    uop_mem[initvar] = _RAND_1[31:0];
  `endif // RANDOMIZE_MEM_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {4{`RANDOM}};
  insn = _RAND_2[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  state = _RAND_3[2:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  uops_read = _RAND_4[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{`RANDOM}};
  uops_data = _RAND_5[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  biases_read = _RAND_6[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {4{`RANDOM}};
  biases_data_0 = _RAND_7[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {4{`RANDOM}};
  biases_data_1 = _RAND_8[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {4{`RANDOM}};
  biases_data_2 = _RAND_9[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {4{`RANDOM}};
  biases_data_3 = _RAND_10[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_11 = {1{`RANDOM}};
  out_mem_write = _RAND_11[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_12 = {1{`RANDOM}};
  uop_cntr_val = _RAND_12[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_13 = {1{`RANDOM}};
  acc_cntr_val = _RAND_13[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_14 = {1{`RANDOM}};
  dst_offset_in = _RAND_14[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_15 = {1{`RANDOM}};
  pop_prev_dep_ready = _RAND_15[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_16 = {1{`RANDOM}};
  pop_next_dep_ready = _RAND_16[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_17 = {1{`RANDOM}};
  push_prev_dep_ready = _RAND_17[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_18 = {1{`RANDOM}};
  push_next_dep_ready = _RAND_18[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_19 = {1{`RANDOM}};
  gemm_queue_ready = _RAND_19[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_20 = {1{`RANDOM}};
  _T_337 = _RAND_20[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_21 = {16{`RANDOM}};
  dst_vector = _RAND_21[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_22 = {16{`RANDOM}};
  src_vector = _RAND_22[511:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_400_en & acc_mem__T_400_mask) begin
      acc_mem[acc_mem__T_400_addr] <= acc_mem__T_400_data; // @[Compute.scala 33:20]
    end
    if(uop_mem__T_358_en & uop_mem__T_358_mask) begin
      uop_mem[uop_mem__T_358_addr] <= uop_mem__T_358_data; // @[Compute.scala 34:20]
    end
    if (gemm_queue_ready) begin
      insn <= io_gemm_queue_data;
    end
    if (reset) begin
      state <= 3'h0;
    end else begin
      if (gemm_queue_ready) begin
        state <= 3'h2;
      end else begin
        if (_T_277) begin
          state <= 3'h4;
        end else begin
          if (_T_275) begin
            state <= 3'h2;
          end else begin
            if (_T_273) begin
              state <= 3'h1;
            end else begin
              if (_T_264) begin
                if (_T_265) begin
                  state <= 3'h3;
                end else begin
                  state <= 3'h4;
                end
              end
            end
          end
        end
      end
    end
    if (_T_296) begin
      if (_T_355) begin
        uops_read <= 1'h0;
      end else begin
        uops_read <= _T_347;
      end
    end else begin
      uops_read <= _T_347;
    end
    if (_T_296) begin
      uops_data <= io_uops_readdata;
    end
    biases_read <= acc_cntr_en & _T_386;
    if (_T_305) begin
      if (3'h0 == _T_392) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_305) begin
      if (3'h1 == _T_392) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_305) begin
      if (3'h2 == _T_392) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_305) begin
      if (3'h3 == _T_392) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    if (reset) begin
      out_mem_write <= 1'h0;
    end else begin
      if (_T_1222) begin
        out_mem_write <= 1'h0;
      end else begin
        out_mem_write <= _T_1213;
      end
    end
    if (gemm_queue_ready) begin
      uop_cntr_val <= 16'h0;
    end else begin
      if (_T_299) begin
        uop_cntr_val <= _T_302;
      end
    end
    if (gemm_queue_ready) begin
      acc_cntr_val <= 16'h0;
    end else begin
      if (_T_308) begin
        acc_cntr_val <= _T_311;
      end
    end
    if (gemm_queue_ready) begin
      dst_offset_in <= 16'h0;
    end else begin
      if (_T_317) begin
        dst_offset_in <= _T_320;
      end
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_281) begin
          pop_prev_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      pop_next_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_next_dep_ready <= 1'h0;
      end else begin
        if (_T_284) begin
          pop_next_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      push_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        push_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_289) begin
          push_prev_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      push_next_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        push_next_dep_ready <= 1'h0;
      end else begin
        if (_T_292) begin
          push_next_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      gemm_queue_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        gemm_queue_ready <= 1'h0;
      end else begin
        gemm_queue_ready <= _T_329;
      end
    end
    _T_337 <= opcode_finish_en & io_done_read;
    if (_T_314) begin
      dst_vector <= acc_mem__T_422_data;
    end
    if (_T_314) begin
      src_vector <= acc_mem__T_424_data;
    end
  end
endmodule
