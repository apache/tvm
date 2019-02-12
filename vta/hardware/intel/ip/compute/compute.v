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
  wire [511:0] acc_mem__T_419_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_419_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_421_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_421_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_397_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_397_addr; // @[Compute.scala 33:20]
  wire  acc_mem__T_397_mask; // @[Compute.scala 33:20]
  wire  acc_mem__T_397_en; // @[Compute.scala 33:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem_uop_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem_uop_addr; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_355_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_355_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_355_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_355_en; // @[Compute.scala 34:20]
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
  reg [127:0] biases_data_0; // @[Compute.scala 86:24]
  reg [127:0] _RAND_7;
  reg [127:0] biases_data_1; // @[Compute.scala 86:24]
  reg [127:0] _RAND_8;
  reg [127:0] biases_data_2; // @[Compute.scala 86:24]
  reg [127:0] _RAND_9;
  reg [127:0] biases_data_3; // @[Compute.scala 86:24]
  reg [127:0] _RAND_10;
  reg  out_mem_write; // @[Compute.scala 88:31]
  reg [31:0] _RAND_11;
  wire  _T_234; // @[Compute.scala 92:37]
  wire  uop_cntr_en; // @[Compute.scala 92:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 94:25]
  reg [31:0] _RAND_12;
  wire  _T_236; // @[Compute.scala 95:38]
  wire  _T_237; // @[Compute.scala 95:56]
  wire  uop_cntr_wrap; // @[Compute.scala 95:71]
  wire [18:0] acc_cntr_max; // @[Compute.scala 97:29]
  wire  _T_239; // @[Compute.scala 98:37]
  wire  acc_cntr_en; // @[Compute.scala 98:59]
  reg [15:0] acc_cntr_val; // @[Compute.scala 100:25]
  reg [31:0] _RAND_13;
  wire [18:0] _GEN_295; // @[Compute.scala 101:38]
  wire  _T_241; // @[Compute.scala 101:38]
  wire  _T_242; // @[Compute.scala 101:56]
  wire  acc_cntr_wrap; // @[Compute.scala 101:71]
  wire  _T_246; // @[Compute.scala 104:37]
  wire  out_cntr_en; // @[Compute.scala 104:56]
  reg [15:0] dst_offset_in; // @[Compute.scala 106:25]
  reg [31:0] _RAND_14;
  wire  _T_248; // @[Compute.scala 107:38]
  wire  _T_249; // @[Compute.scala 107:56]
  wire  out_cntr_wrap; // @[Compute.scala 107:71]
  reg  pop_prev_dep_ready; // @[Compute.scala 110:35]
  reg [31:0] _RAND_15;
  reg  pop_next_dep_ready; // @[Compute.scala 111:35]
  reg [31:0] _RAND_16;
  wire  push_prev_dep_valid; // @[Compute.scala 112:43]
  wire  push_next_dep_valid; // @[Compute.scala 113:43]
  reg  push_prev_dep_ready; // @[Compute.scala 114:36]
  reg [31:0] _RAND_17;
  reg  push_next_dep_ready; // @[Compute.scala 115:36]
  reg [31:0] _RAND_18;
  reg  gemm_queue_ready; // @[Compute.scala 117:33]
  reg [31:0] _RAND_19;
  wire  _T_260; // @[Compute.scala 120:23]
  wire  _T_261; // @[Compute.scala 120:40]
  wire  _T_262; // @[Compute.scala 121:25]
  wire [2:0] _GEN_0; // @[Compute.scala 121:43]
  wire [2:0] _GEN_1; // @[Compute.scala 120:58]
  wire  _T_264; // @[Compute.scala 129:18]
  wire  _T_266; // @[Compute.scala 129:41]
  wire  _T_267; // @[Compute.scala 129:38]
  wire  _T_268; // @[Compute.scala 129:14]
  wire  _T_269; // @[Compute.scala 129:79]
  wire  _T_270; // @[Compute.scala 129:62]
  wire [2:0] _GEN_2; // @[Compute.scala 129:97]
  wire  _T_271; // @[Compute.scala 130:38]
  wire  _T_272; // @[Compute.scala 130:14]
  wire [2:0] _GEN_3; // @[Compute.scala 130:63]
  wire  _T_273; // @[Compute.scala 131:38]
  wire  _T_274; // @[Compute.scala 131:14]
  wire [2:0] _GEN_4; // @[Compute.scala 131:63]
  wire  _T_277; // @[Compute.scala 138:22]
  wire  _T_278; // @[Compute.scala 138:30]
  wire  _GEN_5; // @[Compute.scala 138:57]
  wire  _T_280; // @[Compute.scala 141:22]
  wire  _T_281; // @[Compute.scala 141:30]
  wire  _GEN_6; // @[Compute.scala 141:57]
  wire  _T_285; // @[Compute.scala 148:29]
  wire  _T_286; // @[Compute.scala 148:55]
  wire  _GEN_7; // @[Compute.scala 148:64]
  wire  _T_288; // @[Compute.scala 151:29]
  wire  _T_289; // @[Compute.scala 151:55]
  wire  _GEN_8; // @[Compute.scala 151:64]
  wire  _T_292; // @[Compute.scala 156:22]
  wire  _T_293; // @[Compute.scala 156:19]
  wire  _T_294; // @[Compute.scala 156:37]
  wire  _T_295; // @[Compute.scala 156:61]
  wire  _T_296; // @[Compute.scala 156:45]
  wire [16:0] _T_298; // @[Compute.scala 157:34]
  wire [15:0] _T_299; // @[Compute.scala 157:34]
  wire [15:0] _GEN_9; // @[Compute.scala 156:77]
  wire  _T_301; // @[Compute.scala 159:24]
  wire  _T_302; // @[Compute.scala 159:21]
  wire  _T_303; // @[Compute.scala 159:39]
  wire  _T_304; // @[Compute.scala 159:63]
  wire  _T_305; // @[Compute.scala 159:47]
  wire [16:0] _T_307; // @[Compute.scala 160:34]
  wire [15:0] _T_308; // @[Compute.scala 160:34]
  wire [15:0] _GEN_10; // @[Compute.scala 159:79]
  wire  _T_310; // @[Compute.scala 162:26]
  wire  _T_311; // @[Compute.scala 162:23]
  wire  _T_312; // @[Compute.scala 162:41]
  wire  _T_313; // @[Compute.scala 162:65]
  wire  _T_314; // @[Compute.scala 162:49]
  wire [16:0] _T_316; // @[Compute.scala 163:34]
  wire [15:0] _T_317; // @[Compute.scala 163:34]
  wire [15:0] _GEN_11; // @[Compute.scala 162:81]
  wire  _GEN_16; // @[Compute.scala 167:27]
  wire  _GEN_17; // @[Compute.scala 167:27]
  wire  _GEN_18; // @[Compute.scala 167:27]
  wire  _GEN_19; // @[Compute.scala 167:27]
  wire [2:0] _GEN_20; // @[Compute.scala 167:27]
  wire  _T_325; // @[Compute.scala 180:52]
  wire  _T_326; // @[Compute.scala 180:43]
  wire  _GEN_21; // @[Compute.scala 182:27]
  reg  _T_334; // @[Compute.scala 186:30]
  reg [31:0] _RAND_20;
  wire [31:0] _GEN_297; // @[Compute.scala 190:33]
  wire [32:0] _T_337; // @[Compute.scala 190:33]
  wire [31:0] _T_338; // @[Compute.scala 190:33]
  wire [34:0] _GEN_298; // @[Compute.scala 190:49]
  wire [34:0] uop_dram_addr; // @[Compute.scala 190:49]
  wire [16:0] _T_340; // @[Compute.scala 191:33]
  wire [15:0] uop_sram_addr; // @[Compute.scala 191:33]
  wire  _T_342; // @[Compute.scala 192:31]
  wire  _T_343; // @[Compute.scala 192:28]
  wire  _T_344; // @[Compute.scala 192:46]
  wire [16:0] _T_349; // @[Compute.scala 199:42]
  wire [16:0] _T_350; // @[Compute.scala 199:42]
  wire [15:0] _T_351; // @[Compute.scala 199:42]
  wire  _T_352; // @[Compute.scala 199:24]
  wire  _GEN_22; // @[Compute.scala 199:50]
  wire [31:0] _GEN_299; // @[Compute.scala 204:36]
  wire [32:0] _T_356; // @[Compute.scala 204:36]
  wire [31:0] _T_357; // @[Compute.scala 204:36]
  wire [31:0] _GEN_300; // @[Compute.scala 204:47]
  wire [32:0] _T_358; // @[Compute.scala 204:47]
  wire [31:0] _T_359; // @[Compute.scala 204:47]
  wire [34:0] _GEN_301; // @[Compute.scala 204:58]
  wire [34:0] _T_361; // @[Compute.scala 204:58]
  wire [35:0] _T_363; // @[Compute.scala 204:66]
  wire [35:0] _GEN_302; // @[Compute.scala 204:76]
  wire [36:0] _T_364; // @[Compute.scala 204:76]
  wire [35:0] _T_365; // @[Compute.scala 204:76]
  wire [42:0] _GEN_303; // @[Compute.scala 204:92]
  wire [42:0] acc_dram_addr; // @[Compute.scala 204:92]
  wire [19:0] _GEN_304; // @[Compute.scala 205:36]
  wire [20:0] _T_367; // @[Compute.scala 205:36]
  wire [19:0] _T_368; // @[Compute.scala 205:36]
  wire [19:0] _GEN_305; // @[Compute.scala 205:47]
  wire [20:0] _T_369; // @[Compute.scala 205:47]
  wire [19:0] _T_370; // @[Compute.scala 205:47]
  wire [22:0] _GEN_306; // @[Compute.scala 205:58]
  wire [22:0] _T_372; // @[Compute.scala 205:58]
  wire [23:0] _T_374; // @[Compute.scala 205:66]
  wire [23:0] _GEN_307; // @[Compute.scala 205:76]
  wire [24:0] _T_375; // @[Compute.scala 205:76]
  wire [23:0] _T_376; // @[Compute.scala 205:76]
  wire [23:0] _T_378; // @[Compute.scala 205:92]
  wire [24:0] _T_380; // @[Compute.scala 205:100]
  wire [24:0] _T_381; // @[Compute.scala 205:100]
  wire [23:0] acc_sram_addr; // @[Compute.scala 205:100]
  wire  _T_383; // @[Compute.scala 206:33]
  wire [15:0] _GEN_12; // @[Compute.scala 212:30]
  wire [2:0] _T_389; // @[Compute.scala 212:30]
  wire [127:0] _GEN_25; // @[Compute.scala 212:55]
  wire [127:0] _GEN_26; // @[Compute.scala 212:55]
  wire [127:0] _GEN_27; // @[Compute.scala 212:55]
  wire [127:0] _GEN_28; // @[Compute.scala 212:55]
  wire  _T_395; // @[Compute.scala 213:50]
  wire [255:0] _T_398; // @[Cat.scala 30:58]
  wire [255:0] _T_399; // @[Cat.scala 30:58]
  wire [1:0] alu_opcode; // @[Compute.scala 223:24]
  wire  use_imm; // @[Compute.scala 224:21]
  wire [15:0] imm_raw; // @[Compute.scala 225:21]
  wire [15:0] _T_401; // @[Compute.scala 226:25]
  wire  _T_403; // @[Compute.scala 226:32]
  wire [31:0] _T_405; // @[Cat.scala 30:58]
  wire [16:0] _T_407; // @[Cat.scala 30:58]
  wire [31:0] _T_408; // @[Compute.scala 226:16]
  wire [31:0] imm; // @[Compute.scala 226:89]
  wire [10:0] _T_409; // @[Compute.scala 234:20]
  wire [15:0] _GEN_308; // @[Compute.scala 234:47]
  wire [16:0] _T_410; // @[Compute.scala 234:47]
  wire [15:0] dst_idx; // @[Compute.scala 234:47]
  wire [10:0] _T_411; // @[Compute.scala 235:20]
  wire [15:0] _GEN_309; // @[Compute.scala 235:47]
  wire [16:0] _T_412; // @[Compute.scala 235:47]
  wire [15:0] src_idx; // @[Compute.scala 235:47]
  reg [511:0] dst_vector; // @[Compute.scala 238:23]
  reg [511:0] _RAND_21;
  reg [511:0] src_vector; // @[Compute.scala 239:23]
  reg [511:0] _RAND_22;
  wire  alu_opcode_min_en; // @[Compute.scala 260:38]
  wire  alu_opcode_max_en; // @[Compute.scala 261:38]
  wire  _T_853; // @[Compute.scala 280:20]
  wire [31:0] _T_854; // @[Compute.scala 283:31]
  wire [31:0] _T_855; // @[Compute.scala 283:72]
  wire [31:0] _T_856; // @[Compute.scala 284:31]
  wire [31:0] _T_857; // @[Compute.scala 284:72]
  wire [31:0] _T_858; // @[Compute.scala 283:31]
  wire [31:0] _T_859; // @[Compute.scala 283:72]
  wire [31:0] _T_860; // @[Compute.scala 284:31]
  wire [31:0] _T_861; // @[Compute.scala 284:72]
  wire [31:0] _T_862; // @[Compute.scala 283:31]
  wire [31:0] _T_863; // @[Compute.scala 283:72]
  wire [31:0] _T_864; // @[Compute.scala 284:31]
  wire [31:0] _T_865; // @[Compute.scala 284:72]
  wire [31:0] _T_866; // @[Compute.scala 283:31]
  wire [31:0] _T_867; // @[Compute.scala 283:72]
  wire [31:0] _T_868; // @[Compute.scala 284:31]
  wire [31:0] _T_869; // @[Compute.scala 284:72]
  wire [31:0] _T_870; // @[Compute.scala 283:31]
  wire [31:0] _T_871; // @[Compute.scala 283:72]
  wire [31:0] _T_872; // @[Compute.scala 284:31]
  wire [31:0] _T_873; // @[Compute.scala 284:72]
  wire [31:0] _T_874; // @[Compute.scala 283:31]
  wire [31:0] _T_875; // @[Compute.scala 283:72]
  wire [31:0] _T_876; // @[Compute.scala 284:31]
  wire [31:0] _T_877; // @[Compute.scala 284:72]
  wire [31:0] _T_878; // @[Compute.scala 283:31]
  wire [31:0] _T_879; // @[Compute.scala 283:72]
  wire [31:0] _T_880; // @[Compute.scala 284:31]
  wire [31:0] _T_881; // @[Compute.scala 284:72]
  wire [31:0] _T_882; // @[Compute.scala 283:31]
  wire [31:0] _T_883; // @[Compute.scala 283:72]
  wire [31:0] _T_884; // @[Compute.scala 284:31]
  wire [31:0] _T_885; // @[Compute.scala 284:72]
  wire [31:0] _T_886; // @[Compute.scala 283:31]
  wire [31:0] _T_887; // @[Compute.scala 283:72]
  wire [31:0] _T_888; // @[Compute.scala 284:31]
  wire [31:0] _T_889; // @[Compute.scala 284:72]
  wire [31:0] _T_890; // @[Compute.scala 283:31]
  wire [31:0] _T_891; // @[Compute.scala 283:72]
  wire [31:0] _T_892; // @[Compute.scala 284:31]
  wire [31:0] _T_893; // @[Compute.scala 284:72]
  wire [31:0] _T_894; // @[Compute.scala 283:31]
  wire [31:0] _T_895; // @[Compute.scala 283:72]
  wire [31:0] _T_896; // @[Compute.scala 284:31]
  wire [31:0] _T_897; // @[Compute.scala 284:72]
  wire [31:0] _T_898; // @[Compute.scala 283:31]
  wire [31:0] _T_899; // @[Compute.scala 283:72]
  wire [31:0] _T_900; // @[Compute.scala 284:31]
  wire [31:0] _T_901; // @[Compute.scala 284:72]
  wire [31:0] _T_902; // @[Compute.scala 283:31]
  wire [31:0] _T_903; // @[Compute.scala 283:72]
  wire [31:0] _T_904; // @[Compute.scala 284:31]
  wire [31:0] _T_905; // @[Compute.scala 284:72]
  wire [31:0] _T_906; // @[Compute.scala 283:31]
  wire [31:0] _T_907; // @[Compute.scala 283:72]
  wire [31:0] _T_908; // @[Compute.scala 284:31]
  wire [31:0] _T_909; // @[Compute.scala 284:72]
  wire [31:0] _T_910; // @[Compute.scala 283:31]
  wire [31:0] _T_911; // @[Compute.scala 283:72]
  wire [31:0] _T_912; // @[Compute.scala 284:31]
  wire [31:0] _T_913; // @[Compute.scala 284:72]
  wire [31:0] _T_914; // @[Compute.scala 283:31]
  wire [31:0] _T_915; // @[Compute.scala 283:72]
  wire [31:0] _T_916; // @[Compute.scala 284:31]
  wire [31:0] _T_917; // @[Compute.scala 284:72]
  wire [31:0] _GEN_51; // @[Compute.scala 281:30]
  wire [31:0] _GEN_52; // @[Compute.scala 281:30]
  wire [31:0] _GEN_53; // @[Compute.scala 281:30]
  wire [31:0] _GEN_54; // @[Compute.scala 281:30]
  wire [31:0] _GEN_55; // @[Compute.scala 281:30]
  wire [31:0] _GEN_56; // @[Compute.scala 281:30]
  wire [31:0] _GEN_57; // @[Compute.scala 281:30]
  wire [31:0] _GEN_58; // @[Compute.scala 281:30]
  wire [31:0] _GEN_59; // @[Compute.scala 281:30]
  wire [31:0] _GEN_60; // @[Compute.scala 281:30]
  wire [31:0] _GEN_61; // @[Compute.scala 281:30]
  wire [31:0] _GEN_62; // @[Compute.scala 281:30]
  wire [31:0] _GEN_63; // @[Compute.scala 281:30]
  wire [31:0] _GEN_64; // @[Compute.scala 281:30]
  wire [31:0] _GEN_65; // @[Compute.scala 281:30]
  wire [31:0] _GEN_66; // @[Compute.scala 281:30]
  wire [31:0] _GEN_67; // @[Compute.scala 281:30]
  wire [31:0] _GEN_68; // @[Compute.scala 281:30]
  wire [31:0] _GEN_69; // @[Compute.scala 281:30]
  wire [31:0] _GEN_70; // @[Compute.scala 281:30]
  wire [31:0] _GEN_71; // @[Compute.scala 281:30]
  wire [31:0] _GEN_72; // @[Compute.scala 281:30]
  wire [31:0] _GEN_73; // @[Compute.scala 281:30]
  wire [31:0] _GEN_74; // @[Compute.scala 281:30]
  wire [31:0] _GEN_75; // @[Compute.scala 281:30]
  wire [31:0] _GEN_76; // @[Compute.scala 281:30]
  wire [31:0] _GEN_77; // @[Compute.scala 281:30]
  wire [31:0] _GEN_78; // @[Compute.scala 281:30]
  wire [31:0] _GEN_79; // @[Compute.scala 281:30]
  wire [31:0] _GEN_80; // @[Compute.scala 281:30]
  wire [31:0] _GEN_81; // @[Compute.scala 281:30]
  wire [31:0] _GEN_82; // @[Compute.scala 281:30]
  wire [31:0] _GEN_83; // @[Compute.scala 292:20]
  wire [31:0] _GEN_84; // @[Compute.scala 292:20]
  wire [31:0] _GEN_85; // @[Compute.scala 292:20]
  wire [31:0] _GEN_86; // @[Compute.scala 292:20]
  wire [31:0] _GEN_87; // @[Compute.scala 292:20]
  wire [31:0] _GEN_88; // @[Compute.scala 292:20]
  wire [31:0] _GEN_89; // @[Compute.scala 292:20]
  wire [31:0] _GEN_90; // @[Compute.scala 292:20]
  wire [31:0] _GEN_91; // @[Compute.scala 292:20]
  wire [31:0] _GEN_92; // @[Compute.scala 292:20]
  wire [31:0] _GEN_93; // @[Compute.scala 292:20]
  wire [31:0] _GEN_94; // @[Compute.scala 292:20]
  wire [31:0] _GEN_95; // @[Compute.scala 292:20]
  wire [31:0] _GEN_96; // @[Compute.scala 292:20]
  wire [31:0] _GEN_97; // @[Compute.scala 292:20]
  wire [31:0] _GEN_98; // @[Compute.scala 292:20]
  wire [31:0] src_0_0; // @[Compute.scala 280:36]
  wire [31:0] src_1_0; // @[Compute.scala 280:36]
  wire  _T_982; // @[Compute.scala 297:34]
  wire [31:0] _T_983; // @[Compute.scala 297:24]
  wire [31:0] mix_val_0; // @[Compute.scala 280:36]
  wire [7:0] _T_984; // @[Compute.scala 299:37]
  wire [31:0] _T_985; // @[Compute.scala 300:30]
  wire [31:0] _T_986; // @[Compute.scala 300:59]
  wire [32:0] _T_987; // @[Compute.scala 300:49]
  wire [31:0] _T_988; // @[Compute.scala 300:49]
  wire [31:0] _T_989; // @[Compute.scala 300:79]
  wire [31:0] add_val_0; // @[Compute.scala 280:36]
  wire [31:0] add_res_0; // @[Compute.scala 280:36]
  wire [7:0] _T_990; // @[Compute.scala 302:37]
  wire [4:0] _T_992; // @[Compute.scala 303:60]
  wire [31:0] _T_993; // @[Compute.scala 303:49]
  wire [31:0] _T_994; // @[Compute.scala 303:84]
  wire [31:0] shr_val_0; // @[Compute.scala 280:36]
  wire [31:0] shr_res_0; // @[Compute.scala 280:36]
  wire [7:0] _T_995; // @[Compute.scala 305:37]
  wire [31:0] src_0_1; // @[Compute.scala 280:36]
  wire [31:0] src_1_1; // @[Compute.scala 280:36]
  wire  _T_996; // @[Compute.scala 297:34]
  wire [31:0] _T_997; // @[Compute.scala 297:24]
  wire [31:0] mix_val_1; // @[Compute.scala 280:36]
  wire [7:0] _T_998; // @[Compute.scala 299:37]
  wire [31:0] _T_999; // @[Compute.scala 300:30]
  wire [31:0] _T_1000; // @[Compute.scala 300:59]
  wire [32:0] _T_1001; // @[Compute.scala 300:49]
  wire [31:0] _T_1002; // @[Compute.scala 300:49]
  wire [31:0] _T_1003; // @[Compute.scala 300:79]
  wire [31:0] add_val_1; // @[Compute.scala 280:36]
  wire [31:0] add_res_1; // @[Compute.scala 280:36]
  wire [7:0] _T_1004; // @[Compute.scala 302:37]
  wire [4:0] _T_1006; // @[Compute.scala 303:60]
  wire [31:0] _T_1007; // @[Compute.scala 303:49]
  wire [31:0] _T_1008; // @[Compute.scala 303:84]
  wire [31:0] shr_val_1; // @[Compute.scala 280:36]
  wire [31:0] shr_res_1; // @[Compute.scala 280:36]
  wire [7:0] _T_1009; // @[Compute.scala 305:37]
  wire [31:0] src_0_2; // @[Compute.scala 280:36]
  wire [31:0] src_1_2; // @[Compute.scala 280:36]
  wire  _T_1010; // @[Compute.scala 297:34]
  wire [31:0] _T_1011; // @[Compute.scala 297:24]
  wire [31:0] mix_val_2; // @[Compute.scala 280:36]
  wire [7:0] _T_1012; // @[Compute.scala 299:37]
  wire [31:0] _T_1013; // @[Compute.scala 300:30]
  wire [31:0] _T_1014; // @[Compute.scala 300:59]
  wire [32:0] _T_1015; // @[Compute.scala 300:49]
  wire [31:0] _T_1016; // @[Compute.scala 300:49]
  wire [31:0] _T_1017; // @[Compute.scala 300:79]
  wire [31:0] add_val_2; // @[Compute.scala 280:36]
  wire [31:0] add_res_2; // @[Compute.scala 280:36]
  wire [7:0] _T_1018; // @[Compute.scala 302:37]
  wire [4:0] _T_1020; // @[Compute.scala 303:60]
  wire [31:0] _T_1021; // @[Compute.scala 303:49]
  wire [31:0] _T_1022; // @[Compute.scala 303:84]
  wire [31:0] shr_val_2; // @[Compute.scala 280:36]
  wire [31:0] shr_res_2; // @[Compute.scala 280:36]
  wire [7:0] _T_1023; // @[Compute.scala 305:37]
  wire [31:0] src_0_3; // @[Compute.scala 280:36]
  wire [31:0] src_1_3; // @[Compute.scala 280:36]
  wire  _T_1024; // @[Compute.scala 297:34]
  wire [31:0] _T_1025; // @[Compute.scala 297:24]
  wire [31:0] mix_val_3; // @[Compute.scala 280:36]
  wire [7:0] _T_1026; // @[Compute.scala 299:37]
  wire [31:0] _T_1027; // @[Compute.scala 300:30]
  wire [31:0] _T_1028; // @[Compute.scala 300:59]
  wire [32:0] _T_1029; // @[Compute.scala 300:49]
  wire [31:0] _T_1030; // @[Compute.scala 300:49]
  wire [31:0] _T_1031; // @[Compute.scala 300:79]
  wire [31:0] add_val_3; // @[Compute.scala 280:36]
  wire [31:0] add_res_3; // @[Compute.scala 280:36]
  wire [7:0] _T_1032; // @[Compute.scala 302:37]
  wire [4:0] _T_1034; // @[Compute.scala 303:60]
  wire [31:0] _T_1035; // @[Compute.scala 303:49]
  wire [31:0] _T_1036; // @[Compute.scala 303:84]
  wire [31:0] shr_val_3; // @[Compute.scala 280:36]
  wire [31:0] shr_res_3; // @[Compute.scala 280:36]
  wire [7:0] _T_1037; // @[Compute.scala 305:37]
  wire [31:0] src_0_4; // @[Compute.scala 280:36]
  wire [31:0] src_1_4; // @[Compute.scala 280:36]
  wire  _T_1038; // @[Compute.scala 297:34]
  wire [31:0] _T_1039; // @[Compute.scala 297:24]
  wire [31:0] mix_val_4; // @[Compute.scala 280:36]
  wire [7:0] _T_1040; // @[Compute.scala 299:37]
  wire [31:0] _T_1041; // @[Compute.scala 300:30]
  wire [31:0] _T_1042; // @[Compute.scala 300:59]
  wire [32:0] _T_1043; // @[Compute.scala 300:49]
  wire [31:0] _T_1044; // @[Compute.scala 300:49]
  wire [31:0] _T_1045; // @[Compute.scala 300:79]
  wire [31:0] add_val_4; // @[Compute.scala 280:36]
  wire [31:0] add_res_4; // @[Compute.scala 280:36]
  wire [7:0] _T_1046; // @[Compute.scala 302:37]
  wire [4:0] _T_1048; // @[Compute.scala 303:60]
  wire [31:0] _T_1049; // @[Compute.scala 303:49]
  wire [31:0] _T_1050; // @[Compute.scala 303:84]
  wire [31:0] shr_val_4; // @[Compute.scala 280:36]
  wire [31:0] shr_res_4; // @[Compute.scala 280:36]
  wire [7:0] _T_1051; // @[Compute.scala 305:37]
  wire [31:0] src_0_5; // @[Compute.scala 280:36]
  wire [31:0] src_1_5; // @[Compute.scala 280:36]
  wire  _T_1052; // @[Compute.scala 297:34]
  wire [31:0] _T_1053; // @[Compute.scala 297:24]
  wire [31:0] mix_val_5; // @[Compute.scala 280:36]
  wire [7:0] _T_1054; // @[Compute.scala 299:37]
  wire [31:0] _T_1055; // @[Compute.scala 300:30]
  wire [31:0] _T_1056; // @[Compute.scala 300:59]
  wire [32:0] _T_1057; // @[Compute.scala 300:49]
  wire [31:0] _T_1058; // @[Compute.scala 300:49]
  wire [31:0] _T_1059; // @[Compute.scala 300:79]
  wire [31:0] add_val_5; // @[Compute.scala 280:36]
  wire [31:0] add_res_5; // @[Compute.scala 280:36]
  wire [7:0] _T_1060; // @[Compute.scala 302:37]
  wire [4:0] _T_1062; // @[Compute.scala 303:60]
  wire [31:0] _T_1063; // @[Compute.scala 303:49]
  wire [31:0] _T_1064; // @[Compute.scala 303:84]
  wire [31:0] shr_val_5; // @[Compute.scala 280:36]
  wire [31:0] shr_res_5; // @[Compute.scala 280:36]
  wire [7:0] _T_1065; // @[Compute.scala 305:37]
  wire [31:0] src_0_6; // @[Compute.scala 280:36]
  wire [31:0] src_1_6; // @[Compute.scala 280:36]
  wire  _T_1066; // @[Compute.scala 297:34]
  wire [31:0] _T_1067; // @[Compute.scala 297:24]
  wire [31:0] mix_val_6; // @[Compute.scala 280:36]
  wire [7:0] _T_1068; // @[Compute.scala 299:37]
  wire [31:0] _T_1069; // @[Compute.scala 300:30]
  wire [31:0] _T_1070; // @[Compute.scala 300:59]
  wire [32:0] _T_1071; // @[Compute.scala 300:49]
  wire [31:0] _T_1072; // @[Compute.scala 300:49]
  wire [31:0] _T_1073; // @[Compute.scala 300:79]
  wire [31:0] add_val_6; // @[Compute.scala 280:36]
  wire [31:0] add_res_6; // @[Compute.scala 280:36]
  wire [7:0] _T_1074; // @[Compute.scala 302:37]
  wire [4:0] _T_1076; // @[Compute.scala 303:60]
  wire [31:0] _T_1077; // @[Compute.scala 303:49]
  wire [31:0] _T_1078; // @[Compute.scala 303:84]
  wire [31:0] shr_val_6; // @[Compute.scala 280:36]
  wire [31:0] shr_res_6; // @[Compute.scala 280:36]
  wire [7:0] _T_1079; // @[Compute.scala 305:37]
  wire [31:0] src_0_7; // @[Compute.scala 280:36]
  wire [31:0] src_1_7; // @[Compute.scala 280:36]
  wire  _T_1080; // @[Compute.scala 297:34]
  wire [31:0] _T_1081; // @[Compute.scala 297:24]
  wire [31:0] mix_val_7; // @[Compute.scala 280:36]
  wire [7:0] _T_1082; // @[Compute.scala 299:37]
  wire [31:0] _T_1083; // @[Compute.scala 300:30]
  wire [31:0] _T_1084; // @[Compute.scala 300:59]
  wire [32:0] _T_1085; // @[Compute.scala 300:49]
  wire [31:0] _T_1086; // @[Compute.scala 300:49]
  wire [31:0] _T_1087; // @[Compute.scala 300:79]
  wire [31:0] add_val_7; // @[Compute.scala 280:36]
  wire [31:0] add_res_7; // @[Compute.scala 280:36]
  wire [7:0] _T_1088; // @[Compute.scala 302:37]
  wire [4:0] _T_1090; // @[Compute.scala 303:60]
  wire [31:0] _T_1091; // @[Compute.scala 303:49]
  wire [31:0] _T_1092; // @[Compute.scala 303:84]
  wire [31:0] shr_val_7; // @[Compute.scala 280:36]
  wire [31:0] shr_res_7; // @[Compute.scala 280:36]
  wire [7:0] _T_1093; // @[Compute.scala 305:37]
  wire [31:0] src_0_8; // @[Compute.scala 280:36]
  wire [31:0] src_1_8; // @[Compute.scala 280:36]
  wire  _T_1094; // @[Compute.scala 297:34]
  wire [31:0] _T_1095; // @[Compute.scala 297:24]
  wire [31:0] mix_val_8; // @[Compute.scala 280:36]
  wire [7:0] _T_1096; // @[Compute.scala 299:37]
  wire [31:0] _T_1097; // @[Compute.scala 300:30]
  wire [31:0] _T_1098; // @[Compute.scala 300:59]
  wire [32:0] _T_1099; // @[Compute.scala 300:49]
  wire [31:0] _T_1100; // @[Compute.scala 300:49]
  wire [31:0] _T_1101; // @[Compute.scala 300:79]
  wire [31:0] add_val_8; // @[Compute.scala 280:36]
  wire [31:0] add_res_8; // @[Compute.scala 280:36]
  wire [7:0] _T_1102; // @[Compute.scala 302:37]
  wire [4:0] _T_1104; // @[Compute.scala 303:60]
  wire [31:0] _T_1105; // @[Compute.scala 303:49]
  wire [31:0] _T_1106; // @[Compute.scala 303:84]
  wire [31:0] shr_val_8; // @[Compute.scala 280:36]
  wire [31:0] shr_res_8; // @[Compute.scala 280:36]
  wire [7:0] _T_1107; // @[Compute.scala 305:37]
  wire [31:0] src_0_9; // @[Compute.scala 280:36]
  wire [31:0] src_1_9; // @[Compute.scala 280:36]
  wire  _T_1108; // @[Compute.scala 297:34]
  wire [31:0] _T_1109; // @[Compute.scala 297:24]
  wire [31:0] mix_val_9; // @[Compute.scala 280:36]
  wire [7:0] _T_1110; // @[Compute.scala 299:37]
  wire [31:0] _T_1111; // @[Compute.scala 300:30]
  wire [31:0] _T_1112; // @[Compute.scala 300:59]
  wire [32:0] _T_1113; // @[Compute.scala 300:49]
  wire [31:0] _T_1114; // @[Compute.scala 300:49]
  wire [31:0] _T_1115; // @[Compute.scala 300:79]
  wire [31:0] add_val_9; // @[Compute.scala 280:36]
  wire [31:0] add_res_9; // @[Compute.scala 280:36]
  wire [7:0] _T_1116; // @[Compute.scala 302:37]
  wire [4:0] _T_1118; // @[Compute.scala 303:60]
  wire [31:0] _T_1119; // @[Compute.scala 303:49]
  wire [31:0] _T_1120; // @[Compute.scala 303:84]
  wire [31:0] shr_val_9; // @[Compute.scala 280:36]
  wire [31:0] shr_res_9; // @[Compute.scala 280:36]
  wire [7:0] _T_1121; // @[Compute.scala 305:37]
  wire [31:0] src_0_10; // @[Compute.scala 280:36]
  wire [31:0] src_1_10; // @[Compute.scala 280:36]
  wire  _T_1122; // @[Compute.scala 297:34]
  wire [31:0] _T_1123; // @[Compute.scala 297:24]
  wire [31:0] mix_val_10; // @[Compute.scala 280:36]
  wire [7:0] _T_1124; // @[Compute.scala 299:37]
  wire [31:0] _T_1125; // @[Compute.scala 300:30]
  wire [31:0] _T_1126; // @[Compute.scala 300:59]
  wire [32:0] _T_1127; // @[Compute.scala 300:49]
  wire [31:0] _T_1128; // @[Compute.scala 300:49]
  wire [31:0] _T_1129; // @[Compute.scala 300:79]
  wire [31:0] add_val_10; // @[Compute.scala 280:36]
  wire [31:0] add_res_10; // @[Compute.scala 280:36]
  wire [7:0] _T_1130; // @[Compute.scala 302:37]
  wire [4:0] _T_1132; // @[Compute.scala 303:60]
  wire [31:0] _T_1133; // @[Compute.scala 303:49]
  wire [31:0] _T_1134; // @[Compute.scala 303:84]
  wire [31:0] shr_val_10; // @[Compute.scala 280:36]
  wire [31:0] shr_res_10; // @[Compute.scala 280:36]
  wire [7:0] _T_1135; // @[Compute.scala 305:37]
  wire [31:0] src_0_11; // @[Compute.scala 280:36]
  wire [31:0] src_1_11; // @[Compute.scala 280:36]
  wire  _T_1136; // @[Compute.scala 297:34]
  wire [31:0] _T_1137; // @[Compute.scala 297:24]
  wire [31:0] mix_val_11; // @[Compute.scala 280:36]
  wire [7:0] _T_1138; // @[Compute.scala 299:37]
  wire [31:0] _T_1139; // @[Compute.scala 300:30]
  wire [31:0] _T_1140; // @[Compute.scala 300:59]
  wire [32:0] _T_1141; // @[Compute.scala 300:49]
  wire [31:0] _T_1142; // @[Compute.scala 300:49]
  wire [31:0] _T_1143; // @[Compute.scala 300:79]
  wire [31:0] add_val_11; // @[Compute.scala 280:36]
  wire [31:0] add_res_11; // @[Compute.scala 280:36]
  wire [7:0] _T_1144; // @[Compute.scala 302:37]
  wire [4:0] _T_1146; // @[Compute.scala 303:60]
  wire [31:0] _T_1147; // @[Compute.scala 303:49]
  wire [31:0] _T_1148; // @[Compute.scala 303:84]
  wire [31:0] shr_val_11; // @[Compute.scala 280:36]
  wire [31:0] shr_res_11; // @[Compute.scala 280:36]
  wire [7:0] _T_1149; // @[Compute.scala 305:37]
  wire [31:0] src_0_12; // @[Compute.scala 280:36]
  wire [31:0] src_1_12; // @[Compute.scala 280:36]
  wire  _T_1150; // @[Compute.scala 297:34]
  wire [31:0] _T_1151; // @[Compute.scala 297:24]
  wire [31:0] mix_val_12; // @[Compute.scala 280:36]
  wire [7:0] _T_1152; // @[Compute.scala 299:37]
  wire [31:0] _T_1153; // @[Compute.scala 300:30]
  wire [31:0] _T_1154; // @[Compute.scala 300:59]
  wire [32:0] _T_1155; // @[Compute.scala 300:49]
  wire [31:0] _T_1156; // @[Compute.scala 300:49]
  wire [31:0] _T_1157; // @[Compute.scala 300:79]
  wire [31:0] add_val_12; // @[Compute.scala 280:36]
  wire [31:0] add_res_12; // @[Compute.scala 280:36]
  wire [7:0] _T_1158; // @[Compute.scala 302:37]
  wire [4:0] _T_1160; // @[Compute.scala 303:60]
  wire [31:0] _T_1161; // @[Compute.scala 303:49]
  wire [31:0] _T_1162; // @[Compute.scala 303:84]
  wire [31:0] shr_val_12; // @[Compute.scala 280:36]
  wire [31:0] shr_res_12; // @[Compute.scala 280:36]
  wire [7:0] _T_1163; // @[Compute.scala 305:37]
  wire [31:0] src_0_13; // @[Compute.scala 280:36]
  wire [31:0] src_1_13; // @[Compute.scala 280:36]
  wire  _T_1164; // @[Compute.scala 297:34]
  wire [31:0] _T_1165; // @[Compute.scala 297:24]
  wire [31:0] mix_val_13; // @[Compute.scala 280:36]
  wire [7:0] _T_1166; // @[Compute.scala 299:37]
  wire [31:0] _T_1167; // @[Compute.scala 300:30]
  wire [31:0] _T_1168; // @[Compute.scala 300:59]
  wire [32:0] _T_1169; // @[Compute.scala 300:49]
  wire [31:0] _T_1170; // @[Compute.scala 300:49]
  wire [31:0] _T_1171; // @[Compute.scala 300:79]
  wire [31:0] add_val_13; // @[Compute.scala 280:36]
  wire [31:0] add_res_13; // @[Compute.scala 280:36]
  wire [7:0] _T_1172; // @[Compute.scala 302:37]
  wire [4:0] _T_1174; // @[Compute.scala 303:60]
  wire [31:0] _T_1175; // @[Compute.scala 303:49]
  wire [31:0] _T_1176; // @[Compute.scala 303:84]
  wire [31:0] shr_val_13; // @[Compute.scala 280:36]
  wire [31:0] shr_res_13; // @[Compute.scala 280:36]
  wire [7:0] _T_1177; // @[Compute.scala 305:37]
  wire [31:0] src_0_14; // @[Compute.scala 280:36]
  wire [31:0] src_1_14; // @[Compute.scala 280:36]
  wire  _T_1178; // @[Compute.scala 297:34]
  wire [31:0] _T_1179; // @[Compute.scala 297:24]
  wire [31:0] mix_val_14; // @[Compute.scala 280:36]
  wire [7:0] _T_1180; // @[Compute.scala 299:37]
  wire [31:0] _T_1181; // @[Compute.scala 300:30]
  wire [31:0] _T_1182; // @[Compute.scala 300:59]
  wire [32:0] _T_1183; // @[Compute.scala 300:49]
  wire [31:0] _T_1184; // @[Compute.scala 300:49]
  wire [31:0] _T_1185; // @[Compute.scala 300:79]
  wire [31:0] add_val_14; // @[Compute.scala 280:36]
  wire [31:0] add_res_14; // @[Compute.scala 280:36]
  wire [7:0] _T_1186; // @[Compute.scala 302:37]
  wire [4:0] _T_1188; // @[Compute.scala 303:60]
  wire [31:0] _T_1189; // @[Compute.scala 303:49]
  wire [31:0] _T_1190; // @[Compute.scala 303:84]
  wire [31:0] shr_val_14; // @[Compute.scala 280:36]
  wire [31:0] shr_res_14; // @[Compute.scala 280:36]
  wire [7:0] _T_1191; // @[Compute.scala 305:37]
  wire [31:0] src_0_15; // @[Compute.scala 280:36]
  wire [31:0] src_1_15; // @[Compute.scala 280:36]
  wire  _T_1192; // @[Compute.scala 297:34]
  wire [31:0] _T_1193; // @[Compute.scala 297:24]
  wire [31:0] mix_val_15; // @[Compute.scala 280:36]
  wire [7:0] _T_1194; // @[Compute.scala 299:37]
  wire [31:0] _T_1195; // @[Compute.scala 300:30]
  wire [31:0] _T_1196; // @[Compute.scala 300:59]
  wire [32:0] _T_1197; // @[Compute.scala 300:49]
  wire [31:0] _T_1198; // @[Compute.scala 300:49]
  wire [31:0] _T_1199; // @[Compute.scala 300:79]
  wire [31:0] add_val_15; // @[Compute.scala 280:36]
  wire [31:0] add_res_15; // @[Compute.scala 280:36]
  wire [7:0] _T_1200; // @[Compute.scala 302:37]
  wire [4:0] _T_1202; // @[Compute.scala 303:60]
  wire [31:0] _T_1203; // @[Compute.scala 303:49]
  wire [31:0] _T_1204; // @[Compute.scala 303:84]
  wire [31:0] shr_val_15; // @[Compute.scala 280:36]
  wire [31:0] shr_res_15; // @[Compute.scala 280:36]
  wire [7:0] _T_1205; // @[Compute.scala 305:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 280:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 280:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 280:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 280:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 310:48]
  wire  alu_opcode_add_en; // @[Compute.scala 311:39]
  wire  _T_1208; // @[Compute.scala 312:37]
  wire  _T_1209; // @[Compute.scala 312:34]
  wire  _T_1210; // @[Compute.scala 312:52]
  wire [4:0] _T_1212; // @[Compute.scala 313:58]
  wire [4:0] _T_1213; // @[Compute.scala 313:58]
  wire [3:0] _T_1214; // @[Compute.scala 313:58]
  wire [15:0] _GEN_310; // @[Compute.scala 313:40]
  wire  _T_1215; // @[Compute.scala 313:40]
  wire  _T_1216; // @[Compute.scala 313:23]
  wire  _T_1219; // @[Compute.scala 313:66]
  wire  _GEN_275; // @[Compute.scala 313:85]
  wire [16:0] _T_1222; // @[Compute.scala 316:34]
  wire [16:0] _T_1223; // @[Compute.scala 316:34]
  wire [15:0] _T_1224; // @[Compute.scala 316:34]
  wire [22:0] _GEN_311; // @[Compute.scala 316:41]
  wire [22:0] _T_1226; // @[Compute.scala 316:41]
  wire [63:0] _T_1233; // @[Cat.scala 30:58]
  wire [127:0] _T_1241; // @[Cat.scala 30:58]
  wire [63:0] _T_1248; // @[Cat.scala 30:58]
  wire [127:0] _T_1256; // @[Cat.scala 30:58]
  wire [63:0] _T_1263; // @[Cat.scala 30:58]
  wire [127:0] _T_1271; // @[Cat.scala 30:58]
  wire [127:0] _T_1272; // @[Compute.scala 320:8]
  assign acc_mem__T_419_addr = dst_idx[7:0];
  assign acc_mem__T_419_data = acc_mem[acc_mem__T_419_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_421_addr = src_idx[7:0];
  assign acc_mem__T_421_data = acc_mem[acc_mem__T_421_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_397_data = {_T_399,_T_398};
  assign acc_mem__T_397_addr = acc_sram_addr[7:0];
  assign acc_mem__T_397_mask = 1'h1;
  assign acc_mem__T_397_en = _T_302 ? _T_395 : 1'h0;
  assign uop_mem_uop_addr = 10'h0;
  assign uop_mem_uop_data = uop_mem[uop_mem_uop_addr]; // @[Compute.scala 34:20]
  assign uop_mem__T_355_data = uops_data;
  assign uop_mem__T_355_addr = uop_sram_addr[9:0];
  assign uop_mem__T_355_mask = 1'h1;
  assign uop_mem__T_355_en = 1'h1;
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
  assign _T_234 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 92:37]
  assign uop_cntr_en = _T_234 & insn_valid; // @[Compute.scala 92:59]
  assign _T_236 = uop_cntr_val == uop_cntr_max; // @[Compute.scala 95:38]
  assign _T_237 = _T_236 & uop_cntr_en; // @[Compute.scala 95:56]
  assign uop_cntr_wrap = _T_237 & busy; // @[Compute.scala 95:71]
  assign acc_cntr_max = uop_cntr_max * 16'h4; // @[Compute.scala 97:29]
  assign _T_239 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 98:37]
  assign acc_cntr_en = _T_239 & insn_valid; // @[Compute.scala 98:59]
  assign _GEN_295 = {{3'd0}, acc_cntr_val}; // @[Compute.scala 101:38]
  assign _T_241 = _GEN_295 == acc_cntr_max; // @[Compute.scala 101:38]
  assign _T_242 = _T_241 & acc_cntr_en; // @[Compute.scala 101:56]
  assign acc_cntr_wrap = _T_242 & busy; // @[Compute.scala 101:71]
  assign _T_246 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 104:37]
  assign out_cntr_en = _T_246 & insn_valid; // @[Compute.scala 104:56]
  assign _T_248 = dst_offset_in == 16'h9; // @[Compute.scala 107:38]
  assign _T_249 = _T_248 & out_cntr_en; // @[Compute.scala 107:56]
  assign out_cntr_wrap = _T_249 & busy; // @[Compute.scala 107:71]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Compute.scala 112:43]
  assign push_next_dep_valid = push_next_dep & push; // @[Compute.scala 113:43]
  assign _T_260 = uop_cntr_wrap | acc_cntr_wrap; // @[Compute.scala 120:23]
  assign _T_261 = _T_260 | out_cntr_wrap; // @[Compute.scala 120:40]
  assign _T_262 = push_prev_dep | push_next_dep; // @[Compute.scala 121:25]
  assign _GEN_0 = _T_262 ? 3'h3 : 3'h4; // @[Compute.scala 121:43]
  assign _GEN_1 = _T_261 ? _GEN_0 : state; // @[Compute.scala 120:58]
  assign _T_264 = pop_prev_dep_ready == 1'h0; // @[Compute.scala 129:18]
  assign _T_266 = pop_next_dep_ready == 1'h0; // @[Compute.scala 129:41]
  assign _T_267 = _T_264 & _T_266; // @[Compute.scala 129:38]
  assign _T_268 = busy & _T_267; // @[Compute.scala 129:14]
  assign _T_269 = pop_prev_dep | pop_next_dep; // @[Compute.scala 129:79]
  assign _T_270 = _T_268 & _T_269; // @[Compute.scala 129:62]
  assign _GEN_2 = _T_270 ? 3'h1 : _GEN_1; // @[Compute.scala 129:97]
  assign _T_271 = pop_prev_dep_ready | pop_next_dep_ready; // @[Compute.scala 130:38]
  assign _T_272 = dump & _T_271; // @[Compute.scala 130:14]
  assign _GEN_3 = _T_272 ? 3'h2 : _GEN_2; // @[Compute.scala 130:63]
  assign _T_273 = push_prev_dep_ready | push_next_dep_ready; // @[Compute.scala 131:38]
  assign _T_274 = push & _T_273; // @[Compute.scala 131:14]
  assign _GEN_4 = _T_274 ? 3'h4 : _GEN_3; // @[Compute.scala 131:63]
  assign _T_277 = pop_prev_dep & dump; // @[Compute.scala 138:22]
  assign _T_278 = _T_277 & io_l2g_dep_queue_valid; // @[Compute.scala 138:30]
  assign _GEN_5 = _T_278 ? 1'h1 : pop_prev_dep_ready; // @[Compute.scala 138:57]
  assign _T_280 = pop_next_dep & dump; // @[Compute.scala 141:22]
  assign _T_281 = _T_280 & io_s2g_dep_queue_valid; // @[Compute.scala 141:30]
  assign _GEN_6 = _T_281 ? 1'h1 : pop_next_dep_ready; // @[Compute.scala 141:57]
  assign _T_285 = push_prev_dep_valid & io_g2l_dep_queue_ready; // @[Compute.scala 148:29]
  assign _T_286 = _T_285 & push; // @[Compute.scala 148:55]
  assign _GEN_7 = _T_286 ? 1'h1 : push_prev_dep_ready; // @[Compute.scala 148:64]
  assign _T_288 = push_next_dep_valid & io_g2s_dep_queue_ready; // @[Compute.scala 151:29]
  assign _T_289 = _T_288 & push; // @[Compute.scala 151:55]
  assign _GEN_8 = _T_289 ? 1'h1 : push_next_dep_ready; // @[Compute.scala 151:64]
  assign _T_292 = io_uops_waitrequest == 1'h0; // @[Compute.scala 156:22]
  assign _T_293 = uops_read & _T_292; // @[Compute.scala 156:19]
  assign _T_294 = _T_293 & busy; // @[Compute.scala 156:37]
  assign _T_295 = uop_cntr_val < uop_cntr_max; // @[Compute.scala 156:61]
  assign _T_296 = _T_294 & _T_295; // @[Compute.scala 156:45]
  assign _T_298 = uop_cntr_val + 16'h1; // @[Compute.scala 157:34]
  assign _T_299 = uop_cntr_val + 16'h1; // @[Compute.scala 157:34]
  assign _GEN_9 = _T_296 ? _T_299 : uop_cntr_val; // @[Compute.scala 156:77]
  assign _T_301 = io_biases_waitrequest == 1'h0; // @[Compute.scala 159:24]
  assign _T_302 = biases_read & _T_301; // @[Compute.scala 159:21]
  assign _T_303 = _T_302 & busy; // @[Compute.scala 159:39]
  assign _T_304 = _GEN_295 < acc_cntr_max; // @[Compute.scala 159:63]
  assign _T_305 = _T_303 & _T_304; // @[Compute.scala 159:47]
  assign _T_307 = acc_cntr_val + 16'h1; // @[Compute.scala 160:34]
  assign _T_308 = acc_cntr_val + 16'h1; // @[Compute.scala 160:34]
  assign _GEN_10 = _T_305 ? _T_308 : acc_cntr_val; // @[Compute.scala 159:79]
  assign _T_310 = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 162:26]
  assign _T_311 = out_mem_write & _T_310; // @[Compute.scala 162:23]
  assign _T_312 = _T_311 & busy; // @[Compute.scala 162:41]
  assign _T_313 = dst_offset_in < 16'h9; // @[Compute.scala 162:65]
  assign _T_314 = _T_312 & _T_313; // @[Compute.scala 162:49]
  assign _T_316 = dst_offset_in + 16'h1; // @[Compute.scala 163:34]
  assign _T_317 = dst_offset_in + 16'h1; // @[Compute.scala 163:34]
  assign _GEN_11 = _T_314 ? _T_317 : dst_offset_in; // @[Compute.scala 162:81]
  assign _GEN_16 = gemm_queue_ready ? 1'h0 : _GEN_5; // @[Compute.scala 167:27]
  assign _GEN_17 = gemm_queue_ready ? 1'h0 : _GEN_6; // @[Compute.scala 167:27]
  assign _GEN_18 = gemm_queue_ready ? 1'h0 : _GEN_7; // @[Compute.scala 167:27]
  assign _GEN_19 = gemm_queue_ready ? 1'h0 : _GEN_8; // @[Compute.scala 167:27]
  assign _GEN_20 = gemm_queue_ready ? 3'h2 : _GEN_4; // @[Compute.scala 167:27]
  assign _T_325 = idle | done; // @[Compute.scala 180:52]
  assign _T_326 = io_gemm_queue_valid & _T_325; // @[Compute.scala 180:43]
  assign _GEN_21 = gemm_queue_ready ? 1'h0 : _T_326; // @[Compute.scala 182:27]
  assign _GEN_297 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 190:33]
  assign _T_337 = dram_base + _GEN_297; // @[Compute.scala 190:33]
  assign _T_338 = dram_base + _GEN_297; // @[Compute.scala 190:33]
  assign _GEN_298 = {{3'd0}, _T_338}; // @[Compute.scala 190:49]
  assign uop_dram_addr = _GEN_298 << 2'h2; // @[Compute.scala 190:49]
  assign _T_340 = sram_base + uop_cntr_val; // @[Compute.scala 191:33]
  assign uop_sram_addr = sram_base + uop_cntr_val; // @[Compute.scala 191:33]
  assign _T_342 = uop_cntr_wrap == 1'h0; // @[Compute.scala 192:31]
  assign _T_343 = uop_cntr_en & _T_342; // @[Compute.scala 192:28]
  assign _T_344 = _T_343 & busy; // @[Compute.scala 192:46]
  assign _T_349 = uop_cntr_max - 16'h1; // @[Compute.scala 199:42]
  assign _T_350 = $unsigned(_T_349); // @[Compute.scala 199:42]
  assign _T_351 = _T_350[15:0]; // @[Compute.scala 199:42]
  assign _T_352 = uop_cntr_val == _T_351; // @[Compute.scala 199:24]
  assign _GEN_22 = _T_352 ? 1'h0 : _T_344; // @[Compute.scala 199:50]
  assign _GEN_299 = {{12'd0}, y_offset}; // @[Compute.scala 204:36]
  assign _T_356 = dram_base + _GEN_299; // @[Compute.scala 204:36]
  assign _T_357 = dram_base + _GEN_299; // @[Compute.scala 204:36]
  assign _GEN_300 = {{28'd0}, x_pad_0}; // @[Compute.scala 204:47]
  assign _T_358 = _T_357 + _GEN_300; // @[Compute.scala 204:47]
  assign _T_359 = _T_357 + _GEN_300; // @[Compute.scala 204:47]
  assign _GEN_301 = {{3'd0}, _T_359}; // @[Compute.scala 204:58]
  assign _T_361 = _GEN_301 << 2'h2; // @[Compute.scala 204:58]
  assign _T_363 = _T_361 * 35'h1; // @[Compute.scala 204:66]
  assign _GEN_302 = {{20'd0}, acc_cntr_val}; // @[Compute.scala 204:76]
  assign _T_364 = _T_363 + _GEN_302; // @[Compute.scala 204:76]
  assign _T_365 = _T_363 + _GEN_302; // @[Compute.scala 204:76]
  assign _GEN_303 = {{7'd0}, _T_365}; // @[Compute.scala 204:92]
  assign acc_dram_addr = _GEN_303 << 3'h4; // @[Compute.scala 204:92]
  assign _GEN_304 = {{4'd0}, sram_base}; // @[Compute.scala 205:36]
  assign _T_367 = _GEN_304 + y_offset; // @[Compute.scala 205:36]
  assign _T_368 = _GEN_304 + y_offset; // @[Compute.scala 205:36]
  assign _GEN_305 = {{16'd0}, x_pad_0}; // @[Compute.scala 205:47]
  assign _T_369 = _T_368 + _GEN_305; // @[Compute.scala 205:47]
  assign _T_370 = _T_368 + _GEN_305; // @[Compute.scala 205:47]
  assign _GEN_306 = {{3'd0}, _T_370}; // @[Compute.scala 205:58]
  assign _T_372 = _GEN_306 << 2'h2; // @[Compute.scala 205:58]
  assign _T_374 = _T_372 * 23'h1; // @[Compute.scala 205:66]
  assign _GEN_307 = {{8'd0}, acc_cntr_val}; // @[Compute.scala 205:76]
  assign _T_375 = _T_374 + _GEN_307; // @[Compute.scala 205:76]
  assign _T_376 = _T_374 + _GEN_307; // @[Compute.scala 205:76]
  assign _T_378 = _T_376 >> 2'h2; // @[Compute.scala 205:92]
  assign _T_380 = _T_378 - 24'h1; // @[Compute.scala 205:100]
  assign _T_381 = $unsigned(_T_380); // @[Compute.scala 205:100]
  assign acc_sram_addr = _T_381[23:0]; // @[Compute.scala 205:100]
  assign _T_383 = done == 1'h0; // @[Compute.scala 206:33]
  assign _GEN_12 = acc_cntr_val % 16'h4; // @[Compute.scala 212:30]
  assign _T_389 = _GEN_12[2:0]; // @[Compute.scala 212:30]
  assign _GEN_25 = 3'h0 == _T_389 ? io_biases_readdata : biases_data_0; // @[Compute.scala 212:55]
  assign _GEN_26 = 3'h1 == _T_389 ? io_biases_readdata : biases_data_1; // @[Compute.scala 212:55]
  assign _GEN_27 = 3'h2 == _T_389 ? io_biases_readdata : biases_data_2; // @[Compute.scala 212:55]
  assign _GEN_28 = 3'h3 == _T_389 ? io_biases_readdata : biases_data_3; // @[Compute.scala 212:55]
  assign _T_395 = _T_389 == 3'h0; // @[Compute.scala 213:50]
  assign _T_398 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_399 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 223:24]
  assign use_imm = insn[110]; // @[Compute.scala 224:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 225:21]
  assign _T_401 = $signed(imm_raw); // @[Compute.scala 226:25]
  assign _T_403 = $signed(_T_401) < $signed(16'sh0); // @[Compute.scala 226:32]
  assign _T_405 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_407 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_408 = _T_403 ? _T_405 : {{15'd0}, _T_407}; // @[Compute.scala 226:16]
  assign imm = $signed(_T_408); // @[Compute.scala 226:89]
  assign _T_409 = uop_mem_uop_data[10:0]; // @[Compute.scala 234:20]
  assign _GEN_308 = {{5'd0}, _T_409}; // @[Compute.scala 234:47]
  assign _T_410 = _GEN_308 + dst_offset_in; // @[Compute.scala 234:47]
  assign dst_idx = _GEN_308 + dst_offset_in; // @[Compute.scala 234:47]
  assign _T_411 = uop_mem_uop_data[21:11]; // @[Compute.scala 235:20]
  assign _GEN_309 = {{5'd0}, _T_411}; // @[Compute.scala 235:47]
  assign _T_412 = _GEN_309 + dst_offset_in; // @[Compute.scala 235:47]
  assign src_idx = _GEN_309 + dst_offset_in; // @[Compute.scala 235:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 260:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 261:38]
  assign _T_853 = insn_valid & out_cntr_en; // @[Compute.scala 280:20]
  assign _T_854 = src_vector[31:0]; // @[Compute.scala 283:31]
  assign _T_855 = $signed(_T_854); // @[Compute.scala 283:72]
  assign _T_856 = dst_vector[31:0]; // @[Compute.scala 284:31]
  assign _T_857 = $signed(_T_856); // @[Compute.scala 284:72]
  assign _T_858 = src_vector[63:32]; // @[Compute.scala 283:31]
  assign _T_859 = $signed(_T_858); // @[Compute.scala 283:72]
  assign _T_860 = dst_vector[63:32]; // @[Compute.scala 284:31]
  assign _T_861 = $signed(_T_860); // @[Compute.scala 284:72]
  assign _T_862 = src_vector[95:64]; // @[Compute.scala 283:31]
  assign _T_863 = $signed(_T_862); // @[Compute.scala 283:72]
  assign _T_864 = dst_vector[95:64]; // @[Compute.scala 284:31]
  assign _T_865 = $signed(_T_864); // @[Compute.scala 284:72]
  assign _T_866 = src_vector[127:96]; // @[Compute.scala 283:31]
  assign _T_867 = $signed(_T_866); // @[Compute.scala 283:72]
  assign _T_868 = dst_vector[127:96]; // @[Compute.scala 284:31]
  assign _T_869 = $signed(_T_868); // @[Compute.scala 284:72]
  assign _T_870 = src_vector[159:128]; // @[Compute.scala 283:31]
  assign _T_871 = $signed(_T_870); // @[Compute.scala 283:72]
  assign _T_872 = dst_vector[159:128]; // @[Compute.scala 284:31]
  assign _T_873 = $signed(_T_872); // @[Compute.scala 284:72]
  assign _T_874 = src_vector[191:160]; // @[Compute.scala 283:31]
  assign _T_875 = $signed(_T_874); // @[Compute.scala 283:72]
  assign _T_876 = dst_vector[191:160]; // @[Compute.scala 284:31]
  assign _T_877 = $signed(_T_876); // @[Compute.scala 284:72]
  assign _T_878 = src_vector[223:192]; // @[Compute.scala 283:31]
  assign _T_879 = $signed(_T_878); // @[Compute.scala 283:72]
  assign _T_880 = dst_vector[223:192]; // @[Compute.scala 284:31]
  assign _T_881 = $signed(_T_880); // @[Compute.scala 284:72]
  assign _T_882 = src_vector[255:224]; // @[Compute.scala 283:31]
  assign _T_883 = $signed(_T_882); // @[Compute.scala 283:72]
  assign _T_884 = dst_vector[255:224]; // @[Compute.scala 284:31]
  assign _T_885 = $signed(_T_884); // @[Compute.scala 284:72]
  assign _T_886 = src_vector[287:256]; // @[Compute.scala 283:31]
  assign _T_887 = $signed(_T_886); // @[Compute.scala 283:72]
  assign _T_888 = dst_vector[287:256]; // @[Compute.scala 284:31]
  assign _T_889 = $signed(_T_888); // @[Compute.scala 284:72]
  assign _T_890 = src_vector[319:288]; // @[Compute.scala 283:31]
  assign _T_891 = $signed(_T_890); // @[Compute.scala 283:72]
  assign _T_892 = dst_vector[319:288]; // @[Compute.scala 284:31]
  assign _T_893 = $signed(_T_892); // @[Compute.scala 284:72]
  assign _T_894 = src_vector[351:320]; // @[Compute.scala 283:31]
  assign _T_895 = $signed(_T_894); // @[Compute.scala 283:72]
  assign _T_896 = dst_vector[351:320]; // @[Compute.scala 284:31]
  assign _T_897 = $signed(_T_896); // @[Compute.scala 284:72]
  assign _T_898 = src_vector[383:352]; // @[Compute.scala 283:31]
  assign _T_899 = $signed(_T_898); // @[Compute.scala 283:72]
  assign _T_900 = dst_vector[383:352]; // @[Compute.scala 284:31]
  assign _T_901 = $signed(_T_900); // @[Compute.scala 284:72]
  assign _T_902 = src_vector[415:384]; // @[Compute.scala 283:31]
  assign _T_903 = $signed(_T_902); // @[Compute.scala 283:72]
  assign _T_904 = dst_vector[415:384]; // @[Compute.scala 284:31]
  assign _T_905 = $signed(_T_904); // @[Compute.scala 284:72]
  assign _T_906 = src_vector[447:416]; // @[Compute.scala 283:31]
  assign _T_907 = $signed(_T_906); // @[Compute.scala 283:72]
  assign _T_908 = dst_vector[447:416]; // @[Compute.scala 284:31]
  assign _T_909 = $signed(_T_908); // @[Compute.scala 284:72]
  assign _T_910 = src_vector[479:448]; // @[Compute.scala 283:31]
  assign _T_911 = $signed(_T_910); // @[Compute.scala 283:72]
  assign _T_912 = dst_vector[479:448]; // @[Compute.scala 284:31]
  assign _T_913 = $signed(_T_912); // @[Compute.scala 284:72]
  assign _T_914 = src_vector[511:480]; // @[Compute.scala 283:31]
  assign _T_915 = $signed(_T_914); // @[Compute.scala 283:72]
  assign _T_916 = dst_vector[511:480]; // @[Compute.scala 284:31]
  assign _T_917 = $signed(_T_916); // @[Compute.scala 284:72]
  assign _GEN_51 = alu_opcode_max_en ? $signed(_T_855) : $signed(_T_857); // @[Compute.scala 281:30]
  assign _GEN_52 = alu_opcode_max_en ? $signed(_T_857) : $signed(_T_855); // @[Compute.scala 281:30]
  assign _GEN_53 = alu_opcode_max_en ? $signed(_T_859) : $signed(_T_861); // @[Compute.scala 281:30]
  assign _GEN_54 = alu_opcode_max_en ? $signed(_T_861) : $signed(_T_859); // @[Compute.scala 281:30]
  assign _GEN_55 = alu_opcode_max_en ? $signed(_T_863) : $signed(_T_865); // @[Compute.scala 281:30]
  assign _GEN_56 = alu_opcode_max_en ? $signed(_T_865) : $signed(_T_863); // @[Compute.scala 281:30]
  assign _GEN_57 = alu_opcode_max_en ? $signed(_T_867) : $signed(_T_869); // @[Compute.scala 281:30]
  assign _GEN_58 = alu_opcode_max_en ? $signed(_T_869) : $signed(_T_867); // @[Compute.scala 281:30]
  assign _GEN_59 = alu_opcode_max_en ? $signed(_T_871) : $signed(_T_873); // @[Compute.scala 281:30]
  assign _GEN_60 = alu_opcode_max_en ? $signed(_T_873) : $signed(_T_871); // @[Compute.scala 281:30]
  assign _GEN_61 = alu_opcode_max_en ? $signed(_T_875) : $signed(_T_877); // @[Compute.scala 281:30]
  assign _GEN_62 = alu_opcode_max_en ? $signed(_T_877) : $signed(_T_875); // @[Compute.scala 281:30]
  assign _GEN_63 = alu_opcode_max_en ? $signed(_T_879) : $signed(_T_881); // @[Compute.scala 281:30]
  assign _GEN_64 = alu_opcode_max_en ? $signed(_T_881) : $signed(_T_879); // @[Compute.scala 281:30]
  assign _GEN_65 = alu_opcode_max_en ? $signed(_T_883) : $signed(_T_885); // @[Compute.scala 281:30]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_885) : $signed(_T_883); // @[Compute.scala 281:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_887) : $signed(_T_889); // @[Compute.scala 281:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_889) : $signed(_T_887); // @[Compute.scala 281:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_891) : $signed(_T_893); // @[Compute.scala 281:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_893) : $signed(_T_891); // @[Compute.scala 281:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_895) : $signed(_T_897); // @[Compute.scala 281:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_897) : $signed(_T_895); // @[Compute.scala 281:30]
  assign _GEN_73 = alu_opcode_max_en ? $signed(_T_899) : $signed(_T_901); // @[Compute.scala 281:30]
  assign _GEN_74 = alu_opcode_max_en ? $signed(_T_901) : $signed(_T_899); // @[Compute.scala 281:30]
  assign _GEN_75 = alu_opcode_max_en ? $signed(_T_903) : $signed(_T_905); // @[Compute.scala 281:30]
  assign _GEN_76 = alu_opcode_max_en ? $signed(_T_905) : $signed(_T_903); // @[Compute.scala 281:30]
  assign _GEN_77 = alu_opcode_max_en ? $signed(_T_907) : $signed(_T_909); // @[Compute.scala 281:30]
  assign _GEN_78 = alu_opcode_max_en ? $signed(_T_909) : $signed(_T_907); // @[Compute.scala 281:30]
  assign _GEN_79 = alu_opcode_max_en ? $signed(_T_911) : $signed(_T_913); // @[Compute.scala 281:30]
  assign _GEN_80 = alu_opcode_max_en ? $signed(_T_913) : $signed(_T_911); // @[Compute.scala 281:30]
  assign _GEN_81 = alu_opcode_max_en ? $signed(_T_915) : $signed(_T_917); // @[Compute.scala 281:30]
  assign _GEN_82 = alu_opcode_max_en ? $signed(_T_917) : $signed(_T_915); // @[Compute.scala 281:30]
  assign _GEN_83 = use_imm ? $signed(imm) : $signed(_GEN_52); // @[Compute.scala 292:20]
  assign _GEN_84 = use_imm ? $signed(imm) : $signed(_GEN_54); // @[Compute.scala 292:20]
  assign _GEN_85 = use_imm ? $signed(imm) : $signed(_GEN_56); // @[Compute.scala 292:20]
  assign _GEN_86 = use_imm ? $signed(imm) : $signed(_GEN_58); // @[Compute.scala 292:20]
  assign _GEN_87 = use_imm ? $signed(imm) : $signed(_GEN_60); // @[Compute.scala 292:20]
  assign _GEN_88 = use_imm ? $signed(imm) : $signed(_GEN_62); // @[Compute.scala 292:20]
  assign _GEN_89 = use_imm ? $signed(imm) : $signed(_GEN_64); // @[Compute.scala 292:20]
  assign _GEN_90 = use_imm ? $signed(imm) : $signed(_GEN_66); // @[Compute.scala 292:20]
  assign _GEN_91 = use_imm ? $signed(imm) : $signed(_GEN_68); // @[Compute.scala 292:20]
  assign _GEN_92 = use_imm ? $signed(imm) : $signed(_GEN_70); // @[Compute.scala 292:20]
  assign _GEN_93 = use_imm ? $signed(imm) : $signed(_GEN_72); // @[Compute.scala 292:20]
  assign _GEN_94 = use_imm ? $signed(imm) : $signed(_GEN_74); // @[Compute.scala 292:20]
  assign _GEN_95 = use_imm ? $signed(imm) : $signed(_GEN_76); // @[Compute.scala 292:20]
  assign _GEN_96 = use_imm ? $signed(imm) : $signed(_GEN_78); // @[Compute.scala 292:20]
  assign _GEN_97 = use_imm ? $signed(imm) : $signed(_GEN_80); // @[Compute.scala 292:20]
  assign _GEN_98 = use_imm ? $signed(imm) : $signed(_GEN_82); // @[Compute.scala 292:20]
  assign src_0_0 = _T_853 ? $signed(_GEN_51) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_0 = _T_853 ? $signed(_GEN_83) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_982 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 297:34]
  assign _T_983 = _T_982 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 297:24]
  assign mix_val_0 = _T_853 ? $signed(_T_983) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_984 = mix_val_0[7:0]; // @[Compute.scala 299:37]
  assign _T_985 = $unsigned(src_0_0); // @[Compute.scala 300:30]
  assign _T_986 = $unsigned(src_1_0); // @[Compute.scala 300:59]
  assign _T_987 = _T_985 + _T_986; // @[Compute.scala 300:49]
  assign _T_988 = _T_985 + _T_986; // @[Compute.scala 300:49]
  assign _T_989 = $signed(_T_988); // @[Compute.scala 300:79]
  assign add_val_0 = _T_853 ? $signed(_T_989) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_0 = _T_853 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_990 = add_res_0[7:0]; // @[Compute.scala 302:37]
  assign _T_992 = src_1_0[4:0]; // @[Compute.scala 303:60]
  assign _T_993 = _T_985 >> _T_992; // @[Compute.scala 303:49]
  assign _T_994 = $signed(_T_993); // @[Compute.scala 303:84]
  assign shr_val_0 = _T_853 ? $signed(_T_994) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_0 = _T_853 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_995 = shr_res_0[7:0]; // @[Compute.scala 305:37]
  assign src_0_1 = _T_853 ? $signed(_GEN_53) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_1 = _T_853 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_996 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 297:34]
  assign _T_997 = _T_996 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 297:24]
  assign mix_val_1 = _T_853 ? $signed(_T_997) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_998 = mix_val_1[7:0]; // @[Compute.scala 299:37]
  assign _T_999 = $unsigned(src_0_1); // @[Compute.scala 300:30]
  assign _T_1000 = $unsigned(src_1_1); // @[Compute.scala 300:59]
  assign _T_1001 = _T_999 + _T_1000; // @[Compute.scala 300:49]
  assign _T_1002 = _T_999 + _T_1000; // @[Compute.scala 300:49]
  assign _T_1003 = $signed(_T_1002); // @[Compute.scala 300:79]
  assign add_val_1 = _T_853 ? $signed(_T_1003) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_1 = _T_853 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1004 = add_res_1[7:0]; // @[Compute.scala 302:37]
  assign _T_1006 = src_1_1[4:0]; // @[Compute.scala 303:60]
  assign _T_1007 = _T_999 >> _T_1006; // @[Compute.scala 303:49]
  assign _T_1008 = $signed(_T_1007); // @[Compute.scala 303:84]
  assign shr_val_1 = _T_853 ? $signed(_T_1008) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_1 = _T_853 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1009 = shr_res_1[7:0]; // @[Compute.scala 305:37]
  assign src_0_2 = _T_853 ? $signed(_GEN_55) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_2 = _T_853 ? $signed(_GEN_85) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1010 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 297:34]
  assign _T_1011 = _T_1010 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 297:24]
  assign mix_val_2 = _T_853 ? $signed(_T_1011) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1012 = mix_val_2[7:0]; // @[Compute.scala 299:37]
  assign _T_1013 = $unsigned(src_0_2); // @[Compute.scala 300:30]
  assign _T_1014 = $unsigned(src_1_2); // @[Compute.scala 300:59]
  assign _T_1015 = _T_1013 + _T_1014; // @[Compute.scala 300:49]
  assign _T_1016 = _T_1013 + _T_1014; // @[Compute.scala 300:49]
  assign _T_1017 = $signed(_T_1016); // @[Compute.scala 300:79]
  assign add_val_2 = _T_853 ? $signed(_T_1017) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_2 = _T_853 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1018 = add_res_2[7:0]; // @[Compute.scala 302:37]
  assign _T_1020 = src_1_2[4:0]; // @[Compute.scala 303:60]
  assign _T_1021 = _T_1013 >> _T_1020; // @[Compute.scala 303:49]
  assign _T_1022 = $signed(_T_1021); // @[Compute.scala 303:84]
  assign shr_val_2 = _T_853 ? $signed(_T_1022) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_2 = _T_853 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1023 = shr_res_2[7:0]; // @[Compute.scala 305:37]
  assign src_0_3 = _T_853 ? $signed(_GEN_57) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_3 = _T_853 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1024 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 297:34]
  assign _T_1025 = _T_1024 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 297:24]
  assign mix_val_3 = _T_853 ? $signed(_T_1025) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1026 = mix_val_3[7:0]; // @[Compute.scala 299:37]
  assign _T_1027 = $unsigned(src_0_3); // @[Compute.scala 300:30]
  assign _T_1028 = $unsigned(src_1_3); // @[Compute.scala 300:59]
  assign _T_1029 = _T_1027 + _T_1028; // @[Compute.scala 300:49]
  assign _T_1030 = _T_1027 + _T_1028; // @[Compute.scala 300:49]
  assign _T_1031 = $signed(_T_1030); // @[Compute.scala 300:79]
  assign add_val_3 = _T_853 ? $signed(_T_1031) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_3 = _T_853 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1032 = add_res_3[7:0]; // @[Compute.scala 302:37]
  assign _T_1034 = src_1_3[4:0]; // @[Compute.scala 303:60]
  assign _T_1035 = _T_1027 >> _T_1034; // @[Compute.scala 303:49]
  assign _T_1036 = $signed(_T_1035); // @[Compute.scala 303:84]
  assign shr_val_3 = _T_853 ? $signed(_T_1036) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_3 = _T_853 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1037 = shr_res_3[7:0]; // @[Compute.scala 305:37]
  assign src_0_4 = _T_853 ? $signed(_GEN_59) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_4 = _T_853 ? $signed(_GEN_87) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1038 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 297:34]
  assign _T_1039 = _T_1038 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 297:24]
  assign mix_val_4 = _T_853 ? $signed(_T_1039) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1040 = mix_val_4[7:0]; // @[Compute.scala 299:37]
  assign _T_1041 = $unsigned(src_0_4); // @[Compute.scala 300:30]
  assign _T_1042 = $unsigned(src_1_4); // @[Compute.scala 300:59]
  assign _T_1043 = _T_1041 + _T_1042; // @[Compute.scala 300:49]
  assign _T_1044 = _T_1041 + _T_1042; // @[Compute.scala 300:49]
  assign _T_1045 = $signed(_T_1044); // @[Compute.scala 300:79]
  assign add_val_4 = _T_853 ? $signed(_T_1045) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_4 = _T_853 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1046 = add_res_4[7:0]; // @[Compute.scala 302:37]
  assign _T_1048 = src_1_4[4:0]; // @[Compute.scala 303:60]
  assign _T_1049 = _T_1041 >> _T_1048; // @[Compute.scala 303:49]
  assign _T_1050 = $signed(_T_1049); // @[Compute.scala 303:84]
  assign shr_val_4 = _T_853 ? $signed(_T_1050) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_4 = _T_853 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1051 = shr_res_4[7:0]; // @[Compute.scala 305:37]
  assign src_0_5 = _T_853 ? $signed(_GEN_61) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_5 = _T_853 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1052 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 297:34]
  assign _T_1053 = _T_1052 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 297:24]
  assign mix_val_5 = _T_853 ? $signed(_T_1053) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1054 = mix_val_5[7:0]; // @[Compute.scala 299:37]
  assign _T_1055 = $unsigned(src_0_5); // @[Compute.scala 300:30]
  assign _T_1056 = $unsigned(src_1_5); // @[Compute.scala 300:59]
  assign _T_1057 = _T_1055 + _T_1056; // @[Compute.scala 300:49]
  assign _T_1058 = _T_1055 + _T_1056; // @[Compute.scala 300:49]
  assign _T_1059 = $signed(_T_1058); // @[Compute.scala 300:79]
  assign add_val_5 = _T_853 ? $signed(_T_1059) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_5 = _T_853 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1060 = add_res_5[7:0]; // @[Compute.scala 302:37]
  assign _T_1062 = src_1_5[4:0]; // @[Compute.scala 303:60]
  assign _T_1063 = _T_1055 >> _T_1062; // @[Compute.scala 303:49]
  assign _T_1064 = $signed(_T_1063); // @[Compute.scala 303:84]
  assign shr_val_5 = _T_853 ? $signed(_T_1064) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_5 = _T_853 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1065 = shr_res_5[7:0]; // @[Compute.scala 305:37]
  assign src_0_6 = _T_853 ? $signed(_GEN_63) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_6 = _T_853 ? $signed(_GEN_89) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1066 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 297:34]
  assign _T_1067 = _T_1066 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 297:24]
  assign mix_val_6 = _T_853 ? $signed(_T_1067) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1068 = mix_val_6[7:0]; // @[Compute.scala 299:37]
  assign _T_1069 = $unsigned(src_0_6); // @[Compute.scala 300:30]
  assign _T_1070 = $unsigned(src_1_6); // @[Compute.scala 300:59]
  assign _T_1071 = _T_1069 + _T_1070; // @[Compute.scala 300:49]
  assign _T_1072 = _T_1069 + _T_1070; // @[Compute.scala 300:49]
  assign _T_1073 = $signed(_T_1072); // @[Compute.scala 300:79]
  assign add_val_6 = _T_853 ? $signed(_T_1073) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_6 = _T_853 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1074 = add_res_6[7:0]; // @[Compute.scala 302:37]
  assign _T_1076 = src_1_6[4:0]; // @[Compute.scala 303:60]
  assign _T_1077 = _T_1069 >> _T_1076; // @[Compute.scala 303:49]
  assign _T_1078 = $signed(_T_1077); // @[Compute.scala 303:84]
  assign shr_val_6 = _T_853 ? $signed(_T_1078) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_6 = _T_853 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1079 = shr_res_6[7:0]; // @[Compute.scala 305:37]
  assign src_0_7 = _T_853 ? $signed(_GEN_65) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_7 = _T_853 ? $signed(_GEN_90) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1080 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 297:34]
  assign _T_1081 = _T_1080 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 297:24]
  assign mix_val_7 = _T_853 ? $signed(_T_1081) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1082 = mix_val_7[7:0]; // @[Compute.scala 299:37]
  assign _T_1083 = $unsigned(src_0_7); // @[Compute.scala 300:30]
  assign _T_1084 = $unsigned(src_1_7); // @[Compute.scala 300:59]
  assign _T_1085 = _T_1083 + _T_1084; // @[Compute.scala 300:49]
  assign _T_1086 = _T_1083 + _T_1084; // @[Compute.scala 300:49]
  assign _T_1087 = $signed(_T_1086); // @[Compute.scala 300:79]
  assign add_val_7 = _T_853 ? $signed(_T_1087) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_7 = _T_853 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1088 = add_res_7[7:0]; // @[Compute.scala 302:37]
  assign _T_1090 = src_1_7[4:0]; // @[Compute.scala 303:60]
  assign _T_1091 = _T_1083 >> _T_1090; // @[Compute.scala 303:49]
  assign _T_1092 = $signed(_T_1091); // @[Compute.scala 303:84]
  assign shr_val_7 = _T_853 ? $signed(_T_1092) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_7 = _T_853 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1093 = shr_res_7[7:0]; // @[Compute.scala 305:37]
  assign src_0_8 = _T_853 ? $signed(_GEN_67) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_8 = _T_853 ? $signed(_GEN_91) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1094 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 297:34]
  assign _T_1095 = _T_1094 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 297:24]
  assign mix_val_8 = _T_853 ? $signed(_T_1095) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1096 = mix_val_8[7:0]; // @[Compute.scala 299:37]
  assign _T_1097 = $unsigned(src_0_8); // @[Compute.scala 300:30]
  assign _T_1098 = $unsigned(src_1_8); // @[Compute.scala 300:59]
  assign _T_1099 = _T_1097 + _T_1098; // @[Compute.scala 300:49]
  assign _T_1100 = _T_1097 + _T_1098; // @[Compute.scala 300:49]
  assign _T_1101 = $signed(_T_1100); // @[Compute.scala 300:79]
  assign add_val_8 = _T_853 ? $signed(_T_1101) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_8 = _T_853 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1102 = add_res_8[7:0]; // @[Compute.scala 302:37]
  assign _T_1104 = src_1_8[4:0]; // @[Compute.scala 303:60]
  assign _T_1105 = _T_1097 >> _T_1104; // @[Compute.scala 303:49]
  assign _T_1106 = $signed(_T_1105); // @[Compute.scala 303:84]
  assign shr_val_8 = _T_853 ? $signed(_T_1106) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_8 = _T_853 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1107 = shr_res_8[7:0]; // @[Compute.scala 305:37]
  assign src_0_9 = _T_853 ? $signed(_GEN_69) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_9 = _T_853 ? $signed(_GEN_92) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1108 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 297:34]
  assign _T_1109 = _T_1108 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 297:24]
  assign mix_val_9 = _T_853 ? $signed(_T_1109) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1110 = mix_val_9[7:0]; // @[Compute.scala 299:37]
  assign _T_1111 = $unsigned(src_0_9); // @[Compute.scala 300:30]
  assign _T_1112 = $unsigned(src_1_9); // @[Compute.scala 300:59]
  assign _T_1113 = _T_1111 + _T_1112; // @[Compute.scala 300:49]
  assign _T_1114 = _T_1111 + _T_1112; // @[Compute.scala 300:49]
  assign _T_1115 = $signed(_T_1114); // @[Compute.scala 300:79]
  assign add_val_9 = _T_853 ? $signed(_T_1115) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_9 = _T_853 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1116 = add_res_9[7:0]; // @[Compute.scala 302:37]
  assign _T_1118 = src_1_9[4:0]; // @[Compute.scala 303:60]
  assign _T_1119 = _T_1111 >> _T_1118; // @[Compute.scala 303:49]
  assign _T_1120 = $signed(_T_1119); // @[Compute.scala 303:84]
  assign shr_val_9 = _T_853 ? $signed(_T_1120) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_9 = _T_853 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1121 = shr_res_9[7:0]; // @[Compute.scala 305:37]
  assign src_0_10 = _T_853 ? $signed(_GEN_71) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_10 = _T_853 ? $signed(_GEN_93) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1122 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 297:34]
  assign _T_1123 = _T_1122 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 297:24]
  assign mix_val_10 = _T_853 ? $signed(_T_1123) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1124 = mix_val_10[7:0]; // @[Compute.scala 299:37]
  assign _T_1125 = $unsigned(src_0_10); // @[Compute.scala 300:30]
  assign _T_1126 = $unsigned(src_1_10); // @[Compute.scala 300:59]
  assign _T_1127 = _T_1125 + _T_1126; // @[Compute.scala 300:49]
  assign _T_1128 = _T_1125 + _T_1126; // @[Compute.scala 300:49]
  assign _T_1129 = $signed(_T_1128); // @[Compute.scala 300:79]
  assign add_val_10 = _T_853 ? $signed(_T_1129) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_10 = _T_853 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1130 = add_res_10[7:0]; // @[Compute.scala 302:37]
  assign _T_1132 = src_1_10[4:0]; // @[Compute.scala 303:60]
  assign _T_1133 = _T_1125 >> _T_1132; // @[Compute.scala 303:49]
  assign _T_1134 = $signed(_T_1133); // @[Compute.scala 303:84]
  assign shr_val_10 = _T_853 ? $signed(_T_1134) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_10 = _T_853 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1135 = shr_res_10[7:0]; // @[Compute.scala 305:37]
  assign src_0_11 = _T_853 ? $signed(_GEN_73) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_11 = _T_853 ? $signed(_GEN_94) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1136 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 297:34]
  assign _T_1137 = _T_1136 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 297:24]
  assign mix_val_11 = _T_853 ? $signed(_T_1137) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1138 = mix_val_11[7:0]; // @[Compute.scala 299:37]
  assign _T_1139 = $unsigned(src_0_11); // @[Compute.scala 300:30]
  assign _T_1140 = $unsigned(src_1_11); // @[Compute.scala 300:59]
  assign _T_1141 = _T_1139 + _T_1140; // @[Compute.scala 300:49]
  assign _T_1142 = _T_1139 + _T_1140; // @[Compute.scala 300:49]
  assign _T_1143 = $signed(_T_1142); // @[Compute.scala 300:79]
  assign add_val_11 = _T_853 ? $signed(_T_1143) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_11 = _T_853 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1144 = add_res_11[7:0]; // @[Compute.scala 302:37]
  assign _T_1146 = src_1_11[4:0]; // @[Compute.scala 303:60]
  assign _T_1147 = _T_1139 >> _T_1146; // @[Compute.scala 303:49]
  assign _T_1148 = $signed(_T_1147); // @[Compute.scala 303:84]
  assign shr_val_11 = _T_853 ? $signed(_T_1148) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_11 = _T_853 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1149 = shr_res_11[7:0]; // @[Compute.scala 305:37]
  assign src_0_12 = _T_853 ? $signed(_GEN_75) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_12 = _T_853 ? $signed(_GEN_95) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1150 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 297:34]
  assign _T_1151 = _T_1150 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 297:24]
  assign mix_val_12 = _T_853 ? $signed(_T_1151) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1152 = mix_val_12[7:0]; // @[Compute.scala 299:37]
  assign _T_1153 = $unsigned(src_0_12); // @[Compute.scala 300:30]
  assign _T_1154 = $unsigned(src_1_12); // @[Compute.scala 300:59]
  assign _T_1155 = _T_1153 + _T_1154; // @[Compute.scala 300:49]
  assign _T_1156 = _T_1153 + _T_1154; // @[Compute.scala 300:49]
  assign _T_1157 = $signed(_T_1156); // @[Compute.scala 300:79]
  assign add_val_12 = _T_853 ? $signed(_T_1157) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_12 = _T_853 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1158 = add_res_12[7:0]; // @[Compute.scala 302:37]
  assign _T_1160 = src_1_12[4:0]; // @[Compute.scala 303:60]
  assign _T_1161 = _T_1153 >> _T_1160; // @[Compute.scala 303:49]
  assign _T_1162 = $signed(_T_1161); // @[Compute.scala 303:84]
  assign shr_val_12 = _T_853 ? $signed(_T_1162) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_12 = _T_853 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1163 = shr_res_12[7:0]; // @[Compute.scala 305:37]
  assign src_0_13 = _T_853 ? $signed(_GEN_77) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_13 = _T_853 ? $signed(_GEN_96) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1164 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 297:34]
  assign _T_1165 = _T_1164 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 297:24]
  assign mix_val_13 = _T_853 ? $signed(_T_1165) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1166 = mix_val_13[7:0]; // @[Compute.scala 299:37]
  assign _T_1167 = $unsigned(src_0_13); // @[Compute.scala 300:30]
  assign _T_1168 = $unsigned(src_1_13); // @[Compute.scala 300:59]
  assign _T_1169 = _T_1167 + _T_1168; // @[Compute.scala 300:49]
  assign _T_1170 = _T_1167 + _T_1168; // @[Compute.scala 300:49]
  assign _T_1171 = $signed(_T_1170); // @[Compute.scala 300:79]
  assign add_val_13 = _T_853 ? $signed(_T_1171) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_13 = _T_853 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1172 = add_res_13[7:0]; // @[Compute.scala 302:37]
  assign _T_1174 = src_1_13[4:0]; // @[Compute.scala 303:60]
  assign _T_1175 = _T_1167 >> _T_1174; // @[Compute.scala 303:49]
  assign _T_1176 = $signed(_T_1175); // @[Compute.scala 303:84]
  assign shr_val_13 = _T_853 ? $signed(_T_1176) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_13 = _T_853 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1177 = shr_res_13[7:0]; // @[Compute.scala 305:37]
  assign src_0_14 = _T_853 ? $signed(_GEN_79) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_14 = _T_853 ? $signed(_GEN_97) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1178 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 297:34]
  assign _T_1179 = _T_1178 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 297:24]
  assign mix_val_14 = _T_853 ? $signed(_T_1179) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1180 = mix_val_14[7:0]; // @[Compute.scala 299:37]
  assign _T_1181 = $unsigned(src_0_14); // @[Compute.scala 300:30]
  assign _T_1182 = $unsigned(src_1_14); // @[Compute.scala 300:59]
  assign _T_1183 = _T_1181 + _T_1182; // @[Compute.scala 300:49]
  assign _T_1184 = _T_1181 + _T_1182; // @[Compute.scala 300:49]
  assign _T_1185 = $signed(_T_1184); // @[Compute.scala 300:79]
  assign add_val_14 = _T_853 ? $signed(_T_1185) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_14 = _T_853 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1186 = add_res_14[7:0]; // @[Compute.scala 302:37]
  assign _T_1188 = src_1_14[4:0]; // @[Compute.scala 303:60]
  assign _T_1189 = _T_1181 >> _T_1188; // @[Compute.scala 303:49]
  assign _T_1190 = $signed(_T_1189); // @[Compute.scala 303:84]
  assign shr_val_14 = _T_853 ? $signed(_T_1190) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_14 = _T_853 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1191 = shr_res_14[7:0]; // @[Compute.scala 305:37]
  assign src_0_15 = _T_853 ? $signed(_GEN_81) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign src_1_15 = _T_853 ? $signed(_GEN_98) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1192 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 297:34]
  assign _T_1193 = _T_1192 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 297:24]
  assign mix_val_15 = _T_853 ? $signed(_T_1193) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1194 = mix_val_15[7:0]; // @[Compute.scala 299:37]
  assign _T_1195 = $unsigned(src_0_15); // @[Compute.scala 300:30]
  assign _T_1196 = $unsigned(src_1_15); // @[Compute.scala 300:59]
  assign _T_1197 = _T_1195 + _T_1196; // @[Compute.scala 300:49]
  assign _T_1198 = _T_1195 + _T_1196; // @[Compute.scala 300:49]
  assign _T_1199 = $signed(_T_1198); // @[Compute.scala 300:79]
  assign add_val_15 = _T_853 ? $signed(_T_1199) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign add_res_15 = _T_853 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1200 = add_res_15[7:0]; // @[Compute.scala 302:37]
  assign _T_1202 = src_1_15[4:0]; // @[Compute.scala 303:60]
  assign _T_1203 = _T_1195 >> _T_1202; // @[Compute.scala 303:49]
  assign _T_1204 = $signed(_T_1203); // @[Compute.scala 303:84]
  assign shr_val_15 = _T_853 ? $signed(_T_1204) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign shr_res_15 = _T_853 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 280:36]
  assign _T_1205 = shr_res_15[7:0]; // @[Compute.scala 305:37]
  assign short_cmp_res_0 = _T_853 ? _T_984 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_0 = _T_853 ? _T_990 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_0 = _T_853 ? _T_995 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_1 = _T_853 ? _T_998 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_1 = _T_853 ? _T_1004 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_1 = _T_853 ? _T_1009 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_2 = _T_853 ? _T_1012 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_2 = _T_853 ? _T_1018 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_2 = _T_853 ? _T_1023 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_3 = _T_853 ? _T_1026 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_3 = _T_853 ? _T_1032 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_3 = _T_853 ? _T_1037 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_4 = _T_853 ? _T_1040 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_4 = _T_853 ? _T_1046 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_4 = _T_853 ? _T_1051 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_5 = _T_853 ? _T_1054 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_5 = _T_853 ? _T_1060 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_5 = _T_853 ? _T_1065 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_6 = _T_853 ? _T_1068 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_6 = _T_853 ? _T_1074 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_6 = _T_853 ? _T_1079 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_7 = _T_853 ? _T_1082 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_7 = _T_853 ? _T_1088 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_7 = _T_853 ? _T_1093 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_8 = _T_853 ? _T_1096 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_8 = _T_853 ? _T_1102 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_8 = _T_853 ? _T_1107 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_9 = _T_853 ? _T_1110 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_9 = _T_853 ? _T_1116 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_9 = _T_853 ? _T_1121 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_10 = _T_853 ? _T_1124 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_10 = _T_853 ? _T_1130 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_10 = _T_853 ? _T_1135 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_11 = _T_853 ? _T_1138 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_11 = _T_853 ? _T_1144 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_11 = _T_853 ? _T_1149 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_12 = _T_853 ? _T_1152 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_12 = _T_853 ? _T_1158 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_12 = _T_853 ? _T_1163 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_13 = _T_853 ? _T_1166 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_13 = _T_853 ? _T_1172 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_13 = _T_853 ? _T_1177 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_14 = _T_853 ? _T_1180 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_14 = _T_853 ? _T_1186 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_14 = _T_853 ? _T_1191 : 8'h0; // @[Compute.scala 280:36]
  assign short_cmp_res_15 = _T_853 ? _T_1194 : 8'h0; // @[Compute.scala 280:36]
  assign short_add_res_15 = _T_853 ? _T_1200 : 8'h0; // @[Compute.scala 280:36]
  assign short_shr_res_15 = _T_853 ? _T_1205 : 8'h0; // @[Compute.scala 280:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 310:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 311:39]
  assign _T_1208 = out_cntr_wrap == 1'h0; // @[Compute.scala 312:37]
  assign _T_1209 = opcode_alu_en & _T_1208; // @[Compute.scala 312:34]
  assign _T_1210 = _T_1209 & busy; // @[Compute.scala 312:52]
  assign _T_1212 = 4'h9 - 4'h1; // @[Compute.scala 313:58]
  assign _T_1213 = $unsigned(_T_1212); // @[Compute.scala 313:58]
  assign _T_1214 = _T_1213[3:0]; // @[Compute.scala 313:58]
  assign _GEN_310 = {{12'd0}, _T_1214}; // @[Compute.scala 313:40]
  assign _T_1215 = dst_offset_in == _GEN_310; // @[Compute.scala 313:40]
  assign _T_1216 = out_mem_write & _T_1215; // @[Compute.scala 313:23]
  assign _T_1219 = _T_1216 & _T_310; // @[Compute.scala 313:66]
  assign _GEN_275 = _T_1219 ? 1'h0 : _T_1210; // @[Compute.scala 313:85]
  assign _T_1222 = dst_idx - 16'h1; // @[Compute.scala 316:34]
  assign _T_1223 = $unsigned(_T_1222); // @[Compute.scala 316:34]
  assign _T_1224 = _T_1223[15:0]; // @[Compute.scala 316:34]
  assign _GEN_311 = {{7'd0}, _T_1224}; // @[Compute.scala 316:41]
  assign _T_1226 = _GEN_311 << 3'h4; // @[Compute.scala 316:41]
  assign _T_1233 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1241 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1233}; // @[Cat.scala 30:58]
  assign _T_1248 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1256 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1248}; // @[Cat.scala 30:58]
  assign _T_1263 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1271 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1263}; // @[Cat.scala 30:58]
  assign _T_1272 = alu_opcode_add_en ? _T_1256 : _T_1271; // @[Compute.scala 320:8]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 185:23]
  assign io_done_readdata = _T_334; // @[Compute.scala 186:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 194:19]
  assign io_uops_read = uops_read; // @[Compute.scala 193:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 32'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 207:21]
  assign io_biases_read = biases_read; // @[Compute.scala 208:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = gemm_queue_ready; // @[Compute.scala 181:23]
  assign io_l2g_dep_queue_ready = pop_prev_dep_ready & dump; // @[Compute.scala 134:26]
  assign io_s2g_dep_queue_ready = pop_next_dep_ready & dump; // @[Compute.scala 135:26]
  assign io_g2l_dep_queue_valid = push_prev_dep & push; // @[Compute.scala 146:26]
  assign io_g2l_dep_queue_data = 1'h1; // @[Compute.scala 144:25]
  assign io_g2s_dep_queue_valid = push_next_dep & push; // @[Compute.scala 147:26]
  assign io_g2s_dep_queue_data = 1'h1; // @[Compute.scala 145:25]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = _T_1226[16:0]; // @[Compute.scala 316:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_write; // @[Compute.scala 317:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1241 : _T_1272; // @[Compute.scala 319:24]
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
  _T_334 = _RAND_20[0:0];
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
    if(acc_mem__T_397_en & acc_mem__T_397_mask) begin
      acc_mem[acc_mem__T_397_addr] <= acc_mem__T_397_data; // @[Compute.scala 33:20]
    end
    if(uop_mem__T_355_en & uop_mem__T_355_mask) begin
      uop_mem[uop_mem__T_355_addr] <= uop_mem__T_355_data; // @[Compute.scala 34:20]
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
        if (_T_274) begin
          state <= 3'h4;
        end else begin
          if (_T_272) begin
            state <= 3'h2;
          end else begin
            if (_T_270) begin
              state <= 3'h1;
            end else begin
              if (_T_261) begin
                if (_T_262) begin
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
    if (_T_293) begin
      if (_T_352) begin
        uops_read <= 1'h0;
      end else begin
        uops_read <= _T_344;
      end
    end else begin
      uops_read <= _T_344;
    end
    if (_T_293) begin
      uops_data <= io_uops_readdata;
    end
    biases_read <= acc_cntr_en & _T_383;
    if (_T_302) begin
      if (3'h0 == _T_389) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_302) begin
      if (3'h1 == _T_389) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_302) begin
      if (3'h2 == _T_389) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_302) begin
      if (3'h3 == _T_389) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    if (reset) begin
      out_mem_write <= 1'h0;
    end else begin
      if (_T_1219) begin
        out_mem_write <= 1'h0;
      end else begin
        out_mem_write <= _T_1210;
      end
    end
    if (gemm_queue_ready) begin
      uop_cntr_val <= 16'h0;
    end else begin
      if (_T_296) begin
        uop_cntr_val <= _T_299;
      end
    end
    if (gemm_queue_ready) begin
      acc_cntr_val <= 16'h0;
    end else begin
      if (_T_305) begin
        acc_cntr_val <= _T_308;
      end
    end
    if (gemm_queue_ready) begin
      dst_offset_in <= 16'h0;
    end else begin
      if (_T_314) begin
        dst_offset_in <= _T_317;
      end
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_278) begin
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
        if (_T_281) begin
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
        if (_T_286) begin
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
        if (_T_289) begin
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
        gemm_queue_ready <= _T_326;
      end
    end
    _T_334 <= opcode_finish_en & io_done_read;
    if (_T_311) begin
      dst_vector <= acc_mem__T_419_data;
    end
    if (_T_311) begin
      src_vector <= acc_mem__T_421_data;
    end
  end
endmodule
