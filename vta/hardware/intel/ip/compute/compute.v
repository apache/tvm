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
  input  [127:0] io_uops_readdata,
  output         io_uops_write,
  output [127:0] io_uops_writedata,
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
  wire [511:0] acc_mem__T_469_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_469_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_471_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_471_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_449_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_449_addr; // @[Compute.scala 33:20]
  wire  acc_mem__T_449_mask; // @[Compute.scala 33:20]
  wire  acc_mem__T_449_en; // @[Compute.scala 33:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem_uop_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem_uop_addr; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_382_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_382_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_382_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_382_en; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_388_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_388_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_388_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_388_en; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_394_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_394_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_394_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_394_en; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_400_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_400_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_400_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_400_en; // @[Compute.scala 34:20]
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
  wire [15:0] x_size; // @[Compute.scala 49:25]
  wire [3:0] y_pad_0; // @[Compute.scala 51:25]
  wire [3:0] x_pad_0; // @[Compute.scala 53:25]
  wire [3:0] x_pad_1; // @[Compute.scala 54:25]
  reg [15:0] uop_bgn; // @[Compute.scala 57:20]
  reg [31:0] _RAND_3;
  wire [12:0] _T_203; // @[Compute.scala 58:18]
  reg [15:0] uop_end; // @[Compute.scala 59:20]
  reg [31:0] _RAND_4;
  wire [13:0] _T_205; // @[Compute.scala 60:18]
  wire [13:0] iter_out; // @[Compute.scala 61:22]
  wire [13:0] iter_in; // @[Compute.scala 62:21]
  wire [1:0] alu_opcode; // @[Compute.scala 69:24]
  wire  use_imm; // @[Compute.scala 70:21]
  wire [15:0] imm_raw; // @[Compute.scala 71:21]
  wire [15:0] _T_206; // @[Compute.scala 72:25]
  wire  _T_208; // @[Compute.scala 72:32]
  wire [31:0] _T_210; // @[Cat.scala 30:58]
  wire [16:0] _T_212; // @[Cat.scala 30:58]
  wire [31:0] _T_213; // @[Compute.scala 72:16]
  wire [31:0] imm; // @[Compute.scala 72:89]
  wire [15:0] _GEN_318; // @[Compute.scala 76:30]
  wire [15:0] _GEN_320; // @[Compute.scala 77:30]
  wire [16:0] _T_217; // @[Compute.scala 77:30]
  wire [15:0] _T_218; // @[Compute.scala 77:30]
  wire [15:0] _GEN_321; // @[Compute.scala 77:39]
  wire [16:0] _T_219; // @[Compute.scala 77:39]
  wire [15:0] x_size_total; // @[Compute.scala 77:39]
  wire [19:0] y_offset; // @[Compute.scala 78:31]
  wire  opcode_finish_en; // @[Compute.scala 81:34]
  wire  _T_222; // @[Compute.scala 82:32]
  wire  _T_224; // @[Compute.scala 82:60]
  wire  opcode_load_en; // @[Compute.scala 82:50]
  wire  opcode_gemm_en; // @[Compute.scala 83:32]
  wire  opcode_alu_en; // @[Compute.scala 84:31]
  wire  memory_type_uop_en; // @[Compute.scala 86:40]
  wire  memory_type_acc_en; // @[Compute.scala 87:40]
  reg [2:0] state; // @[Compute.scala 90:22]
  reg [31:0] _RAND_5;
  wire  idle; // @[Compute.scala 92:20]
  wire  dump; // @[Compute.scala 93:20]
  wire  busy; // @[Compute.scala 94:20]
  wire  push; // @[Compute.scala 95:20]
  wire  done; // @[Compute.scala 96:20]
  reg  uops_read; // @[Compute.scala 99:24]
  reg [31:0] _RAND_6;
  reg  biases_read; // @[Compute.scala 102:24]
  reg [31:0] _RAND_7;
  reg [127:0] biases_data_0; // @[Compute.scala 105:24]
  reg [127:0] _RAND_8;
  reg [127:0] biases_data_1; // @[Compute.scala 105:24]
  reg [127:0] _RAND_9;
  reg [127:0] biases_data_2; // @[Compute.scala 105:24]
  reg [127:0] _RAND_10;
  reg [127:0] biases_data_3; // @[Compute.scala 105:24]
  reg [127:0] _RAND_11;
  reg  out_mem_write; // @[Compute.scala 107:31]
  reg [31:0] _RAND_12;
  wire [15:0] uop_cntr_max_val; // @[Compute.scala 110:33]
  wire  _T_248; // @[Compute.scala 111:43]
  wire [15:0] uop_cntr_max; // @[Compute.scala 111:25]
  wire  _T_250; // @[Compute.scala 112:37]
  wire  uop_cntr_en; // @[Compute.scala 112:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 114:25]
  reg [31:0] _RAND_13;
  wire  _T_252; // @[Compute.scala 115:38]
  wire  _T_253; // @[Compute.scala 115:56]
  wire  uop_cntr_wrap; // @[Compute.scala 115:71]
  wire [18:0] _T_255; // @[Compute.scala 117:29]
  wire [19:0] _T_257; // @[Compute.scala 117:46]
  wire [18:0] acc_cntr_max; // @[Compute.scala 117:46]
  wire  _T_258; // @[Compute.scala 118:37]
  wire  acc_cntr_en; // @[Compute.scala 118:59]
  reg [15:0] acc_cntr_val; // @[Compute.scala 120:25]
  reg [31:0] _RAND_14;
  wire [18:0] _GEN_323; // @[Compute.scala 121:38]
  wire  _T_260; // @[Compute.scala 121:38]
  wire  _T_261; // @[Compute.scala 121:56]
  wire  acc_cntr_wrap; // @[Compute.scala 121:71]
  wire [16:0] _T_262; // @[Compute.scala 123:34]
  wire [16:0] _T_263; // @[Compute.scala 123:34]
  wire [15:0] upc_cntr_max_val; // @[Compute.scala 123:34]
  wire  _T_265; // @[Compute.scala 124:43]
  wire [15:0] upc_cntr_max; // @[Compute.scala 124:25]
  wire [27:0] _T_267; // @[Compute.scala 125:30]
  wire [27:0] _GEN_324; // @[Compute.scala 125:41]
  wire [43:0] _T_268; // @[Compute.scala 125:41]
  wire [44:0] _T_270; // @[Compute.scala 125:56]
  wire [43:0] out_cntr_max; // @[Compute.scala 125:56]
  wire  _T_271; // @[Compute.scala 126:37]
  wire  out_cntr_en; // @[Compute.scala 126:56]
  reg [15:0] out_cntr_val; // @[Compute.scala 128:25]
  reg [31:0] _RAND_15;
  wire [43:0] _GEN_325; // @[Compute.scala 129:38]
  wire  _T_273; // @[Compute.scala 129:38]
  wire  _T_274; // @[Compute.scala 129:56]
  wire  out_cntr_wrap; // @[Compute.scala 129:71]
  reg  pop_prev_dep_ready; // @[Compute.scala 132:35]
  reg [31:0] _RAND_16;
  reg  pop_next_dep_ready; // @[Compute.scala 133:35]
  reg [31:0] _RAND_17;
  wire  push_prev_dep_valid; // @[Compute.scala 134:43]
  wire  push_next_dep_valid; // @[Compute.scala 135:43]
  reg  push_prev_dep_ready; // @[Compute.scala 136:36]
  reg [31:0] _RAND_18;
  reg  push_next_dep_ready; // @[Compute.scala 137:36]
  reg [31:0] _RAND_19;
  reg  gemm_queue_ready; // @[Compute.scala 139:33]
  reg [31:0] _RAND_20;
  reg  finish_wrap; // @[Compute.scala 142:28]
  reg [31:0] _RAND_21;
  wire  _T_287; // @[Compute.scala 144:68]
  wire  _T_288; // @[Compute.scala 145:68]
  wire  _T_289; // @[Compute.scala 146:68]
  wire  _T_290; // @[Compute.scala 147:68]
  wire  _GEN_0; // @[Compute.scala 147:31]
  wire  _GEN_1; // @[Compute.scala 146:31]
  wire  _GEN_2; // @[Compute.scala 145:31]
  wire  _GEN_3; // @[Compute.scala 144:31]
  wire  _GEN_4; // @[Compute.scala 143:27]
  wire  _T_293; // @[Compute.scala 150:23]
  wire  _T_294; // @[Compute.scala 150:40]
  wire  _T_295; // @[Compute.scala 150:57]
  wire  _T_296; // @[Compute.scala 151:25]
  wire [2:0] _GEN_5; // @[Compute.scala 151:43]
  wire [2:0] _GEN_6; // @[Compute.scala 150:73]
  wire  _T_298; // @[Compute.scala 159:18]
  wire  _T_300; // @[Compute.scala 159:41]
  wire  _T_301; // @[Compute.scala 159:38]
  wire  _T_302; // @[Compute.scala 159:14]
  wire  _T_303; // @[Compute.scala 159:79]
  wire  _T_304; // @[Compute.scala 159:62]
  wire [2:0] _GEN_7; // @[Compute.scala 159:97]
  wire  _T_305; // @[Compute.scala 160:38]
  wire  _T_306; // @[Compute.scala 160:14]
  wire [2:0] _GEN_8; // @[Compute.scala 160:63]
  wire  _T_307; // @[Compute.scala 161:38]
  wire  _T_308; // @[Compute.scala 161:14]
  wire [2:0] _GEN_9; // @[Compute.scala 161:63]
  wire  _T_311; // @[Compute.scala 168:22]
  wire  _T_312; // @[Compute.scala 168:30]
  wire  _GEN_10; // @[Compute.scala 168:57]
  wire  _T_314; // @[Compute.scala 171:22]
  wire  _T_315; // @[Compute.scala 171:30]
  wire  _GEN_11; // @[Compute.scala 171:57]
  wire  _T_319; // @[Compute.scala 178:29]
  wire  _T_320; // @[Compute.scala 178:55]
  wire  _GEN_12; // @[Compute.scala 178:64]
  wire  _T_322; // @[Compute.scala 181:29]
  wire  _T_323; // @[Compute.scala 181:55]
  wire  _GEN_13; // @[Compute.scala 181:64]
  wire  _T_326; // @[Compute.scala 186:22]
  wire  _T_327; // @[Compute.scala 186:19]
  wire  _T_328; // @[Compute.scala 186:37]
  wire  _T_329; // @[Compute.scala 186:61]
  wire  _T_330; // @[Compute.scala 186:45]
  wire [16:0] _T_332; // @[Compute.scala 187:34]
  wire [15:0] _T_333; // @[Compute.scala 187:34]
  wire [15:0] _GEN_14; // @[Compute.scala 186:77]
  wire  _T_335; // @[Compute.scala 189:24]
  wire  _T_336; // @[Compute.scala 189:21]
  wire  _T_337; // @[Compute.scala 189:39]
  wire  _T_338; // @[Compute.scala 189:63]
  wire  _T_339; // @[Compute.scala 189:47]
  wire [16:0] _T_341; // @[Compute.scala 190:34]
  wire [15:0] _T_342; // @[Compute.scala 190:34]
  wire [15:0] _GEN_15; // @[Compute.scala 189:79]
  wire  _T_344; // @[Compute.scala 192:26]
  wire  _T_345; // @[Compute.scala 192:23]
  wire  _T_346; // @[Compute.scala 192:41]
  wire  _T_347; // @[Compute.scala 192:65]
  wire  _T_348; // @[Compute.scala 192:49]
  wire [16:0] _T_350; // @[Compute.scala 193:34]
  wire [15:0] _T_351; // @[Compute.scala 193:34]
  wire [15:0] _GEN_16; // @[Compute.scala 192:81]
  wire  _GEN_21; // @[Compute.scala 197:27]
  wire  _GEN_22; // @[Compute.scala 197:27]
  wire  _GEN_23; // @[Compute.scala 197:27]
  wire  _GEN_24; // @[Compute.scala 197:27]
  wire [2:0] _GEN_25; // @[Compute.scala 197:27]
  wire  _T_359; // @[Compute.scala 210:52]
  wire  _T_360; // @[Compute.scala 210:43]
  wire  _GEN_26; // @[Compute.scala 212:27]
  wire [31:0] _GEN_328; // @[Compute.scala 222:33]
  wire [32:0] _T_365; // @[Compute.scala 222:33]
  wire [31:0] _T_366; // @[Compute.scala 222:33]
  wire [38:0] _GEN_329; // @[Compute.scala 222:49]
  wire [38:0] uop_dram_addr; // @[Compute.scala 222:49]
  wire [16:0] _T_368; // @[Compute.scala 223:33]
  wire [15:0] _T_369; // @[Compute.scala 223:33]
  wire [18:0] _GEN_330; // @[Compute.scala 223:49]
  wire [18:0] uop_sram_addr; // @[Compute.scala 223:49]
  wire  _T_372; // @[Compute.scala 224:31]
  wire  _T_373; // @[Compute.scala 224:28]
  wire  _T_374; // @[Compute.scala 224:46]
  wire [19:0] _T_379; // @[Compute.scala 232:29]
  wire [18:0] _T_380; // @[Compute.scala 232:29]
  wire [19:0] _T_385; // @[Compute.scala 232:29]
  wire [18:0] _T_386; // @[Compute.scala 232:29]
  wire [19:0] _T_391; // @[Compute.scala 232:29]
  wire [18:0] _T_392; // @[Compute.scala 232:29]
  wire [19:0] _T_397; // @[Compute.scala 232:29]
  wire [18:0] _T_398; // @[Compute.scala 232:29]
  wire [16:0] _T_403; // @[Compute.scala 234:42]
  wire [16:0] _T_404; // @[Compute.scala 234:42]
  wire [15:0] _T_405; // @[Compute.scala 234:42]
  wire  _T_406; // @[Compute.scala 234:24]
  wire  _GEN_27; // @[Compute.scala 234:50]
  wire [31:0] _GEN_331; // @[Compute.scala 238:36]
  wire [32:0] _T_408; // @[Compute.scala 238:36]
  wire [31:0] _T_409; // @[Compute.scala 238:36]
  wire [31:0] _GEN_332; // @[Compute.scala 238:47]
  wire [32:0] _T_410; // @[Compute.scala 238:47]
  wire [31:0] _T_411; // @[Compute.scala 238:47]
  wire [34:0] _GEN_333; // @[Compute.scala 238:58]
  wire [34:0] _T_413; // @[Compute.scala 238:58]
  wire [35:0] _T_415; // @[Compute.scala 238:66]
  wire [35:0] _GEN_334; // @[Compute.scala 238:76]
  wire [36:0] _T_416; // @[Compute.scala 238:76]
  wire [35:0] _T_417; // @[Compute.scala 238:76]
  wire [42:0] _GEN_335; // @[Compute.scala 238:92]
  wire [42:0] acc_dram_addr; // @[Compute.scala 238:92]
  wire [19:0] _GEN_336; // @[Compute.scala 239:36]
  wire [20:0] _T_419; // @[Compute.scala 239:36]
  wire [19:0] _T_420; // @[Compute.scala 239:36]
  wire [19:0] _GEN_337; // @[Compute.scala 239:47]
  wire [20:0] _T_421; // @[Compute.scala 239:47]
  wire [19:0] _T_422; // @[Compute.scala 239:47]
  wire [22:0] _GEN_338; // @[Compute.scala 239:58]
  wire [22:0] _T_424; // @[Compute.scala 239:58]
  wire [23:0] _T_426; // @[Compute.scala 239:66]
  wire [23:0] _GEN_339; // @[Compute.scala 239:76]
  wire [24:0] _T_427; // @[Compute.scala 239:76]
  wire [23:0] _T_428; // @[Compute.scala 239:76]
  wire [23:0] _T_430; // @[Compute.scala 239:92]
  wire [24:0] _T_432; // @[Compute.scala 239:121]
  wire [24:0] _T_433; // @[Compute.scala 239:121]
  wire [23:0] acc_sram_addr; // @[Compute.scala 239:121]
  wire  _T_435; // @[Compute.scala 240:33]
  wire [15:0] _GEN_17; // @[Compute.scala 246:30]
  wire [2:0] _T_441; // @[Compute.scala 246:30]
  wire [127:0] _GEN_40; // @[Compute.scala 246:48]
  wire [127:0] _GEN_41; // @[Compute.scala 246:48]
  wire [127:0] _GEN_42; // @[Compute.scala 246:48]
  wire [127:0] _GEN_43; // @[Compute.scala 246:48]
  wire  _T_447; // @[Compute.scala 250:43]
  wire [255:0] _T_450; // @[Cat.scala 30:58]
  wire [255:0] _T_451; // @[Cat.scala 30:58]
  wire [27:0] _GEN_340; // @[Compute.scala 261:30]
  wire [27:0] _GEN_18; // @[Compute.scala 261:30]
  wire [15:0] it_in; // @[Compute.scala 261:30]
  wire [31:0] _T_455; // @[Compute.scala 262:46]
  wire [32:0] _T_456; // @[Compute.scala 262:38]
  wire [31:0] dst_offset_in; // @[Compute.scala 262:38]
  wire [10:0] _T_459; // @[Compute.scala 264:20]
  wire [31:0] _GEN_341; // @[Compute.scala 264:47]
  wire [32:0] _T_460; // @[Compute.scala 264:47]
  wire [31:0] dst_idx; // @[Compute.scala 264:47]
  wire [10:0] _T_461; // @[Compute.scala 265:20]
  wire [31:0] _GEN_342; // @[Compute.scala 265:47]
  wire [32:0] _T_462; // @[Compute.scala 265:47]
  wire [31:0] src_idx; // @[Compute.scala 265:47]
  reg [511:0] dst_vector; // @[Compute.scala 268:23]
  reg [511:0] _RAND_22;
  reg [511:0] src_vector; // @[Compute.scala 269:23]
  reg [511:0] _RAND_23;
  wire  alu_opcode_min_en; // @[Compute.scala 287:38]
  wire  alu_opcode_max_en; // @[Compute.scala 288:38]
  wire  _T_903; // @[Compute.scala 307:20]
  wire [31:0] _T_904; // @[Compute.scala 310:31]
  wire [31:0] _T_905; // @[Compute.scala 310:72]
  wire [31:0] _T_906; // @[Compute.scala 311:31]
  wire [31:0] _T_907; // @[Compute.scala 311:72]
  wire [31:0] _T_908; // @[Compute.scala 310:31]
  wire [31:0] _T_909; // @[Compute.scala 310:72]
  wire [31:0] _T_910; // @[Compute.scala 311:31]
  wire [31:0] _T_911; // @[Compute.scala 311:72]
  wire [31:0] _T_912; // @[Compute.scala 310:31]
  wire [31:0] _T_913; // @[Compute.scala 310:72]
  wire [31:0] _T_914; // @[Compute.scala 311:31]
  wire [31:0] _T_915; // @[Compute.scala 311:72]
  wire [31:0] _T_916; // @[Compute.scala 310:31]
  wire [31:0] _T_917; // @[Compute.scala 310:72]
  wire [31:0] _T_918; // @[Compute.scala 311:31]
  wire [31:0] _T_919; // @[Compute.scala 311:72]
  wire [31:0] _T_920; // @[Compute.scala 310:31]
  wire [31:0] _T_921; // @[Compute.scala 310:72]
  wire [31:0] _T_922; // @[Compute.scala 311:31]
  wire [31:0] _T_923; // @[Compute.scala 311:72]
  wire [31:0] _T_924; // @[Compute.scala 310:31]
  wire [31:0] _T_925; // @[Compute.scala 310:72]
  wire [31:0] _T_926; // @[Compute.scala 311:31]
  wire [31:0] _T_927; // @[Compute.scala 311:72]
  wire [31:0] _T_928; // @[Compute.scala 310:31]
  wire [31:0] _T_929; // @[Compute.scala 310:72]
  wire [31:0] _T_930; // @[Compute.scala 311:31]
  wire [31:0] _T_931; // @[Compute.scala 311:72]
  wire [31:0] _T_932; // @[Compute.scala 310:31]
  wire [31:0] _T_933; // @[Compute.scala 310:72]
  wire [31:0] _T_934; // @[Compute.scala 311:31]
  wire [31:0] _T_935; // @[Compute.scala 311:72]
  wire [31:0] _T_936; // @[Compute.scala 310:31]
  wire [31:0] _T_937; // @[Compute.scala 310:72]
  wire [31:0] _T_938; // @[Compute.scala 311:31]
  wire [31:0] _T_939; // @[Compute.scala 311:72]
  wire [31:0] _T_940; // @[Compute.scala 310:31]
  wire [31:0] _T_941; // @[Compute.scala 310:72]
  wire [31:0] _T_942; // @[Compute.scala 311:31]
  wire [31:0] _T_943; // @[Compute.scala 311:72]
  wire [31:0] _T_944; // @[Compute.scala 310:31]
  wire [31:0] _T_945; // @[Compute.scala 310:72]
  wire [31:0] _T_946; // @[Compute.scala 311:31]
  wire [31:0] _T_947; // @[Compute.scala 311:72]
  wire [31:0] _T_948; // @[Compute.scala 310:31]
  wire [31:0] _T_949; // @[Compute.scala 310:72]
  wire [31:0] _T_950; // @[Compute.scala 311:31]
  wire [31:0] _T_951; // @[Compute.scala 311:72]
  wire [31:0] _T_952; // @[Compute.scala 310:31]
  wire [31:0] _T_953; // @[Compute.scala 310:72]
  wire [31:0] _T_954; // @[Compute.scala 311:31]
  wire [31:0] _T_955; // @[Compute.scala 311:72]
  wire [31:0] _T_956; // @[Compute.scala 310:31]
  wire [31:0] _T_957; // @[Compute.scala 310:72]
  wire [31:0] _T_958; // @[Compute.scala 311:31]
  wire [31:0] _T_959; // @[Compute.scala 311:72]
  wire [31:0] _T_960; // @[Compute.scala 310:31]
  wire [31:0] _T_961; // @[Compute.scala 310:72]
  wire [31:0] _T_962; // @[Compute.scala 311:31]
  wire [31:0] _T_963; // @[Compute.scala 311:72]
  wire [31:0] _T_964; // @[Compute.scala 310:31]
  wire [31:0] _T_965; // @[Compute.scala 310:72]
  wire [31:0] _T_966; // @[Compute.scala 311:31]
  wire [31:0] _T_967; // @[Compute.scala 311:72]
  wire [31:0] _GEN_66; // @[Compute.scala 308:30]
  wire [31:0] _GEN_67; // @[Compute.scala 308:30]
  wire [31:0] _GEN_68; // @[Compute.scala 308:30]
  wire [31:0] _GEN_69; // @[Compute.scala 308:30]
  wire [31:0] _GEN_70; // @[Compute.scala 308:30]
  wire [31:0] _GEN_71; // @[Compute.scala 308:30]
  wire [31:0] _GEN_72; // @[Compute.scala 308:30]
  wire [31:0] _GEN_73; // @[Compute.scala 308:30]
  wire [31:0] _GEN_74; // @[Compute.scala 308:30]
  wire [31:0] _GEN_75; // @[Compute.scala 308:30]
  wire [31:0] _GEN_76; // @[Compute.scala 308:30]
  wire [31:0] _GEN_77; // @[Compute.scala 308:30]
  wire [31:0] _GEN_78; // @[Compute.scala 308:30]
  wire [31:0] _GEN_79; // @[Compute.scala 308:30]
  wire [31:0] _GEN_80; // @[Compute.scala 308:30]
  wire [31:0] _GEN_81; // @[Compute.scala 308:30]
  wire [31:0] _GEN_82; // @[Compute.scala 308:30]
  wire [31:0] _GEN_83; // @[Compute.scala 308:30]
  wire [31:0] _GEN_84; // @[Compute.scala 308:30]
  wire [31:0] _GEN_85; // @[Compute.scala 308:30]
  wire [31:0] _GEN_86; // @[Compute.scala 308:30]
  wire [31:0] _GEN_87; // @[Compute.scala 308:30]
  wire [31:0] _GEN_88; // @[Compute.scala 308:30]
  wire [31:0] _GEN_89; // @[Compute.scala 308:30]
  wire [31:0] _GEN_90; // @[Compute.scala 308:30]
  wire [31:0] _GEN_91; // @[Compute.scala 308:30]
  wire [31:0] _GEN_92; // @[Compute.scala 308:30]
  wire [31:0] _GEN_93; // @[Compute.scala 308:30]
  wire [31:0] _GEN_94; // @[Compute.scala 308:30]
  wire [31:0] _GEN_95; // @[Compute.scala 308:30]
  wire [31:0] _GEN_96; // @[Compute.scala 308:30]
  wire [31:0] _GEN_97; // @[Compute.scala 308:30]
  wire [31:0] _GEN_98; // @[Compute.scala 319:20]
  wire [31:0] _GEN_99; // @[Compute.scala 319:20]
  wire [31:0] _GEN_100; // @[Compute.scala 319:20]
  wire [31:0] _GEN_101; // @[Compute.scala 319:20]
  wire [31:0] _GEN_102; // @[Compute.scala 319:20]
  wire [31:0] _GEN_103; // @[Compute.scala 319:20]
  wire [31:0] _GEN_104; // @[Compute.scala 319:20]
  wire [31:0] _GEN_105; // @[Compute.scala 319:20]
  wire [31:0] _GEN_106; // @[Compute.scala 319:20]
  wire [31:0] _GEN_107; // @[Compute.scala 319:20]
  wire [31:0] _GEN_108; // @[Compute.scala 319:20]
  wire [31:0] _GEN_109; // @[Compute.scala 319:20]
  wire [31:0] _GEN_110; // @[Compute.scala 319:20]
  wire [31:0] _GEN_111; // @[Compute.scala 319:20]
  wire [31:0] _GEN_112; // @[Compute.scala 319:20]
  wire [31:0] _GEN_113; // @[Compute.scala 319:20]
  wire [31:0] src_0_0; // @[Compute.scala 307:36]
  wire [31:0] src_1_0; // @[Compute.scala 307:36]
  wire  _T_1032; // @[Compute.scala 324:34]
  wire [31:0] _T_1033; // @[Compute.scala 324:24]
  wire [31:0] mix_val_0; // @[Compute.scala 307:36]
  wire [7:0] _T_1034; // @[Compute.scala 326:37]
  wire [31:0] _T_1035; // @[Compute.scala 327:30]
  wire [31:0] _T_1036; // @[Compute.scala 327:59]
  wire [32:0] _T_1037; // @[Compute.scala 327:49]
  wire [31:0] _T_1038; // @[Compute.scala 327:49]
  wire [31:0] _T_1039; // @[Compute.scala 327:79]
  wire [31:0] add_val_0; // @[Compute.scala 307:36]
  wire [31:0] add_res_0; // @[Compute.scala 307:36]
  wire [7:0] _T_1040; // @[Compute.scala 329:37]
  wire [4:0] _T_1042; // @[Compute.scala 330:60]
  wire [31:0] _T_1043; // @[Compute.scala 330:49]
  wire [31:0] _T_1044; // @[Compute.scala 330:84]
  wire [31:0] shr_val_0; // @[Compute.scala 307:36]
  wire [31:0] shr_res_0; // @[Compute.scala 307:36]
  wire [7:0] _T_1045; // @[Compute.scala 332:37]
  wire [31:0] src_0_1; // @[Compute.scala 307:36]
  wire [31:0] src_1_1; // @[Compute.scala 307:36]
  wire  _T_1046; // @[Compute.scala 324:34]
  wire [31:0] _T_1047; // @[Compute.scala 324:24]
  wire [31:0] mix_val_1; // @[Compute.scala 307:36]
  wire [7:0] _T_1048; // @[Compute.scala 326:37]
  wire [31:0] _T_1049; // @[Compute.scala 327:30]
  wire [31:0] _T_1050; // @[Compute.scala 327:59]
  wire [32:0] _T_1051; // @[Compute.scala 327:49]
  wire [31:0] _T_1052; // @[Compute.scala 327:49]
  wire [31:0] _T_1053; // @[Compute.scala 327:79]
  wire [31:0] add_val_1; // @[Compute.scala 307:36]
  wire [31:0] add_res_1; // @[Compute.scala 307:36]
  wire [7:0] _T_1054; // @[Compute.scala 329:37]
  wire [4:0] _T_1056; // @[Compute.scala 330:60]
  wire [31:0] _T_1057; // @[Compute.scala 330:49]
  wire [31:0] _T_1058; // @[Compute.scala 330:84]
  wire [31:0] shr_val_1; // @[Compute.scala 307:36]
  wire [31:0] shr_res_1; // @[Compute.scala 307:36]
  wire [7:0] _T_1059; // @[Compute.scala 332:37]
  wire [31:0] src_0_2; // @[Compute.scala 307:36]
  wire [31:0] src_1_2; // @[Compute.scala 307:36]
  wire  _T_1060; // @[Compute.scala 324:34]
  wire [31:0] _T_1061; // @[Compute.scala 324:24]
  wire [31:0] mix_val_2; // @[Compute.scala 307:36]
  wire [7:0] _T_1062; // @[Compute.scala 326:37]
  wire [31:0] _T_1063; // @[Compute.scala 327:30]
  wire [31:0] _T_1064; // @[Compute.scala 327:59]
  wire [32:0] _T_1065; // @[Compute.scala 327:49]
  wire [31:0] _T_1066; // @[Compute.scala 327:49]
  wire [31:0] _T_1067; // @[Compute.scala 327:79]
  wire [31:0] add_val_2; // @[Compute.scala 307:36]
  wire [31:0] add_res_2; // @[Compute.scala 307:36]
  wire [7:0] _T_1068; // @[Compute.scala 329:37]
  wire [4:0] _T_1070; // @[Compute.scala 330:60]
  wire [31:0] _T_1071; // @[Compute.scala 330:49]
  wire [31:0] _T_1072; // @[Compute.scala 330:84]
  wire [31:0] shr_val_2; // @[Compute.scala 307:36]
  wire [31:0] shr_res_2; // @[Compute.scala 307:36]
  wire [7:0] _T_1073; // @[Compute.scala 332:37]
  wire [31:0] src_0_3; // @[Compute.scala 307:36]
  wire [31:0] src_1_3; // @[Compute.scala 307:36]
  wire  _T_1074; // @[Compute.scala 324:34]
  wire [31:0] _T_1075; // @[Compute.scala 324:24]
  wire [31:0] mix_val_3; // @[Compute.scala 307:36]
  wire [7:0] _T_1076; // @[Compute.scala 326:37]
  wire [31:0] _T_1077; // @[Compute.scala 327:30]
  wire [31:0] _T_1078; // @[Compute.scala 327:59]
  wire [32:0] _T_1079; // @[Compute.scala 327:49]
  wire [31:0] _T_1080; // @[Compute.scala 327:49]
  wire [31:0] _T_1081; // @[Compute.scala 327:79]
  wire [31:0] add_val_3; // @[Compute.scala 307:36]
  wire [31:0] add_res_3; // @[Compute.scala 307:36]
  wire [7:0] _T_1082; // @[Compute.scala 329:37]
  wire [4:0] _T_1084; // @[Compute.scala 330:60]
  wire [31:0] _T_1085; // @[Compute.scala 330:49]
  wire [31:0] _T_1086; // @[Compute.scala 330:84]
  wire [31:0] shr_val_3; // @[Compute.scala 307:36]
  wire [31:0] shr_res_3; // @[Compute.scala 307:36]
  wire [7:0] _T_1087; // @[Compute.scala 332:37]
  wire [31:0] src_0_4; // @[Compute.scala 307:36]
  wire [31:0] src_1_4; // @[Compute.scala 307:36]
  wire  _T_1088; // @[Compute.scala 324:34]
  wire [31:0] _T_1089; // @[Compute.scala 324:24]
  wire [31:0] mix_val_4; // @[Compute.scala 307:36]
  wire [7:0] _T_1090; // @[Compute.scala 326:37]
  wire [31:0] _T_1091; // @[Compute.scala 327:30]
  wire [31:0] _T_1092; // @[Compute.scala 327:59]
  wire [32:0] _T_1093; // @[Compute.scala 327:49]
  wire [31:0] _T_1094; // @[Compute.scala 327:49]
  wire [31:0] _T_1095; // @[Compute.scala 327:79]
  wire [31:0] add_val_4; // @[Compute.scala 307:36]
  wire [31:0] add_res_4; // @[Compute.scala 307:36]
  wire [7:0] _T_1096; // @[Compute.scala 329:37]
  wire [4:0] _T_1098; // @[Compute.scala 330:60]
  wire [31:0] _T_1099; // @[Compute.scala 330:49]
  wire [31:0] _T_1100; // @[Compute.scala 330:84]
  wire [31:0] shr_val_4; // @[Compute.scala 307:36]
  wire [31:0] shr_res_4; // @[Compute.scala 307:36]
  wire [7:0] _T_1101; // @[Compute.scala 332:37]
  wire [31:0] src_0_5; // @[Compute.scala 307:36]
  wire [31:0] src_1_5; // @[Compute.scala 307:36]
  wire  _T_1102; // @[Compute.scala 324:34]
  wire [31:0] _T_1103; // @[Compute.scala 324:24]
  wire [31:0] mix_val_5; // @[Compute.scala 307:36]
  wire [7:0] _T_1104; // @[Compute.scala 326:37]
  wire [31:0] _T_1105; // @[Compute.scala 327:30]
  wire [31:0] _T_1106; // @[Compute.scala 327:59]
  wire [32:0] _T_1107; // @[Compute.scala 327:49]
  wire [31:0] _T_1108; // @[Compute.scala 327:49]
  wire [31:0] _T_1109; // @[Compute.scala 327:79]
  wire [31:0] add_val_5; // @[Compute.scala 307:36]
  wire [31:0] add_res_5; // @[Compute.scala 307:36]
  wire [7:0] _T_1110; // @[Compute.scala 329:37]
  wire [4:0] _T_1112; // @[Compute.scala 330:60]
  wire [31:0] _T_1113; // @[Compute.scala 330:49]
  wire [31:0] _T_1114; // @[Compute.scala 330:84]
  wire [31:0] shr_val_5; // @[Compute.scala 307:36]
  wire [31:0] shr_res_5; // @[Compute.scala 307:36]
  wire [7:0] _T_1115; // @[Compute.scala 332:37]
  wire [31:0] src_0_6; // @[Compute.scala 307:36]
  wire [31:0] src_1_6; // @[Compute.scala 307:36]
  wire  _T_1116; // @[Compute.scala 324:34]
  wire [31:0] _T_1117; // @[Compute.scala 324:24]
  wire [31:0] mix_val_6; // @[Compute.scala 307:36]
  wire [7:0] _T_1118; // @[Compute.scala 326:37]
  wire [31:0] _T_1119; // @[Compute.scala 327:30]
  wire [31:0] _T_1120; // @[Compute.scala 327:59]
  wire [32:0] _T_1121; // @[Compute.scala 327:49]
  wire [31:0] _T_1122; // @[Compute.scala 327:49]
  wire [31:0] _T_1123; // @[Compute.scala 327:79]
  wire [31:0] add_val_6; // @[Compute.scala 307:36]
  wire [31:0] add_res_6; // @[Compute.scala 307:36]
  wire [7:0] _T_1124; // @[Compute.scala 329:37]
  wire [4:0] _T_1126; // @[Compute.scala 330:60]
  wire [31:0] _T_1127; // @[Compute.scala 330:49]
  wire [31:0] _T_1128; // @[Compute.scala 330:84]
  wire [31:0] shr_val_6; // @[Compute.scala 307:36]
  wire [31:0] shr_res_6; // @[Compute.scala 307:36]
  wire [7:0] _T_1129; // @[Compute.scala 332:37]
  wire [31:0] src_0_7; // @[Compute.scala 307:36]
  wire [31:0] src_1_7; // @[Compute.scala 307:36]
  wire  _T_1130; // @[Compute.scala 324:34]
  wire [31:0] _T_1131; // @[Compute.scala 324:24]
  wire [31:0] mix_val_7; // @[Compute.scala 307:36]
  wire [7:0] _T_1132; // @[Compute.scala 326:37]
  wire [31:0] _T_1133; // @[Compute.scala 327:30]
  wire [31:0] _T_1134; // @[Compute.scala 327:59]
  wire [32:0] _T_1135; // @[Compute.scala 327:49]
  wire [31:0] _T_1136; // @[Compute.scala 327:49]
  wire [31:0] _T_1137; // @[Compute.scala 327:79]
  wire [31:0] add_val_7; // @[Compute.scala 307:36]
  wire [31:0] add_res_7; // @[Compute.scala 307:36]
  wire [7:0] _T_1138; // @[Compute.scala 329:37]
  wire [4:0] _T_1140; // @[Compute.scala 330:60]
  wire [31:0] _T_1141; // @[Compute.scala 330:49]
  wire [31:0] _T_1142; // @[Compute.scala 330:84]
  wire [31:0] shr_val_7; // @[Compute.scala 307:36]
  wire [31:0] shr_res_7; // @[Compute.scala 307:36]
  wire [7:0] _T_1143; // @[Compute.scala 332:37]
  wire [31:0] src_0_8; // @[Compute.scala 307:36]
  wire [31:0] src_1_8; // @[Compute.scala 307:36]
  wire  _T_1144; // @[Compute.scala 324:34]
  wire [31:0] _T_1145; // @[Compute.scala 324:24]
  wire [31:0] mix_val_8; // @[Compute.scala 307:36]
  wire [7:0] _T_1146; // @[Compute.scala 326:37]
  wire [31:0] _T_1147; // @[Compute.scala 327:30]
  wire [31:0] _T_1148; // @[Compute.scala 327:59]
  wire [32:0] _T_1149; // @[Compute.scala 327:49]
  wire [31:0] _T_1150; // @[Compute.scala 327:49]
  wire [31:0] _T_1151; // @[Compute.scala 327:79]
  wire [31:0] add_val_8; // @[Compute.scala 307:36]
  wire [31:0] add_res_8; // @[Compute.scala 307:36]
  wire [7:0] _T_1152; // @[Compute.scala 329:37]
  wire [4:0] _T_1154; // @[Compute.scala 330:60]
  wire [31:0] _T_1155; // @[Compute.scala 330:49]
  wire [31:0] _T_1156; // @[Compute.scala 330:84]
  wire [31:0] shr_val_8; // @[Compute.scala 307:36]
  wire [31:0] shr_res_8; // @[Compute.scala 307:36]
  wire [7:0] _T_1157; // @[Compute.scala 332:37]
  wire [31:0] src_0_9; // @[Compute.scala 307:36]
  wire [31:0] src_1_9; // @[Compute.scala 307:36]
  wire  _T_1158; // @[Compute.scala 324:34]
  wire [31:0] _T_1159; // @[Compute.scala 324:24]
  wire [31:0] mix_val_9; // @[Compute.scala 307:36]
  wire [7:0] _T_1160; // @[Compute.scala 326:37]
  wire [31:0] _T_1161; // @[Compute.scala 327:30]
  wire [31:0] _T_1162; // @[Compute.scala 327:59]
  wire [32:0] _T_1163; // @[Compute.scala 327:49]
  wire [31:0] _T_1164; // @[Compute.scala 327:49]
  wire [31:0] _T_1165; // @[Compute.scala 327:79]
  wire [31:0] add_val_9; // @[Compute.scala 307:36]
  wire [31:0] add_res_9; // @[Compute.scala 307:36]
  wire [7:0] _T_1166; // @[Compute.scala 329:37]
  wire [4:0] _T_1168; // @[Compute.scala 330:60]
  wire [31:0] _T_1169; // @[Compute.scala 330:49]
  wire [31:0] _T_1170; // @[Compute.scala 330:84]
  wire [31:0] shr_val_9; // @[Compute.scala 307:36]
  wire [31:0] shr_res_9; // @[Compute.scala 307:36]
  wire [7:0] _T_1171; // @[Compute.scala 332:37]
  wire [31:0] src_0_10; // @[Compute.scala 307:36]
  wire [31:0] src_1_10; // @[Compute.scala 307:36]
  wire  _T_1172; // @[Compute.scala 324:34]
  wire [31:0] _T_1173; // @[Compute.scala 324:24]
  wire [31:0] mix_val_10; // @[Compute.scala 307:36]
  wire [7:0] _T_1174; // @[Compute.scala 326:37]
  wire [31:0] _T_1175; // @[Compute.scala 327:30]
  wire [31:0] _T_1176; // @[Compute.scala 327:59]
  wire [32:0] _T_1177; // @[Compute.scala 327:49]
  wire [31:0] _T_1178; // @[Compute.scala 327:49]
  wire [31:0] _T_1179; // @[Compute.scala 327:79]
  wire [31:0] add_val_10; // @[Compute.scala 307:36]
  wire [31:0] add_res_10; // @[Compute.scala 307:36]
  wire [7:0] _T_1180; // @[Compute.scala 329:37]
  wire [4:0] _T_1182; // @[Compute.scala 330:60]
  wire [31:0] _T_1183; // @[Compute.scala 330:49]
  wire [31:0] _T_1184; // @[Compute.scala 330:84]
  wire [31:0] shr_val_10; // @[Compute.scala 307:36]
  wire [31:0] shr_res_10; // @[Compute.scala 307:36]
  wire [7:0] _T_1185; // @[Compute.scala 332:37]
  wire [31:0] src_0_11; // @[Compute.scala 307:36]
  wire [31:0] src_1_11; // @[Compute.scala 307:36]
  wire  _T_1186; // @[Compute.scala 324:34]
  wire [31:0] _T_1187; // @[Compute.scala 324:24]
  wire [31:0] mix_val_11; // @[Compute.scala 307:36]
  wire [7:0] _T_1188; // @[Compute.scala 326:37]
  wire [31:0] _T_1189; // @[Compute.scala 327:30]
  wire [31:0] _T_1190; // @[Compute.scala 327:59]
  wire [32:0] _T_1191; // @[Compute.scala 327:49]
  wire [31:0] _T_1192; // @[Compute.scala 327:49]
  wire [31:0] _T_1193; // @[Compute.scala 327:79]
  wire [31:0] add_val_11; // @[Compute.scala 307:36]
  wire [31:0] add_res_11; // @[Compute.scala 307:36]
  wire [7:0] _T_1194; // @[Compute.scala 329:37]
  wire [4:0] _T_1196; // @[Compute.scala 330:60]
  wire [31:0] _T_1197; // @[Compute.scala 330:49]
  wire [31:0] _T_1198; // @[Compute.scala 330:84]
  wire [31:0] shr_val_11; // @[Compute.scala 307:36]
  wire [31:0] shr_res_11; // @[Compute.scala 307:36]
  wire [7:0] _T_1199; // @[Compute.scala 332:37]
  wire [31:0] src_0_12; // @[Compute.scala 307:36]
  wire [31:0] src_1_12; // @[Compute.scala 307:36]
  wire  _T_1200; // @[Compute.scala 324:34]
  wire [31:0] _T_1201; // @[Compute.scala 324:24]
  wire [31:0] mix_val_12; // @[Compute.scala 307:36]
  wire [7:0] _T_1202; // @[Compute.scala 326:37]
  wire [31:0] _T_1203; // @[Compute.scala 327:30]
  wire [31:0] _T_1204; // @[Compute.scala 327:59]
  wire [32:0] _T_1205; // @[Compute.scala 327:49]
  wire [31:0] _T_1206; // @[Compute.scala 327:49]
  wire [31:0] _T_1207; // @[Compute.scala 327:79]
  wire [31:0] add_val_12; // @[Compute.scala 307:36]
  wire [31:0] add_res_12; // @[Compute.scala 307:36]
  wire [7:0] _T_1208; // @[Compute.scala 329:37]
  wire [4:0] _T_1210; // @[Compute.scala 330:60]
  wire [31:0] _T_1211; // @[Compute.scala 330:49]
  wire [31:0] _T_1212; // @[Compute.scala 330:84]
  wire [31:0] shr_val_12; // @[Compute.scala 307:36]
  wire [31:0] shr_res_12; // @[Compute.scala 307:36]
  wire [7:0] _T_1213; // @[Compute.scala 332:37]
  wire [31:0] src_0_13; // @[Compute.scala 307:36]
  wire [31:0] src_1_13; // @[Compute.scala 307:36]
  wire  _T_1214; // @[Compute.scala 324:34]
  wire [31:0] _T_1215; // @[Compute.scala 324:24]
  wire [31:0] mix_val_13; // @[Compute.scala 307:36]
  wire [7:0] _T_1216; // @[Compute.scala 326:37]
  wire [31:0] _T_1217; // @[Compute.scala 327:30]
  wire [31:0] _T_1218; // @[Compute.scala 327:59]
  wire [32:0] _T_1219; // @[Compute.scala 327:49]
  wire [31:0] _T_1220; // @[Compute.scala 327:49]
  wire [31:0] _T_1221; // @[Compute.scala 327:79]
  wire [31:0] add_val_13; // @[Compute.scala 307:36]
  wire [31:0] add_res_13; // @[Compute.scala 307:36]
  wire [7:0] _T_1222; // @[Compute.scala 329:37]
  wire [4:0] _T_1224; // @[Compute.scala 330:60]
  wire [31:0] _T_1225; // @[Compute.scala 330:49]
  wire [31:0] _T_1226; // @[Compute.scala 330:84]
  wire [31:0] shr_val_13; // @[Compute.scala 307:36]
  wire [31:0] shr_res_13; // @[Compute.scala 307:36]
  wire [7:0] _T_1227; // @[Compute.scala 332:37]
  wire [31:0] src_0_14; // @[Compute.scala 307:36]
  wire [31:0] src_1_14; // @[Compute.scala 307:36]
  wire  _T_1228; // @[Compute.scala 324:34]
  wire [31:0] _T_1229; // @[Compute.scala 324:24]
  wire [31:0] mix_val_14; // @[Compute.scala 307:36]
  wire [7:0] _T_1230; // @[Compute.scala 326:37]
  wire [31:0] _T_1231; // @[Compute.scala 327:30]
  wire [31:0] _T_1232; // @[Compute.scala 327:59]
  wire [32:0] _T_1233; // @[Compute.scala 327:49]
  wire [31:0] _T_1234; // @[Compute.scala 327:49]
  wire [31:0] _T_1235; // @[Compute.scala 327:79]
  wire [31:0] add_val_14; // @[Compute.scala 307:36]
  wire [31:0] add_res_14; // @[Compute.scala 307:36]
  wire [7:0] _T_1236; // @[Compute.scala 329:37]
  wire [4:0] _T_1238; // @[Compute.scala 330:60]
  wire [31:0] _T_1239; // @[Compute.scala 330:49]
  wire [31:0] _T_1240; // @[Compute.scala 330:84]
  wire [31:0] shr_val_14; // @[Compute.scala 307:36]
  wire [31:0] shr_res_14; // @[Compute.scala 307:36]
  wire [7:0] _T_1241; // @[Compute.scala 332:37]
  wire [31:0] src_0_15; // @[Compute.scala 307:36]
  wire [31:0] src_1_15; // @[Compute.scala 307:36]
  wire  _T_1242; // @[Compute.scala 324:34]
  wire [31:0] _T_1243; // @[Compute.scala 324:24]
  wire [31:0] mix_val_15; // @[Compute.scala 307:36]
  wire [7:0] _T_1244; // @[Compute.scala 326:37]
  wire [31:0] _T_1245; // @[Compute.scala 327:30]
  wire [31:0] _T_1246; // @[Compute.scala 327:59]
  wire [32:0] _T_1247; // @[Compute.scala 327:49]
  wire [31:0] _T_1248; // @[Compute.scala 327:49]
  wire [31:0] _T_1249; // @[Compute.scala 327:79]
  wire [31:0] add_val_15; // @[Compute.scala 307:36]
  wire [31:0] add_res_15; // @[Compute.scala 307:36]
  wire [7:0] _T_1250; // @[Compute.scala 329:37]
  wire [4:0] _T_1252; // @[Compute.scala 330:60]
  wire [31:0] _T_1253; // @[Compute.scala 330:49]
  wire [31:0] _T_1254; // @[Compute.scala 330:84]
  wire [31:0] shr_val_15; // @[Compute.scala 307:36]
  wire [31:0] shr_res_15; // @[Compute.scala 307:36]
  wire [7:0] _T_1255; // @[Compute.scala 332:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 307:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 307:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 307:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 307:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 337:48]
  wire  alu_opcode_add_en; // @[Compute.scala 338:39]
  wire  _T_1257; // @[Compute.scala 339:34]
  wire  _T_1259; // @[Compute.scala 339:45]
  wire  _T_1260; // @[Compute.scala 339:42]
  wire [44:0] _T_1262; // @[Compute.scala 340:58]
  wire [44:0] _T_1263; // @[Compute.scala 340:58]
  wire [43:0] _T_1264; // @[Compute.scala 340:58]
  wire  _T_1265; // @[Compute.scala 340:40]
  wire  _T_1266; // @[Compute.scala 340:23]
  wire  _T_1269; // @[Compute.scala 340:66]
  wire  _GEN_290; // @[Compute.scala 340:85]
  reg [31:0] out_mem_address; // @[Compute.scala 343:28]
  reg [31:0] _RAND_24;
  wire [38:0] _GEN_344; // @[Compute.scala 347:41]
  wire [38:0] _T_1275; // @[Compute.scala 347:41]
  wire [63:0] _T_1282; // @[Cat.scala 30:58]
  wire [127:0] _T_1290; // @[Cat.scala 30:58]
  wire [63:0] _T_1297; // @[Cat.scala 30:58]
  wire [127:0] _T_1305; // @[Cat.scala 30:58]
  wire [63:0] _T_1312; // @[Cat.scala 30:58]
  wire [127:0] _T_1320; // @[Cat.scala 30:58]
  wire [127:0] _T_1321; // @[Compute.scala 351:8]
  assign acc_mem__T_469_addr = dst_idx[7:0];
  assign acc_mem__T_469_data = acc_mem[acc_mem__T_469_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_471_addr = src_idx[7:0];
  assign acc_mem__T_471_data = acc_mem[acc_mem__T_471_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_449_data = {_T_451,_T_450};
  assign acc_mem__T_449_addr = acc_sram_addr[7:0];
  assign acc_mem__T_449_mask = 1'h1;
  assign acc_mem__T_449_en = _T_336 ? _T_447 : 1'h0;
  assign uop_mem_uop_addr = out_cntr_val[9:0];
  assign uop_mem_uop_data = uop_mem[uop_mem_uop_addr]; // @[Compute.scala 34:20]
  assign uop_mem__T_382_data = io_uops_readdata[31:0];
  assign uop_mem__T_382_addr = _T_380[9:0];
  assign uop_mem__T_382_mask = 1'h1;
  assign uop_mem__T_382_en = uops_read & _T_326;
  assign uop_mem__T_388_data = io_uops_readdata[63:32];
  assign uop_mem__T_388_addr = _T_386[9:0];
  assign uop_mem__T_388_mask = 1'h1;
  assign uop_mem__T_388_en = uops_read & _T_326;
  assign uop_mem__T_394_data = io_uops_readdata[95:64];
  assign uop_mem__T_394_addr = _T_392[9:0];
  assign uop_mem__T_394_mask = 1'h1;
  assign uop_mem__T_394_en = uops_read & _T_326;
  assign uop_mem__T_400_data = io_uops_readdata[127:96];
  assign uop_mem__T_400_addr = _T_398[9:0];
  assign uop_mem__T_400_mask = 1'h1;
  assign uop_mem__T_400_en = uops_read & _T_326;
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
  assign x_size = insn[95:80]; // @[Compute.scala 49:25]
  assign y_pad_0 = insn[115:112]; // @[Compute.scala 51:25]
  assign x_pad_0 = insn[123:120]; // @[Compute.scala 53:25]
  assign x_pad_1 = insn[127:124]; // @[Compute.scala 54:25]
  assign _T_203 = insn[20:8]; // @[Compute.scala 58:18]
  assign _T_205 = insn[34:21]; // @[Compute.scala 60:18]
  assign iter_out = insn[48:35]; // @[Compute.scala 61:22]
  assign iter_in = insn[62:49]; // @[Compute.scala 62:21]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 69:24]
  assign use_imm = insn[110]; // @[Compute.scala 70:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 71:21]
  assign _T_206 = $signed(imm_raw); // @[Compute.scala 72:25]
  assign _T_208 = $signed(_T_206) < $signed(16'sh0); // @[Compute.scala 72:32]
  assign _T_210 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_212 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_213 = _T_208 ? _T_210 : {{15'd0}, _T_212}; // @[Compute.scala 72:16]
  assign imm = $signed(_T_213); // @[Compute.scala 72:89]
  assign _GEN_318 = {{12'd0}, y_pad_0}; // @[Compute.scala 76:30]
  assign _GEN_320 = {{12'd0}, x_pad_0}; // @[Compute.scala 77:30]
  assign _T_217 = _GEN_320 + x_size; // @[Compute.scala 77:30]
  assign _T_218 = _GEN_320 + x_size; // @[Compute.scala 77:30]
  assign _GEN_321 = {{12'd0}, x_pad_1}; // @[Compute.scala 77:39]
  assign _T_219 = _T_218 + _GEN_321; // @[Compute.scala 77:39]
  assign x_size_total = _T_218 + _GEN_321; // @[Compute.scala 77:39]
  assign y_offset = x_size_total * _GEN_318; // @[Compute.scala 78:31]
  assign opcode_finish_en = opcode == 3'h3; // @[Compute.scala 81:34]
  assign _T_222 = opcode == 3'h0; // @[Compute.scala 82:32]
  assign _T_224 = opcode == 3'h1; // @[Compute.scala 82:60]
  assign opcode_load_en = _T_222 | _T_224; // @[Compute.scala 82:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Compute.scala 83:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Compute.scala 84:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Compute.scala 86:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Compute.scala 87:40]
  assign idle = state == 3'h0; // @[Compute.scala 92:20]
  assign dump = state == 3'h1; // @[Compute.scala 93:20]
  assign busy = state == 3'h2; // @[Compute.scala 94:20]
  assign push = state == 3'h3; // @[Compute.scala 95:20]
  assign done = state == 3'h4; // @[Compute.scala 96:20]
  assign uop_cntr_max_val = x_size >> 2'h2; // @[Compute.scala 110:33]
  assign _T_248 = uop_cntr_max_val == 16'h0; // @[Compute.scala 111:43]
  assign uop_cntr_max = _T_248 ? 16'h1 : uop_cntr_max_val; // @[Compute.scala 111:25]
  assign _T_250 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 112:37]
  assign uop_cntr_en = _T_250 & insn_valid; // @[Compute.scala 112:59]
  assign _T_252 = uop_cntr_val == uop_cntr_max; // @[Compute.scala 115:38]
  assign _T_253 = _T_252 & uop_cntr_en; // @[Compute.scala 115:56]
  assign uop_cntr_wrap = _T_253 & busy; // @[Compute.scala 115:71]
  assign _T_255 = x_size * 16'h4; // @[Compute.scala 117:29]
  assign _T_257 = _T_255 + 19'h1; // @[Compute.scala 117:46]
  assign acc_cntr_max = _T_255 + 19'h1; // @[Compute.scala 117:46]
  assign _T_258 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 118:37]
  assign acc_cntr_en = _T_258 & insn_valid; // @[Compute.scala 118:59]
  assign _GEN_323 = {{3'd0}, acc_cntr_val}; // @[Compute.scala 121:38]
  assign _T_260 = _GEN_323 == acc_cntr_max; // @[Compute.scala 121:38]
  assign _T_261 = _T_260 & acc_cntr_en; // @[Compute.scala 121:56]
  assign acc_cntr_wrap = _T_261 & busy; // @[Compute.scala 121:71]
  assign _T_262 = uop_end - uop_bgn; // @[Compute.scala 123:34]
  assign _T_263 = $unsigned(_T_262); // @[Compute.scala 123:34]
  assign upc_cntr_max_val = _T_263[15:0]; // @[Compute.scala 123:34]
  assign _T_265 = upc_cntr_max_val <= 16'h0; // @[Compute.scala 124:43]
  assign upc_cntr_max = _T_265 ? 16'h1 : upc_cntr_max_val; // @[Compute.scala 124:25]
  assign _T_267 = iter_in * iter_out; // @[Compute.scala 125:30]
  assign _GEN_324 = {{12'd0}, upc_cntr_max}; // @[Compute.scala 125:41]
  assign _T_268 = _T_267 * _GEN_324; // @[Compute.scala 125:41]
  assign _T_270 = _T_268 + 44'h1; // @[Compute.scala 125:56]
  assign out_cntr_max = _T_268 + 44'h1; // @[Compute.scala 125:56]
  assign _T_271 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 126:37]
  assign out_cntr_en = _T_271 & insn_valid; // @[Compute.scala 126:56]
  assign _GEN_325 = {{28'd0}, out_cntr_val}; // @[Compute.scala 129:38]
  assign _T_273 = _GEN_325 == out_cntr_max; // @[Compute.scala 129:38]
  assign _T_274 = _T_273 & out_cntr_en; // @[Compute.scala 129:56]
  assign out_cntr_wrap = _T_274 & busy; // @[Compute.scala 129:71]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Compute.scala 134:43]
  assign push_next_dep_valid = push_next_dep & push; // @[Compute.scala 135:43]
  assign _T_287 = pop_prev_dep_ready & busy; // @[Compute.scala 144:68]
  assign _T_288 = pop_next_dep_ready & busy; // @[Compute.scala 145:68]
  assign _T_289 = push_prev_dep_ready & busy; // @[Compute.scala 146:68]
  assign _T_290 = push_next_dep_ready & busy; // @[Compute.scala 147:68]
  assign _GEN_0 = push_next_dep ? _T_290 : 1'h0; // @[Compute.scala 147:31]
  assign _GEN_1 = push_prev_dep ? _T_289 : _GEN_0; // @[Compute.scala 146:31]
  assign _GEN_2 = pop_next_dep ? _T_288 : _GEN_1; // @[Compute.scala 145:31]
  assign _GEN_3 = pop_prev_dep ? _T_287 : _GEN_2; // @[Compute.scala 144:31]
  assign _GEN_4 = opcode_finish_en ? _GEN_3 : 1'h0; // @[Compute.scala 143:27]
  assign _T_293 = uop_cntr_wrap | acc_cntr_wrap; // @[Compute.scala 150:23]
  assign _T_294 = _T_293 | out_cntr_wrap; // @[Compute.scala 150:40]
  assign _T_295 = _T_294 | finish_wrap; // @[Compute.scala 150:57]
  assign _T_296 = push_prev_dep | push_next_dep; // @[Compute.scala 151:25]
  assign _GEN_5 = _T_296 ? 3'h3 : 3'h4; // @[Compute.scala 151:43]
  assign _GEN_6 = _T_295 ? _GEN_5 : state; // @[Compute.scala 150:73]
  assign _T_298 = pop_prev_dep_ready == 1'h0; // @[Compute.scala 159:18]
  assign _T_300 = pop_next_dep_ready == 1'h0; // @[Compute.scala 159:41]
  assign _T_301 = _T_298 & _T_300; // @[Compute.scala 159:38]
  assign _T_302 = busy & _T_301; // @[Compute.scala 159:14]
  assign _T_303 = pop_prev_dep | pop_next_dep; // @[Compute.scala 159:79]
  assign _T_304 = _T_302 & _T_303; // @[Compute.scala 159:62]
  assign _GEN_7 = _T_304 ? 3'h1 : _GEN_6; // @[Compute.scala 159:97]
  assign _T_305 = pop_prev_dep_ready | pop_next_dep_ready; // @[Compute.scala 160:38]
  assign _T_306 = dump & _T_305; // @[Compute.scala 160:14]
  assign _GEN_8 = _T_306 ? 3'h2 : _GEN_7; // @[Compute.scala 160:63]
  assign _T_307 = push_prev_dep_ready | push_next_dep_ready; // @[Compute.scala 161:38]
  assign _T_308 = push & _T_307; // @[Compute.scala 161:14]
  assign _GEN_9 = _T_308 ? 3'h4 : _GEN_8; // @[Compute.scala 161:63]
  assign _T_311 = pop_prev_dep & dump; // @[Compute.scala 168:22]
  assign _T_312 = _T_311 & io_l2g_dep_queue_valid; // @[Compute.scala 168:30]
  assign _GEN_10 = _T_312 ? 1'h1 : pop_prev_dep_ready; // @[Compute.scala 168:57]
  assign _T_314 = pop_next_dep & dump; // @[Compute.scala 171:22]
  assign _T_315 = _T_314 & io_s2g_dep_queue_valid; // @[Compute.scala 171:30]
  assign _GEN_11 = _T_315 ? 1'h1 : pop_next_dep_ready; // @[Compute.scala 171:57]
  assign _T_319 = push_prev_dep_valid & io_g2l_dep_queue_ready; // @[Compute.scala 178:29]
  assign _T_320 = _T_319 & push; // @[Compute.scala 178:55]
  assign _GEN_12 = _T_320 ? 1'h1 : push_prev_dep_ready; // @[Compute.scala 178:64]
  assign _T_322 = push_next_dep_valid & io_g2s_dep_queue_ready; // @[Compute.scala 181:29]
  assign _T_323 = _T_322 & push; // @[Compute.scala 181:55]
  assign _GEN_13 = _T_323 ? 1'h1 : push_next_dep_ready; // @[Compute.scala 181:64]
  assign _T_326 = io_uops_waitrequest == 1'h0; // @[Compute.scala 186:22]
  assign _T_327 = uops_read & _T_326; // @[Compute.scala 186:19]
  assign _T_328 = _T_327 & busy; // @[Compute.scala 186:37]
  assign _T_329 = uop_cntr_val < uop_cntr_max; // @[Compute.scala 186:61]
  assign _T_330 = _T_328 & _T_329; // @[Compute.scala 186:45]
  assign _T_332 = uop_cntr_val + 16'h1; // @[Compute.scala 187:34]
  assign _T_333 = uop_cntr_val + 16'h1; // @[Compute.scala 187:34]
  assign _GEN_14 = _T_330 ? _T_333 : uop_cntr_val; // @[Compute.scala 186:77]
  assign _T_335 = io_biases_waitrequest == 1'h0; // @[Compute.scala 189:24]
  assign _T_336 = biases_read & _T_335; // @[Compute.scala 189:21]
  assign _T_337 = _T_336 & busy; // @[Compute.scala 189:39]
  assign _T_338 = _GEN_323 < acc_cntr_max; // @[Compute.scala 189:63]
  assign _T_339 = _T_337 & _T_338; // @[Compute.scala 189:47]
  assign _T_341 = acc_cntr_val + 16'h1; // @[Compute.scala 190:34]
  assign _T_342 = acc_cntr_val + 16'h1; // @[Compute.scala 190:34]
  assign _GEN_15 = _T_339 ? _T_342 : acc_cntr_val; // @[Compute.scala 189:79]
  assign _T_344 = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 192:26]
  assign _T_345 = out_mem_write & _T_344; // @[Compute.scala 192:23]
  assign _T_346 = _T_345 & busy; // @[Compute.scala 192:41]
  assign _T_347 = _GEN_325 < out_cntr_max; // @[Compute.scala 192:65]
  assign _T_348 = _T_346 & _T_347; // @[Compute.scala 192:49]
  assign _T_350 = out_cntr_val + 16'h1; // @[Compute.scala 193:34]
  assign _T_351 = out_cntr_val + 16'h1; // @[Compute.scala 193:34]
  assign _GEN_16 = _T_348 ? _T_351 : out_cntr_val; // @[Compute.scala 192:81]
  assign _GEN_21 = gemm_queue_ready ? 1'h0 : _GEN_10; // @[Compute.scala 197:27]
  assign _GEN_22 = gemm_queue_ready ? 1'h0 : _GEN_11; // @[Compute.scala 197:27]
  assign _GEN_23 = gemm_queue_ready ? 1'h0 : _GEN_12; // @[Compute.scala 197:27]
  assign _GEN_24 = gemm_queue_ready ? 1'h0 : _GEN_13; // @[Compute.scala 197:27]
  assign _GEN_25 = gemm_queue_ready ? 3'h2 : _GEN_9; // @[Compute.scala 197:27]
  assign _T_359 = idle | done; // @[Compute.scala 210:52]
  assign _T_360 = io_gemm_queue_valid & _T_359; // @[Compute.scala 210:43]
  assign _GEN_26 = gemm_queue_ready ? 1'h0 : _T_360; // @[Compute.scala 212:27]
  assign _GEN_328 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 222:33]
  assign _T_365 = dram_base + _GEN_328; // @[Compute.scala 222:33]
  assign _T_366 = dram_base + _GEN_328; // @[Compute.scala 222:33]
  assign _GEN_329 = {{7'd0}, _T_366}; // @[Compute.scala 222:49]
  assign uop_dram_addr = _GEN_329 << 3'h4; // @[Compute.scala 222:49]
  assign _T_368 = sram_base + uop_cntr_val; // @[Compute.scala 223:33]
  assign _T_369 = sram_base + uop_cntr_val; // @[Compute.scala 223:33]
  assign _GEN_330 = {{3'd0}, _T_369}; // @[Compute.scala 223:49]
  assign uop_sram_addr = _GEN_330 << 2'h2; // @[Compute.scala 223:49]
  assign _T_372 = uop_cntr_wrap == 1'h0; // @[Compute.scala 224:31]
  assign _T_373 = uop_cntr_en & _T_372; // @[Compute.scala 224:28]
  assign _T_374 = _T_373 & busy; // @[Compute.scala 224:46]
  assign _T_379 = {{1'd0}, uop_sram_addr}; // @[Compute.scala 232:29]
  assign _T_380 = _T_379[18:0]; // @[Compute.scala 232:29]
  assign _T_385 = uop_sram_addr + 19'h1; // @[Compute.scala 232:29]
  assign _T_386 = uop_sram_addr + 19'h1; // @[Compute.scala 232:29]
  assign _T_391 = uop_sram_addr + 19'h2; // @[Compute.scala 232:29]
  assign _T_392 = uop_sram_addr + 19'h2; // @[Compute.scala 232:29]
  assign _T_397 = uop_sram_addr + 19'h3; // @[Compute.scala 232:29]
  assign _T_398 = uop_sram_addr + 19'h3; // @[Compute.scala 232:29]
  assign _T_403 = uop_cntr_max - 16'h1; // @[Compute.scala 234:42]
  assign _T_404 = $unsigned(_T_403); // @[Compute.scala 234:42]
  assign _T_405 = _T_404[15:0]; // @[Compute.scala 234:42]
  assign _T_406 = uop_cntr_val == _T_405; // @[Compute.scala 234:24]
  assign _GEN_27 = _T_406 ? 1'h0 : _T_374; // @[Compute.scala 234:50]
  assign _GEN_331 = {{12'd0}, y_offset}; // @[Compute.scala 238:36]
  assign _T_408 = dram_base + _GEN_331; // @[Compute.scala 238:36]
  assign _T_409 = dram_base + _GEN_331; // @[Compute.scala 238:36]
  assign _GEN_332 = {{28'd0}, x_pad_0}; // @[Compute.scala 238:47]
  assign _T_410 = _T_409 + _GEN_332; // @[Compute.scala 238:47]
  assign _T_411 = _T_409 + _GEN_332; // @[Compute.scala 238:47]
  assign _GEN_333 = {{3'd0}, _T_411}; // @[Compute.scala 238:58]
  assign _T_413 = _GEN_333 << 2'h2; // @[Compute.scala 238:58]
  assign _T_415 = _T_413 * 35'h1; // @[Compute.scala 238:66]
  assign _GEN_334 = {{20'd0}, acc_cntr_val}; // @[Compute.scala 238:76]
  assign _T_416 = _T_415 + _GEN_334; // @[Compute.scala 238:76]
  assign _T_417 = _T_415 + _GEN_334; // @[Compute.scala 238:76]
  assign _GEN_335 = {{7'd0}, _T_417}; // @[Compute.scala 238:92]
  assign acc_dram_addr = _GEN_335 << 3'h4; // @[Compute.scala 238:92]
  assign _GEN_336 = {{4'd0}, sram_base}; // @[Compute.scala 239:36]
  assign _T_419 = _GEN_336 + y_offset; // @[Compute.scala 239:36]
  assign _T_420 = _GEN_336 + y_offset; // @[Compute.scala 239:36]
  assign _GEN_337 = {{16'd0}, x_pad_0}; // @[Compute.scala 239:47]
  assign _T_421 = _T_420 + _GEN_337; // @[Compute.scala 239:47]
  assign _T_422 = _T_420 + _GEN_337; // @[Compute.scala 239:47]
  assign _GEN_338 = {{3'd0}, _T_422}; // @[Compute.scala 239:58]
  assign _T_424 = _GEN_338 << 2'h2; // @[Compute.scala 239:58]
  assign _T_426 = _T_424 * 23'h1; // @[Compute.scala 239:66]
  assign _GEN_339 = {{8'd0}, acc_cntr_val}; // @[Compute.scala 239:76]
  assign _T_427 = _T_426 + _GEN_339; // @[Compute.scala 239:76]
  assign _T_428 = _T_426 + _GEN_339; // @[Compute.scala 239:76]
  assign _T_430 = _T_428 >> 2'h2; // @[Compute.scala 239:92]
  assign _T_432 = _T_430 - 24'h1; // @[Compute.scala 239:121]
  assign _T_433 = $unsigned(_T_432); // @[Compute.scala 239:121]
  assign acc_sram_addr = _T_433[23:0]; // @[Compute.scala 239:121]
  assign _T_435 = done == 1'h0; // @[Compute.scala 240:33]
  assign _GEN_17 = acc_cntr_val % 16'h4; // @[Compute.scala 246:30]
  assign _T_441 = _GEN_17[2:0]; // @[Compute.scala 246:30]
  assign _GEN_40 = 3'h0 == _T_441 ? io_biases_readdata : biases_data_0; // @[Compute.scala 246:48]
  assign _GEN_41 = 3'h1 == _T_441 ? io_biases_readdata : biases_data_1; // @[Compute.scala 246:48]
  assign _GEN_42 = 3'h2 == _T_441 ? io_biases_readdata : biases_data_2; // @[Compute.scala 246:48]
  assign _GEN_43 = 3'h3 == _T_441 ? io_biases_readdata : biases_data_3; // @[Compute.scala 246:48]
  assign _T_447 = _T_441 == 3'h0; // @[Compute.scala 250:43]
  assign _T_450 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_451 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign _GEN_340 = {{12'd0}, out_cntr_val}; // @[Compute.scala 261:30]
  assign _GEN_18 = _GEN_340 % _T_267; // @[Compute.scala 261:30]
  assign it_in = _GEN_18[15:0]; // @[Compute.scala 261:30]
  assign _T_455 = it_in * 16'h1; // @[Compute.scala 262:46]
  assign _T_456 = {{1'd0}, _T_455}; // @[Compute.scala 262:38]
  assign dst_offset_in = _T_456[31:0]; // @[Compute.scala 262:38]
  assign _T_459 = uop_mem_uop_data[10:0]; // @[Compute.scala 264:20]
  assign _GEN_341 = {{21'd0}, _T_459}; // @[Compute.scala 264:47]
  assign _T_460 = _GEN_341 + dst_offset_in; // @[Compute.scala 264:47]
  assign dst_idx = _GEN_341 + dst_offset_in; // @[Compute.scala 264:47]
  assign _T_461 = uop_mem_uop_data[21:11]; // @[Compute.scala 265:20]
  assign _GEN_342 = {{21'd0}, _T_461}; // @[Compute.scala 265:47]
  assign _T_462 = _GEN_342 + dst_offset_in; // @[Compute.scala 265:47]
  assign src_idx = _GEN_342 + dst_offset_in; // @[Compute.scala 265:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 287:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 288:38]
  assign _T_903 = insn_valid & out_cntr_en; // @[Compute.scala 307:20]
  assign _T_904 = src_vector[31:0]; // @[Compute.scala 310:31]
  assign _T_905 = $signed(_T_904); // @[Compute.scala 310:72]
  assign _T_906 = dst_vector[31:0]; // @[Compute.scala 311:31]
  assign _T_907 = $signed(_T_906); // @[Compute.scala 311:72]
  assign _T_908 = src_vector[63:32]; // @[Compute.scala 310:31]
  assign _T_909 = $signed(_T_908); // @[Compute.scala 310:72]
  assign _T_910 = dst_vector[63:32]; // @[Compute.scala 311:31]
  assign _T_911 = $signed(_T_910); // @[Compute.scala 311:72]
  assign _T_912 = src_vector[95:64]; // @[Compute.scala 310:31]
  assign _T_913 = $signed(_T_912); // @[Compute.scala 310:72]
  assign _T_914 = dst_vector[95:64]; // @[Compute.scala 311:31]
  assign _T_915 = $signed(_T_914); // @[Compute.scala 311:72]
  assign _T_916 = src_vector[127:96]; // @[Compute.scala 310:31]
  assign _T_917 = $signed(_T_916); // @[Compute.scala 310:72]
  assign _T_918 = dst_vector[127:96]; // @[Compute.scala 311:31]
  assign _T_919 = $signed(_T_918); // @[Compute.scala 311:72]
  assign _T_920 = src_vector[159:128]; // @[Compute.scala 310:31]
  assign _T_921 = $signed(_T_920); // @[Compute.scala 310:72]
  assign _T_922 = dst_vector[159:128]; // @[Compute.scala 311:31]
  assign _T_923 = $signed(_T_922); // @[Compute.scala 311:72]
  assign _T_924 = src_vector[191:160]; // @[Compute.scala 310:31]
  assign _T_925 = $signed(_T_924); // @[Compute.scala 310:72]
  assign _T_926 = dst_vector[191:160]; // @[Compute.scala 311:31]
  assign _T_927 = $signed(_T_926); // @[Compute.scala 311:72]
  assign _T_928 = src_vector[223:192]; // @[Compute.scala 310:31]
  assign _T_929 = $signed(_T_928); // @[Compute.scala 310:72]
  assign _T_930 = dst_vector[223:192]; // @[Compute.scala 311:31]
  assign _T_931 = $signed(_T_930); // @[Compute.scala 311:72]
  assign _T_932 = src_vector[255:224]; // @[Compute.scala 310:31]
  assign _T_933 = $signed(_T_932); // @[Compute.scala 310:72]
  assign _T_934 = dst_vector[255:224]; // @[Compute.scala 311:31]
  assign _T_935 = $signed(_T_934); // @[Compute.scala 311:72]
  assign _T_936 = src_vector[287:256]; // @[Compute.scala 310:31]
  assign _T_937 = $signed(_T_936); // @[Compute.scala 310:72]
  assign _T_938 = dst_vector[287:256]; // @[Compute.scala 311:31]
  assign _T_939 = $signed(_T_938); // @[Compute.scala 311:72]
  assign _T_940 = src_vector[319:288]; // @[Compute.scala 310:31]
  assign _T_941 = $signed(_T_940); // @[Compute.scala 310:72]
  assign _T_942 = dst_vector[319:288]; // @[Compute.scala 311:31]
  assign _T_943 = $signed(_T_942); // @[Compute.scala 311:72]
  assign _T_944 = src_vector[351:320]; // @[Compute.scala 310:31]
  assign _T_945 = $signed(_T_944); // @[Compute.scala 310:72]
  assign _T_946 = dst_vector[351:320]; // @[Compute.scala 311:31]
  assign _T_947 = $signed(_T_946); // @[Compute.scala 311:72]
  assign _T_948 = src_vector[383:352]; // @[Compute.scala 310:31]
  assign _T_949 = $signed(_T_948); // @[Compute.scala 310:72]
  assign _T_950 = dst_vector[383:352]; // @[Compute.scala 311:31]
  assign _T_951 = $signed(_T_950); // @[Compute.scala 311:72]
  assign _T_952 = src_vector[415:384]; // @[Compute.scala 310:31]
  assign _T_953 = $signed(_T_952); // @[Compute.scala 310:72]
  assign _T_954 = dst_vector[415:384]; // @[Compute.scala 311:31]
  assign _T_955 = $signed(_T_954); // @[Compute.scala 311:72]
  assign _T_956 = src_vector[447:416]; // @[Compute.scala 310:31]
  assign _T_957 = $signed(_T_956); // @[Compute.scala 310:72]
  assign _T_958 = dst_vector[447:416]; // @[Compute.scala 311:31]
  assign _T_959 = $signed(_T_958); // @[Compute.scala 311:72]
  assign _T_960 = src_vector[479:448]; // @[Compute.scala 310:31]
  assign _T_961 = $signed(_T_960); // @[Compute.scala 310:72]
  assign _T_962 = dst_vector[479:448]; // @[Compute.scala 311:31]
  assign _T_963 = $signed(_T_962); // @[Compute.scala 311:72]
  assign _T_964 = src_vector[511:480]; // @[Compute.scala 310:31]
  assign _T_965 = $signed(_T_964); // @[Compute.scala 310:72]
  assign _T_966 = dst_vector[511:480]; // @[Compute.scala 311:31]
  assign _T_967 = $signed(_T_966); // @[Compute.scala 311:72]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_905) : $signed(_T_907); // @[Compute.scala 308:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_907) : $signed(_T_905); // @[Compute.scala 308:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_909) : $signed(_T_911); // @[Compute.scala 308:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_911) : $signed(_T_909); // @[Compute.scala 308:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_913) : $signed(_T_915); // @[Compute.scala 308:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_915) : $signed(_T_913); // @[Compute.scala 308:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_917) : $signed(_T_919); // @[Compute.scala 308:30]
  assign _GEN_73 = alu_opcode_max_en ? $signed(_T_919) : $signed(_T_917); // @[Compute.scala 308:30]
  assign _GEN_74 = alu_opcode_max_en ? $signed(_T_921) : $signed(_T_923); // @[Compute.scala 308:30]
  assign _GEN_75 = alu_opcode_max_en ? $signed(_T_923) : $signed(_T_921); // @[Compute.scala 308:30]
  assign _GEN_76 = alu_opcode_max_en ? $signed(_T_925) : $signed(_T_927); // @[Compute.scala 308:30]
  assign _GEN_77 = alu_opcode_max_en ? $signed(_T_927) : $signed(_T_925); // @[Compute.scala 308:30]
  assign _GEN_78 = alu_opcode_max_en ? $signed(_T_929) : $signed(_T_931); // @[Compute.scala 308:30]
  assign _GEN_79 = alu_opcode_max_en ? $signed(_T_931) : $signed(_T_929); // @[Compute.scala 308:30]
  assign _GEN_80 = alu_opcode_max_en ? $signed(_T_933) : $signed(_T_935); // @[Compute.scala 308:30]
  assign _GEN_81 = alu_opcode_max_en ? $signed(_T_935) : $signed(_T_933); // @[Compute.scala 308:30]
  assign _GEN_82 = alu_opcode_max_en ? $signed(_T_937) : $signed(_T_939); // @[Compute.scala 308:30]
  assign _GEN_83 = alu_opcode_max_en ? $signed(_T_939) : $signed(_T_937); // @[Compute.scala 308:30]
  assign _GEN_84 = alu_opcode_max_en ? $signed(_T_941) : $signed(_T_943); // @[Compute.scala 308:30]
  assign _GEN_85 = alu_opcode_max_en ? $signed(_T_943) : $signed(_T_941); // @[Compute.scala 308:30]
  assign _GEN_86 = alu_opcode_max_en ? $signed(_T_945) : $signed(_T_947); // @[Compute.scala 308:30]
  assign _GEN_87 = alu_opcode_max_en ? $signed(_T_947) : $signed(_T_945); // @[Compute.scala 308:30]
  assign _GEN_88 = alu_opcode_max_en ? $signed(_T_949) : $signed(_T_951); // @[Compute.scala 308:30]
  assign _GEN_89 = alu_opcode_max_en ? $signed(_T_951) : $signed(_T_949); // @[Compute.scala 308:30]
  assign _GEN_90 = alu_opcode_max_en ? $signed(_T_953) : $signed(_T_955); // @[Compute.scala 308:30]
  assign _GEN_91 = alu_opcode_max_en ? $signed(_T_955) : $signed(_T_953); // @[Compute.scala 308:30]
  assign _GEN_92 = alu_opcode_max_en ? $signed(_T_957) : $signed(_T_959); // @[Compute.scala 308:30]
  assign _GEN_93 = alu_opcode_max_en ? $signed(_T_959) : $signed(_T_957); // @[Compute.scala 308:30]
  assign _GEN_94 = alu_opcode_max_en ? $signed(_T_961) : $signed(_T_963); // @[Compute.scala 308:30]
  assign _GEN_95 = alu_opcode_max_en ? $signed(_T_963) : $signed(_T_961); // @[Compute.scala 308:30]
  assign _GEN_96 = alu_opcode_max_en ? $signed(_T_965) : $signed(_T_967); // @[Compute.scala 308:30]
  assign _GEN_97 = alu_opcode_max_en ? $signed(_T_967) : $signed(_T_965); // @[Compute.scala 308:30]
  assign _GEN_98 = use_imm ? $signed(imm) : $signed(_GEN_67); // @[Compute.scala 319:20]
  assign _GEN_99 = use_imm ? $signed(imm) : $signed(_GEN_69); // @[Compute.scala 319:20]
  assign _GEN_100 = use_imm ? $signed(imm) : $signed(_GEN_71); // @[Compute.scala 319:20]
  assign _GEN_101 = use_imm ? $signed(imm) : $signed(_GEN_73); // @[Compute.scala 319:20]
  assign _GEN_102 = use_imm ? $signed(imm) : $signed(_GEN_75); // @[Compute.scala 319:20]
  assign _GEN_103 = use_imm ? $signed(imm) : $signed(_GEN_77); // @[Compute.scala 319:20]
  assign _GEN_104 = use_imm ? $signed(imm) : $signed(_GEN_79); // @[Compute.scala 319:20]
  assign _GEN_105 = use_imm ? $signed(imm) : $signed(_GEN_81); // @[Compute.scala 319:20]
  assign _GEN_106 = use_imm ? $signed(imm) : $signed(_GEN_83); // @[Compute.scala 319:20]
  assign _GEN_107 = use_imm ? $signed(imm) : $signed(_GEN_85); // @[Compute.scala 319:20]
  assign _GEN_108 = use_imm ? $signed(imm) : $signed(_GEN_87); // @[Compute.scala 319:20]
  assign _GEN_109 = use_imm ? $signed(imm) : $signed(_GEN_89); // @[Compute.scala 319:20]
  assign _GEN_110 = use_imm ? $signed(imm) : $signed(_GEN_91); // @[Compute.scala 319:20]
  assign _GEN_111 = use_imm ? $signed(imm) : $signed(_GEN_93); // @[Compute.scala 319:20]
  assign _GEN_112 = use_imm ? $signed(imm) : $signed(_GEN_95); // @[Compute.scala 319:20]
  assign _GEN_113 = use_imm ? $signed(imm) : $signed(_GEN_97); // @[Compute.scala 319:20]
  assign src_0_0 = _T_903 ? $signed(_GEN_66) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_0 = _T_903 ? $signed(_GEN_98) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1032 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 324:34]
  assign _T_1033 = _T_1032 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 324:24]
  assign mix_val_0 = _T_903 ? $signed(_T_1033) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1034 = mix_val_0[7:0]; // @[Compute.scala 326:37]
  assign _T_1035 = $unsigned(src_0_0); // @[Compute.scala 327:30]
  assign _T_1036 = $unsigned(src_1_0); // @[Compute.scala 327:59]
  assign _T_1037 = _T_1035 + _T_1036; // @[Compute.scala 327:49]
  assign _T_1038 = _T_1035 + _T_1036; // @[Compute.scala 327:49]
  assign _T_1039 = $signed(_T_1038); // @[Compute.scala 327:79]
  assign add_val_0 = _T_903 ? $signed(_T_1039) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_0 = _T_903 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1040 = add_res_0[7:0]; // @[Compute.scala 329:37]
  assign _T_1042 = src_1_0[4:0]; // @[Compute.scala 330:60]
  assign _T_1043 = _T_1035 >> _T_1042; // @[Compute.scala 330:49]
  assign _T_1044 = $signed(_T_1043); // @[Compute.scala 330:84]
  assign shr_val_0 = _T_903 ? $signed(_T_1044) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_0 = _T_903 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1045 = shr_res_0[7:0]; // @[Compute.scala 332:37]
  assign src_0_1 = _T_903 ? $signed(_GEN_68) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_1 = _T_903 ? $signed(_GEN_99) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1046 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 324:34]
  assign _T_1047 = _T_1046 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 324:24]
  assign mix_val_1 = _T_903 ? $signed(_T_1047) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1048 = mix_val_1[7:0]; // @[Compute.scala 326:37]
  assign _T_1049 = $unsigned(src_0_1); // @[Compute.scala 327:30]
  assign _T_1050 = $unsigned(src_1_1); // @[Compute.scala 327:59]
  assign _T_1051 = _T_1049 + _T_1050; // @[Compute.scala 327:49]
  assign _T_1052 = _T_1049 + _T_1050; // @[Compute.scala 327:49]
  assign _T_1053 = $signed(_T_1052); // @[Compute.scala 327:79]
  assign add_val_1 = _T_903 ? $signed(_T_1053) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_1 = _T_903 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1054 = add_res_1[7:0]; // @[Compute.scala 329:37]
  assign _T_1056 = src_1_1[4:0]; // @[Compute.scala 330:60]
  assign _T_1057 = _T_1049 >> _T_1056; // @[Compute.scala 330:49]
  assign _T_1058 = $signed(_T_1057); // @[Compute.scala 330:84]
  assign shr_val_1 = _T_903 ? $signed(_T_1058) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_1 = _T_903 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1059 = shr_res_1[7:0]; // @[Compute.scala 332:37]
  assign src_0_2 = _T_903 ? $signed(_GEN_70) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_2 = _T_903 ? $signed(_GEN_100) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1060 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 324:34]
  assign _T_1061 = _T_1060 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 324:24]
  assign mix_val_2 = _T_903 ? $signed(_T_1061) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1062 = mix_val_2[7:0]; // @[Compute.scala 326:37]
  assign _T_1063 = $unsigned(src_0_2); // @[Compute.scala 327:30]
  assign _T_1064 = $unsigned(src_1_2); // @[Compute.scala 327:59]
  assign _T_1065 = _T_1063 + _T_1064; // @[Compute.scala 327:49]
  assign _T_1066 = _T_1063 + _T_1064; // @[Compute.scala 327:49]
  assign _T_1067 = $signed(_T_1066); // @[Compute.scala 327:79]
  assign add_val_2 = _T_903 ? $signed(_T_1067) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_2 = _T_903 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1068 = add_res_2[7:0]; // @[Compute.scala 329:37]
  assign _T_1070 = src_1_2[4:0]; // @[Compute.scala 330:60]
  assign _T_1071 = _T_1063 >> _T_1070; // @[Compute.scala 330:49]
  assign _T_1072 = $signed(_T_1071); // @[Compute.scala 330:84]
  assign shr_val_2 = _T_903 ? $signed(_T_1072) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_2 = _T_903 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1073 = shr_res_2[7:0]; // @[Compute.scala 332:37]
  assign src_0_3 = _T_903 ? $signed(_GEN_72) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_3 = _T_903 ? $signed(_GEN_101) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1074 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 324:34]
  assign _T_1075 = _T_1074 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 324:24]
  assign mix_val_3 = _T_903 ? $signed(_T_1075) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1076 = mix_val_3[7:0]; // @[Compute.scala 326:37]
  assign _T_1077 = $unsigned(src_0_3); // @[Compute.scala 327:30]
  assign _T_1078 = $unsigned(src_1_3); // @[Compute.scala 327:59]
  assign _T_1079 = _T_1077 + _T_1078; // @[Compute.scala 327:49]
  assign _T_1080 = _T_1077 + _T_1078; // @[Compute.scala 327:49]
  assign _T_1081 = $signed(_T_1080); // @[Compute.scala 327:79]
  assign add_val_3 = _T_903 ? $signed(_T_1081) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_3 = _T_903 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1082 = add_res_3[7:0]; // @[Compute.scala 329:37]
  assign _T_1084 = src_1_3[4:0]; // @[Compute.scala 330:60]
  assign _T_1085 = _T_1077 >> _T_1084; // @[Compute.scala 330:49]
  assign _T_1086 = $signed(_T_1085); // @[Compute.scala 330:84]
  assign shr_val_3 = _T_903 ? $signed(_T_1086) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_3 = _T_903 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1087 = shr_res_3[7:0]; // @[Compute.scala 332:37]
  assign src_0_4 = _T_903 ? $signed(_GEN_74) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_4 = _T_903 ? $signed(_GEN_102) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1088 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 324:34]
  assign _T_1089 = _T_1088 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 324:24]
  assign mix_val_4 = _T_903 ? $signed(_T_1089) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1090 = mix_val_4[7:0]; // @[Compute.scala 326:37]
  assign _T_1091 = $unsigned(src_0_4); // @[Compute.scala 327:30]
  assign _T_1092 = $unsigned(src_1_4); // @[Compute.scala 327:59]
  assign _T_1093 = _T_1091 + _T_1092; // @[Compute.scala 327:49]
  assign _T_1094 = _T_1091 + _T_1092; // @[Compute.scala 327:49]
  assign _T_1095 = $signed(_T_1094); // @[Compute.scala 327:79]
  assign add_val_4 = _T_903 ? $signed(_T_1095) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_4 = _T_903 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1096 = add_res_4[7:0]; // @[Compute.scala 329:37]
  assign _T_1098 = src_1_4[4:0]; // @[Compute.scala 330:60]
  assign _T_1099 = _T_1091 >> _T_1098; // @[Compute.scala 330:49]
  assign _T_1100 = $signed(_T_1099); // @[Compute.scala 330:84]
  assign shr_val_4 = _T_903 ? $signed(_T_1100) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_4 = _T_903 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1101 = shr_res_4[7:0]; // @[Compute.scala 332:37]
  assign src_0_5 = _T_903 ? $signed(_GEN_76) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_5 = _T_903 ? $signed(_GEN_103) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1102 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 324:34]
  assign _T_1103 = _T_1102 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 324:24]
  assign mix_val_5 = _T_903 ? $signed(_T_1103) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1104 = mix_val_5[7:0]; // @[Compute.scala 326:37]
  assign _T_1105 = $unsigned(src_0_5); // @[Compute.scala 327:30]
  assign _T_1106 = $unsigned(src_1_5); // @[Compute.scala 327:59]
  assign _T_1107 = _T_1105 + _T_1106; // @[Compute.scala 327:49]
  assign _T_1108 = _T_1105 + _T_1106; // @[Compute.scala 327:49]
  assign _T_1109 = $signed(_T_1108); // @[Compute.scala 327:79]
  assign add_val_5 = _T_903 ? $signed(_T_1109) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_5 = _T_903 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1110 = add_res_5[7:0]; // @[Compute.scala 329:37]
  assign _T_1112 = src_1_5[4:0]; // @[Compute.scala 330:60]
  assign _T_1113 = _T_1105 >> _T_1112; // @[Compute.scala 330:49]
  assign _T_1114 = $signed(_T_1113); // @[Compute.scala 330:84]
  assign shr_val_5 = _T_903 ? $signed(_T_1114) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_5 = _T_903 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1115 = shr_res_5[7:0]; // @[Compute.scala 332:37]
  assign src_0_6 = _T_903 ? $signed(_GEN_78) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_6 = _T_903 ? $signed(_GEN_104) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1116 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 324:34]
  assign _T_1117 = _T_1116 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 324:24]
  assign mix_val_6 = _T_903 ? $signed(_T_1117) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1118 = mix_val_6[7:0]; // @[Compute.scala 326:37]
  assign _T_1119 = $unsigned(src_0_6); // @[Compute.scala 327:30]
  assign _T_1120 = $unsigned(src_1_6); // @[Compute.scala 327:59]
  assign _T_1121 = _T_1119 + _T_1120; // @[Compute.scala 327:49]
  assign _T_1122 = _T_1119 + _T_1120; // @[Compute.scala 327:49]
  assign _T_1123 = $signed(_T_1122); // @[Compute.scala 327:79]
  assign add_val_6 = _T_903 ? $signed(_T_1123) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_6 = _T_903 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1124 = add_res_6[7:0]; // @[Compute.scala 329:37]
  assign _T_1126 = src_1_6[4:0]; // @[Compute.scala 330:60]
  assign _T_1127 = _T_1119 >> _T_1126; // @[Compute.scala 330:49]
  assign _T_1128 = $signed(_T_1127); // @[Compute.scala 330:84]
  assign shr_val_6 = _T_903 ? $signed(_T_1128) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_6 = _T_903 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1129 = shr_res_6[7:0]; // @[Compute.scala 332:37]
  assign src_0_7 = _T_903 ? $signed(_GEN_80) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_7 = _T_903 ? $signed(_GEN_105) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1130 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 324:34]
  assign _T_1131 = _T_1130 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 324:24]
  assign mix_val_7 = _T_903 ? $signed(_T_1131) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1132 = mix_val_7[7:0]; // @[Compute.scala 326:37]
  assign _T_1133 = $unsigned(src_0_7); // @[Compute.scala 327:30]
  assign _T_1134 = $unsigned(src_1_7); // @[Compute.scala 327:59]
  assign _T_1135 = _T_1133 + _T_1134; // @[Compute.scala 327:49]
  assign _T_1136 = _T_1133 + _T_1134; // @[Compute.scala 327:49]
  assign _T_1137 = $signed(_T_1136); // @[Compute.scala 327:79]
  assign add_val_7 = _T_903 ? $signed(_T_1137) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_7 = _T_903 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1138 = add_res_7[7:0]; // @[Compute.scala 329:37]
  assign _T_1140 = src_1_7[4:0]; // @[Compute.scala 330:60]
  assign _T_1141 = _T_1133 >> _T_1140; // @[Compute.scala 330:49]
  assign _T_1142 = $signed(_T_1141); // @[Compute.scala 330:84]
  assign shr_val_7 = _T_903 ? $signed(_T_1142) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_7 = _T_903 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1143 = shr_res_7[7:0]; // @[Compute.scala 332:37]
  assign src_0_8 = _T_903 ? $signed(_GEN_82) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_8 = _T_903 ? $signed(_GEN_106) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1144 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 324:34]
  assign _T_1145 = _T_1144 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 324:24]
  assign mix_val_8 = _T_903 ? $signed(_T_1145) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1146 = mix_val_8[7:0]; // @[Compute.scala 326:37]
  assign _T_1147 = $unsigned(src_0_8); // @[Compute.scala 327:30]
  assign _T_1148 = $unsigned(src_1_8); // @[Compute.scala 327:59]
  assign _T_1149 = _T_1147 + _T_1148; // @[Compute.scala 327:49]
  assign _T_1150 = _T_1147 + _T_1148; // @[Compute.scala 327:49]
  assign _T_1151 = $signed(_T_1150); // @[Compute.scala 327:79]
  assign add_val_8 = _T_903 ? $signed(_T_1151) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_8 = _T_903 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1152 = add_res_8[7:0]; // @[Compute.scala 329:37]
  assign _T_1154 = src_1_8[4:0]; // @[Compute.scala 330:60]
  assign _T_1155 = _T_1147 >> _T_1154; // @[Compute.scala 330:49]
  assign _T_1156 = $signed(_T_1155); // @[Compute.scala 330:84]
  assign shr_val_8 = _T_903 ? $signed(_T_1156) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_8 = _T_903 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1157 = shr_res_8[7:0]; // @[Compute.scala 332:37]
  assign src_0_9 = _T_903 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_9 = _T_903 ? $signed(_GEN_107) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1158 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 324:34]
  assign _T_1159 = _T_1158 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 324:24]
  assign mix_val_9 = _T_903 ? $signed(_T_1159) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1160 = mix_val_9[7:0]; // @[Compute.scala 326:37]
  assign _T_1161 = $unsigned(src_0_9); // @[Compute.scala 327:30]
  assign _T_1162 = $unsigned(src_1_9); // @[Compute.scala 327:59]
  assign _T_1163 = _T_1161 + _T_1162; // @[Compute.scala 327:49]
  assign _T_1164 = _T_1161 + _T_1162; // @[Compute.scala 327:49]
  assign _T_1165 = $signed(_T_1164); // @[Compute.scala 327:79]
  assign add_val_9 = _T_903 ? $signed(_T_1165) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_9 = _T_903 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1166 = add_res_9[7:0]; // @[Compute.scala 329:37]
  assign _T_1168 = src_1_9[4:0]; // @[Compute.scala 330:60]
  assign _T_1169 = _T_1161 >> _T_1168; // @[Compute.scala 330:49]
  assign _T_1170 = $signed(_T_1169); // @[Compute.scala 330:84]
  assign shr_val_9 = _T_903 ? $signed(_T_1170) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_9 = _T_903 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1171 = shr_res_9[7:0]; // @[Compute.scala 332:37]
  assign src_0_10 = _T_903 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_10 = _T_903 ? $signed(_GEN_108) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1172 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 324:34]
  assign _T_1173 = _T_1172 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 324:24]
  assign mix_val_10 = _T_903 ? $signed(_T_1173) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1174 = mix_val_10[7:0]; // @[Compute.scala 326:37]
  assign _T_1175 = $unsigned(src_0_10); // @[Compute.scala 327:30]
  assign _T_1176 = $unsigned(src_1_10); // @[Compute.scala 327:59]
  assign _T_1177 = _T_1175 + _T_1176; // @[Compute.scala 327:49]
  assign _T_1178 = _T_1175 + _T_1176; // @[Compute.scala 327:49]
  assign _T_1179 = $signed(_T_1178); // @[Compute.scala 327:79]
  assign add_val_10 = _T_903 ? $signed(_T_1179) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_10 = _T_903 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1180 = add_res_10[7:0]; // @[Compute.scala 329:37]
  assign _T_1182 = src_1_10[4:0]; // @[Compute.scala 330:60]
  assign _T_1183 = _T_1175 >> _T_1182; // @[Compute.scala 330:49]
  assign _T_1184 = $signed(_T_1183); // @[Compute.scala 330:84]
  assign shr_val_10 = _T_903 ? $signed(_T_1184) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_10 = _T_903 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1185 = shr_res_10[7:0]; // @[Compute.scala 332:37]
  assign src_0_11 = _T_903 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_11 = _T_903 ? $signed(_GEN_109) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1186 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 324:34]
  assign _T_1187 = _T_1186 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 324:24]
  assign mix_val_11 = _T_903 ? $signed(_T_1187) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1188 = mix_val_11[7:0]; // @[Compute.scala 326:37]
  assign _T_1189 = $unsigned(src_0_11); // @[Compute.scala 327:30]
  assign _T_1190 = $unsigned(src_1_11); // @[Compute.scala 327:59]
  assign _T_1191 = _T_1189 + _T_1190; // @[Compute.scala 327:49]
  assign _T_1192 = _T_1189 + _T_1190; // @[Compute.scala 327:49]
  assign _T_1193 = $signed(_T_1192); // @[Compute.scala 327:79]
  assign add_val_11 = _T_903 ? $signed(_T_1193) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_11 = _T_903 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1194 = add_res_11[7:0]; // @[Compute.scala 329:37]
  assign _T_1196 = src_1_11[4:0]; // @[Compute.scala 330:60]
  assign _T_1197 = _T_1189 >> _T_1196; // @[Compute.scala 330:49]
  assign _T_1198 = $signed(_T_1197); // @[Compute.scala 330:84]
  assign shr_val_11 = _T_903 ? $signed(_T_1198) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_11 = _T_903 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1199 = shr_res_11[7:0]; // @[Compute.scala 332:37]
  assign src_0_12 = _T_903 ? $signed(_GEN_90) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_12 = _T_903 ? $signed(_GEN_110) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1200 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 324:34]
  assign _T_1201 = _T_1200 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 324:24]
  assign mix_val_12 = _T_903 ? $signed(_T_1201) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1202 = mix_val_12[7:0]; // @[Compute.scala 326:37]
  assign _T_1203 = $unsigned(src_0_12); // @[Compute.scala 327:30]
  assign _T_1204 = $unsigned(src_1_12); // @[Compute.scala 327:59]
  assign _T_1205 = _T_1203 + _T_1204; // @[Compute.scala 327:49]
  assign _T_1206 = _T_1203 + _T_1204; // @[Compute.scala 327:49]
  assign _T_1207 = $signed(_T_1206); // @[Compute.scala 327:79]
  assign add_val_12 = _T_903 ? $signed(_T_1207) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_12 = _T_903 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1208 = add_res_12[7:0]; // @[Compute.scala 329:37]
  assign _T_1210 = src_1_12[4:0]; // @[Compute.scala 330:60]
  assign _T_1211 = _T_1203 >> _T_1210; // @[Compute.scala 330:49]
  assign _T_1212 = $signed(_T_1211); // @[Compute.scala 330:84]
  assign shr_val_12 = _T_903 ? $signed(_T_1212) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_12 = _T_903 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1213 = shr_res_12[7:0]; // @[Compute.scala 332:37]
  assign src_0_13 = _T_903 ? $signed(_GEN_92) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_13 = _T_903 ? $signed(_GEN_111) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1214 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 324:34]
  assign _T_1215 = _T_1214 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 324:24]
  assign mix_val_13 = _T_903 ? $signed(_T_1215) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1216 = mix_val_13[7:0]; // @[Compute.scala 326:37]
  assign _T_1217 = $unsigned(src_0_13); // @[Compute.scala 327:30]
  assign _T_1218 = $unsigned(src_1_13); // @[Compute.scala 327:59]
  assign _T_1219 = _T_1217 + _T_1218; // @[Compute.scala 327:49]
  assign _T_1220 = _T_1217 + _T_1218; // @[Compute.scala 327:49]
  assign _T_1221 = $signed(_T_1220); // @[Compute.scala 327:79]
  assign add_val_13 = _T_903 ? $signed(_T_1221) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_13 = _T_903 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1222 = add_res_13[7:0]; // @[Compute.scala 329:37]
  assign _T_1224 = src_1_13[4:0]; // @[Compute.scala 330:60]
  assign _T_1225 = _T_1217 >> _T_1224; // @[Compute.scala 330:49]
  assign _T_1226 = $signed(_T_1225); // @[Compute.scala 330:84]
  assign shr_val_13 = _T_903 ? $signed(_T_1226) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_13 = _T_903 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1227 = shr_res_13[7:0]; // @[Compute.scala 332:37]
  assign src_0_14 = _T_903 ? $signed(_GEN_94) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_14 = _T_903 ? $signed(_GEN_112) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1228 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 324:34]
  assign _T_1229 = _T_1228 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 324:24]
  assign mix_val_14 = _T_903 ? $signed(_T_1229) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1230 = mix_val_14[7:0]; // @[Compute.scala 326:37]
  assign _T_1231 = $unsigned(src_0_14); // @[Compute.scala 327:30]
  assign _T_1232 = $unsigned(src_1_14); // @[Compute.scala 327:59]
  assign _T_1233 = _T_1231 + _T_1232; // @[Compute.scala 327:49]
  assign _T_1234 = _T_1231 + _T_1232; // @[Compute.scala 327:49]
  assign _T_1235 = $signed(_T_1234); // @[Compute.scala 327:79]
  assign add_val_14 = _T_903 ? $signed(_T_1235) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_14 = _T_903 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1236 = add_res_14[7:0]; // @[Compute.scala 329:37]
  assign _T_1238 = src_1_14[4:0]; // @[Compute.scala 330:60]
  assign _T_1239 = _T_1231 >> _T_1238; // @[Compute.scala 330:49]
  assign _T_1240 = $signed(_T_1239); // @[Compute.scala 330:84]
  assign shr_val_14 = _T_903 ? $signed(_T_1240) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_14 = _T_903 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1241 = shr_res_14[7:0]; // @[Compute.scala 332:37]
  assign src_0_15 = _T_903 ? $signed(_GEN_96) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign src_1_15 = _T_903 ? $signed(_GEN_113) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1242 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 324:34]
  assign _T_1243 = _T_1242 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 324:24]
  assign mix_val_15 = _T_903 ? $signed(_T_1243) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1244 = mix_val_15[7:0]; // @[Compute.scala 326:37]
  assign _T_1245 = $unsigned(src_0_15); // @[Compute.scala 327:30]
  assign _T_1246 = $unsigned(src_1_15); // @[Compute.scala 327:59]
  assign _T_1247 = _T_1245 + _T_1246; // @[Compute.scala 327:49]
  assign _T_1248 = _T_1245 + _T_1246; // @[Compute.scala 327:49]
  assign _T_1249 = $signed(_T_1248); // @[Compute.scala 327:79]
  assign add_val_15 = _T_903 ? $signed(_T_1249) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign add_res_15 = _T_903 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1250 = add_res_15[7:0]; // @[Compute.scala 329:37]
  assign _T_1252 = src_1_15[4:0]; // @[Compute.scala 330:60]
  assign _T_1253 = _T_1245 >> _T_1252; // @[Compute.scala 330:49]
  assign _T_1254 = $signed(_T_1253); // @[Compute.scala 330:84]
  assign shr_val_15 = _T_903 ? $signed(_T_1254) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign shr_res_15 = _T_903 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 307:36]
  assign _T_1255 = shr_res_15[7:0]; // @[Compute.scala 332:37]
  assign short_cmp_res_0 = _T_903 ? _T_1034 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_0 = _T_903 ? _T_1040 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_0 = _T_903 ? _T_1045 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_1 = _T_903 ? _T_1048 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_1 = _T_903 ? _T_1054 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_1 = _T_903 ? _T_1059 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_2 = _T_903 ? _T_1062 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_2 = _T_903 ? _T_1068 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_2 = _T_903 ? _T_1073 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_3 = _T_903 ? _T_1076 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_3 = _T_903 ? _T_1082 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_3 = _T_903 ? _T_1087 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_4 = _T_903 ? _T_1090 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_4 = _T_903 ? _T_1096 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_4 = _T_903 ? _T_1101 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_5 = _T_903 ? _T_1104 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_5 = _T_903 ? _T_1110 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_5 = _T_903 ? _T_1115 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_6 = _T_903 ? _T_1118 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_6 = _T_903 ? _T_1124 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_6 = _T_903 ? _T_1129 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_7 = _T_903 ? _T_1132 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_7 = _T_903 ? _T_1138 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_7 = _T_903 ? _T_1143 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_8 = _T_903 ? _T_1146 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_8 = _T_903 ? _T_1152 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_8 = _T_903 ? _T_1157 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_9 = _T_903 ? _T_1160 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_9 = _T_903 ? _T_1166 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_9 = _T_903 ? _T_1171 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_10 = _T_903 ? _T_1174 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_10 = _T_903 ? _T_1180 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_10 = _T_903 ? _T_1185 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_11 = _T_903 ? _T_1188 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_11 = _T_903 ? _T_1194 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_11 = _T_903 ? _T_1199 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_12 = _T_903 ? _T_1202 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_12 = _T_903 ? _T_1208 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_12 = _T_903 ? _T_1213 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_13 = _T_903 ? _T_1216 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_13 = _T_903 ? _T_1222 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_13 = _T_903 ? _T_1227 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_14 = _T_903 ? _T_1230 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_14 = _T_903 ? _T_1236 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_14 = _T_903 ? _T_1241 : 8'h0; // @[Compute.scala 307:36]
  assign short_cmp_res_15 = _T_903 ? _T_1244 : 8'h0; // @[Compute.scala 307:36]
  assign short_add_res_15 = _T_903 ? _T_1250 : 8'h0; // @[Compute.scala 307:36]
  assign short_shr_res_15 = _T_903 ? _T_1255 : 8'h0; // @[Compute.scala 307:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 337:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 338:39]
  assign _T_1257 = opcode_alu_en & busy; // @[Compute.scala 339:34]
  assign _T_1259 = out_cntr_wrap == 1'h0; // @[Compute.scala 339:45]
  assign _T_1260 = _T_1257 & _T_1259; // @[Compute.scala 339:42]
  assign _T_1262 = out_cntr_max - 44'h1; // @[Compute.scala 340:58]
  assign _T_1263 = $unsigned(_T_1262); // @[Compute.scala 340:58]
  assign _T_1264 = _T_1263[43:0]; // @[Compute.scala 340:58]
  assign _T_1265 = _GEN_325 == _T_1264; // @[Compute.scala 340:40]
  assign _T_1266 = out_mem_write & _T_1265; // @[Compute.scala 340:23]
  assign _T_1269 = _T_1266 & _T_344; // @[Compute.scala 340:66]
  assign _GEN_290 = _T_1269 ? 1'h0 : _T_1260; // @[Compute.scala 340:85]
  assign _GEN_344 = {{7'd0}, out_mem_address}; // @[Compute.scala 347:41]
  assign _T_1275 = _GEN_344 << 3'h4; // @[Compute.scala 347:41]
  assign _T_1282 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1290 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1282}; // @[Cat.scala 30:58]
  assign _T_1297 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1305 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1297}; // @[Cat.scala 30:58]
  assign _T_1312 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1320 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1312}; // @[Cat.scala 30:58]
  assign _T_1321 = alu_opcode_add_en ? _T_1305 : _T_1320; // @[Compute.scala 351:8]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 215:23]
  assign io_done_readdata = opcode == 3'h3; // @[Compute.scala 218:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 226:19]
  assign io_uops_read = uops_read; // @[Compute.scala 225:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 128'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 241:21]
  assign io_biases_read = biases_read; // @[Compute.scala 242:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = gemm_queue_ready; // @[Compute.scala 211:23]
  assign io_l2g_dep_queue_ready = pop_prev_dep_ready & dump; // @[Compute.scala 164:26]
  assign io_s2g_dep_queue_ready = pop_next_dep_ready & dump; // @[Compute.scala 165:26]
  assign io_g2l_dep_queue_valid = push_prev_dep & push; // @[Compute.scala 176:26]
  assign io_g2l_dep_queue_data = 1'h1; // @[Compute.scala 174:25]
  assign io_g2s_dep_queue_valid = push_next_dep & push; // @[Compute.scala 177:26]
  assign io_g2s_dep_queue_data = 1'h1; // @[Compute.scala 175:25]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = _T_1275[16:0]; // @[Compute.scala 347:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_write; // @[Compute.scala 348:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1290 : _T_1321; // @[Compute.scala 350:24]
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
  uop_bgn = _RAND_3[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  uop_end = _RAND_4[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{`RANDOM}};
  state = _RAND_5[2:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  uops_read = _RAND_6[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {1{`RANDOM}};
  biases_read = _RAND_7[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {4{`RANDOM}};
  biases_data_0 = _RAND_8[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {4{`RANDOM}};
  biases_data_1 = _RAND_9[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {4{`RANDOM}};
  biases_data_2 = _RAND_10[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_11 = {4{`RANDOM}};
  biases_data_3 = _RAND_11[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_12 = {1{`RANDOM}};
  out_mem_write = _RAND_12[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_13 = {1{`RANDOM}};
  uop_cntr_val = _RAND_13[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_14 = {1{`RANDOM}};
  acc_cntr_val = _RAND_14[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_15 = {1{`RANDOM}};
  out_cntr_val = _RAND_15[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_16 = {1{`RANDOM}};
  pop_prev_dep_ready = _RAND_16[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_17 = {1{`RANDOM}};
  pop_next_dep_ready = _RAND_17[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_18 = {1{`RANDOM}};
  push_prev_dep_ready = _RAND_18[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_19 = {1{`RANDOM}};
  push_next_dep_ready = _RAND_19[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_20 = {1{`RANDOM}};
  gemm_queue_ready = _RAND_20[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_21 = {1{`RANDOM}};
  finish_wrap = _RAND_21[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_22 = {16{`RANDOM}};
  dst_vector = _RAND_22[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_23 = {16{`RANDOM}};
  src_vector = _RAND_23[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_24 = {1{`RANDOM}};
  out_mem_address = _RAND_24[31:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_449_en & acc_mem__T_449_mask) begin
      acc_mem[acc_mem__T_449_addr] <= acc_mem__T_449_data; // @[Compute.scala 33:20]
    end
    if(uop_mem__T_382_en & uop_mem__T_382_mask) begin
      uop_mem[uop_mem__T_382_addr] <= uop_mem__T_382_data; // @[Compute.scala 34:20]
    end
    if(uop_mem__T_388_en & uop_mem__T_388_mask) begin
      uop_mem[uop_mem__T_388_addr] <= uop_mem__T_388_data; // @[Compute.scala 34:20]
    end
    if(uop_mem__T_394_en & uop_mem__T_394_mask) begin
      uop_mem[uop_mem__T_394_addr] <= uop_mem__T_394_data; // @[Compute.scala 34:20]
    end
    if(uop_mem__T_400_en & uop_mem__T_400_mask) begin
      uop_mem[uop_mem__T_400_addr] <= uop_mem__T_400_data; // @[Compute.scala 34:20]
    end
    if (gemm_queue_ready) begin
      insn <= io_gemm_queue_data;
    end
    uop_bgn <= {{3'd0}, _T_203};
    uop_end <= {{2'd0}, _T_205};
    if (reset) begin
      state <= 3'h0;
    end else begin
      if (gemm_queue_ready) begin
        state <= 3'h2;
      end else begin
        if (_T_308) begin
          state <= 3'h4;
        end else begin
          if (_T_306) begin
            state <= 3'h2;
          end else begin
            if (_T_304) begin
              state <= 3'h1;
            end else begin
              if (_T_295) begin
                if (_T_296) begin
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
    if (_T_327) begin
      if (_T_406) begin
        uops_read <= 1'h0;
      end else begin
        uops_read <= _T_374;
      end
    end else begin
      uops_read <= _T_374;
    end
    biases_read <= acc_cntr_en & _T_435;
    if (_T_336) begin
      if (3'h0 == _T_441) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_336) begin
      if (3'h1 == _T_441) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_336) begin
      if (3'h2 == _T_441) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_336) begin
      if (3'h3 == _T_441) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    if (reset) begin
      out_mem_write <= 1'h0;
    end else begin
      if (_T_1269) begin
        out_mem_write <= 1'h0;
      end else begin
        out_mem_write <= _T_1260;
      end
    end
    if (gemm_queue_ready) begin
      uop_cntr_val <= 16'h0;
    end else begin
      if (_T_330) begin
        uop_cntr_val <= _T_333;
      end
    end
    if (gemm_queue_ready) begin
      acc_cntr_val <= 16'h0;
    end else begin
      if (_T_339) begin
        acc_cntr_val <= _T_342;
      end
    end
    if (gemm_queue_ready) begin
      out_cntr_val <= 16'h0;
    end else begin
      if (_T_348) begin
        out_cntr_val <= _T_351;
      end
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_312) begin
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
        if (_T_315) begin
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
        if (_T_320) begin
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
        if (_T_323) begin
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
        gemm_queue_ready <= _T_360;
      end
    end
    if (reset) begin
      finish_wrap <= 1'h0;
    end else begin
      if (opcode_finish_en) begin
        if (pop_prev_dep) begin
          finish_wrap <= _T_287;
        end else begin
          if (pop_next_dep) begin
            finish_wrap <= _T_288;
          end else begin
            if (push_prev_dep) begin
              finish_wrap <= _T_289;
            end else begin
              if (push_next_dep) begin
                finish_wrap <= _T_290;
              end else begin
                finish_wrap <= 1'h0;
              end
            end
          end
        end
      end else begin
        finish_wrap <= 1'h0;
      end
    end
    if (_T_345) begin
      dst_vector <= acc_mem__T_469_data;
    end
    if (_T_345) begin
      src_vector <= acc_mem__T_471_data;
    end
    if (_T_344) begin
      out_mem_address <= dst_idx;
    end
  end
endmodule
