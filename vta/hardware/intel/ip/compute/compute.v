module Control(
  input          clock,
  input          reset,
  output         io_done_readdata,
  output         io_uops_ready,
  input          io_uops_valid,
  input  [31:0]  io_uops_data,
  output         io_biases_ready,
  input          io_biases_valid,
  input  [511:0] io_biases_data,
  output         io_gemm_queue_ready,
  input          io_gemm_queue_valid,
  input  [127:0] io_gemm_queue_data,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata,
  input  [31:0]  io_uop_mem_readdata,
  output         io_uop_mem_write,
  output [31:0]  io_uop_mem_writedata
);
  reg [511:0] acc_mem [0:255] /* synthesis ramstyle = "M20K" */; // @[Control.scala 22:20]
  reg [511:0] _RAND_0;
  wire [511:0] acc_mem__T_196_data; // @[Control.scala 22:20]
  wire [7:0] acc_mem__T_196_addr; // @[Control.scala 22:20]
  wire [511:0] acc_mem__T_199_data; // @[Control.scala 22:20]
  wire [7:0] acc_mem__T_199_addr; // @[Control.scala 22:20]
  wire [511:0] acc_mem__T_181_data; // @[Control.scala 22:20]
  wire [7:0] acc_mem__T_181_addr; // @[Control.scala 22:20]
  wire  acc_mem__T_181_mask; // @[Control.scala 22:20]
  wire  acc_mem__T_181_en; // @[Control.scala 22:20]
  reg [127:0] insn; // @[Control.scala 24:28]
  reg [127:0] _RAND_1;
  wire  insn_valid; // @[Control.scala 25:30]
  wire [2:0] opcode; // @[Control.scala 27:29]
  reg [31:0] uops_data; // @[Control.scala 33:28]
  reg [31:0] _RAND_2;
  wire [1:0] memory_type; // @[Control.scala 35:25]
  wire [15:0] sram_base; // @[Control.scala 36:25]
  wire [15:0] x_size; // @[Control.scala 39:25]
  wire [3:0] y_pad_0; // @[Control.scala 41:25]
  wire [3:0] x_pad_0; // @[Control.scala 43:25]
  wire [3:0] x_pad_1; // @[Control.scala 44:25]
  wire [15:0] _GEN_249; // @[Control.scala 48:30]
  wire [15:0] _GEN_251; // @[Control.scala 49:30]
  wire [16:0] _T_95; // @[Control.scala 49:30]
  wire [15:0] _T_96; // @[Control.scala 49:30]
  wire [15:0] _GEN_252; // @[Control.scala 49:39]
  wire [16:0] _T_97; // @[Control.scala 49:39]
  wire [15:0] x_size_total; // @[Control.scala 49:39]
  wire [19:0] y_offset; // @[Control.scala 50:31]
  wire  _T_100; // @[Control.scala 54:32]
  wire  _T_102; // @[Control.scala 54:60]
  wire  opcode_load_en; // @[Control.scala 54:50]
  wire  opcode_gemm_en; // @[Control.scala 55:32]
  wire  opcode_alu_en; // @[Control.scala 56:31]
  wire  memory_type_uop_en; // @[Control.scala 58:40]
  wire  memory_type_acc_en; // @[Control.scala 59:40]
  wire  acc_x_cntr_en; // @[Control.scala 63:39]
  reg [7:0] acc_x_cntr_val; // @[Control.scala 65:31]
  reg [31:0] _RAND_3;
  wire  _T_110; // @[Control.scala 67:23]
  wire  _T_112; // @[Control.scala 68:26]
  wire [8:0] _T_116; // @[Control.scala 72:40]
  wire [7:0] _T_117; // @[Control.scala 72:40]
  wire [7:0] _GEN_0; // @[Control.scala 68:54]
  wire [7:0] _GEN_2; // @[Control.scala 67:43]
  wire  acc_x_cntr_wrap; // @[Control.scala 67:43]
  wire  in_loop_cntr_en; // @[Control.scala 81:40]
  reg [7:0] dst_offset_in; // @[Control.scala 83:33]
  reg [31:0] _RAND_4;
  wire  _T_124; // @[Control.scala 85:28]
  wire  _T_125; // @[Control.scala 85:25]
  wire  _T_127; // @[Control.scala 86:28]
  wire [8:0] _T_131; // @[Control.scala 90:44]
  wire [7:0] _T_132; // @[Control.scala 90:44]
  wire [7:0] _GEN_4; // @[Control.scala 86:58]
  wire [7:0] _GEN_6; // @[Control.scala 85:53]
  wire  in_loop_cntr_wrap; // @[Control.scala 85:53]
  wire  _T_135; // @[Control.scala 100:33]
  wire  _T_138; // @[Control.scala 101:35]
  wire  _T_139; // @[Control.scala 101:32]
  wire  _T_142; // @[Control.scala 102:37]
  wire  _T_143; // @[Control.scala 102:34]
  wire  _T_147; // @[Control.scala 101:17]
  wire  busy; // @[Control.scala 100:17]
  wire  _T_149; // @[Control.scala 106:32]
  wire  _T_150; // @[Control.scala 106:29]
  wire  _T_153; // @[Control.scala 115:23]
  wire  _T_154; // @[Control.scala 115:45]
  wire [19:0] _GEN_254; // @[Control.scala 142:33]
  wire [20:0] _T_173; // @[Control.scala 142:33]
  wire [19:0] _T_174; // @[Control.scala 142:33]
  wire [19:0] _GEN_255; // @[Control.scala 142:44]
  wire [20:0] _T_175; // @[Control.scala 142:44]
  wire [19:0] _T_176; // @[Control.scala 142:44]
  wire [20:0] _T_178; // @[Control.scala 142:55]
  wire [20:0] _GEN_256; // @[Control.scala 142:65]
  wire [21:0] _T_179; // @[Control.scala 142:65]
  wire [20:0] acc_mem_addr; // @[Control.scala 142:65]
  wire [1:0] alu_opcode; // @[Control.scala 164:24]
  wire  use_imm; // @[Control.scala 165:21]
  wire [15:0] imm_raw; // @[Control.scala 166:21]
  wire [15:0] _T_182; // @[Control.scala 167:25]
  wire  _T_184; // @[Control.scala 167:32]
  wire [31:0] _T_186; // @[Cat.scala 30:58]
  wire [16:0] _T_188; // @[Cat.scala 30:58]
  wire [31:0] _T_189; // @[Control.scala 167:16]
  wire [31:0] imm; // @[Control.scala 167:89]
  wire [10:0] _T_191; // @[Control.scala 176:20]
  wire [10:0] _GEN_257; // @[Control.scala 176:47]
  wire [11:0] _T_192; // @[Control.scala 176:47]
  wire [10:0] dst_idx; // @[Control.scala 176:47]
  wire [10:0] _T_193; // @[Control.scala 177:20]
  wire [11:0] _T_194; // @[Control.scala 177:47]
  wire [10:0] src_idx; // @[Control.scala 177:47]
  reg [511:0] dst_vector; // @[Control.scala 179:27]
  reg [511:0] _RAND_5;
  reg [511:0] src_vector; // @[Control.scala 180:27]
  reg [511:0] _RAND_6;
  reg [10:0] out_mem_addr; // @[Control.scala 193:30]
  reg [31:0] _RAND_7;
  reg  out_mem_write_en; // @[Control.scala 194:34]
  reg [31:0] _RAND_8;
  wire  alu_opcode_min_en; // @[Control.scala 196:38]
  wire  alu_opcode_max_en; // @[Control.scala 197:38]
  wire  _T_634; // @[Control.scala 215:20]
  wire [31:0] _T_635; // @[Control.scala 235:31]
  wire [31:0] _T_636; // @[Control.scala 235:72]
  wire [31:0] _T_637; // @[Control.scala 236:31]
  wire [31:0] _T_638; // @[Control.scala 236:72]
  wire [31:0] _T_639; // @[Control.scala 235:31]
  wire [31:0] _T_640; // @[Control.scala 235:72]
  wire [31:0] _T_641; // @[Control.scala 236:31]
  wire [31:0] _T_642; // @[Control.scala 236:72]
  wire [31:0] _T_643; // @[Control.scala 235:31]
  wire [31:0] _T_644; // @[Control.scala 235:72]
  wire [31:0] _T_645; // @[Control.scala 236:31]
  wire [31:0] _T_646; // @[Control.scala 236:72]
  wire [31:0] _T_647; // @[Control.scala 235:31]
  wire [31:0] _T_648; // @[Control.scala 235:72]
  wire [31:0] _T_649; // @[Control.scala 236:31]
  wire [31:0] _T_650; // @[Control.scala 236:72]
  wire [31:0] _T_651; // @[Control.scala 235:31]
  wire [31:0] _T_652; // @[Control.scala 235:72]
  wire [31:0] _T_653; // @[Control.scala 236:31]
  wire [31:0] _T_654; // @[Control.scala 236:72]
  wire [31:0] _T_655; // @[Control.scala 235:31]
  wire [31:0] _T_656; // @[Control.scala 235:72]
  wire [31:0] _T_657; // @[Control.scala 236:31]
  wire [31:0] _T_658; // @[Control.scala 236:72]
  wire [31:0] _T_659; // @[Control.scala 235:31]
  wire [31:0] _T_660; // @[Control.scala 235:72]
  wire [31:0] _T_661; // @[Control.scala 236:31]
  wire [31:0] _T_662; // @[Control.scala 236:72]
  wire [31:0] _T_663; // @[Control.scala 235:31]
  wire [31:0] _T_664; // @[Control.scala 235:72]
  wire [31:0] _T_665; // @[Control.scala 236:31]
  wire [31:0] _T_666; // @[Control.scala 236:72]
  wire [31:0] _T_667; // @[Control.scala 235:31]
  wire [31:0] _T_668; // @[Control.scala 235:72]
  wire [31:0] _T_669; // @[Control.scala 236:31]
  wire [31:0] _T_670; // @[Control.scala 236:72]
  wire [31:0] _T_671; // @[Control.scala 235:31]
  wire [31:0] _T_672; // @[Control.scala 235:72]
  wire [31:0] _T_673; // @[Control.scala 236:31]
  wire [31:0] _T_674; // @[Control.scala 236:72]
  wire [31:0] _T_675; // @[Control.scala 235:31]
  wire [31:0] _T_676; // @[Control.scala 235:72]
  wire [31:0] _T_677; // @[Control.scala 236:31]
  wire [31:0] _T_678; // @[Control.scala 236:72]
  wire [31:0] _T_679; // @[Control.scala 235:31]
  wire [31:0] _T_680; // @[Control.scala 235:72]
  wire [31:0] _T_681; // @[Control.scala 236:31]
  wire [31:0] _T_682; // @[Control.scala 236:72]
  wire [31:0] _T_683; // @[Control.scala 235:31]
  wire [31:0] _T_684; // @[Control.scala 235:72]
  wire [31:0] _T_685; // @[Control.scala 236:31]
  wire [31:0] _T_686; // @[Control.scala 236:72]
  wire [31:0] _T_687; // @[Control.scala 235:31]
  wire [31:0] _T_688; // @[Control.scala 235:72]
  wire [31:0] _T_689; // @[Control.scala 236:31]
  wire [31:0] _T_690; // @[Control.scala 236:72]
  wire [31:0] _T_691; // @[Control.scala 235:31]
  wire [31:0] _T_692; // @[Control.scala 235:72]
  wire [31:0] _T_693; // @[Control.scala 236:31]
  wire [31:0] _T_694; // @[Control.scala 236:72]
  wire [31:0] _T_695; // @[Control.scala 235:31]
  wire [31:0] _T_696; // @[Control.scala 235:72]
  wire [31:0] _T_697; // @[Control.scala 236:31]
  wire [31:0] _T_698; // @[Control.scala 236:72]
  wire [31:0] _GEN_17; // @[Control.scala 233:30]
  wire [31:0] _GEN_18; // @[Control.scala 233:30]
  wire [31:0] _GEN_19; // @[Control.scala 233:30]
  wire [31:0] _GEN_20; // @[Control.scala 233:30]
  wire [31:0] _GEN_21; // @[Control.scala 233:30]
  wire [31:0] _GEN_22; // @[Control.scala 233:30]
  wire [31:0] _GEN_23; // @[Control.scala 233:30]
  wire [31:0] _GEN_24; // @[Control.scala 233:30]
  wire [31:0] _GEN_25; // @[Control.scala 233:30]
  wire [31:0] _GEN_26; // @[Control.scala 233:30]
  wire [31:0] _GEN_27; // @[Control.scala 233:30]
  wire [31:0] _GEN_28; // @[Control.scala 233:30]
  wire [31:0] _GEN_29; // @[Control.scala 233:30]
  wire [31:0] _GEN_30; // @[Control.scala 233:30]
  wire [31:0] _GEN_31; // @[Control.scala 233:30]
  wire [31:0] _GEN_32; // @[Control.scala 233:30]
  wire [31:0] _GEN_33; // @[Control.scala 233:30]
  wire [31:0] _GEN_34; // @[Control.scala 233:30]
  wire [31:0] _GEN_35; // @[Control.scala 233:30]
  wire [31:0] _GEN_36; // @[Control.scala 233:30]
  wire [31:0] _GEN_37; // @[Control.scala 233:30]
  wire [31:0] _GEN_38; // @[Control.scala 233:30]
  wire [31:0] _GEN_39; // @[Control.scala 233:30]
  wire [31:0] _GEN_40; // @[Control.scala 233:30]
  wire [31:0] _GEN_41; // @[Control.scala 233:30]
  wire [31:0] _GEN_42; // @[Control.scala 233:30]
  wire [31:0] _GEN_43; // @[Control.scala 233:30]
  wire [31:0] _GEN_44; // @[Control.scala 233:30]
  wire [31:0] _GEN_45; // @[Control.scala 233:30]
  wire [31:0] _GEN_46; // @[Control.scala 233:30]
  wire [31:0] _GEN_47; // @[Control.scala 233:30]
  wire [31:0] _GEN_48; // @[Control.scala 233:30]
  wire [31:0] _GEN_49; // @[Control.scala 244:20]
  wire [31:0] _GEN_50; // @[Control.scala 244:20]
  wire [31:0] _GEN_51; // @[Control.scala 244:20]
  wire [31:0] _GEN_52; // @[Control.scala 244:20]
  wire [31:0] _GEN_53; // @[Control.scala 244:20]
  wire [31:0] _GEN_54; // @[Control.scala 244:20]
  wire [31:0] _GEN_55; // @[Control.scala 244:20]
  wire [31:0] _GEN_56; // @[Control.scala 244:20]
  wire [31:0] _GEN_57; // @[Control.scala 244:20]
  wire [31:0] _GEN_58; // @[Control.scala 244:20]
  wire [31:0] _GEN_59; // @[Control.scala 244:20]
  wire [31:0] _GEN_60; // @[Control.scala 244:20]
  wire [31:0] _GEN_61; // @[Control.scala 244:20]
  wire [31:0] _GEN_62; // @[Control.scala 244:20]
  wire [31:0] _GEN_63; // @[Control.scala 244:20]
  wire [31:0] _GEN_64; // @[Control.scala 244:20]
  wire [31:0] src_0_0; // @[Control.scala 215:40]
  wire [31:0] src_1_0; // @[Control.scala 215:40]
  wire  _T_763; // @[Control.scala 249:34]
  wire [31:0] _T_764; // @[Control.scala 249:24]
  wire [31:0] mix_val_0; // @[Control.scala 215:40]
  wire [7:0] _T_765; // @[Control.scala 251:37]
  wire [31:0] _T_766; // @[Control.scala 252:30]
  wire [31:0] _T_767; // @[Control.scala 252:59]
  wire [32:0] _T_768; // @[Control.scala 252:49]
  wire [31:0] _T_769; // @[Control.scala 252:49]
  wire [31:0] _T_770; // @[Control.scala 252:79]
  wire [31:0] add_val_0; // @[Control.scala 215:40]
  wire [31:0] add_res_0; // @[Control.scala 215:40]
  wire [7:0] _T_771; // @[Control.scala 254:37]
  wire [4:0] _T_773; // @[Control.scala 255:60]
  wire [31:0] _T_774; // @[Control.scala 255:49]
  wire [31:0] _T_775; // @[Control.scala 255:84]
  wire [31:0] shr_val_0; // @[Control.scala 215:40]
  wire [31:0] shr_res_0; // @[Control.scala 215:40]
  wire [7:0] _T_776; // @[Control.scala 257:37]
  wire [31:0] src_0_1; // @[Control.scala 215:40]
  wire [31:0] src_1_1; // @[Control.scala 215:40]
  wire  _T_777; // @[Control.scala 249:34]
  wire [31:0] _T_778; // @[Control.scala 249:24]
  wire [31:0] mix_val_1; // @[Control.scala 215:40]
  wire [7:0] _T_779; // @[Control.scala 251:37]
  wire [31:0] _T_780; // @[Control.scala 252:30]
  wire [31:0] _T_781; // @[Control.scala 252:59]
  wire [32:0] _T_782; // @[Control.scala 252:49]
  wire [31:0] _T_783; // @[Control.scala 252:49]
  wire [31:0] _T_784; // @[Control.scala 252:79]
  wire [31:0] add_val_1; // @[Control.scala 215:40]
  wire [31:0] add_res_1; // @[Control.scala 215:40]
  wire [7:0] _T_785; // @[Control.scala 254:37]
  wire [4:0] _T_787; // @[Control.scala 255:60]
  wire [31:0] _T_788; // @[Control.scala 255:49]
  wire [31:0] _T_789; // @[Control.scala 255:84]
  wire [31:0] shr_val_1; // @[Control.scala 215:40]
  wire [31:0] shr_res_1; // @[Control.scala 215:40]
  wire [7:0] _T_790; // @[Control.scala 257:37]
  wire [31:0] src_0_2; // @[Control.scala 215:40]
  wire [31:0] src_1_2; // @[Control.scala 215:40]
  wire  _T_791; // @[Control.scala 249:34]
  wire [31:0] _T_792; // @[Control.scala 249:24]
  wire [31:0] mix_val_2; // @[Control.scala 215:40]
  wire [7:0] _T_793; // @[Control.scala 251:37]
  wire [31:0] _T_794; // @[Control.scala 252:30]
  wire [31:0] _T_795; // @[Control.scala 252:59]
  wire [32:0] _T_796; // @[Control.scala 252:49]
  wire [31:0] _T_797; // @[Control.scala 252:49]
  wire [31:0] _T_798; // @[Control.scala 252:79]
  wire [31:0] add_val_2; // @[Control.scala 215:40]
  wire [31:0] add_res_2; // @[Control.scala 215:40]
  wire [7:0] _T_799; // @[Control.scala 254:37]
  wire [4:0] _T_801; // @[Control.scala 255:60]
  wire [31:0] _T_802; // @[Control.scala 255:49]
  wire [31:0] _T_803; // @[Control.scala 255:84]
  wire [31:0] shr_val_2; // @[Control.scala 215:40]
  wire [31:0] shr_res_2; // @[Control.scala 215:40]
  wire [7:0] _T_804; // @[Control.scala 257:37]
  wire [31:0] src_0_3; // @[Control.scala 215:40]
  wire [31:0] src_1_3; // @[Control.scala 215:40]
  wire  _T_805; // @[Control.scala 249:34]
  wire [31:0] _T_806; // @[Control.scala 249:24]
  wire [31:0] mix_val_3; // @[Control.scala 215:40]
  wire [7:0] _T_807; // @[Control.scala 251:37]
  wire [31:0] _T_808; // @[Control.scala 252:30]
  wire [31:0] _T_809; // @[Control.scala 252:59]
  wire [32:0] _T_810; // @[Control.scala 252:49]
  wire [31:0] _T_811; // @[Control.scala 252:49]
  wire [31:0] _T_812; // @[Control.scala 252:79]
  wire [31:0] add_val_3; // @[Control.scala 215:40]
  wire [31:0] add_res_3; // @[Control.scala 215:40]
  wire [7:0] _T_813; // @[Control.scala 254:37]
  wire [4:0] _T_815; // @[Control.scala 255:60]
  wire [31:0] _T_816; // @[Control.scala 255:49]
  wire [31:0] _T_817; // @[Control.scala 255:84]
  wire [31:0] shr_val_3; // @[Control.scala 215:40]
  wire [31:0] shr_res_3; // @[Control.scala 215:40]
  wire [7:0] _T_818; // @[Control.scala 257:37]
  wire [31:0] src_0_4; // @[Control.scala 215:40]
  wire [31:0] src_1_4; // @[Control.scala 215:40]
  wire  _T_819; // @[Control.scala 249:34]
  wire [31:0] _T_820; // @[Control.scala 249:24]
  wire [31:0] mix_val_4; // @[Control.scala 215:40]
  wire [7:0] _T_821; // @[Control.scala 251:37]
  wire [31:0] _T_822; // @[Control.scala 252:30]
  wire [31:0] _T_823; // @[Control.scala 252:59]
  wire [32:0] _T_824; // @[Control.scala 252:49]
  wire [31:0] _T_825; // @[Control.scala 252:49]
  wire [31:0] _T_826; // @[Control.scala 252:79]
  wire [31:0] add_val_4; // @[Control.scala 215:40]
  wire [31:0] add_res_4; // @[Control.scala 215:40]
  wire [7:0] _T_827; // @[Control.scala 254:37]
  wire [4:0] _T_829; // @[Control.scala 255:60]
  wire [31:0] _T_830; // @[Control.scala 255:49]
  wire [31:0] _T_831; // @[Control.scala 255:84]
  wire [31:0] shr_val_4; // @[Control.scala 215:40]
  wire [31:0] shr_res_4; // @[Control.scala 215:40]
  wire [7:0] _T_832; // @[Control.scala 257:37]
  wire [31:0] src_0_5; // @[Control.scala 215:40]
  wire [31:0] src_1_5; // @[Control.scala 215:40]
  wire  _T_833; // @[Control.scala 249:34]
  wire [31:0] _T_834; // @[Control.scala 249:24]
  wire [31:0] mix_val_5; // @[Control.scala 215:40]
  wire [7:0] _T_835; // @[Control.scala 251:37]
  wire [31:0] _T_836; // @[Control.scala 252:30]
  wire [31:0] _T_837; // @[Control.scala 252:59]
  wire [32:0] _T_838; // @[Control.scala 252:49]
  wire [31:0] _T_839; // @[Control.scala 252:49]
  wire [31:0] _T_840; // @[Control.scala 252:79]
  wire [31:0] add_val_5; // @[Control.scala 215:40]
  wire [31:0] add_res_5; // @[Control.scala 215:40]
  wire [7:0] _T_841; // @[Control.scala 254:37]
  wire [4:0] _T_843; // @[Control.scala 255:60]
  wire [31:0] _T_844; // @[Control.scala 255:49]
  wire [31:0] _T_845; // @[Control.scala 255:84]
  wire [31:0] shr_val_5; // @[Control.scala 215:40]
  wire [31:0] shr_res_5; // @[Control.scala 215:40]
  wire [7:0] _T_846; // @[Control.scala 257:37]
  wire [31:0] src_0_6; // @[Control.scala 215:40]
  wire [31:0] src_1_6; // @[Control.scala 215:40]
  wire  _T_847; // @[Control.scala 249:34]
  wire [31:0] _T_848; // @[Control.scala 249:24]
  wire [31:0] mix_val_6; // @[Control.scala 215:40]
  wire [7:0] _T_849; // @[Control.scala 251:37]
  wire [31:0] _T_850; // @[Control.scala 252:30]
  wire [31:0] _T_851; // @[Control.scala 252:59]
  wire [32:0] _T_852; // @[Control.scala 252:49]
  wire [31:0] _T_853; // @[Control.scala 252:49]
  wire [31:0] _T_854; // @[Control.scala 252:79]
  wire [31:0] add_val_6; // @[Control.scala 215:40]
  wire [31:0] add_res_6; // @[Control.scala 215:40]
  wire [7:0] _T_855; // @[Control.scala 254:37]
  wire [4:0] _T_857; // @[Control.scala 255:60]
  wire [31:0] _T_858; // @[Control.scala 255:49]
  wire [31:0] _T_859; // @[Control.scala 255:84]
  wire [31:0] shr_val_6; // @[Control.scala 215:40]
  wire [31:0] shr_res_6; // @[Control.scala 215:40]
  wire [7:0] _T_860; // @[Control.scala 257:37]
  wire [31:0] src_0_7; // @[Control.scala 215:40]
  wire [31:0] src_1_7; // @[Control.scala 215:40]
  wire  _T_861; // @[Control.scala 249:34]
  wire [31:0] _T_862; // @[Control.scala 249:24]
  wire [31:0] mix_val_7; // @[Control.scala 215:40]
  wire [7:0] _T_863; // @[Control.scala 251:37]
  wire [31:0] _T_864; // @[Control.scala 252:30]
  wire [31:0] _T_865; // @[Control.scala 252:59]
  wire [32:0] _T_866; // @[Control.scala 252:49]
  wire [31:0] _T_867; // @[Control.scala 252:49]
  wire [31:0] _T_868; // @[Control.scala 252:79]
  wire [31:0] add_val_7; // @[Control.scala 215:40]
  wire [31:0] add_res_7; // @[Control.scala 215:40]
  wire [7:0] _T_869; // @[Control.scala 254:37]
  wire [4:0] _T_871; // @[Control.scala 255:60]
  wire [31:0] _T_872; // @[Control.scala 255:49]
  wire [31:0] _T_873; // @[Control.scala 255:84]
  wire [31:0] shr_val_7; // @[Control.scala 215:40]
  wire [31:0] shr_res_7; // @[Control.scala 215:40]
  wire [7:0] _T_874; // @[Control.scala 257:37]
  wire [31:0] src_0_8; // @[Control.scala 215:40]
  wire [31:0] src_1_8; // @[Control.scala 215:40]
  wire  _T_875; // @[Control.scala 249:34]
  wire [31:0] _T_876; // @[Control.scala 249:24]
  wire [31:0] mix_val_8; // @[Control.scala 215:40]
  wire [7:0] _T_877; // @[Control.scala 251:37]
  wire [31:0] _T_878; // @[Control.scala 252:30]
  wire [31:0] _T_879; // @[Control.scala 252:59]
  wire [32:0] _T_880; // @[Control.scala 252:49]
  wire [31:0] _T_881; // @[Control.scala 252:49]
  wire [31:0] _T_882; // @[Control.scala 252:79]
  wire [31:0] add_val_8; // @[Control.scala 215:40]
  wire [31:0] add_res_8; // @[Control.scala 215:40]
  wire [7:0] _T_883; // @[Control.scala 254:37]
  wire [4:0] _T_885; // @[Control.scala 255:60]
  wire [31:0] _T_886; // @[Control.scala 255:49]
  wire [31:0] _T_887; // @[Control.scala 255:84]
  wire [31:0] shr_val_8; // @[Control.scala 215:40]
  wire [31:0] shr_res_8; // @[Control.scala 215:40]
  wire [7:0] _T_888; // @[Control.scala 257:37]
  wire [31:0] src_0_9; // @[Control.scala 215:40]
  wire [31:0] src_1_9; // @[Control.scala 215:40]
  wire  _T_889; // @[Control.scala 249:34]
  wire [31:0] _T_890; // @[Control.scala 249:24]
  wire [31:0] mix_val_9; // @[Control.scala 215:40]
  wire [7:0] _T_891; // @[Control.scala 251:37]
  wire [31:0] _T_892; // @[Control.scala 252:30]
  wire [31:0] _T_893; // @[Control.scala 252:59]
  wire [32:0] _T_894; // @[Control.scala 252:49]
  wire [31:0] _T_895; // @[Control.scala 252:49]
  wire [31:0] _T_896; // @[Control.scala 252:79]
  wire [31:0] add_val_9; // @[Control.scala 215:40]
  wire [31:0] add_res_9; // @[Control.scala 215:40]
  wire [7:0] _T_897; // @[Control.scala 254:37]
  wire [4:0] _T_899; // @[Control.scala 255:60]
  wire [31:0] _T_900; // @[Control.scala 255:49]
  wire [31:0] _T_901; // @[Control.scala 255:84]
  wire [31:0] shr_val_9; // @[Control.scala 215:40]
  wire [31:0] shr_res_9; // @[Control.scala 215:40]
  wire [7:0] _T_902; // @[Control.scala 257:37]
  wire [31:0] src_0_10; // @[Control.scala 215:40]
  wire [31:0] src_1_10; // @[Control.scala 215:40]
  wire  _T_903; // @[Control.scala 249:34]
  wire [31:0] _T_904; // @[Control.scala 249:24]
  wire [31:0] mix_val_10; // @[Control.scala 215:40]
  wire [7:0] _T_905; // @[Control.scala 251:37]
  wire [31:0] _T_906; // @[Control.scala 252:30]
  wire [31:0] _T_907; // @[Control.scala 252:59]
  wire [32:0] _T_908; // @[Control.scala 252:49]
  wire [31:0] _T_909; // @[Control.scala 252:49]
  wire [31:0] _T_910; // @[Control.scala 252:79]
  wire [31:0] add_val_10; // @[Control.scala 215:40]
  wire [31:0] add_res_10; // @[Control.scala 215:40]
  wire [7:0] _T_911; // @[Control.scala 254:37]
  wire [4:0] _T_913; // @[Control.scala 255:60]
  wire [31:0] _T_914; // @[Control.scala 255:49]
  wire [31:0] _T_915; // @[Control.scala 255:84]
  wire [31:0] shr_val_10; // @[Control.scala 215:40]
  wire [31:0] shr_res_10; // @[Control.scala 215:40]
  wire [7:0] _T_916; // @[Control.scala 257:37]
  wire [31:0] src_0_11; // @[Control.scala 215:40]
  wire [31:0] src_1_11; // @[Control.scala 215:40]
  wire  _T_917; // @[Control.scala 249:34]
  wire [31:0] _T_918; // @[Control.scala 249:24]
  wire [31:0] mix_val_11; // @[Control.scala 215:40]
  wire [7:0] _T_919; // @[Control.scala 251:37]
  wire [31:0] _T_920; // @[Control.scala 252:30]
  wire [31:0] _T_921; // @[Control.scala 252:59]
  wire [32:0] _T_922; // @[Control.scala 252:49]
  wire [31:0] _T_923; // @[Control.scala 252:49]
  wire [31:0] _T_924; // @[Control.scala 252:79]
  wire [31:0] add_val_11; // @[Control.scala 215:40]
  wire [31:0] add_res_11; // @[Control.scala 215:40]
  wire [7:0] _T_925; // @[Control.scala 254:37]
  wire [4:0] _T_927; // @[Control.scala 255:60]
  wire [31:0] _T_928; // @[Control.scala 255:49]
  wire [31:0] _T_929; // @[Control.scala 255:84]
  wire [31:0] shr_val_11; // @[Control.scala 215:40]
  wire [31:0] shr_res_11; // @[Control.scala 215:40]
  wire [7:0] _T_930; // @[Control.scala 257:37]
  wire [31:0] src_0_12; // @[Control.scala 215:40]
  wire [31:0] src_1_12; // @[Control.scala 215:40]
  wire  _T_931; // @[Control.scala 249:34]
  wire [31:0] _T_932; // @[Control.scala 249:24]
  wire [31:0] mix_val_12; // @[Control.scala 215:40]
  wire [7:0] _T_933; // @[Control.scala 251:37]
  wire [31:0] _T_934; // @[Control.scala 252:30]
  wire [31:0] _T_935; // @[Control.scala 252:59]
  wire [32:0] _T_936; // @[Control.scala 252:49]
  wire [31:0] _T_937; // @[Control.scala 252:49]
  wire [31:0] _T_938; // @[Control.scala 252:79]
  wire [31:0] add_val_12; // @[Control.scala 215:40]
  wire [31:0] add_res_12; // @[Control.scala 215:40]
  wire [7:0] _T_939; // @[Control.scala 254:37]
  wire [4:0] _T_941; // @[Control.scala 255:60]
  wire [31:0] _T_942; // @[Control.scala 255:49]
  wire [31:0] _T_943; // @[Control.scala 255:84]
  wire [31:0] shr_val_12; // @[Control.scala 215:40]
  wire [31:0] shr_res_12; // @[Control.scala 215:40]
  wire [7:0] _T_944; // @[Control.scala 257:37]
  wire [31:0] src_0_13; // @[Control.scala 215:40]
  wire [31:0] src_1_13; // @[Control.scala 215:40]
  wire  _T_945; // @[Control.scala 249:34]
  wire [31:0] _T_946; // @[Control.scala 249:24]
  wire [31:0] mix_val_13; // @[Control.scala 215:40]
  wire [7:0] _T_947; // @[Control.scala 251:37]
  wire [31:0] _T_948; // @[Control.scala 252:30]
  wire [31:0] _T_949; // @[Control.scala 252:59]
  wire [32:0] _T_950; // @[Control.scala 252:49]
  wire [31:0] _T_951; // @[Control.scala 252:49]
  wire [31:0] _T_952; // @[Control.scala 252:79]
  wire [31:0] add_val_13; // @[Control.scala 215:40]
  wire [31:0] add_res_13; // @[Control.scala 215:40]
  wire [7:0] _T_953; // @[Control.scala 254:37]
  wire [4:0] _T_955; // @[Control.scala 255:60]
  wire [31:0] _T_956; // @[Control.scala 255:49]
  wire [31:0] _T_957; // @[Control.scala 255:84]
  wire [31:0] shr_val_13; // @[Control.scala 215:40]
  wire [31:0] shr_res_13; // @[Control.scala 215:40]
  wire [7:0] _T_958; // @[Control.scala 257:37]
  wire [31:0] src_0_14; // @[Control.scala 215:40]
  wire [31:0] src_1_14; // @[Control.scala 215:40]
  wire  _T_959; // @[Control.scala 249:34]
  wire [31:0] _T_960; // @[Control.scala 249:24]
  wire [31:0] mix_val_14; // @[Control.scala 215:40]
  wire [7:0] _T_961; // @[Control.scala 251:37]
  wire [31:0] _T_962; // @[Control.scala 252:30]
  wire [31:0] _T_963; // @[Control.scala 252:59]
  wire [32:0] _T_964; // @[Control.scala 252:49]
  wire [31:0] _T_965; // @[Control.scala 252:49]
  wire [31:0] _T_966; // @[Control.scala 252:79]
  wire [31:0] add_val_14; // @[Control.scala 215:40]
  wire [31:0] add_res_14; // @[Control.scala 215:40]
  wire [7:0] _T_967; // @[Control.scala 254:37]
  wire [4:0] _T_969; // @[Control.scala 255:60]
  wire [31:0] _T_970; // @[Control.scala 255:49]
  wire [31:0] _T_971; // @[Control.scala 255:84]
  wire [31:0] shr_val_14; // @[Control.scala 215:40]
  wire [31:0] shr_res_14; // @[Control.scala 215:40]
  wire [7:0] _T_972; // @[Control.scala 257:37]
  wire [31:0] src_0_15; // @[Control.scala 215:40]
  wire [31:0] src_1_15; // @[Control.scala 215:40]
  wire  _T_973; // @[Control.scala 249:34]
  wire [31:0] _T_974; // @[Control.scala 249:24]
  wire [31:0] mix_val_15; // @[Control.scala 215:40]
  wire [7:0] _T_975; // @[Control.scala 251:37]
  wire [31:0] _T_976; // @[Control.scala 252:30]
  wire [31:0] _T_977; // @[Control.scala 252:59]
  wire [32:0] _T_978; // @[Control.scala 252:49]
  wire [31:0] _T_979; // @[Control.scala 252:49]
  wire [31:0] _T_980; // @[Control.scala 252:79]
  wire [31:0] add_val_15; // @[Control.scala 215:40]
  wire [31:0] add_res_15; // @[Control.scala 215:40]
  wire [7:0] _T_981; // @[Control.scala 254:37]
  wire [4:0] _T_983; // @[Control.scala 255:60]
  wire [31:0] _T_984; // @[Control.scala 255:49]
  wire [31:0] _T_985; // @[Control.scala 255:84]
  wire [31:0] shr_val_15; // @[Control.scala 215:40]
  wire [31:0] shr_res_15; // @[Control.scala 215:40]
  wire [7:0] _T_986; // @[Control.scala 257:37]
  wire [7:0] short_cmp_res_0; // @[Control.scala 215:40]
  wire [7:0] short_add_res_0; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_0; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_1; // @[Control.scala 215:40]
  wire [7:0] short_add_res_1; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_1; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_2; // @[Control.scala 215:40]
  wire [7:0] short_add_res_2; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_2; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_3; // @[Control.scala 215:40]
  wire [7:0] short_add_res_3; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_3; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_4; // @[Control.scala 215:40]
  wire [7:0] short_add_res_4; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_4; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_5; // @[Control.scala 215:40]
  wire [7:0] short_add_res_5; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_5; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_6; // @[Control.scala 215:40]
  wire [7:0] short_add_res_6; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_6; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_7; // @[Control.scala 215:40]
  wire [7:0] short_add_res_7; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_7; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_8; // @[Control.scala 215:40]
  wire [7:0] short_add_res_8; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_8; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_9; // @[Control.scala 215:40]
  wire [7:0] short_add_res_9; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_9; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_10; // @[Control.scala 215:40]
  wire [7:0] short_add_res_10; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_10; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_11; // @[Control.scala 215:40]
  wire [7:0] short_add_res_11; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_11; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_12; // @[Control.scala 215:40]
  wire [7:0] short_add_res_12; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_12; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_13; // @[Control.scala 215:40]
  wire [7:0] short_add_res_13; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_13; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_14; // @[Control.scala 215:40]
  wire [7:0] short_add_res_14; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_14; // @[Control.scala 215:40]
  wire [7:0] short_cmp_res_15; // @[Control.scala 215:40]
  wire [7:0] short_add_res_15; // @[Control.scala 215:40]
  wire [7:0] short_shr_res_15; // @[Control.scala 215:40]
  wire  alu_opcode_minmax_en; // @[Control.scala 271:48]
  wire  alu_opcode_add_en; // @[Control.scala 272:39]
  wire [63:0] _T_995; // @[Cat.scala 30:58]
  wire [127:0] _T_1003; // @[Cat.scala 30:58]
  wire [63:0] _T_1010; // @[Cat.scala 30:58]
  wire [127:0] _T_1018; // @[Cat.scala 30:58]
  wire [63:0] _T_1025; // @[Cat.scala 30:58]
  wire [127:0] _T_1033; // @[Cat.scala 30:58]
  wire [127:0] _T_1034; // @[Control.scala 274:30]
  assign acc_mem__T_196_addr = dst_idx[7:0];
  assign acc_mem__T_196_data = acc_mem[acc_mem__T_196_addr]; // @[Control.scala 22:20]
  assign acc_mem__T_199_addr = src_idx[7:0];
  assign acc_mem__T_199_data = acc_mem[acc_mem__T_199_addr]; // @[Control.scala 22:20]
  assign acc_mem__T_181_data = io_biases_data;
  assign acc_mem__T_181_addr = acc_mem_addr[7:0];
  assign acc_mem__T_181_mask = 1'h1;
  assign acc_mem__T_181_en = opcode_load_en & memory_type_acc_en;
  assign insn_valid = insn != 128'h0; // @[Control.scala 25:30]
  assign opcode = insn[2:0]; // @[Control.scala 27:29]
  assign memory_type = insn[8:7]; // @[Control.scala 35:25]
  assign sram_base = insn[24:9]; // @[Control.scala 36:25]
  assign x_size = insn[95:80]; // @[Control.scala 39:25]
  assign y_pad_0 = insn[115:112]; // @[Control.scala 41:25]
  assign x_pad_0 = insn[123:120]; // @[Control.scala 43:25]
  assign x_pad_1 = insn[127:124]; // @[Control.scala 44:25]
  assign _GEN_249 = {{12'd0}, y_pad_0}; // @[Control.scala 48:30]
  assign _GEN_251 = {{12'd0}, x_pad_0}; // @[Control.scala 49:30]
  assign _T_95 = _GEN_251 + x_size; // @[Control.scala 49:30]
  assign _T_96 = _GEN_251 + x_size; // @[Control.scala 49:30]
  assign _GEN_252 = {{12'd0}, x_pad_1}; // @[Control.scala 49:39]
  assign _T_97 = _T_96 + _GEN_252; // @[Control.scala 49:39]
  assign x_size_total = _T_96 + _GEN_252; // @[Control.scala 49:39]
  assign y_offset = x_size_total * _GEN_249; // @[Control.scala 50:31]
  assign _T_100 = opcode == 3'h0; // @[Control.scala 54:32]
  assign _T_102 = opcode == 3'h1; // @[Control.scala 54:60]
  assign opcode_load_en = _T_100 | _T_102; // @[Control.scala 54:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Control.scala 55:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Control.scala 56:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Control.scala 58:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Control.scala 59:40]
  assign acc_x_cntr_en = opcode_load_en & memory_type_acc_en; // @[Control.scala 63:39]
  assign _T_110 = acc_x_cntr_en & io_biases_valid; // @[Control.scala 67:23]
  assign _T_112 = acc_x_cntr_val == 8'h7; // @[Control.scala 68:26]
  assign _T_116 = acc_x_cntr_val + 8'h1; // @[Control.scala 72:40]
  assign _T_117 = acc_x_cntr_val + 8'h1; // @[Control.scala 72:40]
  assign _GEN_0 = _T_112 ? 8'h0 : _T_117; // @[Control.scala 68:54]
  assign _GEN_2 = _T_110 ? _GEN_0 : acc_x_cntr_val; // @[Control.scala 67:43]
  assign acc_x_cntr_wrap = _T_110 ? _T_112 : 1'h0; // @[Control.scala 67:43]
  assign in_loop_cntr_en = opcode_alu_en | opcode_gemm_en; // @[Control.scala 81:40]
  assign _T_124 = io_out_mem_waitrequest == 1'h0; // @[Control.scala 85:28]
  assign _T_125 = in_loop_cntr_en & _T_124; // @[Control.scala 85:25]
  assign _T_127 = dst_offset_in == 8'h7; // @[Control.scala 86:28]
  assign _T_131 = dst_offset_in + 8'h1; // @[Control.scala 90:44]
  assign _T_132 = dst_offset_in + 8'h1; // @[Control.scala 90:44]
  assign _GEN_4 = _T_127 ? 8'h0 : _T_132; // @[Control.scala 86:58]
  assign _GEN_6 = _T_125 ? _GEN_4 : dst_offset_in; // @[Control.scala 85:53]
  assign in_loop_cntr_wrap = _T_125 ? _T_127 : 1'h0; // @[Control.scala 85:53]
  assign _T_135 = opcode_load_en & memory_type_uop_en; // @[Control.scala 100:33]
  assign _T_138 = acc_x_cntr_wrap == 1'h0; // @[Control.scala 101:35]
  assign _T_139 = acc_x_cntr_en & _T_138; // @[Control.scala 101:32]
  assign _T_142 = in_loop_cntr_wrap == 1'h0; // @[Control.scala 102:37]
  assign _T_143 = in_loop_cntr_en & _T_142; // @[Control.scala 102:34]
  assign _T_147 = _T_139 ? 1'h1 : _T_143; // @[Control.scala 101:17]
  assign busy = _T_135 ? 1'h0 : _T_147; // @[Control.scala 100:17]
  assign _T_149 = busy == 1'h0; // @[Control.scala 106:32]
  assign _T_150 = io_gemm_queue_valid & _T_149; // @[Control.scala 106:29]
  assign _T_153 = io_uops_valid & memory_type_uop_en; // @[Control.scala 115:23]
  assign _T_154 = _T_153 & insn_valid; // @[Control.scala 115:45]
  assign _GEN_254 = {{4'd0}, sram_base}; // @[Control.scala 142:33]
  assign _T_173 = _GEN_254 + y_offset; // @[Control.scala 142:33]
  assign _T_174 = _GEN_254 + y_offset; // @[Control.scala 142:33]
  assign _GEN_255 = {{16'd0}, x_pad_0}; // @[Control.scala 142:44]
  assign _T_175 = _T_174 + _GEN_255; // @[Control.scala 142:44]
  assign _T_176 = _T_174 + _GEN_255; // @[Control.scala 142:44]
  assign _T_178 = _T_176 * 20'h1; // @[Control.scala 142:55]
  assign _GEN_256 = {{13'd0}, acc_x_cntr_val}; // @[Control.scala 142:65]
  assign _T_179 = _T_178 + _GEN_256; // @[Control.scala 142:65]
  assign acc_mem_addr = _T_178 + _GEN_256; // @[Control.scala 142:65]
  assign alu_opcode = insn[109:108]; // @[Control.scala 164:24]
  assign use_imm = insn[110]; // @[Control.scala 165:21]
  assign imm_raw = insn[126:111]; // @[Control.scala 166:21]
  assign _T_182 = $signed(imm_raw); // @[Control.scala 167:25]
  assign _T_184 = $signed(_T_182) < $signed(16'sh0); // @[Control.scala 167:32]
  assign _T_186 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_188 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_189 = _T_184 ? _T_186 : {{15'd0}, _T_188}; // @[Control.scala 167:16]
  assign imm = $signed(_T_189); // @[Control.scala 167:89]
  assign _T_191 = io_uop_mem_readdata[10:0]; // @[Control.scala 176:20]
  assign _GEN_257 = {{3'd0}, dst_offset_in}; // @[Control.scala 176:47]
  assign _T_192 = _T_191 + _GEN_257; // @[Control.scala 176:47]
  assign dst_idx = _T_191 + _GEN_257; // @[Control.scala 176:47]
  assign _T_193 = io_uop_mem_readdata[21:11]; // @[Control.scala 177:20]
  assign _T_194 = _T_193 + _GEN_257; // @[Control.scala 177:47]
  assign src_idx = _T_193 + _GEN_257; // @[Control.scala 177:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Control.scala 196:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Control.scala 197:38]
  assign _T_634 = insn_valid & in_loop_cntr_en; // @[Control.scala 215:20]
  assign _T_635 = src_vector[31:0]; // @[Control.scala 235:31]
  assign _T_636 = $signed(_T_635); // @[Control.scala 235:72]
  assign _T_637 = dst_vector[31:0]; // @[Control.scala 236:31]
  assign _T_638 = $signed(_T_637); // @[Control.scala 236:72]
  assign _T_639 = src_vector[63:32]; // @[Control.scala 235:31]
  assign _T_640 = $signed(_T_639); // @[Control.scala 235:72]
  assign _T_641 = dst_vector[63:32]; // @[Control.scala 236:31]
  assign _T_642 = $signed(_T_641); // @[Control.scala 236:72]
  assign _T_643 = src_vector[95:64]; // @[Control.scala 235:31]
  assign _T_644 = $signed(_T_643); // @[Control.scala 235:72]
  assign _T_645 = dst_vector[95:64]; // @[Control.scala 236:31]
  assign _T_646 = $signed(_T_645); // @[Control.scala 236:72]
  assign _T_647 = src_vector[127:96]; // @[Control.scala 235:31]
  assign _T_648 = $signed(_T_647); // @[Control.scala 235:72]
  assign _T_649 = dst_vector[127:96]; // @[Control.scala 236:31]
  assign _T_650 = $signed(_T_649); // @[Control.scala 236:72]
  assign _T_651 = src_vector[159:128]; // @[Control.scala 235:31]
  assign _T_652 = $signed(_T_651); // @[Control.scala 235:72]
  assign _T_653 = dst_vector[159:128]; // @[Control.scala 236:31]
  assign _T_654 = $signed(_T_653); // @[Control.scala 236:72]
  assign _T_655 = src_vector[191:160]; // @[Control.scala 235:31]
  assign _T_656 = $signed(_T_655); // @[Control.scala 235:72]
  assign _T_657 = dst_vector[191:160]; // @[Control.scala 236:31]
  assign _T_658 = $signed(_T_657); // @[Control.scala 236:72]
  assign _T_659 = src_vector[223:192]; // @[Control.scala 235:31]
  assign _T_660 = $signed(_T_659); // @[Control.scala 235:72]
  assign _T_661 = dst_vector[223:192]; // @[Control.scala 236:31]
  assign _T_662 = $signed(_T_661); // @[Control.scala 236:72]
  assign _T_663 = src_vector[255:224]; // @[Control.scala 235:31]
  assign _T_664 = $signed(_T_663); // @[Control.scala 235:72]
  assign _T_665 = dst_vector[255:224]; // @[Control.scala 236:31]
  assign _T_666 = $signed(_T_665); // @[Control.scala 236:72]
  assign _T_667 = src_vector[287:256]; // @[Control.scala 235:31]
  assign _T_668 = $signed(_T_667); // @[Control.scala 235:72]
  assign _T_669 = dst_vector[287:256]; // @[Control.scala 236:31]
  assign _T_670 = $signed(_T_669); // @[Control.scala 236:72]
  assign _T_671 = src_vector[319:288]; // @[Control.scala 235:31]
  assign _T_672 = $signed(_T_671); // @[Control.scala 235:72]
  assign _T_673 = dst_vector[319:288]; // @[Control.scala 236:31]
  assign _T_674 = $signed(_T_673); // @[Control.scala 236:72]
  assign _T_675 = src_vector[351:320]; // @[Control.scala 235:31]
  assign _T_676 = $signed(_T_675); // @[Control.scala 235:72]
  assign _T_677 = dst_vector[351:320]; // @[Control.scala 236:31]
  assign _T_678 = $signed(_T_677); // @[Control.scala 236:72]
  assign _T_679 = src_vector[383:352]; // @[Control.scala 235:31]
  assign _T_680 = $signed(_T_679); // @[Control.scala 235:72]
  assign _T_681 = dst_vector[383:352]; // @[Control.scala 236:31]
  assign _T_682 = $signed(_T_681); // @[Control.scala 236:72]
  assign _T_683 = src_vector[415:384]; // @[Control.scala 235:31]
  assign _T_684 = $signed(_T_683); // @[Control.scala 235:72]
  assign _T_685 = dst_vector[415:384]; // @[Control.scala 236:31]
  assign _T_686 = $signed(_T_685); // @[Control.scala 236:72]
  assign _T_687 = src_vector[447:416]; // @[Control.scala 235:31]
  assign _T_688 = $signed(_T_687); // @[Control.scala 235:72]
  assign _T_689 = dst_vector[447:416]; // @[Control.scala 236:31]
  assign _T_690 = $signed(_T_689); // @[Control.scala 236:72]
  assign _T_691 = src_vector[479:448]; // @[Control.scala 235:31]
  assign _T_692 = $signed(_T_691); // @[Control.scala 235:72]
  assign _T_693 = dst_vector[479:448]; // @[Control.scala 236:31]
  assign _T_694 = $signed(_T_693); // @[Control.scala 236:72]
  assign _T_695 = src_vector[511:480]; // @[Control.scala 235:31]
  assign _T_696 = $signed(_T_695); // @[Control.scala 235:72]
  assign _T_697 = dst_vector[511:480]; // @[Control.scala 236:31]
  assign _T_698 = $signed(_T_697); // @[Control.scala 236:72]
  assign _GEN_17 = alu_opcode_max_en ? $signed(_T_636) : $signed(_T_638); // @[Control.scala 233:30]
  assign _GEN_18 = alu_opcode_max_en ? $signed(_T_638) : $signed(_T_636); // @[Control.scala 233:30]
  assign _GEN_19 = alu_opcode_max_en ? $signed(_T_640) : $signed(_T_642); // @[Control.scala 233:30]
  assign _GEN_20 = alu_opcode_max_en ? $signed(_T_642) : $signed(_T_640); // @[Control.scala 233:30]
  assign _GEN_21 = alu_opcode_max_en ? $signed(_T_644) : $signed(_T_646); // @[Control.scala 233:30]
  assign _GEN_22 = alu_opcode_max_en ? $signed(_T_646) : $signed(_T_644); // @[Control.scala 233:30]
  assign _GEN_23 = alu_opcode_max_en ? $signed(_T_648) : $signed(_T_650); // @[Control.scala 233:30]
  assign _GEN_24 = alu_opcode_max_en ? $signed(_T_650) : $signed(_T_648); // @[Control.scala 233:30]
  assign _GEN_25 = alu_opcode_max_en ? $signed(_T_652) : $signed(_T_654); // @[Control.scala 233:30]
  assign _GEN_26 = alu_opcode_max_en ? $signed(_T_654) : $signed(_T_652); // @[Control.scala 233:30]
  assign _GEN_27 = alu_opcode_max_en ? $signed(_T_656) : $signed(_T_658); // @[Control.scala 233:30]
  assign _GEN_28 = alu_opcode_max_en ? $signed(_T_658) : $signed(_T_656); // @[Control.scala 233:30]
  assign _GEN_29 = alu_opcode_max_en ? $signed(_T_660) : $signed(_T_662); // @[Control.scala 233:30]
  assign _GEN_30 = alu_opcode_max_en ? $signed(_T_662) : $signed(_T_660); // @[Control.scala 233:30]
  assign _GEN_31 = alu_opcode_max_en ? $signed(_T_664) : $signed(_T_666); // @[Control.scala 233:30]
  assign _GEN_32 = alu_opcode_max_en ? $signed(_T_666) : $signed(_T_664); // @[Control.scala 233:30]
  assign _GEN_33 = alu_opcode_max_en ? $signed(_T_668) : $signed(_T_670); // @[Control.scala 233:30]
  assign _GEN_34 = alu_opcode_max_en ? $signed(_T_670) : $signed(_T_668); // @[Control.scala 233:30]
  assign _GEN_35 = alu_opcode_max_en ? $signed(_T_672) : $signed(_T_674); // @[Control.scala 233:30]
  assign _GEN_36 = alu_opcode_max_en ? $signed(_T_674) : $signed(_T_672); // @[Control.scala 233:30]
  assign _GEN_37 = alu_opcode_max_en ? $signed(_T_676) : $signed(_T_678); // @[Control.scala 233:30]
  assign _GEN_38 = alu_opcode_max_en ? $signed(_T_678) : $signed(_T_676); // @[Control.scala 233:30]
  assign _GEN_39 = alu_opcode_max_en ? $signed(_T_680) : $signed(_T_682); // @[Control.scala 233:30]
  assign _GEN_40 = alu_opcode_max_en ? $signed(_T_682) : $signed(_T_680); // @[Control.scala 233:30]
  assign _GEN_41 = alu_opcode_max_en ? $signed(_T_684) : $signed(_T_686); // @[Control.scala 233:30]
  assign _GEN_42 = alu_opcode_max_en ? $signed(_T_686) : $signed(_T_684); // @[Control.scala 233:30]
  assign _GEN_43 = alu_opcode_max_en ? $signed(_T_688) : $signed(_T_690); // @[Control.scala 233:30]
  assign _GEN_44 = alu_opcode_max_en ? $signed(_T_690) : $signed(_T_688); // @[Control.scala 233:30]
  assign _GEN_45 = alu_opcode_max_en ? $signed(_T_692) : $signed(_T_694); // @[Control.scala 233:30]
  assign _GEN_46 = alu_opcode_max_en ? $signed(_T_694) : $signed(_T_692); // @[Control.scala 233:30]
  assign _GEN_47 = alu_opcode_max_en ? $signed(_T_696) : $signed(_T_698); // @[Control.scala 233:30]
  assign _GEN_48 = alu_opcode_max_en ? $signed(_T_698) : $signed(_T_696); // @[Control.scala 233:30]
  assign _GEN_49 = use_imm ? $signed(imm) : $signed(_GEN_18); // @[Control.scala 244:20]
  assign _GEN_50 = use_imm ? $signed(imm) : $signed(_GEN_20); // @[Control.scala 244:20]
  assign _GEN_51 = use_imm ? $signed(imm) : $signed(_GEN_22); // @[Control.scala 244:20]
  assign _GEN_52 = use_imm ? $signed(imm) : $signed(_GEN_24); // @[Control.scala 244:20]
  assign _GEN_53 = use_imm ? $signed(imm) : $signed(_GEN_26); // @[Control.scala 244:20]
  assign _GEN_54 = use_imm ? $signed(imm) : $signed(_GEN_28); // @[Control.scala 244:20]
  assign _GEN_55 = use_imm ? $signed(imm) : $signed(_GEN_30); // @[Control.scala 244:20]
  assign _GEN_56 = use_imm ? $signed(imm) : $signed(_GEN_32); // @[Control.scala 244:20]
  assign _GEN_57 = use_imm ? $signed(imm) : $signed(_GEN_34); // @[Control.scala 244:20]
  assign _GEN_58 = use_imm ? $signed(imm) : $signed(_GEN_36); // @[Control.scala 244:20]
  assign _GEN_59 = use_imm ? $signed(imm) : $signed(_GEN_38); // @[Control.scala 244:20]
  assign _GEN_60 = use_imm ? $signed(imm) : $signed(_GEN_40); // @[Control.scala 244:20]
  assign _GEN_61 = use_imm ? $signed(imm) : $signed(_GEN_42); // @[Control.scala 244:20]
  assign _GEN_62 = use_imm ? $signed(imm) : $signed(_GEN_44); // @[Control.scala 244:20]
  assign _GEN_63 = use_imm ? $signed(imm) : $signed(_GEN_46); // @[Control.scala 244:20]
  assign _GEN_64 = use_imm ? $signed(imm) : $signed(_GEN_48); // @[Control.scala 244:20]
  assign src_0_0 = _T_634 ? $signed(_GEN_17) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_0 = _T_634 ? $signed(_GEN_49) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_763 = $signed(src_0_0) < $signed(src_1_0); // @[Control.scala 249:34]
  assign _T_764 = _T_763 ? $signed(src_0_0) : $signed(src_1_0); // @[Control.scala 249:24]
  assign mix_val_0 = _T_634 ? $signed(_T_764) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_765 = mix_val_0[7:0]; // @[Control.scala 251:37]
  assign _T_766 = $unsigned(src_0_0); // @[Control.scala 252:30]
  assign _T_767 = $unsigned(src_1_0); // @[Control.scala 252:59]
  assign _T_768 = _T_766 + _T_767; // @[Control.scala 252:49]
  assign _T_769 = _T_766 + _T_767; // @[Control.scala 252:49]
  assign _T_770 = $signed(_T_769); // @[Control.scala 252:79]
  assign add_val_0 = _T_634 ? $signed(_T_770) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_0 = _T_634 ? $signed(add_val_0) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_771 = add_res_0[7:0]; // @[Control.scala 254:37]
  assign _T_773 = src_1_0[4:0]; // @[Control.scala 255:60]
  assign _T_774 = _T_766 >> _T_773; // @[Control.scala 255:49]
  assign _T_775 = $signed(_T_774); // @[Control.scala 255:84]
  assign shr_val_0 = _T_634 ? $signed(_T_775) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_0 = _T_634 ? $signed(shr_val_0) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_776 = shr_res_0[7:0]; // @[Control.scala 257:37]
  assign src_0_1 = _T_634 ? $signed(_GEN_19) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_1 = _T_634 ? $signed(_GEN_50) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_777 = $signed(src_0_1) < $signed(src_1_1); // @[Control.scala 249:34]
  assign _T_778 = _T_777 ? $signed(src_0_1) : $signed(src_1_1); // @[Control.scala 249:24]
  assign mix_val_1 = _T_634 ? $signed(_T_778) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_779 = mix_val_1[7:0]; // @[Control.scala 251:37]
  assign _T_780 = $unsigned(src_0_1); // @[Control.scala 252:30]
  assign _T_781 = $unsigned(src_1_1); // @[Control.scala 252:59]
  assign _T_782 = _T_780 + _T_781; // @[Control.scala 252:49]
  assign _T_783 = _T_780 + _T_781; // @[Control.scala 252:49]
  assign _T_784 = $signed(_T_783); // @[Control.scala 252:79]
  assign add_val_1 = _T_634 ? $signed(_T_784) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_1 = _T_634 ? $signed(add_val_1) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_785 = add_res_1[7:0]; // @[Control.scala 254:37]
  assign _T_787 = src_1_1[4:0]; // @[Control.scala 255:60]
  assign _T_788 = _T_780 >> _T_787; // @[Control.scala 255:49]
  assign _T_789 = $signed(_T_788); // @[Control.scala 255:84]
  assign shr_val_1 = _T_634 ? $signed(_T_789) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_1 = _T_634 ? $signed(shr_val_1) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_790 = shr_res_1[7:0]; // @[Control.scala 257:37]
  assign src_0_2 = _T_634 ? $signed(_GEN_21) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_2 = _T_634 ? $signed(_GEN_51) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_791 = $signed(src_0_2) < $signed(src_1_2); // @[Control.scala 249:34]
  assign _T_792 = _T_791 ? $signed(src_0_2) : $signed(src_1_2); // @[Control.scala 249:24]
  assign mix_val_2 = _T_634 ? $signed(_T_792) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_793 = mix_val_2[7:0]; // @[Control.scala 251:37]
  assign _T_794 = $unsigned(src_0_2); // @[Control.scala 252:30]
  assign _T_795 = $unsigned(src_1_2); // @[Control.scala 252:59]
  assign _T_796 = _T_794 + _T_795; // @[Control.scala 252:49]
  assign _T_797 = _T_794 + _T_795; // @[Control.scala 252:49]
  assign _T_798 = $signed(_T_797); // @[Control.scala 252:79]
  assign add_val_2 = _T_634 ? $signed(_T_798) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_2 = _T_634 ? $signed(add_val_2) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_799 = add_res_2[7:0]; // @[Control.scala 254:37]
  assign _T_801 = src_1_2[4:0]; // @[Control.scala 255:60]
  assign _T_802 = _T_794 >> _T_801; // @[Control.scala 255:49]
  assign _T_803 = $signed(_T_802); // @[Control.scala 255:84]
  assign shr_val_2 = _T_634 ? $signed(_T_803) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_2 = _T_634 ? $signed(shr_val_2) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_804 = shr_res_2[7:0]; // @[Control.scala 257:37]
  assign src_0_3 = _T_634 ? $signed(_GEN_23) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_3 = _T_634 ? $signed(_GEN_52) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_805 = $signed(src_0_3) < $signed(src_1_3); // @[Control.scala 249:34]
  assign _T_806 = _T_805 ? $signed(src_0_3) : $signed(src_1_3); // @[Control.scala 249:24]
  assign mix_val_3 = _T_634 ? $signed(_T_806) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_807 = mix_val_3[7:0]; // @[Control.scala 251:37]
  assign _T_808 = $unsigned(src_0_3); // @[Control.scala 252:30]
  assign _T_809 = $unsigned(src_1_3); // @[Control.scala 252:59]
  assign _T_810 = _T_808 + _T_809; // @[Control.scala 252:49]
  assign _T_811 = _T_808 + _T_809; // @[Control.scala 252:49]
  assign _T_812 = $signed(_T_811); // @[Control.scala 252:79]
  assign add_val_3 = _T_634 ? $signed(_T_812) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_3 = _T_634 ? $signed(add_val_3) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_813 = add_res_3[7:0]; // @[Control.scala 254:37]
  assign _T_815 = src_1_3[4:0]; // @[Control.scala 255:60]
  assign _T_816 = _T_808 >> _T_815; // @[Control.scala 255:49]
  assign _T_817 = $signed(_T_816); // @[Control.scala 255:84]
  assign shr_val_3 = _T_634 ? $signed(_T_817) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_3 = _T_634 ? $signed(shr_val_3) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_818 = shr_res_3[7:0]; // @[Control.scala 257:37]
  assign src_0_4 = _T_634 ? $signed(_GEN_25) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_4 = _T_634 ? $signed(_GEN_53) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_819 = $signed(src_0_4) < $signed(src_1_4); // @[Control.scala 249:34]
  assign _T_820 = _T_819 ? $signed(src_0_4) : $signed(src_1_4); // @[Control.scala 249:24]
  assign mix_val_4 = _T_634 ? $signed(_T_820) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_821 = mix_val_4[7:0]; // @[Control.scala 251:37]
  assign _T_822 = $unsigned(src_0_4); // @[Control.scala 252:30]
  assign _T_823 = $unsigned(src_1_4); // @[Control.scala 252:59]
  assign _T_824 = _T_822 + _T_823; // @[Control.scala 252:49]
  assign _T_825 = _T_822 + _T_823; // @[Control.scala 252:49]
  assign _T_826 = $signed(_T_825); // @[Control.scala 252:79]
  assign add_val_4 = _T_634 ? $signed(_T_826) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_4 = _T_634 ? $signed(add_val_4) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_827 = add_res_4[7:0]; // @[Control.scala 254:37]
  assign _T_829 = src_1_4[4:0]; // @[Control.scala 255:60]
  assign _T_830 = _T_822 >> _T_829; // @[Control.scala 255:49]
  assign _T_831 = $signed(_T_830); // @[Control.scala 255:84]
  assign shr_val_4 = _T_634 ? $signed(_T_831) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_4 = _T_634 ? $signed(shr_val_4) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_832 = shr_res_4[7:0]; // @[Control.scala 257:37]
  assign src_0_5 = _T_634 ? $signed(_GEN_27) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_5 = _T_634 ? $signed(_GEN_54) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_833 = $signed(src_0_5) < $signed(src_1_5); // @[Control.scala 249:34]
  assign _T_834 = _T_833 ? $signed(src_0_5) : $signed(src_1_5); // @[Control.scala 249:24]
  assign mix_val_5 = _T_634 ? $signed(_T_834) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_835 = mix_val_5[7:0]; // @[Control.scala 251:37]
  assign _T_836 = $unsigned(src_0_5); // @[Control.scala 252:30]
  assign _T_837 = $unsigned(src_1_5); // @[Control.scala 252:59]
  assign _T_838 = _T_836 + _T_837; // @[Control.scala 252:49]
  assign _T_839 = _T_836 + _T_837; // @[Control.scala 252:49]
  assign _T_840 = $signed(_T_839); // @[Control.scala 252:79]
  assign add_val_5 = _T_634 ? $signed(_T_840) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_5 = _T_634 ? $signed(add_val_5) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_841 = add_res_5[7:0]; // @[Control.scala 254:37]
  assign _T_843 = src_1_5[4:0]; // @[Control.scala 255:60]
  assign _T_844 = _T_836 >> _T_843; // @[Control.scala 255:49]
  assign _T_845 = $signed(_T_844); // @[Control.scala 255:84]
  assign shr_val_5 = _T_634 ? $signed(_T_845) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_5 = _T_634 ? $signed(shr_val_5) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_846 = shr_res_5[7:0]; // @[Control.scala 257:37]
  assign src_0_6 = _T_634 ? $signed(_GEN_29) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_6 = _T_634 ? $signed(_GEN_55) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_847 = $signed(src_0_6) < $signed(src_1_6); // @[Control.scala 249:34]
  assign _T_848 = _T_847 ? $signed(src_0_6) : $signed(src_1_6); // @[Control.scala 249:24]
  assign mix_val_6 = _T_634 ? $signed(_T_848) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_849 = mix_val_6[7:0]; // @[Control.scala 251:37]
  assign _T_850 = $unsigned(src_0_6); // @[Control.scala 252:30]
  assign _T_851 = $unsigned(src_1_6); // @[Control.scala 252:59]
  assign _T_852 = _T_850 + _T_851; // @[Control.scala 252:49]
  assign _T_853 = _T_850 + _T_851; // @[Control.scala 252:49]
  assign _T_854 = $signed(_T_853); // @[Control.scala 252:79]
  assign add_val_6 = _T_634 ? $signed(_T_854) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_6 = _T_634 ? $signed(add_val_6) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_855 = add_res_6[7:0]; // @[Control.scala 254:37]
  assign _T_857 = src_1_6[4:0]; // @[Control.scala 255:60]
  assign _T_858 = _T_850 >> _T_857; // @[Control.scala 255:49]
  assign _T_859 = $signed(_T_858); // @[Control.scala 255:84]
  assign shr_val_6 = _T_634 ? $signed(_T_859) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_6 = _T_634 ? $signed(shr_val_6) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_860 = shr_res_6[7:0]; // @[Control.scala 257:37]
  assign src_0_7 = _T_634 ? $signed(_GEN_31) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_7 = _T_634 ? $signed(_GEN_56) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_861 = $signed(src_0_7) < $signed(src_1_7); // @[Control.scala 249:34]
  assign _T_862 = _T_861 ? $signed(src_0_7) : $signed(src_1_7); // @[Control.scala 249:24]
  assign mix_val_7 = _T_634 ? $signed(_T_862) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_863 = mix_val_7[7:0]; // @[Control.scala 251:37]
  assign _T_864 = $unsigned(src_0_7); // @[Control.scala 252:30]
  assign _T_865 = $unsigned(src_1_7); // @[Control.scala 252:59]
  assign _T_866 = _T_864 + _T_865; // @[Control.scala 252:49]
  assign _T_867 = _T_864 + _T_865; // @[Control.scala 252:49]
  assign _T_868 = $signed(_T_867); // @[Control.scala 252:79]
  assign add_val_7 = _T_634 ? $signed(_T_868) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_7 = _T_634 ? $signed(add_val_7) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_869 = add_res_7[7:0]; // @[Control.scala 254:37]
  assign _T_871 = src_1_7[4:0]; // @[Control.scala 255:60]
  assign _T_872 = _T_864 >> _T_871; // @[Control.scala 255:49]
  assign _T_873 = $signed(_T_872); // @[Control.scala 255:84]
  assign shr_val_7 = _T_634 ? $signed(_T_873) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_7 = _T_634 ? $signed(shr_val_7) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_874 = shr_res_7[7:0]; // @[Control.scala 257:37]
  assign src_0_8 = _T_634 ? $signed(_GEN_33) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_8 = _T_634 ? $signed(_GEN_57) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_875 = $signed(src_0_8) < $signed(src_1_8); // @[Control.scala 249:34]
  assign _T_876 = _T_875 ? $signed(src_0_8) : $signed(src_1_8); // @[Control.scala 249:24]
  assign mix_val_8 = _T_634 ? $signed(_T_876) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_877 = mix_val_8[7:0]; // @[Control.scala 251:37]
  assign _T_878 = $unsigned(src_0_8); // @[Control.scala 252:30]
  assign _T_879 = $unsigned(src_1_8); // @[Control.scala 252:59]
  assign _T_880 = _T_878 + _T_879; // @[Control.scala 252:49]
  assign _T_881 = _T_878 + _T_879; // @[Control.scala 252:49]
  assign _T_882 = $signed(_T_881); // @[Control.scala 252:79]
  assign add_val_8 = _T_634 ? $signed(_T_882) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_8 = _T_634 ? $signed(add_val_8) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_883 = add_res_8[7:0]; // @[Control.scala 254:37]
  assign _T_885 = src_1_8[4:0]; // @[Control.scala 255:60]
  assign _T_886 = _T_878 >> _T_885; // @[Control.scala 255:49]
  assign _T_887 = $signed(_T_886); // @[Control.scala 255:84]
  assign shr_val_8 = _T_634 ? $signed(_T_887) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_8 = _T_634 ? $signed(shr_val_8) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_888 = shr_res_8[7:0]; // @[Control.scala 257:37]
  assign src_0_9 = _T_634 ? $signed(_GEN_35) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_9 = _T_634 ? $signed(_GEN_58) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_889 = $signed(src_0_9) < $signed(src_1_9); // @[Control.scala 249:34]
  assign _T_890 = _T_889 ? $signed(src_0_9) : $signed(src_1_9); // @[Control.scala 249:24]
  assign mix_val_9 = _T_634 ? $signed(_T_890) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_891 = mix_val_9[7:0]; // @[Control.scala 251:37]
  assign _T_892 = $unsigned(src_0_9); // @[Control.scala 252:30]
  assign _T_893 = $unsigned(src_1_9); // @[Control.scala 252:59]
  assign _T_894 = _T_892 + _T_893; // @[Control.scala 252:49]
  assign _T_895 = _T_892 + _T_893; // @[Control.scala 252:49]
  assign _T_896 = $signed(_T_895); // @[Control.scala 252:79]
  assign add_val_9 = _T_634 ? $signed(_T_896) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_9 = _T_634 ? $signed(add_val_9) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_897 = add_res_9[7:0]; // @[Control.scala 254:37]
  assign _T_899 = src_1_9[4:0]; // @[Control.scala 255:60]
  assign _T_900 = _T_892 >> _T_899; // @[Control.scala 255:49]
  assign _T_901 = $signed(_T_900); // @[Control.scala 255:84]
  assign shr_val_9 = _T_634 ? $signed(_T_901) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_9 = _T_634 ? $signed(shr_val_9) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_902 = shr_res_9[7:0]; // @[Control.scala 257:37]
  assign src_0_10 = _T_634 ? $signed(_GEN_37) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_10 = _T_634 ? $signed(_GEN_59) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_903 = $signed(src_0_10) < $signed(src_1_10); // @[Control.scala 249:34]
  assign _T_904 = _T_903 ? $signed(src_0_10) : $signed(src_1_10); // @[Control.scala 249:24]
  assign mix_val_10 = _T_634 ? $signed(_T_904) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_905 = mix_val_10[7:0]; // @[Control.scala 251:37]
  assign _T_906 = $unsigned(src_0_10); // @[Control.scala 252:30]
  assign _T_907 = $unsigned(src_1_10); // @[Control.scala 252:59]
  assign _T_908 = _T_906 + _T_907; // @[Control.scala 252:49]
  assign _T_909 = _T_906 + _T_907; // @[Control.scala 252:49]
  assign _T_910 = $signed(_T_909); // @[Control.scala 252:79]
  assign add_val_10 = _T_634 ? $signed(_T_910) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_10 = _T_634 ? $signed(add_val_10) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_911 = add_res_10[7:0]; // @[Control.scala 254:37]
  assign _T_913 = src_1_10[4:0]; // @[Control.scala 255:60]
  assign _T_914 = _T_906 >> _T_913; // @[Control.scala 255:49]
  assign _T_915 = $signed(_T_914); // @[Control.scala 255:84]
  assign shr_val_10 = _T_634 ? $signed(_T_915) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_10 = _T_634 ? $signed(shr_val_10) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_916 = shr_res_10[7:0]; // @[Control.scala 257:37]
  assign src_0_11 = _T_634 ? $signed(_GEN_39) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_11 = _T_634 ? $signed(_GEN_60) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_917 = $signed(src_0_11) < $signed(src_1_11); // @[Control.scala 249:34]
  assign _T_918 = _T_917 ? $signed(src_0_11) : $signed(src_1_11); // @[Control.scala 249:24]
  assign mix_val_11 = _T_634 ? $signed(_T_918) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_919 = mix_val_11[7:0]; // @[Control.scala 251:37]
  assign _T_920 = $unsigned(src_0_11); // @[Control.scala 252:30]
  assign _T_921 = $unsigned(src_1_11); // @[Control.scala 252:59]
  assign _T_922 = _T_920 + _T_921; // @[Control.scala 252:49]
  assign _T_923 = _T_920 + _T_921; // @[Control.scala 252:49]
  assign _T_924 = $signed(_T_923); // @[Control.scala 252:79]
  assign add_val_11 = _T_634 ? $signed(_T_924) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_11 = _T_634 ? $signed(add_val_11) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_925 = add_res_11[7:0]; // @[Control.scala 254:37]
  assign _T_927 = src_1_11[4:0]; // @[Control.scala 255:60]
  assign _T_928 = _T_920 >> _T_927; // @[Control.scala 255:49]
  assign _T_929 = $signed(_T_928); // @[Control.scala 255:84]
  assign shr_val_11 = _T_634 ? $signed(_T_929) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_11 = _T_634 ? $signed(shr_val_11) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_930 = shr_res_11[7:0]; // @[Control.scala 257:37]
  assign src_0_12 = _T_634 ? $signed(_GEN_41) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_12 = _T_634 ? $signed(_GEN_61) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_931 = $signed(src_0_12) < $signed(src_1_12); // @[Control.scala 249:34]
  assign _T_932 = _T_931 ? $signed(src_0_12) : $signed(src_1_12); // @[Control.scala 249:24]
  assign mix_val_12 = _T_634 ? $signed(_T_932) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_933 = mix_val_12[7:0]; // @[Control.scala 251:37]
  assign _T_934 = $unsigned(src_0_12); // @[Control.scala 252:30]
  assign _T_935 = $unsigned(src_1_12); // @[Control.scala 252:59]
  assign _T_936 = _T_934 + _T_935; // @[Control.scala 252:49]
  assign _T_937 = _T_934 + _T_935; // @[Control.scala 252:49]
  assign _T_938 = $signed(_T_937); // @[Control.scala 252:79]
  assign add_val_12 = _T_634 ? $signed(_T_938) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_12 = _T_634 ? $signed(add_val_12) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_939 = add_res_12[7:0]; // @[Control.scala 254:37]
  assign _T_941 = src_1_12[4:0]; // @[Control.scala 255:60]
  assign _T_942 = _T_934 >> _T_941; // @[Control.scala 255:49]
  assign _T_943 = $signed(_T_942); // @[Control.scala 255:84]
  assign shr_val_12 = _T_634 ? $signed(_T_943) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_12 = _T_634 ? $signed(shr_val_12) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_944 = shr_res_12[7:0]; // @[Control.scala 257:37]
  assign src_0_13 = _T_634 ? $signed(_GEN_43) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_13 = _T_634 ? $signed(_GEN_62) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_945 = $signed(src_0_13) < $signed(src_1_13); // @[Control.scala 249:34]
  assign _T_946 = _T_945 ? $signed(src_0_13) : $signed(src_1_13); // @[Control.scala 249:24]
  assign mix_val_13 = _T_634 ? $signed(_T_946) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_947 = mix_val_13[7:0]; // @[Control.scala 251:37]
  assign _T_948 = $unsigned(src_0_13); // @[Control.scala 252:30]
  assign _T_949 = $unsigned(src_1_13); // @[Control.scala 252:59]
  assign _T_950 = _T_948 + _T_949; // @[Control.scala 252:49]
  assign _T_951 = _T_948 + _T_949; // @[Control.scala 252:49]
  assign _T_952 = $signed(_T_951); // @[Control.scala 252:79]
  assign add_val_13 = _T_634 ? $signed(_T_952) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_13 = _T_634 ? $signed(add_val_13) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_953 = add_res_13[7:0]; // @[Control.scala 254:37]
  assign _T_955 = src_1_13[4:0]; // @[Control.scala 255:60]
  assign _T_956 = _T_948 >> _T_955; // @[Control.scala 255:49]
  assign _T_957 = $signed(_T_956); // @[Control.scala 255:84]
  assign shr_val_13 = _T_634 ? $signed(_T_957) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_13 = _T_634 ? $signed(shr_val_13) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_958 = shr_res_13[7:0]; // @[Control.scala 257:37]
  assign src_0_14 = _T_634 ? $signed(_GEN_45) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_14 = _T_634 ? $signed(_GEN_63) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_959 = $signed(src_0_14) < $signed(src_1_14); // @[Control.scala 249:34]
  assign _T_960 = _T_959 ? $signed(src_0_14) : $signed(src_1_14); // @[Control.scala 249:24]
  assign mix_val_14 = _T_634 ? $signed(_T_960) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_961 = mix_val_14[7:0]; // @[Control.scala 251:37]
  assign _T_962 = $unsigned(src_0_14); // @[Control.scala 252:30]
  assign _T_963 = $unsigned(src_1_14); // @[Control.scala 252:59]
  assign _T_964 = _T_962 + _T_963; // @[Control.scala 252:49]
  assign _T_965 = _T_962 + _T_963; // @[Control.scala 252:49]
  assign _T_966 = $signed(_T_965); // @[Control.scala 252:79]
  assign add_val_14 = _T_634 ? $signed(_T_966) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_14 = _T_634 ? $signed(add_val_14) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_967 = add_res_14[7:0]; // @[Control.scala 254:37]
  assign _T_969 = src_1_14[4:0]; // @[Control.scala 255:60]
  assign _T_970 = _T_962 >> _T_969; // @[Control.scala 255:49]
  assign _T_971 = $signed(_T_970); // @[Control.scala 255:84]
  assign shr_val_14 = _T_634 ? $signed(_T_971) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_14 = _T_634 ? $signed(shr_val_14) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_972 = shr_res_14[7:0]; // @[Control.scala 257:37]
  assign src_0_15 = _T_634 ? $signed(_GEN_47) : $signed(32'sh0); // @[Control.scala 215:40]
  assign src_1_15 = _T_634 ? $signed(_GEN_64) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_973 = $signed(src_0_15) < $signed(src_1_15); // @[Control.scala 249:34]
  assign _T_974 = _T_973 ? $signed(src_0_15) : $signed(src_1_15); // @[Control.scala 249:24]
  assign mix_val_15 = _T_634 ? $signed(_T_974) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_975 = mix_val_15[7:0]; // @[Control.scala 251:37]
  assign _T_976 = $unsigned(src_0_15); // @[Control.scala 252:30]
  assign _T_977 = $unsigned(src_1_15); // @[Control.scala 252:59]
  assign _T_978 = _T_976 + _T_977; // @[Control.scala 252:49]
  assign _T_979 = _T_976 + _T_977; // @[Control.scala 252:49]
  assign _T_980 = $signed(_T_979); // @[Control.scala 252:79]
  assign add_val_15 = _T_634 ? $signed(_T_980) : $signed(32'sh0); // @[Control.scala 215:40]
  assign add_res_15 = _T_634 ? $signed(add_val_15) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_981 = add_res_15[7:0]; // @[Control.scala 254:37]
  assign _T_983 = src_1_15[4:0]; // @[Control.scala 255:60]
  assign _T_984 = _T_976 >> _T_983; // @[Control.scala 255:49]
  assign _T_985 = $signed(_T_984); // @[Control.scala 255:84]
  assign shr_val_15 = _T_634 ? $signed(_T_985) : $signed(32'sh0); // @[Control.scala 215:40]
  assign shr_res_15 = _T_634 ? $signed(shr_val_15) : $signed(32'sh0); // @[Control.scala 215:40]
  assign _T_986 = shr_res_15[7:0]; // @[Control.scala 257:37]
  assign short_cmp_res_0 = _T_634 ? _T_765 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_0 = _T_634 ? _T_771 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_0 = _T_634 ? _T_776 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_1 = _T_634 ? _T_779 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_1 = _T_634 ? _T_785 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_1 = _T_634 ? _T_790 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_2 = _T_634 ? _T_793 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_2 = _T_634 ? _T_799 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_2 = _T_634 ? _T_804 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_3 = _T_634 ? _T_807 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_3 = _T_634 ? _T_813 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_3 = _T_634 ? _T_818 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_4 = _T_634 ? _T_821 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_4 = _T_634 ? _T_827 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_4 = _T_634 ? _T_832 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_5 = _T_634 ? _T_835 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_5 = _T_634 ? _T_841 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_5 = _T_634 ? _T_846 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_6 = _T_634 ? _T_849 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_6 = _T_634 ? _T_855 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_6 = _T_634 ? _T_860 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_7 = _T_634 ? _T_863 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_7 = _T_634 ? _T_869 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_7 = _T_634 ? _T_874 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_8 = _T_634 ? _T_877 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_8 = _T_634 ? _T_883 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_8 = _T_634 ? _T_888 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_9 = _T_634 ? _T_891 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_9 = _T_634 ? _T_897 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_9 = _T_634 ? _T_902 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_10 = _T_634 ? _T_905 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_10 = _T_634 ? _T_911 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_10 = _T_634 ? _T_916 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_11 = _T_634 ? _T_919 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_11 = _T_634 ? _T_925 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_11 = _T_634 ? _T_930 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_12 = _T_634 ? _T_933 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_12 = _T_634 ? _T_939 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_12 = _T_634 ? _T_944 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_13 = _T_634 ? _T_947 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_13 = _T_634 ? _T_953 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_13 = _T_634 ? _T_958 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_14 = _T_634 ? _T_961 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_14 = _T_634 ? _T_967 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_14 = _T_634 ? _T_972 : 8'h0; // @[Control.scala 215:40]
  assign short_cmp_res_15 = _T_634 ? _T_975 : 8'h0; // @[Control.scala 215:40]
  assign short_add_res_15 = _T_634 ? _T_981 : 8'h0; // @[Control.scala 215:40]
  assign short_shr_res_15 = _T_634 ? _T_986 : 8'h0; // @[Control.scala 215:40]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Control.scala 271:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Control.scala 272:39]
  assign _T_995 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1003 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_995}; // @[Cat.scala 30:58]
  assign _T_1010 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1018 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1010}; // @[Cat.scala 30:58]
  assign _T_1025 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1033 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1025}; // @[Cat.scala 30:58]
  assign _T_1034 = alu_opcode_add_en ? _T_1018 : _T_1033; // @[Control.scala 274:30]
  assign io_done_readdata = opcode == 3'h3; // @[Control.scala 132:20]
  assign io_uops_ready = _T_153 & insn_valid; // @[Control.scala 117:19 Control.scala 120:19]
  assign io_biases_ready = opcode_load_en & memory_type_acc_en; // @[Control.scala 146:19]
  assign io_gemm_queue_ready = io_gemm_queue_valid & _T_149; // @[Control.scala 108:25 Control.scala 111:25]
  assign io_out_mem_address = {{6'd0}, out_mem_addr}; // @[Control.scala 268:22]
  assign io_out_mem_write = out_mem_write_en; // @[Control.scala 270:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1003 : _T_1034; // @[Control.scala 273:24]
  assign io_uop_mem_write = opcode_load_en & memory_type_uop_en; // @[Control.scala 137:20]
  assign io_uop_mem_writedata = uops_data; // @[Control.scala 138:24]
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
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {4{`RANDOM}};
  insn = _RAND_1[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  uops_data = _RAND_2[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  acc_x_cntr_val = _RAND_3[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  dst_offset_in = _RAND_4[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {16{`RANDOM}};
  dst_vector = _RAND_5[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {16{`RANDOM}};
  src_vector = _RAND_6[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {1{`RANDOM}};
  out_mem_addr = _RAND_7[10:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {1{`RANDOM}};
  out_mem_write_en = _RAND_8[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_181_en & acc_mem__T_181_mask) begin
      acc_mem[acc_mem__T_181_addr] <= acc_mem__T_181_data; // @[Control.scala 22:20]
    end
    if (_T_150) begin
      insn <= io_gemm_queue_data;
    end
    if (_T_154) begin
      uops_data <= io_uops_data;
    end
    if (reset) begin
      acc_x_cntr_val <= 8'h0;
    end else begin
      if (_T_110) begin
        if (_T_112) begin
          acc_x_cntr_val <= 8'h0;
        end else begin
          acc_x_cntr_val <= _T_117;
        end
      end
    end
    if (reset) begin
      dst_offset_in <= 8'h0;
    end else begin
      if (_T_125) begin
        if (_T_127) begin
          dst_offset_in <= 8'h0;
        end else begin
          dst_offset_in <= _T_132;
        end
      end
    end
    dst_vector <= acc_mem__T_196_data;
    src_vector <= acc_mem__T_199_data;
    out_mem_addr <= _T_191 + _GEN_257;
    out_mem_write_en <= opcode == 3'h4;
  end
endmodule
module MemBlock(
  input         clock,
  output [31:0] io_readdata,
  input         io_write,
  input  [31:0] io_writedata
);
  reg [31:0] mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[MemBlock.scala 25:16]
  reg [31:0] _RAND_0;
  wire [31:0] mem__T_22_data; // @[MemBlock.scala 25:16]
  wire [9:0] mem__T_22_addr; // @[MemBlock.scala 25:16]
  wire [31:0] mem__T_21_data; // @[MemBlock.scala 25:16]
  wire [9:0] mem__T_21_addr; // @[MemBlock.scala 25:16]
  wire  mem__T_21_mask; // @[MemBlock.scala 25:16]
  wire  mem__T_21_en; // @[MemBlock.scala 25:16]
  reg [31:0] readdata_reg; // @[MemBlock.scala 37:29]
  reg [31:0] _RAND_1;
  reg  _T_26; // @[MemBlock.scala 42:16]
  reg [31:0] _RAND_2;
  reg [31:0] _T_35; // @[MemBlock.scala 43:27]
  reg [31:0] _RAND_3;
  assign mem__T_22_addr = 10'h0;
  assign mem__T_22_data = mem[mem__T_22_addr]; // @[MemBlock.scala 25:16]
  assign mem__T_21_data = io_writedata;
  assign mem__T_21_addr = 10'h0;
  assign mem__T_21_mask = 1'h1;
  assign mem__T_21_en = io_write;
  assign io_readdata = _T_26 ? _T_35 : readdata_reg; // @[MemBlock.scala 38:15 MemBlock.scala 43:17]
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
  _RAND_0 = {1{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 1024; initvar = initvar+1)
    mem[initvar] = _RAND_0[31:0];
  `endif // RANDOMIZE_MEM_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  readdata_reg = _RAND_1[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  _T_26 = _RAND_2[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  _T_35 = _RAND_3[31:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(mem__T_21_en & mem__T_21_mask) begin
      mem[mem__T_21_addr] <= mem__T_21_data; // @[MemBlock.scala 25:16]
    end
    readdata_reg <= mem__T_22_data;
    _T_26 <= io_write;
    _T_35 <= io_writedata;
  end
endmodule
module Core(
  input          clock,
  input          reset,
  output         io_done_readdata,
  output         io_uops_ready,
  input          io_uops_valid,
  input  [31:0]  io_uops_data,
  output         io_biases_ready,
  input          io_biases_valid,
  input  [511:0] io_biases_data,
  output         io_gemm_queue_ready,
  input          io_gemm_queue_valid,
  input  [127:0] io_gemm_queue_data,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata
);
  wire  ctrl_clock; // @[Core.scala 157:21]
  wire  ctrl_reset; // @[Core.scala 157:21]
  wire  ctrl_io_done_readdata; // @[Core.scala 157:21]
  wire  ctrl_io_uops_ready; // @[Core.scala 157:21]
  wire  ctrl_io_uops_valid; // @[Core.scala 157:21]
  wire [31:0] ctrl_io_uops_data; // @[Core.scala 157:21]
  wire  ctrl_io_biases_ready; // @[Core.scala 157:21]
  wire  ctrl_io_biases_valid; // @[Core.scala 157:21]
  wire [511:0] ctrl_io_biases_data; // @[Core.scala 157:21]
  wire  ctrl_io_gemm_queue_ready; // @[Core.scala 157:21]
  wire  ctrl_io_gemm_queue_valid; // @[Core.scala 157:21]
  wire [127:0] ctrl_io_gemm_queue_data; // @[Core.scala 157:21]
  wire  ctrl_io_out_mem_waitrequest; // @[Core.scala 157:21]
  wire [16:0] ctrl_io_out_mem_address; // @[Core.scala 157:21]
  wire  ctrl_io_out_mem_write; // @[Core.scala 157:21]
  wire [127:0] ctrl_io_out_mem_writedata; // @[Core.scala 157:21]
  wire [31:0] ctrl_io_uop_mem_readdata; // @[Core.scala 157:21]
  wire  ctrl_io_uop_mem_write; // @[Core.scala 157:21]
  wire [31:0] ctrl_io_uop_mem_writedata; // @[Core.scala 157:21]
  wire  uop_mem_clock; // @[Core.scala 158:23]
  wire [31:0] uop_mem_io_readdata; // @[Core.scala 158:23]
  wire  uop_mem_io_write; // @[Core.scala 158:23]
  wire [31:0] uop_mem_io_writedata; // @[Core.scala 158:23]
  Control ctrl ( // @[Core.scala 157:21]
    .clock(ctrl_clock),
    .reset(ctrl_reset),
    .io_done_readdata(ctrl_io_done_readdata),
    .io_uops_ready(ctrl_io_uops_ready),
    .io_uops_valid(ctrl_io_uops_valid),
    .io_uops_data(ctrl_io_uops_data),
    .io_biases_ready(ctrl_io_biases_ready),
    .io_biases_valid(ctrl_io_biases_valid),
    .io_biases_data(ctrl_io_biases_data),
    .io_gemm_queue_ready(ctrl_io_gemm_queue_ready),
    .io_gemm_queue_valid(ctrl_io_gemm_queue_valid),
    .io_gemm_queue_data(ctrl_io_gemm_queue_data),
    .io_out_mem_waitrequest(ctrl_io_out_mem_waitrequest),
    .io_out_mem_address(ctrl_io_out_mem_address),
    .io_out_mem_write(ctrl_io_out_mem_write),
    .io_out_mem_writedata(ctrl_io_out_mem_writedata),
    .io_uop_mem_readdata(ctrl_io_uop_mem_readdata),
    .io_uop_mem_write(ctrl_io_uop_mem_write),
    .io_uop_mem_writedata(ctrl_io_uop_mem_writedata)
  );
  MemBlock uop_mem ( // @[Core.scala 158:23]
    .clock(uop_mem_clock),
    .io_readdata(uop_mem_io_readdata),
    .io_write(uop_mem_io_write),
    .io_writedata(uop_mem_io_writedata)
  );
  assign io_done_readdata = ctrl_io_done_readdata; // @[Core.scala 162:16]
  assign io_uops_ready = ctrl_io_uops_ready; // @[Core.scala 163:16]
  assign io_biases_ready = ctrl_io_biases_ready; // @[Core.scala 164:18]
  assign io_gemm_queue_ready = ctrl_io_gemm_queue_ready; // @[Core.scala 165:22]
  assign io_out_mem_address = ctrl_io_out_mem_address; // @[Core.scala 166:19]
  assign io_out_mem_write = ctrl_io_out_mem_write; // @[Core.scala 166:19]
  assign io_out_mem_writedata = ctrl_io_out_mem_writedata; // @[Core.scala 166:19]
  assign ctrl_clock = clock;
  assign ctrl_reset = reset;
  assign ctrl_io_uops_valid = io_uops_valid; // @[Core.scala 163:16]
  assign ctrl_io_uops_data = io_uops_data; // @[Core.scala 163:16]
  assign ctrl_io_biases_valid = io_biases_valid; // @[Core.scala 164:18]
  assign ctrl_io_biases_data = io_biases_data; // @[Core.scala 164:18]
  assign ctrl_io_gemm_queue_valid = io_gemm_queue_valid; // @[Core.scala 165:22]
  assign ctrl_io_gemm_queue_data = io_gemm_queue_data; // @[Core.scala 165:22]
  assign ctrl_io_out_mem_waitrequest = io_out_mem_waitrequest; // @[Core.scala 166:19]
  assign ctrl_io_uop_mem_readdata = uop_mem_io_readdata; // @[Core.scala 169:19]
  assign uop_mem_clock = clock;
  assign uop_mem_io_write = ctrl_io_uop_mem_write; // @[Core.scala 169:19]
  assign uop_mem_io_writedata = ctrl_io_uop_mem_writedata; // @[Core.scala 169:19]
endmodule
module Compute(
  input          clock,
  input          reset,
  output         io_done_waitrequest,
  input          io_done_address,
  input          io_done_read,
  output         io_done_readdata,
  input          io_done_write,
  input          io_done_writedata,
  output         io_uops_ready,
  input          io_uops_valid,
  input  [31:0]  io_uops_data,
  output         io_biases_ready,
  input          io_biases_valid,
  input  [511:0] io_biases_data,
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
  wire  core_clock; // @[Compute.scala 40:20]
  wire  core_reset; // @[Compute.scala 40:20]
  wire  core_io_done_readdata; // @[Compute.scala 40:20]
  wire  core_io_uops_ready; // @[Compute.scala 40:20]
  wire  core_io_uops_valid; // @[Compute.scala 40:20]
  wire [31:0] core_io_uops_data; // @[Compute.scala 40:20]
  wire  core_io_biases_ready; // @[Compute.scala 40:20]
  wire  core_io_biases_valid; // @[Compute.scala 40:20]
  wire [511:0] core_io_biases_data; // @[Compute.scala 40:20]
  wire  core_io_gemm_queue_ready; // @[Compute.scala 40:20]
  wire  core_io_gemm_queue_valid; // @[Compute.scala 40:20]
  wire [127:0] core_io_gemm_queue_data; // @[Compute.scala 40:20]
  wire  core_io_out_mem_waitrequest; // @[Compute.scala 40:20]
  wire [16:0] core_io_out_mem_address; // @[Compute.scala 40:20]
  wire  core_io_out_mem_write; // @[Compute.scala 40:20]
  wire [127:0] core_io_out_mem_writedata; // @[Compute.scala 40:20]
  Core core ( // @[Compute.scala 40:20]
    .clock(core_clock),
    .reset(core_reset),
    .io_done_readdata(core_io_done_readdata),
    .io_uops_ready(core_io_uops_ready),
    .io_uops_valid(core_io_uops_valid),
    .io_uops_data(core_io_uops_data),
    .io_biases_ready(core_io_biases_ready),
    .io_biases_valid(core_io_biases_valid),
    .io_biases_data(core_io_biases_data),
    .io_gemm_queue_ready(core_io_gemm_queue_ready),
    .io_gemm_queue_valid(core_io_gemm_queue_valid),
    .io_gemm_queue_data(core_io_gemm_queue_data),
    .io_out_mem_waitrequest(core_io_out_mem_waitrequest),
    .io_out_mem_address(core_io_out_mem_address),
    .io_out_mem_write(core_io_out_mem_write),
    .io_out_mem_writedata(core_io_out_mem_writedata)
  );
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 42:11]
  assign io_done_readdata = core_io_done_readdata; // @[Compute.scala 42:11]
  assign io_uops_ready = core_io_uops_ready; // @[Compute.scala 43:11]
  assign io_biases_ready = core_io_biases_ready; // @[Compute.scala 44:13]
  assign io_gemm_queue_ready = core_io_gemm_queue_ready; // @[Compute.scala 45:17]
  assign io_l2g_dep_queue_ready = 1'h0;
  assign io_s2g_dep_queue_ready = 1'h0;
  assign io_g2l_dep_queue_valid = 1'h0;
  assign io_g2l_dep_queue_data = 1'h0;
  assign io_g2s_dep_queue_valid = 1'h0;
  assign io_g2s_dep_queue_data = 1'h0;
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = core_io_out_mem_address; // @[Compute.scala 52:14]
  assign io_out_mem_read = 1'h0; // @[Compute.scala 52:14]
  assign io_out_mem_write = core_io_out_mem_write; // @[Compute.scala 52:14]
  assign io_out_mem_writedata = core_io_out_mem_writedata; // @[Compute.scala 52:14]
  assign core_clock = clock;
  assign core_reset = reset;
  assign core_io_uops_valid = io_uops_valid; // @[Compute.scala 43:11]
  assign core_io_uops_data = io_uops_data; // @[Compute.scala 43:11]
  assign core_io_biases_valid = io_biases_valid; // @[Compute.scala 44:13]
  assign core_io_biases_data = io_biases_data; // @[Compute.scala 44:13]
  assign core_io_gemm_queue_valid = io_gemm_queue_valid; // @[Compute.scala 45:17]
  assign core_io_gemm_queue_data = io_gemm_queue_data; // @[Compute.scala 45:17]
  assign core_io_out_mem_waitrequest = io_out_mem_waitrequest; // @[Compute.scala 52:14]
endmodule
