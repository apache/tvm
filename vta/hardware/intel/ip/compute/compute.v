module Queue(
  input   clock,
  input   reset,
  output  io_enq_ready,
  input   io_enq_valid,
  input   io_deq_ready,
  output  io_deq_valid,
  output  io_deq_bits
);
  reg  _T_35 [0:15]; // @[Decoupled.scala 215:24]
  reg [31:0] _RAND_0;
  wire  _T_35__T_68_data; // @[Decoupled.scala 215:24]
  wire [3:0] _T_35__T_68_addr; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_data; // @[Decoupled.scala 215:24]
  wire [3:0] _T_35__T_54_addr; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_mask; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_en; // @[Decoupled.scala 215:24]
  reg [3:0] value; // @[Counter.scala 26:33]
  reg [31:0] _RAND_1;
  reg [3:0] value_1; // @[Counter.scala 26:33]
  reg [31:0] _RAND_2;
  reg  _T_42; // @[Decoupled.scala 218:35]
  reg [31:0] _RAND_3;
  wire  _T_43; // @[Decoupled.scala 220:41]
  wire  _T_45; // @[Decoupled.scala 221:36]
  wire  _T_46; // @[Decoupled.scala 221:33]
  wire  _T_47; // @[Decoupled.scala 222:32]
  wire  _T_48; // @[Decoupled.scala 37:37]
  wire  _T_51; // @[Decoupled.scala 37:37]
  wire [4:0] _T_57; // @[Counter.scala 35:22]
  wire [3:0] _T_58; // @[Counter.scala 35:22]
  wire [3:0] _GEN_5; // @[Decoupled.scala 226:17]
  wire [4:0] _T_61; // @[Counter.scala 35:22]
  wire [3:0] _T_62; // @[Counter.scala 35:22]
  wire [3:0] _GEN_6; // @[Decoupled.scala 230:17]
  wire  _T_63; // @[Decoupled.scala 233:16]
  wire  _GEN_7; // @[Decoupled.scala 233:28]
  assign _T_35__T_68_addr = value_1;
  assign _T_35__T_68_data = _T_35[_T_35__T_68_addr]; // @[Decoupled.scala 215:24]
  assign _T_35__T_54_data = 1'h1;
  assign _T_35__T_54_addr = value;
  assign _T_35__T_54_mask = 1'h1;
  assign _T_35__T_54_en = io_enq_ready & io_enq_valid;
  assign _T_43 = value == value_1; // @[Decoupled.scala 220:41]
  assign _T_45 = _T_42 == 1'h0; // @[Decoupled.scala 221:36]
  assign _T_46 = _T_43 & _T_45; // @[Decoupled.scala 221:33]
  assign _T_47 = _T_43 & _T_42; // @[Decoupled.scala 222:32]
  assign _T_48 = io_enq_ready & io_enq_valid; // @[Decoupled.scala 37:37]
  assign _T_51 = io_deq_ready & io_deq_valid; // @[Decoupled.scala 37:37]
  assign _T_57 = value + 4'h1; // @[Counter.scala 35:22]
  assign _T_58 = value + 4'h1; // @[Counter.scala 35:22]
  assign _GEN_5 = _T_48 ? _T_58 : value; // @[Decoupled.scala 226:17]
  assign _T_61 = value_1 + 4'h1; // @[Counter.scala 35:22]
  assign _T_62 = value_1 + 4'h1; // @[Counter.scala 35:22]
  assign _GEN_6 = _T_51 ? _T_62 : value_1; // @[Decoupled.scala 230:17]
  assign _T_63 = _T_48 != _T_51; // @[Decoupled.scala 233:16]
  assign _GEN_7 = _T_63 ? _T_48 : _T_42; // @[Decoupled.scala 233:28]
  assign io_enq_ready = _T_47 == 1'h0; // @[Decoupled.scala 238:16]
  assign io_deq_valid = _T_46 == 1'h0; // @[Decoupled.scala 237:16]
  assign io_deq_bits = _T_35__T_68_data; // @[Decoupled.scala 239:15]
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
  for (initvar = 0; initvar < 16; initvar = initvar+1)
    _T_35[initvar] = _RAND_0[0:0];
  `endif // RANDOMIZE_MEM_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  value = _RAND_1[3:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  value_1 = _RAND_2[3:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  _T_42 = _RAND_3[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(_T_35__T_54_en & _T_35__T_54_mask) begin
      _T_35[_T_35__T_54_addr] <= _T_35__T_54_data; // @[Decoupled.scala 215:24]
    end
    if (reset) begin
      value <= 4'h0;
    end else begin
      if (_T_48) begin
        value <= _T_58;
      end
    end
    if (reset) begin
      value_1 <= 4'h0;
    end else begin
      if (_T_51) begin
        value_1 <= _T_62;
      end
    end
    if (reset) begin
      _T_42 <= 1'h0;
    end else begin
      if (_T_63) begin
        _T_42 <= _T_48;
      end
    end
  end
endmodule
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
  input          io_g2l_dep_queue_ready,
  output         io_g2l_dep_queue_valid,
  output         io_g2l_dep_queue_data,
  input          io_g2s_dep_queue_ready,
  output         io_g2s_dep_queue_valid,
  output         io_g2s_dep_queue_data,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata,
  input  [31:0]  io_uop_mem_readdata,
  output         io_uop_mem_write,
  output [31:0]  io_uop_mem_writedata
);
  reg [511:0] acc_mem [0:255] /* synthesis ramstyle = "M20K" */; // @[Control.scala 26:20]
  reg [511:0] _RAND_0;
  wire [511:0] acc_mem__T_236_data; // @[Control.scala 26:20]
  wire [7:0] acc_mem__T_236_addr; // @[Control.scala 26:20]
  wire [511:0] acc_mem__T_239_data; // @[Control.scala 26:20]
  wire [7:0] acc_mem__T_239_addr; // @[Control.scala 26:20]
  wire [511:0] acc_mem__T_221_data; // @[Control.scala 26:20]
  wire [7:0] acc_mem__T_221_addr; // @[Control.scala 26:20]
  wire  acc_mem__T_221_mask; // @[Control.scala 26:20]
  wire  acc_mem__T_221_en; // @[Control.scala 26:20]
  wire  g2l_queue_clock; // @[Control.scala 306:25]
  wire  g2l_queue_reset; // @[Control.scala 306:25]
  wire  g2l_queue_io_enq_ready; // @[Control.scala 306:25]
  wire  g2l_queue_io_enq_valid; // @[Control.scala 306:25]
  wire  g2l_queue_io_deq_ready; // @[Control.scala 306:25]
  wire  g2l_queue_io_deq_valid; // @[Control.scala 306:25]
  wire  g2l_queue_io_deq_bits; // @[Control.scala 306:25]
  wire  g2s_queue_clock; // @[Control.scala 307:25]
  wire  g2s_queue_reset; // @[Control.scala 307:25]
  wire  g2s_queue_io_enq_ready; // @[Control.scala 307:25]
  wire  g2s_queue_io_enq_valid; // @[Control.scala 307:25]
  wire  g2s_queue_io_deq_ready; // @[Control.scala 307:25]
  wire  g2s_queue_io_deq_valid; // @[Control.scala 307:25]
  wire  g2s_queue_io_deq_bits; // @[Control.scala 307:25]
  reg [127:0] insn; // @[Control.scala 28:28]
  reg [127:0] _RAND_1;
  wire  insn_valid; // @[Control.scala 29:30]
  wire [2:0] opcode; // @[Control.scala 31:29]
  wire  push_prev_dep; // @[Control.scala 34:29]
  wire  push_next_dep; // @[Control.scala 35:29]
  reg [31:0] uops_data; // @[Control.scala 37:28]
  reg [31:0] _RAND_2;
  wire [1:0] memory_type; // @[Control.scala 39:25]
  wire [15:0] sram_base; // @[Control.scala 40:25]
  wire [15:0] x_size; // @[Control.scala 43:25]
  wire [3:0] y_pad_0; // @[Control.scala 45:25]
  wire [3:0] x_pad_0; // @[Control.scala 47:25]
  wire [3:0] x_pad_1; // @[Control.scala 48:25]
  wire [15:0] _GEN_249; // @[Control.scala 52:30]
  wire [15:0] _GEN_251; // @[Control.scala 53:30]
  wire [16:0] _T_135; // @[Control.scala 53:30]
  wire [15:0] _T_136; // @[Control.scala 53:30]
  wire [15:0] _GEN_252; // @[Control.scala 53:39]
  wire [16:0] _T_137; // @[Control.scala 53:39]
  wire [15:0] x_size_total; // @[Control.scala 53:39]
  wire [19:0] y_offset; // @[Control.scala 54:31]
  wire  _T_140; // @[Control.scala 58:32]
  wire  _T_142; // @[Control.scala 58:60]
  wire  opcode_load_en; // @[Control.scala 58:50]
  wire  opcode_gemm_en; // @[Control.scala 59:32]
  wire  opcode_alu_en; // @[Control.scala 60:31]
  wire  memory_type_uop_en; // @[Control.scala 62:40]
  wire  memory_type_acc_en; // @[Control.scala 63:40]
  wire  acc_x_cntr_en; // @[Control.scala 67:39]
  reg [7:0] acc_x_cntr_val; // @[Control.scala 69:31]
  reg [31:0] _RAND_3;
  wire  _T_150; // @[Control.scala 71:23]
  wire  _T_152; // @[Control.scala 72:26]
  wire [8:0] _T_156; // @[Control.scala 76:40]
  wire [7:0] _T_157; // @[Control.scala 76:40]
  wire [7:0] _GEN_0; // @[Control.scala 72:54]
  wire [7:0] _GEN_2; // @[Control.scala 71:43]
  wire  acc_x_cntr_wrap; // @[Control.scala 71:43]
  wire  in_loop_cntr_en; // @[Control.scala 85:40]
  reg [7:0] dst_offset_in; // @[Control.scala 87:33]
  reg [31:0] _RAND_4;
  wire  _T_164; // @[Control.scala 89:28]
  wire  _T_165; // @[Control.scala 89:25]
  wire  _T_167; // @[Control.scala 90:28]
  wire [8:0] _T_171; // @[Control.scala 94:44]
  wire [7:0] _T_172; // @[Control.scala 94:44]
  wire [7:0] _GEN_4; // @[Control.scala 90:58]
  wire [7:0] _GEN_6; // @[Control.scala 89:53]
  wire  in_loop_cntr_wrap; // @[Control.scala 89:53]
  wire  _T_175; // @[Control.scala 104:33]
  wire  _T_178; // @[Control.scala 105:35]
  wire  _T_179; // @[Control.scala 105:32]
  wire  _T_182; // @[Control.scala 106:37]
  wire  _T_183; // @[Control.scala 106:34]
  wire  _T_187; // @[Control.scala 105:17]
  wire  busy; // @[Control.scala 104:17]
  wire  _T_189; // @[Control.scala 110:32]
  wire  _T_190; // @[Control.scala 110:29]
  wire  _T_193; // @[Control.scala 119:23]
  wire  _T_194; // @[Control.scala 119:45]
  wire [19:0] _GEN_254; // @[Control.scala 146:33]
  wire [20:0] _T_213; // @[Control.scala 146:33]
  wire [19:0] _T_214; // @[Control.scala 146:33]
  wire [19:0] _GEN_255; // @[Control.scala 146:44]
  wire [20:0] _T_215; // @[Control.scala 146:44]
  wire [19:0] _T_216; // @[Control.scala 146:44]
  wire [20:0] _T_218; // @[Control.scala 146:55]
  wire [20:0] _GEN_256; // @[Control.scala 146:65]
  wire [21:0] _T_219; // @[Control.scala 146:65]
  wire [20:0] acc_mem_addr; // @[Control.scala 146:65]
  wire [1:0] alu_opcode; // @[Control.scala 168:24]
  wire  use_imm; // @[Control.scala 169:21]
  wire [15:0] imm_raw; // @[Control.scala 170:21]
  wire [15:0] _T_222; // @[Control.scala 171:25]
  wire  _T_224; // @[Control.scala 171:32]
  wire [31:0] _T_226; // @[Cat.scala 30:58]
  wire [16:0] _T_228; // @[Cat.scala 30:58]
  wire [31:0] _T_229; // @[Control.scala 171:16]
  wire [31:0] imm; // @[Control.scala 171:89]
  wire [10:0] _T_231; // @[Control.scala 180:20]
  wire [10:0] _GEN_257; // @[Control.scala 180:47]
  wire [11:0] _T_232; // @[Control.scala 180:47]
  wire [10:0] dst_idx; // @[Control.scala 180:47]
  wire [10:0] _T_233; // @[Control.scala 181:20]
  wire [11:0] _T_234; // @[Control.scala 181:47]
  wire [10:0] src_idx; // @[Control.scala 181:47]
  reg [511:0] dst_vector; // @[Control.scala 183:27]
  reg [511:0] _RAND_5;
  reg [511:0] src_vector; // @[Control.scala 184:27]
  reg [511:0] _RAND_6;
  reg [10:0] out_mem_addr; // @[Control.scala 197:30]
  reg [31:0] _RAND_7;
  reg  out_mem_write_en; // @[Control.scala 198:34]
  reg [31:0] _RAND_8;
  wire  alu_opcode_min_en; // @[Control.scala 200:38]
  wire  alu_opcode_max_en; // @[Control.scala 201:38]
  wire  _T_674; // @[Control.scala 219:20]
  wire [31:0] _T_675; // @[Control.scala 239:31]
  wire [31:0] _T_676; // @[Control.scala 239:72]
  wire [31:0] _T_677; // @[Control.scala 240:31]
  wire [31:0] _T_678; // @[Control.scala 240:72]
  wire [31:0] _T_679; // @[Control.scala 239:31]
  wire [31:0] _T_680; // @[Control.scala 239:72]
  wire [31:0] _T_681; // @[Control.scala 240:31]
  wire [31:0] _T_682; // @[Control.scala 240:72]
  wire [31:0] _T_683; // @[Control.scala 239:31]
  wire [31:0] _T_684; // @[Control.scala 239:72]
  wire [31:0] _T_685; // @[Control.scala 240:31]
  wire [31:0] _T_686; // @[Control.scala 240:72]
  wire [31:0] _T_687; // @[Control.scala 239:31]
  wire [31:0] _T_688; // @[Control.scala 239:72]
  wire [31:0] _T_689; // @[Control.scala 240:31]
  wire [31:0] _T_690; // @[Control.scala 240:72]
  wire [31:0] _T_691; // @[Control.scala 239:31]
  wire [31:0] _T_692; // @[Control.scala 239:72]
  wire [31:0] _T_693; // @[Control.scala 240:31]
  wire [31:0] _T_694; // @[Control.scala 240:72]
  wire [31:0] _T_695; // @[Control.scala 239:31]
  wire [31:0] _T_696; // @[Control.scala 239:72]
  wire [31:0] _T_697; // @[Control.scala 240:31]
  wire [31:0] _T_698; // @[Control.scala 240:72]
  wire [31:0] _T_699; // @[Control.scala 239:31]
  wire [31:0] _T_700; // @[Control.scala 239:72]
  wire [31:0] _T_701; // @[Control.scala 240:31]
  wire [31:0] _T_702; // @[Control.scala 240:72]
  wire [31:0] _T_703; // @[Control.scala 239:31]
  wire [31:0] _T_704; // @[Control.scala 239:72]
  wire [31:0] _T_705; // @[Control.scala 240:31]
  wire [31:0] _T_706; // @[Control.scala 240:72]
  wire [31:0] _T_707; // @[Control.scala 239:31]
  wire [31:0] _T_708; // @[Control.scala 239:72]
  wire [31:0] _T_709; // @[Control.scala 240:31]
  wire [31:0] _T_710; // @[Control.scala 240:72]
  wire [31:0] _T_711; // @[Control.scala 239:31]
  wire [31:0] _T_712; // @[Control.scala 239:72]
  wire [31:0] _T_713; // @[Control.scala 240:31]
  wire [31:0] _T_714; // @[Control.scala 240:72]
  wire [31:0] _T_715; // @[Control.scala 239:31]
  wire [31:0] _T_716; // @[Control.scala 239:72]
  wire [31:0] _T_717; // @[Control.scala 240:31]
  wire [31:0] _T_718; // @[Control.scala 240:72]
  wire [31:0] _T_719; // @[Control.scala 239:31]
  wire [31:0] _T_720; // @[Control.scala 239:72]
  wire [31:0] _T_721; // @[Control.scala 240:31]
  wire [31:0] _T_722; // @[Control.scala 240:72]
  wire [31:0] _T_723; // @[Control.scala 239:31]
  wire [31:0] _T_724; // @[Control.scala 239:72]
  wire [31:0] _T_725; // @[Control.scala 240:31]
  wire [31:0] _T_726; // @[Control.scala 240:72]
  wire [31:0] _T_727; // @[Control.scala 239:31]
  wire [31:0] _T_728; // @[Control.scala 239:72]
  wire [31:0] _T_729; // @[Control.scala 240:31]
  wire [31:0] _T_730; // @[Control.scala 240:72]
  wire [31:0] _T_731; // @[Control.scala 239:31]
  wire [31:0] _T_732; // @[Control.scala 239:72]
  wire [31:0] _T_733; // @[Control.scala 240:31]
  wire [31:0] _T_734; // @[Control.scala 240:72]
  wire [31:0] _T_735; // @[Control.scala 239:31]
  wire [31:0] _T_736; // @[Control.scala 239:72]
  wire [31:0] _T_737; // @[Control.scala 240:31]
  wire [31:0] _T_738; // @[Control.scala 240:72]
  wire [31:0] _GEN_17; // @[Control.scala 237:30]
  wire [31:0] _GEN_18; // @[Control.scala 237:30]
  wire [31:0] _GEN_19; // @[Control.scala 237:30]
  wire [31:0] _GEN_20; // @[Control.scala 237:30]
  wire [31:0] _GEN_21; // @[Control.scala 237:30]
  wire [31:0] _GEN_22; // @[Control.scala 237:30]
  wire [31:0] _GEN_23; // @[Control.scala 237:30]
  wire [31:0] _GEN_24; // @[Control.scala 237:30]
  wire [31:0] _GEN_25; // @[Control.scala 237:30]
  wire [31:0] _GEN_26; // @[Control.scala 237:30]
  wire [31:0] _GEN_27; // @[Control.scala 237:30]
  wire [31:0] _GEN_28; // @[Control.scala 237:30]
  wire [31:0] _GEN_29; // @[Control.scala 237:30]
  wire [31:0] _GEN_30; // @[Control.scala 237:30]
  wire [31:0] _GEN_31; // @[Control.scala 237:30]
  wire [31:0] _GEN_32; // @[Control.scala 237:30]
  wire [31:0] _GEN_33; // @[Control.scala 237:30]
  wire [31:0] _GEN_34; // @[Control.scala 237:30]
  wire [31:0] _GEN_35; // @[Control.scala 237:30]
  wire [31:0] _GEN_36; // @[Control.scala 237:30]
  wire [31:0] _GEN_37; // @[Control.scala 237:30]
  wire [31:0] _GEN_38; // @[Control.scala 237:30]
  wire [31:0] _GEN_39; // @[Control.scala 237:30]
  wire [31:0] _GEN_40; // @[Control.scala 237:30]
  wire [31:0] _GEN_41; // @[Control.scala 237:30]
  wire [31:0] _GEN_42; // @[Control.scala 237:30]
  wire [31:0] _GEN_43; // @[Control.scala 237:30]
  wire [31:0] _GEN_44; // @[Control.scala 237:30]
  wire [31:0] _GEN_45; // @[Control.scala 237:30]
  wire [31:0] _GEN_46; // @[Control.scala 237:30]
  wire [31:0] _GEN_47; // @[Control.scala 237:30]
  wire [31:0] _GEN_48; // @[Control.scala 237:30]
  wire [31:0] _GEN_49; // @[Control.scala 248:20]
  wire [31:0] _GEN_50; // @[Control.scala 248:20]
  wire [31:0] _GEN_51; // @[Control.scala 248:20]
  wire [31:0] _GEN_52; // @[Control.scala 248:20]
  wire [31:0] _GEN_53; // @[Control.scala 248:20]
  wire [31:0] _GEN_54; // @[Control.scala 248:20]
  wire [31:0] _GEN_55; // @[Control.scala 248:20]
  wire [31:0] _GEN_56; // @[Control.scala 248:20]
  wire [31:0] _GEN_57; // @[Control.scala 248:20]
  wire [31:0] _GEN_58; // @[Control.scala 248:20]
  wire [31:0] _GEN_59; // @[Control.scala 248:20]
  wire [31:0] _GEN_60; // @[Control.scala 248:20]
  wire [31:0] _GEN_61; // @[Control.scala 248:20]
  wire [31:0] _GEN_62; // @[Control.scala 248:20]
  wire [31:0] _GEN_63; // @[Control.scala 248:20]
  wire [31:0] _GEN_64; // @[Control.scala 248:20]
  wire [31:0] src_0_0; // @[Control.scala 219:40]
  wire [31:0] src_1_0; // @[Control.scala 219:40]
  wire  _T_803; // @[Control.scala 253:34]
  wire [31:0] _T_804; // @[Control.scala 253:24]
  wire [31:0] mix_val_0; // @[Control.scala 219:40]
  wire [7:0] _T_805; // @[Control.scala 255:37]
  wire [31:0] _T_806; // @[Control.scala 256:30]
  wire [31:0] _T_807; // @[Control.scala 256:59]
  wire [32:0] _T_808; // @[Control.scala 256:49]
  wire [31:0] _T_809; // @[Control.scala 256:49]
  wire [31:0] _T_810; // @[Control.scala 256:79]
  wire [31:0] add_val_0; // @[Control.scala 219:40]
  wire [31:0] add_res_0; // @[Control.scala 219:40]
  wire [7:0] _T_811; // @[Control.scala 258:37]
  wire [4:0] _T_813; // @[Control.scala 259:60]
  wire [31:0] _T_814; // @[Control.scala 259:49]
  wire [31:0] _T_815; // @[Control.scala 259:84]
  wire [31:0] shr_val_0; // @[Control.scala 219:40]
  wire [31:0] shr_res_0; // @[Control.scala 219:40]
  wire [7:0] _T_816; // @[Control.scala 261:37]
  wire [31:0] src_0_1; // @[Control.scala 219:40]
  wire [31:0] src_1_1; // @[Control.scala 219:40]
  wire  _T_817; // @[Control.scala 253:34]
  wire [31:0] _T_818; // @[Control.scala 253:24]
  wire [31:0] mix_val_1; // @[Control.scala 219:40]
  wire [7:0] _T_819; // @[Control.scala 255:37]
  wire [31:0] _T_820; // @[Control.scala 256:30]
  wire [31:0] _T_821; // @[Control.scala 256:59]
  wire [32:0] _T_822; // @[Control.scala 256:49]
  wire [31:0] _T_823; // @[Control.scala 256:49]
  wire [31:0] _T_824; // @[Control.scala 256:79]
  wire [31:0] add_val_1; // @[Control.scala 219:40]
  wire [31:0] add_res_1; // @[Control.scala 219:40]
  wire [7:0] _T_825; // @[Control.scala 258:37]
  wire [4:0] _T_827; // @[Control.scala 259:60]
  wire [31:0] _T_828; // @[Control.scala 259:49]
  wire [31:0] _T_829; // @[Control.scala 259:84]
  wire [31:0] shr_val_1; // @[Control.scala 219:40]
  wire [31:0] shr_res_1; // @[Control.scala 219:40]
  wire [7:0] _T_830; // @[Control.scala 261:37]
  wire [31:0] src_0_2; // @[Control.scala 219:40]
  wire [31:0] src_1_2; // @[Control.scala 219:40]
  wire  _T_831; // @[Control.scala 253:34]
  wire [31:0] _T_832; // @[Control.scala 253:24]
  wire [31:0] mix_val_2; // @[Control.scala 219:40]
  wire [7:0] _T_833; // @[Control.scala 255:37]
  wire [31:0] _T_834; // @[Control.scala 256:30]
  wire [31:0] _T_835; // @[Control.scala 256:59]
  wire [32:0] _T_836; // @[Control.scala 256:49]
  wire [31:0] _T_837; // @[Control.scala 256:49]
  wire [31:0] _T_838; // @[Control.scala 256:79]
  wire [31:0] add_val_2; // @[Control.scala 219:40]
  wire [31:0] add_res_2; // @[Control.scala 219:40]
  wire [7:0] _T_839; // @[Control.scala 258:37]
  wire [4:0] _T_841; // @[Control.scala 259:60]
  wire [31:0] _T_842; // @[Control.scala 259:49]
  wire [31:0] _T_843; // @[Control.scala 259:84]
  wire [31:0] shr_val_2; // @[Control.scala 219:40]
  wire [31:0] shr_res_2; // @[Control.scala 219:40]
  wire [7:0] _T_844; // @[Control.scala 261:37]
  wire [31:0] src_0_3; // @[Control.scala 219:40]
  wire [31:0] src_1_3; // @[Control.scala 219:40]
  wire  _T_845; // @[Control.scala 253:34]
  wire [31:0] _T_846; // @[Control.scala 253:24]
  wire [31:0] mix_val_3; // @[Control.scala 219:40]
  wire [7:0] _T_847; // @[Control.scala 255:37]
  wire [31:0] _T_848; // @[Control.scala 256:30]
  wire [31:0] _T_849; // @[Control.scala 256:59]
  wire [32:0] _T_850; // @[Control.scala 256:49]
  wire [31:0] _T_851; // @[Control.scala 256:49]
  wire [31:0] _T_852; // @[Control.scala 256:79]
  wire [31:0] add_val_3; // @[Control.scala 219:40]
  wire [31:0] add_res_3; // @[Control.scala 219:40]
  wire [7:0] _T_853; // @[Control.scala 258:37]
  wire [4:0] _T_855; // @[Control.scala 259:60]
  wire [31:0] _T_856; // @[Control.scala 259:49]
  wire [31:0] _T_857; // @[Control.scala 259:84]
  wire [31:0] shr_val_3; // @[Control.scala 219:40]
  wire [31:0] shr_res_3; // @[Control.scala 219:40]
  wire [7:0] _T_858; // @[Control.scala 261:37]
  wire [31:0] src_0_4; // @[Control.scala 219:40]
  wire [31:0] src_1_4; // @[Control.scala 219:40]
  wire  _T_859; // @[Control.scala 253:34]
  wire [31:0] _T_860; // @[Control.scala 253:24]
  wire [31:0] mix_val_4; // @[Control.scala 219:40]
  wire [7:0] _T_861; // @[Control.scala 255:37]
  wire [31:0] _T_862; // @[Control.scala 256:30]
  wire [31:0] _T_863; // @[Control.scala 256:59]
  wire [32:0] _T_864; // @[Control.scala 256:49]
  wire [31:0] _T_865; // @[Control.scala 256:49]
  wire [31:0] _T_866; // @[Control.scala 256:79]
  wire [31:0] add_val_4; // @[Control.scala 219:40]
  wire [31:0] add_res_4; // @[Control.scala 219:40]
  wire [7:0] _T_867; // @[Control.scala 258:37]
  wire [4:0] _T_869; // @[Control.scala 259:60]
  wire [31:0] _T_870; // @[Control.scala 259:49]
  wire [31:0] _T_871; // @[Control.scala 259:84]
  wire [31:0] shr_val_4; // @[Control.scala 219:40]
  wire [31:0] shr_res_4; // @[Control.scala 219:40]
  wire [7:0] _T_872; // @[Control.scala 261:37]
  wire [31:0] src_0_5; // @[Control.scala 219:40]
  wire [31:0] src_1_5; // @[Control.scala 219:40]
  wire  _T_873; // @[Control.scala 253:34]
  wire [31:0] _T_874; // @[Control.scala 253:24]
  wire [31:0] mix_val_5; // @[Control.scala 219:40]
  wire [7:0] _T_875; // @[Control.scala 255:37]
  wire [31:0] _T_876; // @[Control.scala 256:30]
  wire [31:0] _T_877; // @[Control.scala 256:59]
  wire [32:0] _T_878; // @[Control.scala 256:49]
  wire [31:0] _T_879; // @[Control.scala 256:49]
  wire [31:0] _T_880; // @[Control.scala 256:79]
  wire [31:0] add_val_5; // @[Control.scala 219:40]
  wire [31:0] add_res_5; // @[Control.scala 219:40]
  wire [7:0] _T_881; // @[Control.scala 258:37]
  wire [4:0] _T_883; // @[Control.scala 259:60]
  wire [31:0] _T_884; // @[Control.scala 259:49]
  wire [31:0] _T_885; // @[Control.scala 259:84]
  wire [31:0] shr_val_5; // @[Control.scala 219:40]
  wire [31:0] shr_res_5; // @[Control.scala 219:40]
  wire [7:0] _T_886; // @[Control.scala 261:37]
  wire [31:0] src_0_6; // @[Control.scala 219:40]
  wire [31:0] src_1_6; // @[Control.scala 219:40]
  wire  _T_887; // @[Control.scala 253:34]
  wire [31:0] _T_888; // @[Control.scala 253:24]
  wire [31:0] mix_val_6; // @[Control.scala 219:40]
  wire [7:0] _T_889; // @[Control.scala 255:37]
  wire [31:0] _T_890; // @[Control.scala 256:30]
  wire [31:0] _T_891; // @[Control.scala 256:59]
  wire [32:0] _T_892; // @[Control.scala 256:49]
  wire [31:0] _T_893; // @[Control.scala 256:49]
  wire [31:0] _T_894; // @[Control.scala 256:79]
  wire [31:0] add_val_6; // @[Control.scala 219:40]
  wire [31:0] add_res_6; // @[Control.scala 219:40]
  wire [7:0] _T_895; // @[Control.scala 258:37]
  wire [4:0] _T_897; // @[Control.scala 259:60]
  wire [31:0] _T_898; // @[Control.scala 259:49]
  wire [31:0] _T_899; // @[Control.scala 259:84]
  wire [31:0] shr_val_6; // @[Control.scala 219:40]
  wire [31:0] shr_res_6; // @[Control.scala 219:40]
  wire [7:0] _T_900; // @[Control.scala 261:37]
  wire [31:0] src_0_7; // @[Control.scala 219:40]
  wire [31:0] src_1_7; // @[Control.scala 219:40]
  wire  _T_901; // @[Control.scala 253:34]
  wire [31:0] _T_902; // @[Control.scala 253:24]
  wire [31:0] mix_val_7; // @[Control.scala 219:40]
  wire [7:0] _T_903; // @[Control.scala 255:37]
  wire [31:0] _T_904; // @[Control.scala 256:30]
  wire [31:0] _T_905; // @[Control.scala 256:59]
  wire [32:0] _T_906; // @[Control.scala 256:49]
  wire [31:0] _T_907; // @[Control.scala 256:49]
  wire [31:0] _T_908; // @[Control.scala 256:79]
  wire [31:0] add_val_7; // @[Control.scala 219:40]
  wire [31:0] add_res_7; // @[Control.scala 219:40]
  wire [7:0] _T_909; // @[Control.scala 258:37]
  wire [4:0] _T_911; // @[Control.scala 259:60]
  wire [31:0] _T_912; // @[Control.scala 259:49]
  wire [31:0] _T_913; // @[Control.scala 259:84]
  wire [31:0] shr_val_7; // @[Control.scala 219:40]
  wire [31:0] shr_res_7; // @[Control.scala 219:40]
  wire [7:0] _T_914; // @[Control.scala 261:37]
  wire [31:0] src_0_8; // @[Control.scala 219:40]
  wire [31:0] src_1_8; // @[Control.scala 219:40]
  wire  _T_915; // @[Control.scala 253:34]
  wire [31:0] _T_916; // @[Control.scala 253:24]
  wire [31:0] mix_val_8; // @[Control.scala 219:40]
  wire [7:0] _T_917; // @[Control.scala 255:37]
  wire [31:0] _T_918; // @[Control.scala 256:30]
  wire [31:0] _T_919; // @[Control.scala 256:59]
  wire [32:0] _T_920; // @[Control.scala 256:49]
  wire [31:0] _T_921; // @[Control.scala 256:49]
  wire [31:0] _T_922; // @[Control.scala 256:79]
  wire [31:0] add_val_8; // @[Control.scala 219:40]
  wire [31:0] add_res_8; // @[Control.scala 219:40]
  wire [7:0] _T_923; // @[Control.scala 258:37]
  wire [4:0] _T_925; // @[Control.scala 259:60]
  wire [31:0] _T_926; // @[Control.scala 259:49]
  wire [31:0] _T_927; // @[Control.scala 259:84]
  wire [31:0] shr_val_8; // @[Control.scala 219:40]
  wire [31:0] shr_res_8; // @[Control.scala 219:40]
  wire [7:0] _T_928; // @[Control.scala 261:37]
  wire [31:0] src_0_9; // @[Control.scala 219:40]
  wire [31:0] src_1_9; // @[Control.scala 219:40]
  wire  _T_929; // @[Control.scala 253:34]
  wire [31:0] _T_930; // @[Control.scala 253:24]
  wire [31:0] mix_val_9; // @[Control.scala 219:40]
  wire [7:0] _T_931; // @[Control.scala 255:37]
  wire [31:0] _T_932; // @[Control.scala 256:30]
  wire [31:0] _T_933; // @[Control.scala 256:59]
  wire [32:0] _T_934; // @[Control.scala 256:49]
  wire [31:0] _T_935; // @[Control.scala 256:49]
  wire [31:0] _T_936; // @[Control.scala 256:79]
  wire [31:0] add_val_9; // @[Control.scala 219:40]
  wire [31:0] add_res_9; // @[Control.scala 219:40]
  wire [7:0] _T_937; // @[Control.scala 258:37]
  wire [4:0] _T_939; // @[Control.scala 259:60]
  wire [31:0] _T_940; // @[Control.scala 259:49]
  wire [31:0] _T_941; // @[Control.scala 259:84]
  wire [31:0] shr_val_9; // @[Control.scala 219:40]
  wire [31:0] shr_res_9; // @[Control.scala 219:40]
  wire [7:0] _T_942; // @[Control.scala 261:37]
  wire [31:0] src_0_10; // @[Control.scala 219:40]
  wire [31:0] src_1_10; // @[Control.scala 219:40]
  wire  _T_943; // @[Control.scala 253:34]
  wire [31:0] _T_944; // @[Control.scala 253:24]
  wire [31:0] mix_val_10; // @[Control.scala 219:40]
  wire [7:0] _T_945; // @[Control.scala 255:37]
  wire [31:0] _T_946; // @[Control.scala 256:30]
  wire [31:0] _T_947; // @[Control.scala 256:59]
  wire [32:0] _T_948; // @[Control.scala 256:49]
  wire [31:0] _T_949; // @[Control.scala 256:49]
  wire [31:0] _T_950; // @[Control.scala 256:79]
  wire [31:0] add_val_10; // @[Control.scala 219:40]
  wire [31:0] add_res_10; // @[Control.scala 219:40]
  wire [7:0] _T_951; // @[Control.scala 258:37]
  wire [4:0] _T_953; // @[Control.scala 259:60]
  wire [31:0] _T_954; // @[Control.scala 259:49]
  wire [31:0] _T_955; // @[Control.scala 259:84]
  wire [31:0] shr_val_10; // @[Control.scala 219:40]
  wire [31:0] shr_res_10; // @[Control.scala 219:40]
  wire [7:0] _T_956; // @[Control.scala 261:37]
  wire [31:0] src_0_11; // @[Control.scala 219:40]
  wire [31:0] src_1_11; // @[Control.scala 219:40]
  wire  _T_957; // @[Control.scala 253:34]
  wire [31:0] _T_958; // @[Control.scala 253:24]
  wire [31:0] mix_val_11; // @[Control.scala 219:40]
  wire [7:0] _T_959; // @[Control.scala 255:37]
  wire [31:0] _T_960; // @[Control.scala 256:30]
  wire [31:0] _T_961; // @[Control.scala 256:59]
  wire [32:0] _T_962; // @[Control.scala 256:49]
  wire [31:0] _T_963; // @[Control.scala 256:49]
  wire [31:0] _T_964; // @[Control.scala 256:79]
  wire [31:0] add_val_11; // @[Control.scala 219:40]
  wire [31:0] add_res_11; // @[Control.scala 219:40]
  wire [7:0] _T_965; // @[Control.scala 258:37]
  wire [4:0] _T_967; // @[Control.scala 259:60]
  wire [31:0] _T_968; // @[Control.scala 259:49]
  wire [31:0] _T_969; // @[Control.scala 259:84]
  wire [31:0] shr_val_11; // @[Control.scala 219:40]
  wire [31:0] shr_res_11; // @[Control.scala 219:40]
  wire [7:0] _T_970; // @[Control.scala 261:37]
  wire [31:0] src_0_12; // @[Control.scala 219:40]
  wire [31:0] src_1_12; // @[Control.scala 219:40]
  wire  _T_971; // @[Control.scala 253:34]
  wire [31:0] _T_972; // @[Control.scala 253:24]
  wire [31:0] mix_val_12; // @[Control.scala 219:40]
  wire [7:0] _T_973; // @[Control.scala 255:37]
  wire [31:0] _T_974; // @[Control.scala 256:30]
  wire [31:0] _T_975; // @[Control.scala 256:59]
  wire [32:0] _T_976; // @[Control.scala 256:49]
  wire [31:0] _T_977; // @[Control.scala 256:49]
  wire [31:0] _T_978; // @[Control.scala 256:79]
  wire [31:0] add_val_12; // @[Control.scala 219:40]
  wire [31:0] add_res_12; // @[Control.scala 219:40]
  wire [7:0] _T_979; // @[Control.scala 258:37]
  wire [4:0] _T_981; // @[Control.scala 259:60]
  wire [31:0] _T_982; // @[Control.scala 259:49]
  wire [31:0] _T_983; // @[Control.scala 259:84]
  wire [31:0] shr_val_12; // @[Control.scala 219:40]
  wire [31:0] shr_res_12; // @[Control.scala 219:40]
  wire [7:0] _T_984; // @[Control.scala 261:37]
  wire [31:0] src_0_13; // @[Control.scala 219:40]
  wire [31:0] src_1_13; // @[Control.scala 219:40]
  wire  _T_985; // @[Control.scala 253:34]
  wire [31:0] _T_986; // @[Control.scala 253:24]
  wire [31:0] mix_val_13; // @[Control.scala 219:40]
  wire [7:0] _T_987; // @[Control.scala 255:37]
  wire [31:0] _T_988; // @[Control.scala 256:30]
  wire [31:0] _T_989; // @[Control.scala 256:59]
  wire [32:0] _T_990; // @[Control.scala 256:49]
  wire [31:0] _T_991; // @[Control.scala 256:49]
  wire [31:0] _T_992; // @[Control.scala 256:79]
  wire [31:0] add_val_13; // @[Control.scala 219:40]
  wire [31:0] add_res_13; // @[Control.scala 219:40]
  wire [7:0] _T_993; // @[Control.scala 258:37]
  wire [4:0] _T_995; // @[Control.scala 259:60]
  wire [31:0] _T_996; // @[Control.scala 259:49]
  wire [31:0] _T_997; // @[Control.scala 259:84]
  wire [31:0] shr_val_13; // @[Control.scala 219:40]
  wire [31:0] shr_res_13; // @[Control.scala 219:40]
  wire [7:0] _T_998; // @[Control.scala 261:37]
  wire [31:0] src_0_14; // @[Control.scala 219:40]
  wire [31:0] src_1_14; // @[Control.scala 219:40]
  wire  _T_999; // @[Control.scala 253:34]
  wire [31:0] _T_1000; // @[Control.scala 253:24]
  wire [31:0] mix_val_14; // @[Control.scala 219:40]
  wire [7:0] _T_1001; // @[Control.scala 255:37]
  wire [31:0] _T_1002; // @[Control.scala 256:30]
  wire [31:0] _T_1003; // @[Control.scala 256:59]
  wire [32:0] _T_1004; // @[Control.scala 256:49]
  wire [31:0] _T_1005; // @[Control.scala 256:49]
  wire [31:0] _T_1006; // @[Control.scala 256:79]
  wire [31:0] add_val_14; // @[Control.scala 219:40]
  wire [31:0] add_res_14; // @[Control.scala 219:40]
  wire [7:0] _T_1007; // @[Control.scala 258:37]
  wire [4:0] _T_1009; // @[Control.scala 259:60]
  wire [31:0] _T_1010; // @[Control.scala 259:49]
  wire [31:0] _T_1011; // @[Control.scala 259:84]
  wire [31:0] shr_val_14; // @[Control.scala 219:40]
  wire [31:0] shr_res_14; // @[Control.scala 219:40]
  wire [7:0] _T_1012; // @[Control.scala 261:37]
  wire [31:0] src_0_15; // @[Control.scala 219:40]
  wire [31:0] src_1_15; // @[Control.scala 219:40]
  wire  _T_1013; // @[Control.scala 253:34]
  wire [31:0] _T_1014; // @[Control.scala 253:24]
  wire [31:0] mix_val_15; // @[Control.scala 219:40]
  wire [7:0] _T_1015; // @[Control.scala 255:37]
  wire [31:0] _T_1016; // @[Control.scala 256:30]
  wire [31:0] _T_1017; // @[Control.scala 256:59]
  wire [32:0] _T_1018; // @[Control.scala 256:49]
  wire [31:0] _T_1019; // @[Control.scala 256:49]
  wire [31:0] _T_1020; // @[Control.scala 256:79]
  wire [31:0] add_val_15; // @[Control.scala 219:40]
  wire [31:0] add_res_15; // @[Control.scala 219:40]
  wire [7:0] _T_1021; // @[Control.scala 258:37]
  wire [4:0] _T_1023; // @[Control.scala 259:60]
  wire [31:0] _T_1024; // @[Control.scala 259:49]
  wire [31:0] _T_1025; // @[Control.scala 259:84]
  wire [31:0] shr_val_15; // @[Control.scala 219:40]
  wire [31:0] shr_res_15; // @[Control.scala 219:40]
  wire [7:0] _T_1026; // @[Control.scala 261:37]
  wire [7:0] short_cmp_res_0; // @[Control.scala 219:40]
  wire [7:0] short_add_res_0; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_0; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_1; // @[Control.scala 219:40]
  wire [7:0] short_add_res_1; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_1; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_2; // @[Control.scala 219:40]
  wire [7:0] short_add_res_2; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_2; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_3; // @[Control.scala 219:40]
  wire [7:0] short_add_res_3; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_3; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_4; // @[Control.scala 219:40]
  wire [7:0] short_add_res_4; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_4; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_5; // @[Control.scala 219:40]
  wire [7:0] short_add_res_5; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_5; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_6; // @[Control.scala 219:40]
  wire [7:0] short_add_res_6; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_6; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_7; // @[Control.scala 219:40]
  wire [7:0] short_add_res_7; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_7; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_8; // @[Control.scala 219:40]
  wire [7:0] short_add_res_8; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_8; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_9; // @[Control.scala 219:40]
  wire [7:0] short_add_res_9; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_9; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_10; // @[Control.scala 219:40]
  wire [7:0] short_add_res_10; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_10; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_11; // @[Control.scala 219:40]
  wire [7:0] short_add_res_11; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_11; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_12; // @[Control.scala 219:40]
  wire [7:0] short_add_res_12; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_12; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_13; // @[Control.scala 219:40]
  wire [7:0] short_add_res_13; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_13; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_14; // @[Control.scala 219:40]
  wire [7:0] short_add_res_14; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_14; // @[Control.scala 219:40]
  wire [7:0] short_cmp_res_15; // @[Control.scala 219:40]
  wire [7:0] short_add_res_15; // @[Control.scala 219:40]
  wire [7:0] short_shr_res_15; // @[Control.scala 219:40]
  wire  alu_opcode_minmax_en; // @[Control.scala 275:48]
  wire  alu_opcode_add_en; // @[Control.scala 276:39]
  wire [63:0] _T_1035; // @[Cat.scala 30:58]
  wire [127:0] _T_1043; // @[Cat.scala 30:58]
  wire [63:0] _T_1050; // @[Cat.scala 30:58]
  wire [127:0] _T_1058; // @[Cat.scala 30:58]
  wire [63:0] _T_1065; // @[Cat.scala 30:58]
  wire [127:0] _T_1073; // @[Cat.scala 30:58]
  wire [127:0] _T_1074; // @[Control.scala 278:30]
  Queue g2l_queue ( // @[Control.scala 306:25]
    .clock(g2l_queue_clock),
    .reset(g2l_queue_reset),
    .io_enq_ready(g2l_queue_io_enq_ready),
    .io_enq_valid(g2l_queue_io_enq_valid),
    .io_deq_ready(g2l_queue_io_deq_ready),
    .io_deq_valid(g2l_queue_io_deq_valid),
    .io_deq_bits(g2l_queue_io_deq_bits)
  );
  Queue g2s_queue ( // @[Control.scala 307:25]
    .clock(g2s_queue_clock),
    .reset(g2s_queue_reset),
    .io_enq_ready(g2s_queue_io_enq_ready),
    .io_enq_valid(g2s_queue_io_enq_valid),
    .io_deq_ready(g2s_queue_io_deq_ready),
    .io_deq_valid(g2s_queue_io_deq_valid),
    .io_deq_bits(g2s_queue_io_deq_bits)
  );
  assign acc_mem__T_236_addr = dst_idx[7:0];
  assign acc_mem__T_236_data = acc_mem[acc_mem__T_236_addr]; // @[Control.scala 26:20]
  assign acc_mem__T_239_addr = src_idx[7:0];
  assign acc_mem__T_239_data = acc_mem[acc_mem__T_239_addr]; // @[Control.scala 26:20]
  assign acc_mem__T_221_data = io_biases_data;
  assign acc_mem__T_221_addr = acc_mem_addr[7:0];
  assign acc_mem__T_221_mask = 1'h1;
  assign acc_mem__T_221_en = opcode_load_en & memory_type_acc_en;
  assign insn_valid = insn != 128'h0; // @[Control.scala 29:30]
  assign opcode = insn[2:0]; // @[Control.scala 31:29]
  assign push_prev_dep = insn[5]; // @[Control.scala 34:29]
  assign push_next_dep = insn[6]; // @[Control.scala 35:29]
  assign memory_type = insn[8:7]; // @[Control.scala 39:25]
  assign sram_base = insn[24:9]; // @[Control.scala 40:25]
  assign x_size = insn[95:80]; // @[Control.scala 43:25]
  assign y_pad_0 = insn[115:112]; // @[Control.scala 45:25]
  assign x_pad_0 = insn[123:120]; // @[Control.scala 47:25]
  assign x_pad_1 = insn[127:124]; // @[Control.scala 48:25]
  assign _GEN_249 = {{12'd0}, y_pad_0}; // @[Control.scala 52:30]
  assign _GEN_251 = {{12'd0}, x_pad_0}; // @[Control.scala 53:30]
  assign _T_135 = _GEN_251 + x_size; // @[Control.scala 53:30]
  assign _T_136 = _GEN_251 + x_size; // @[Control.scala 53:30]
  assign _GEN_252 = {{12'd0}, x_pad_1}; // @[Control.scala 53:39]
  assign _T_137 = _T_136 + _GEN_252; // @[Control.scala 53:39]
  assign x_size_total = _T_136 + _GEN_252; // @[Control.scala 53:39]
  assign y_offset = x_size_total * _GEN_249; // @[Control.scala 54:31]
  assign _T_140 = opcode == 3'h0; // @[Control.scala 58:32]
  assign _T_142 = opcode == 3'h1; // @[Control.scala 58:60]
  assign opcode_load_en = _T_140 | _T_142; // @[Control.scala 58:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Control.scala 59:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Control.scala 60:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Control.scala 62:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Control.scala 63:40]
  assign acc_x_cntr_en = opcode_load_en & memory_type_acc_en; // @[Control.scala 67:39]
  assign _T_150 = acc_x_cntr_en & io_biases_valid; // @[Control.scala 71:23]
  assign _T_152 = acc_x_cntr_val == 8'h7; // @[Control.scala 72:26]
  assign _T_156 = acc_x_cntr_val + 8'h1; // @[Control.scala 76:40]
  assign _T_157 = acc_x_cntr_val + 8'h1; // @[Control.scala 76:40]
  assign _GEN_0 = _T_152 ? 8'h0 : _T_157; // @[Control.scala 72:54]
  assign _GEN_2 = _T_150 ? _GEN_0 : acc_x_cntr_val; // @[Control.scala 71:43]
  assign acc_x_cntr_wrap = _T_150 ? _T_152 : 1'h0; // @[Control.scala 71:43]
  assign in_loop_cntr_en = opcode_alu_en | opcode_gemm_en; // @[Control.scala 85:40]
  assign _T_164 = io_out_mem_waitrequest == 1'h0; // @[Control.scala 89:28]
  assign _T_165 = in_loop_cntr_en & _T_164; // @[Control.scala 89:25]
  assign _T_167 = dst_offset_in == 8'h7; // @[Control.scala 90:28]
  assign _T_171 = dst_offset_in + 8'h1; // @[Control.scala 94:44]
  assign _T_172 = dst_offset_in + 8'h1; // @[Control.scala 94:44]
  assign _GEN_4 = _T_167 ? 8'h0 : _T_172; // @[Control.scala 90:58]
  assign _GEN_6 = _T_165 ? _GEN_4 : dst_offset_in; // @[Control.scala 89:53]
  assign in_loop_cntr_wrap = _T_165 ? _T_167 : 1'h0; // @[Control.scala 89:53]
  assign _T_175 = opcode_load_en & memory_type_uop_en; // @[Control.scala 104:33]
  assign _T_178 = acc_x_cntr_wrap == 1'h0; // @[Control.scala 105:35]
  assign _T_179 = acc_x_cntr_en & _T_178; // @[Control.scala 105:32]
  assign _T_182 = in_loop_cntr_wrap == 1'h0; // @[Control.scala 106:37]
  assign _T_183 = in_loop_cntr_en & _T_182; // @[Control.scala 106:34]
  assign _T_187 = _T_179 ? 1'h1 : _T_183; // @[Control.scala 105:17]
  assign busy = _T_175 ? 1'h0 : _T_187; // @[Control.scala 104:17]
  assign _T_189 = busy == 1'h0; // @[Control.scala 110:32]
  assign _T_190 = io_gemm_queue_valid & _T_189; // @[Control.scala 110:29]
  assign _T_193 = io_uops_valid & memory_type_uop_en; // @[Control.scala 119:23]
  assign _T_194 = _T_193 & insn_valid; // @[Control.scala 119:45]
  assign _GEN_254 = {{4'd0}, sram_base}; // @[Control.scala 146:33]
  assign _T_213 = _GEN_254 + y_offset; // @[Control.scala 146:33]
  assign _T_214 = _GEN_254 + y_offset; // @[Control.scala 146:33]
  assign _GEN_255 = {{16'd0}, x_pad_0}; // @[Control.scala 146:44]
  assign _T_215 = _T_214 + _GEN_255; // @[Control.scala 146:44]
  assign _T_216 = _T_214 + _GEN_255; // @[Control.scala 146:44]
  assign _T_218 = _T_216 * 20'h1; // @[Control.scala 146:55]
  assign _GEN_256 = {{13'd0}, acc_x_cntr_val}; // @[Control.scala 146:65]
  assign _T_219 = _T_218 + _GEN_256; // @[Control.scala 146:65]
  assign acc_mem_addr = _T_218 + _GEN_256; // @[Control.scala 146:65]
  assign alu_opcode = insn[109:108]; // @[Control.scala 168:24]
  assign use_imm = insn[110]; // @[Control.scala 169:21]
  assign imm_raw = insn[126:111]; // @[Control.scala 170:21]
  assign _T_222 = $signed(imm_raw); // @[Control.scala 171:25]
  assign _T_224 = $signed(_T_222) < $signed(16'sh0); // @[Control.scala 171:32]
  assign _T_226 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_228 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_229 = _T_224 ? _T_226 : {{15'd0}, _T_228}; // @[Control.scala 171:16]
  assign imm = $signed(_T_229); // @[Control.scala 171:89]
  assign _T_231 = io_uop_mem_readdata[10:0]; // @[Control.scala 180:20]
  assign _GEN_257 = {{3'd0}, dst_offset_in}; // @[Control.scala 180:47]
  assign _T_232 = _T_231 + _GEN_257; // @[Control.scala 180:47]
  assign dst_idx = _T_231 + _GEN_257; // @[Control.scala 180:47]
  assign _T_233 = io_uop_mem_readdata[21:11]; // @[Control.scala 181:20]
  assign _T_234 = _T_233 + _GEN_257; // @[Control.scala 181:47]
  assign src_idx = _T_233 + _GEN_257; // @[Control.scala 181:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Control.scala 200:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Control.scala 201:38]
  assign _T_674 = insn_valid & in_loop_cntr_en; // @[Control.scala 219:20]
  assign _T_675 = src_vector[31:0]; // @[Control.scala 239:31]
  assign _T_676 = $signed(_T_675); // @[Control.scala 239:72]
  assign _T_677 = dst_vector[31:0]; // @[Control.scala 240:31]
  assign _T_678 = $signed(_T_677); // @[Control.scala 240:72]
  assign _T_679 = src_vector[63:32]; // @[Control.scala 239:31]
  assign _T_680 = $signed(_T_679); // @[Control.scala 239:72]
  assign _T_681 = dst_vector[63:32]; // @[Control.scala 240:31]
  assign _T_682 = $signed(_T_681); // @[Control.scala 240:72]
  assign _T_683 = src_vector[95:64]; // @[Control.scala 239:31]
  assign _T_684 = $signed(_T_683); // @[Control.scala 239:72]
  assign _T_685 = dst_vector[95:64]; // @[Control.scala 240:31]
  assign _T_686 = $signed(_T_685); // @[Control.scala 240:72]
  assign _T_687 = src_vector[127:96]; // @[Control.scala 239:31]
  assign _T_688 = $signed(_T_687); // @[Control.scala 239:72]
  assign _T_689 = dst_vector[127:96]; // @[Control.scala 240:31]
  assign _T_690 = $signed(_T_689); // @[Control.scala 240:72]
  assign _T_691 = src_vector[159:128]; // @[Control.scala 239:31]
  assign _T_692 = $signed(_T_691); // @[Control.scala 239:72]
  assign _T_693 = dst_vector[159:128]; // @[Control.scala 240:31]
  assign _T_694 = $signed(_T_693); // @[Control.scala 240:72]
  assign _T_695 = src_vector[191:160]; // @[Control.scala 239:31]
  assign _T_696 = $signed(_T_695); // @[Control.scala 239:72]
  assign _T_697 = dst_vector[191:160]; // @[Control.scala 240:31]
  assign _T_698 = $signed(_T_697); // @[Control.scala 240:72]
  assign _T_699 = src_vector[223:192]; // @[Control.scala 239:31]
  assign _T_700 = $signed(_T_699); // @[Control.scala 239:72]
  assign _T_701 = dst_vector[223:192]; // @[Control.scala 240:31]
  assign _T_702 = $signed(_T_701); // @[Control.scala 240:72]
  assign _T_703 = src_vector[255:224]; // @[Control.scala 239:31]
  assign _T_704 = $signed(_T_703); // @[Control.scala 239:72]
  assign _T_705 = dst_vector[255:224]; // @[Control.scala 240:31]
  assign _T_706 = $signed(_T_705); // @[Control.scala 240:72]
  assign _T_707 = src_vector[287:256]; // @[Control.scala 239:31]
  assign _T_708 = $signed(_T_707); // @[Control.scala 239:72]
  assign _T_709 = dst_vector[287:256]; // @[Control.scala 240:31]
  assign _T_710 = $signed(_T_709); // @[Control.scala 240:72]
  assign _T_711 = src_vector[319:288]; // @[Control.scala 239:31]
  assign _T_712 = $signed(_T_711); // @[Control.scala 239:72]
  assign _T_713 = dst_vector[319:288]; // @[Control.scala 240:31]
  assign _T_714 = $signed(_T_713); // @[Control.scala 240:72]
  assign _T_715 = src_vector[351:320]; // @[Control.scala 239:31]
  assign _T_716 = $signed(_T_715); // @[Control.scala 239:72]
  assign _T_717 = dst_vector[351:320]; // @[Control.scala 240:31]
  assign _T_718 = $signed(_T_717); // @[Control.scala 240:72]
  assign _T_719 = src_vector[383:352]; // @[Control.scala 239:31]
  assign _T_720 = $signed(_T_719); // @[Control.scala 239:72]
  assign _T_721 = dst_vector[383:352]; // @[Control.scala 240:31]
  assign _T_722 = $signed(_T_721); // @[Control.scala 240:72]
  assign _T_723 = src_vector[415:384]; // @[Control.scala 239:31]
  assign _T_724 = $signed(_T_723); // @[Control.scala 239:72]
  assign _T_725 = dst_vector[415:384]; // @[Control.scala 240:31]
  assign _T_726 = $signed(_T_725); // @[Control.scala 240:72]
  assign _T_727 = src_vector[447:416]; // @[Control.scala 239:31]
  assign _T_728 = $signed(_T_727); // @[Control.scala 239:72]
  assign _T_729 = dst_vector[447:416]; // @[Control.scala 240:31]
  assign _T_730 = $signed(_T_729); // @[Control.scala 240:72]
  assign _T_731 = src_vector[479:448]; // @[Control.scala 239:31]
  assign _T_732 = $signed(_T_731); // @[Control.scala 239:72]
  assign _T_733 = dst_vector[479:448]; // @[Control.scala 240:31]
  assign _T_734 = $signed(_T_733); // @[Control.scala 240:72]
  assign _T_735 = src_vector[511:480]; // @[Control.scala 239:31]
  assign _T_736 = $signed(_T_735); // @[Control.scala 239:72]
  assign _T_737 = dst_vector[511:480]; // @[Control.scala 240:31]
  assign _T_738 = $signed(_T_737); // @[Control.scala 240:72]
  assign _GEN_17 = alu_opcode_max_en ? $signed(_T_676) : $signed(_T_678); // @[Control.scala 237:30]
  assign _GEN_18 = alu_opcode_max_en ? $signed(_T_678) : $signed(_T_676); // @[Control.scala 237:30]
  assign _GEN_19 = alu_opcode_max_en ? $signed(_T_680) : $signed(_T_682); // @[Control.scala 237:30]
  assign _GEN_20 = alu_opcode_max_en ? $signed(_T_682) : $signed(_T_680); // @[Control.scala 237:30]
  assign _GEN_21 = alu_opcode_max_en ? $signed(_T_684) : $signed(_T_686); // @[Control.scala 237:30]
  assign _GEN_22 = alu_opcode_max_en ? $signed(_T_686) : $signed(_T_684); // @[Control.scala 237:30]
  assign _GEN_23 = alu_opcode_max_en ? $signed(_T_688) : $signed(_T_690); // @[Control.scala 237:30]
  assign _GEN_24 = alu_opcode_max_en ? $signed(_T_690) : $signed(_T_688); // @[Control.scala 237:30]
  assign _GEN_25 = alu_opcode_max_en ? $signed(_T_692) : $signed(_T_694); // @[Control.scala 237:30]
  assign _GEN_26 = alu_opcode_max_en ? $signed(_T_694) : $signed(_T_692); // @[Control.scala 237:30]
  assign _GEN_27 = alu_opcode_max_en ? $signed(_T_696) : $signed(_T_698); // @[Control.scala 237:30]
  assign _GEN_28 = alu_opcode_max_en ? $signed(_T_698) : $signed(_T_696); // @[Control.scala 237:30]
  assign _GEN_29 = alu_opcode_max_en ? $signed(_T_700) : $signed(_T_702); // @[Control.scala 237:30]
  assign _GEN_30 = alu_opcode_max_en ? $signed(_T_702) : $signed(_T_700); // @[Control.scala 237:30]
  assign _GEN_31 = alu_opcode_max_en ? $signed(_T_704) : $signed(_T_706); // @[Control.scala 237:30]
  assign _GEN_32 = alu_opcode_max_en ? $signed(_T_706) : $signed(_T_704); // @[Control.scala 237:30]
  assign _GEN_33 = alu_opcode_max_en ? $signed(_T_708) : $signed(_T_710); // @[Control.scala 237:30]
  assign _GEN_34 = alu_opcode_max_en ? $signed(_T_710) : $signed(_T_708); // @[Control.scala 237:30]
  assign _GEN_35 = alu_opcode_max_en ? $signed(_T_712) : $signed(_T_714); // @[Control.scala 237:30]
  assign _GEN_36 = alu_opcode_max_en ? $signed(_T_714) : $signed(_T_712); // @[Control.scala 237:30]
  assign _GEN_37 = alu_opcode_max_en ? $signed(_T_716) : $signed(_T_718); // @[Control.scala 237:30]
  assign _GEN_38 = alu_opcode_max_en ? $signed(_T_718) : $signed(_T_716); // @[Control.scala 237:30]
  assign _GEN_39 = alu_opcode_max_en ? $signed(_T_720) : $signed(_T_722); // @[Control.scala 237:30]
  assign _GEN_40 = alu_opcode_max_en ? $signed(_T_722) : $signed(_T_720); // @[Control.scala 237:30]
  assign _GEN_41 = alu_opcode_max_en ? $signed(_T_724) : $signed(_T_726); // @[Control.scala 237:30]
  assign _GEN_42 = alu_opcode_max_en ? $signed(_T_726) : $signed(_T_724); // @[Control.scala 237:30]
  assign _GEN_43 = alu_opcode_max_en ? $signed(_T_728) : $signed(_T_730); // @[Control.scala 237:30]
  assign _GEN_44 = alu_opcode_max_en ? $signed(_T_730) : $signed(_T_728); // @[Control.scala 237:30]
  assign _GEN_45 = alu_opcode_max_en ? $signed(_T_732) : $signed(_T_734); // @[Control.scala 237:30]
  assign _GEN_46 = alu_opcode_max_en ? $signed(_T_734) : $signed(_T_732); // @[Control.scala 237:30]
  assign _GEN_47 = alu_opcode_max_en ? $signed(_T_736) : $signed(_T_738); // @[Control.scala 237:30]
  assign _GEN_48 = alu_opcode_max_en ? $signed(_T_738) : $signed(_T_736); // @[Control.scala 237:30]
  assign _GEN_49 = use_imm ? $signed(imm) : $signed(_GEN_18); // @[Control.scala 248:20]
  assign _GEN_50 = use_imm ? $signed(imm) : $signed(_GEN_20); // @[Control.scala 248:20]
  assign _GEN_51 = use_imm ? $signed(imm) : $signed(_GEN_22); // @[Control.scala 248:20]
  assign _GEN_52 = use_imm ? $signed(imm) : $signed(_GEN_24); // @[Control.scala 248:20]
  assign _GEN_53 = use_imm ? $signed(imm) : $signed(_GEN_26); // @[Control.scala 248:20]
  assign _GEN_54 = use_imm ? $signed(imm) : $signed(_GEN_28); // @[Control.scala 248:20]
  assign _GEN_55 = use_imm ? $signed(imm) : $signed(_GEN_30); // @[Control.scala 248:20]
  assign _GEN_56 = use_imm ? $signed(imm) : $signed(_GEN_32); // @[Control.scala 248:20]
  assign _GEN_57 = use_imm ? $signed(imm) : $signed(_GEN_34); // @[Control.scala 248:20]
  assign _GEN_58 = use_imm ? $signed(imm) : $signed(_GEN_36); // @[Control.scala 248:20]
  assign _GEN_59 = use_imm ? $signed(imm) : $signed(_GEN_38); // @[Control.scala 248:20]
  assign _GEN_60 = use_imm ? $signed(imm) : $signed(_GEN_40); // @[Control.scala 248:20]
  assign _GEN_61 = use_imm ? $signed(imm) : $signed(_GEN_42); // @[Control.scala 248:20]
  assign _GEN_62 = use_imm ? $signed(imm) : $signed(_GEN_44); // @[Control.scala 248:20]
  assign _GEN_63 = use_imm ? $signed(imm) : $signed(_GEN_46); // @[Control.scala 248:20]
  assign _GEN_64 = use_imm ? $signed(imm) : $signed(_GEN_48); // @[Control.scala 248:20]
  assign src_0_0 = _T_674 ? $signed(_GEN_17) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_0 = _T_674 ? $signed(_GEN_49) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_803 = $signed(src_0_0) < $signed(src_1_0); // @[Control.scala 253:34]
  assign _T_804 = _T_803 ? $signed(src_0_0) : $signed(src_1_0); // @[Control.scala 253:24]
  assign mix_val_0 = _T_674 ? $signed(_T_804) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_805 = mix_val_0[7:0]; // @[Control.scala 255:37]
  assign _T_806 = $unsigned(src_0_0); // @[Control.scala 256:30]
  assign _T_807 = $unsigned(src_1_0); // @[Control.scala 256:59]
  assign _T_808 = _T_806 + _T_807; // @[Control.scala 256:49]
  assign _T_809 = _T_806 + _T_807; // @[Control.scala 256:49]
  assign _T_810 = $signed(_T_809); // @[Control.scala 256:79]
  assign add_val_0 = _T_674 ? $signed(_T_810) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_0 = _T_674 ? $signed(add_val_0) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_811 = add_res_0[7:0]; // @[Control.scala 258:37]
  assign _T_813 = src_1_0[4:0]; // @[Control.scala 259:60]
  assign _T_814 = _T_806 >> _T_813; // @[Control.scala 259:49]
  assign _T_815 = $signed(_T_814); // @[Control.scala 259:84]
  assign shr_val_0 = _T_674 ? $signed(_T_815) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_0 = _T_674 ? $signed(shr_val_0) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_816 = shr_res_0[7:0]; // @[Control.scala 261:37]
  assign src_0_1 = _T_674 ? $signed(_GEN_19) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_1 = _T_674 ? $signed(_GEN_50) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_817 = $signed(src_0_1) < $signed(src_1_1); // @[Control.scala 253:34]
  assign _T_818 = _T_817 ? $signed(src_0_1) : $signed(src_1_1); // @[Control.scala 253:24]
  assign mix_val_1 = _T_674 ? $signed(_T_818) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_819 = mix_val_1[7:0]; // @[Control.scala 255:37]
  assign _T_820 = $unsigned(src_0_1); // @[Control.scala 256:30]
  assign _T_821 = $unsigned(src_1_1); // @[Control.scala 256:59]
  assign _T_822 = _T_820 + _T_821; // @[Control.scala 256:49]
  assign _T_823 = _T_820 + _T_821; // @[Control.scala 256:49]
  assign _T_824 = $signed(_T_823); // @[Control.scala 256:79]
  assign add_val_1 = _T_674 ? $signed(_T_824) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_1 = _T_674 ? $signed(add_val_1) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_825 = add_res_1[7:0]; // @[Control.scala 258:37]
  assign _T_827 = src_1_1[4:0]; // @[Control.scala 259:60]
  assign _T_828 = _T_820 >> _T_827; // @[Control.scala 259:49]
  assign _T_829 = $signed(_T_828); // @[Control.scala 259:84]
  assign shr_val_1 = _T_674 ? $signed(_T_829) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_1 = _T_674 ? $signed(shr_val_1) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_830 = shr_res_1[7:0]; // @[Control.scala 261:37]
  assign src_0_2 = _T_674 ? $signed(_GEN_21) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_2 = _T_674 ? $signed(_GEN_51) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_831 = $signed(src_0_2) < $signed(src_1_2); // @[Control.scala 253:34]
  assign _T_832 = _T_831 ? $signed(src_0_2) : $signed(src_1_2); // @[Control.scala 253:24]
  assign mix_val_2 = _T_674 ? $signed(_T_832) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_833 = mix_val_2[7:0]; // @[Control.scala 255:37]
  assign _T_834 = $unsigned(src_0_2); // @[Control.scala 256:30]
  assign _T_835 = $unsigned(src_1_2); // @[Control.scala 256:59]
  assign _T_836 = _T_834 + _T_835; // @[Control.scala 256:49]
  assign _T_837 = _T_834 + _T_835; // @[Control.scala 256:49]
  assign _T_838 = $signed(_T_837); // @[Control.scala 256:79]
  assign add_val_2 = _T_674 ? $signed(_T_838) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_2 = _T_674 ? $signed(add_val_2) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_839 = add_res_2[7:0]; // @[Control.scala 258:37]
  assign _T_841 = src_1_2[4:0]; // @[Control.scala 259:60]
  assign _T_842 = _T_834 >> _T_841; // @[Control.scala 259:49]
  assign _T_843 = $signed(_T_842); // @[Control.scala 259:84]
  assign shr_val_2 = _T_674 ? $signed(_T_843) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_2 = _T_674 ? $signed(shr_val_2) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_844 = shr_res_2[7:0]; // @[Control.scala 261:37]
  assign src_0_3 = _T_674 ? $signed(_GEN_23) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_3 = _T_674 ? $signed(_GEN_52) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_845 = $signed(src_0_3) < $signed(src_1_3); // @[Control.scala 253:34]
  assign _T_846 = _T_845 ? $signed(src_0_3) : $signed(src_1_3); // @[Control.scala 253:24]
  assign mix_val_3 = _T_674 ? $signed(_T_846) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_847 = mix_val_3[7:0]; // @[Control.scala 255:37]
  assign _T_848 = $unsigned(src_0_3); // @[Control.scala 256:30]
  assign _T_849 = $unsigned(src_1_3); // @[Control.scala 256:59]
  assign _T_850 = _T_848 + _T_849; // @[Control.scala 256:49]
  assign _T_851 = _T_848 + _T_849; // @[Control.scala 256:49]
  assign _T_852 = $signed(_T_851); // @[Control.scala 256:79]
  assign add_val_3 = _T_674 ? $signed(_T_852) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_3 = _T_674 ? $signed(add_val_3) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_853 = add_res_3[7:0]; // @[Control.scala 258:37]
  assign _T_855 = src_1_3[4:0]; // @[Control.scala 259:60]
  assign _T_856 = _T_848 >> _T_855; // @[Control.scala 259:49]
  assign _T_857 = $signed(_T_856); // @[Control.scala 259:84]
  assign shr_val_3 = _T_674 ? $signed(_T_857) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_3 = _T_674 ? $signed(shr_val_3) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_858 = shr_res_3[7:0]; // @[Control.scala 261:37]
  assign src_0_4 = _T_674 ? $signed(_GEN_25) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_4 = _T_674 ? $signed(_GEN_53) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_859 = $signed(src_0_4) < $signed(src_1_4); // @[Control.scala 253:34]
  assign _T_860 = _T_859 ? $signed(src_0_4) : $signed(src_1_4); // @[Control.scala 253:24]
  assign mix_val_4 = _T_674 ? $signed(_T_860) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_861 = mix_val_4[7:0]; // @[Control.scala 255:37]
  assign _T_862 = $unsigned(src_0_4); // @[Control.scala 256:30]
  assign _T_863 = $unsigned(src_1_4); // @[Control.scala 256:59]
  assign _T_864 = _T_862 + _T_863; // @[Control.scala 256:49]
  assign _T_865 = _T_862 + _T_863; // @[Control.scala 256:49]
  assign _T_866 = $signed(_T_865); // @[Control.scala 256:79]
  assign add_val_4 = _T_674 ? $signed(_T_866) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_4 = _T_674 ? $signed(add_val_4) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_867 = add_res_4[7:0]; // @[Control.scala 258:37]
  assign _T_869 = src_1_4[4:0]; // @[Control.scala 259:60]
  assign _T_870 = _T_862 >> _T_869; // @[Control.scala 259:49]
  assign _T_871 = $signed(_T_870); // @[Control.scala 259:84]
  assign shr_val_4 = _T_674 ? $signed(_T_871) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_4 = _T_674 ? $signed(shr_val_4) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_872 = shr_res_4[7:0]; // @[Control.scala 261:37]
  assign src_0_5 = _T_674 ? $signed(_GEN_27) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_5 = _T_674 ? $signed(_GEN_54) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_873 = $signed(src_0_5) < $signed(src_1_5); // @[Control.scala 253:34]
  assign _T_874 = _T_873 ? $signed(src_0_5) : $signed(src_1_5); // @[Control.scala 253:24]
  assign mix_val_5 = _T_674 ? $signed(_T_874) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_875 = mix_val_5[7:0]; // @[Control.scala 255:37]
  assign _T_876 = $unsigned(src_0_5); // @[Control.scala 256:30]
  assign _T_877 = $unsigned(src_1_5); // @[Control.scala 256:59]
  assign _T_878 = _T_876 + _T_877; // @[Control.scala 256:49]
  assign _T_879 = _T_876 + _T_877; // @[Control.scala 256:49]
  assign _T_880 = $signed(_T_879); // @[Control.scala 256:79]
  assign add_val_5 = _T_674 ? $signed(_T_880) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_5 = _T_674 ? $signed(add_val_5) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_881 = add_res_5[7:0]; // @[Control.scala 258:37]
  assign _T_883 = src_1_5[4:0]; // @[Control.scala 259:60]
  assign _T_884 = _T_876 >> _T_883; // @[Control.scala 259:49]
  assign _T_885 = $signed(_T_884); // @[Control.scala 259:84]
  assign shr_val_5 = _T_674 ? $signed(_T_885) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_5 = _T_674 ? $signed(shr_val_5) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_886 = shr_res_5[7:0]; // @[Control.scala 261:37]
  assign src_0_6 = _T_674 ? $signed(_GEN_29) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_6 = _T_674 ? $signed(_GEN_55) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_887 = $signed(src_0_6) < $signed(src_1_6); // @[Control.scala 253:34]
  assign _T_888 = _T_887 ? $signed(src_0_6) : $signed(src_1_6); // @[Control.scala 253:24]
  assign mix_val_6 = _T_674 ? $signed(_T_888) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_889 = mix_val_6[7:0]; // @[Control.scala 255:37]
  assign _T_890 = $unsigned(src_0_6); // @[Control.scala 256:30]
  assign _T_891 = $unsigned(src_1_6); // @[Control.scala 256:59]
  assign _T_892 = _T_890 + _T_891; // @[Control.scala 256:49]
  assign _T_893 = _T_890 + _T_891; // @[Control.scala 256:49]
  assign _T_894 = $signed(_T_893); // @[Control.scala 256:79]
  assign add_val_6 = _T_674 ? $signed(_T_894) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_6 = _T_674 ? $signed(add_val_6) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_895 = add_res_6[7:0]; // @[Control.scala 258:37]
  assign _T_897 = src_1_6[4:0]; // @[Control.scala 259:60]
  assign _T_898 = _T_890 >> _T_897; // @[Control.scala 259:49]
  assign _T_899 = $signed(_T_898); // @[Control.scala 259:84]
  assign shr_val_6 = _T_674 ? $signed(_T_899) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_6 = _T_674 ? $signed(shr_val_6) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_900 = shr_res_6[7:0]; // @[Control.scala 261:37]
  assign src_0_7 = _T_674 ? $signed(_GEN_31) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_7 = _T_674 ? $signed(_GEN_56) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_901 = $signed(src_0_7) < $signed(src_1_7); // @[Control.scala 253:34]
  assign _T_902 = _T_901 ? $signed(src_0_7) : $signed(src_1_7); // @[Control.scala 253:24]
  assign mix_val_7 = _T_674 ? $signed(_T_902) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_903 = mix_val_7[7:0]; // @[Control.scala 255:37]
  assign _T_904 = $unsigned(src_0_7); // @[Control.scala 256:30]
  assign _T_905 = $unsigned(src_1_7); // @[Control.scala 256:59]
  assign _T_906 = _T_904 + _T_905; // @[Control.scala 256:49]
  assign _T_907 = _T_904 + _T_905; // @[Control.scala 256:49]
  assign _T_908 = $signed(_T_907); // @[Control.scala 256:79]
  assign add_val_7 = _T_674 ? $signed(_T_908) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_7 = _T_674 ? $signed(add_val_7) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_909 = add_res_7[7:0]; // @[Control.scala 258:37]
  assign _T_911 = src_1_7[4:0]; // @[Control.scala 259:60]
  assign _T_912 = _T_904 >> _T_911; // @[Control.scala 259:49]
  assign _T_913 = $signed(_T_912); // @[Control.scala 259:84]
  assign shr_val_7 = _T_674 ? $signed(_T_913) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_7 = _T_674 ? $signed(shr_val_7) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_914 = shr_res_7[7:0]; // @[Control.scala 261:37]
  assign src_0_8 = _T_674 ? $signed(_GEN_33) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_8 = _T_674 ? $signed(_GEN_57) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_915 = $signed(src_0_8) < $signed(src_1_8); // @[Control.scala 253:34]
  assign _T_916 = _T_915 ? $signed(src_0_8) : $signed(src_1_8); // @[Control.scala 253:24]
  assign mix_val_8 = _T_674 ? $signed(_T_916) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_917 = mix_val_8[7:0]; // @[Control.scala 255:37]
  assign _T_918 = $unsigned(src_0_8); // @[Control.scala 256:30]
  assign _T_919 = $unsigned(src_1_8); // @[Control.scala 256:59]
  assign _T_920 = _T_918 + _T_919; // @[Control.scala 256:49]
  assign _T_921 = _T_918 + _T_919; // @[Control.scala 256:49]
  assign _T_922 = $signed(_T_921); // @[Control.scala 256:79]
  assign add_val_8 = _T_674 ? $signed(_T_922) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_8 = _T_674 ? $signed(add_val_8) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_923 = add_res_8[7:0]; // @[Control.scala 258:37]
  assign _T_925 = src_1_8[4:0]; // @[Control.scala 259:60]
  assign _T_926 = _T_918 >> _T_925; // @[Control.scala 259:49]
  assign _T_927 = $signed(_T_926); // @[Control.scala 259:84]
  assign shr_val_8 = _T_674 ? $signed(_T_927) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_8 = _T_674 ? $signed(shr_val_8) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_928 = shr_res_8[7:0]; // @[Control.scala 261:37]
  assign src_0_9 = _T_674 ? $signed(_GEN_35) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_9 = _T_674 ? $signed(_GEN_58) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_929 = $signed(src_0_9) < $signed(src_1_9); // @[Control.scala 253:34]
  assign _T_930 = _T_929 ? $signed(src_0_9) : $signed(src_1_9); // @[Control.scala 253:24]
  assign mix_val_9 = _T_674 ? $signed(_T_930) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_931 = mix_val_9[7:0]; // @[Control.scala 255:37]
  assign _T_932 = $unsigned(src_0_9); // @[Control.scala 256:30]
  assign _T_933 = $unsigned(src_1_9); // @[Control.scala 256:59]
  assign _T_934 = _T_932 + _T_933; // @[Control.scala 256:49]
  assign _T_935 = _T_932 + _T_933; // @[Control.scala 256:49]
  assign _T_936 = $signed(_T_935); // @[Control.scala 256:79]
  assign add_val_9 = _T_674 ? $signed(_T_936) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_9 = _T_674 ? $signed(add_val_9) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_937 = add_res_9[7:0]; // @[Control.scala 258:37]
  assign _T_939 = src_1_9[4:0]; // @[Control.scala 259:60]
  assign _T_940 = _T_932 >> _T_939; // @[Control.scala 259:49]
  assign _T_941 = $signed(_T_940); // @[Control.scala 259:84]
  assign shr_val_9 = _T_674 ? $signed(_T_941) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_9 = _T_674 ? $signed(shr_val_9) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_942 = shr_res_9[7:0]; // @[Control.scala 261:37]
  assign src_0_10 = _T_674 ? $signed(_GEN_37) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_10 = _T_674 ? $signed(_GEN_59) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_943 = $signed(src_0_10) < $signed(src_1_10); // @[Control.scala 253:34]
  assign _T_944 = _T_943 ? $signed(src_0_10) : $signed(src_1_10); // @[Control.scala 253:24]
  assign mix_val_10 = _T_674 ? $signed(_T_944) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_945 = mix_val_10[7:0]; // @[Control.scala 255:37]
  assign _T_946 = $unsigned(src_0_10); // @[Control.scala 256:30]
  assign _T_947 = $unsigned(src_1_10); // @[Control.scala 256:59]
  assign _T_948 = _T_946 + _T_947; // @[Control.scala 256:49]
  assign _T_949 = _T_946 + _T_947; // @[Control.scala 256:49]
  assign _T_950 = $signed(_T_949); // @[Control.scala 256:79]
  assign add_val_10 = _T_674 ? $signed(_T_950) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_10 = _T_674 ? $signed(add_val_10) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_951 = add_res_10[7:0]; // @[Control.scala 258:37]
  assign _T_953 = src_1_10[4:0]; // @[Control.scala 259:60]
  assign _T_954 = _T_946 >> _T_953; // @[Control.scala 259:49]
  assign _T_955 = $signed(_T_954); // @[Control.scala 259:84]
  assign shr_val_10 = _T_674 ? $signed(_T_955) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_10 = _T_674 ? $signed(shr_val_10) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_956 = shr_res_10[7:0]; // @[Control.scala 261:37]
  assign src_0_11 = _T_674 ? $signed(_GEN_39) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_11 = _T_674 ? $signed(_GEN_60) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_957 = $signed(src_0_11) < $signed(src_1_11); // @[Control.scala 253:34]
  assign _T_958 = _T_957 ? $signed(src_0_11) : $signed(src_1_11); // @[Control.scala 253:24]
  assign mix_val_11 = _T_674 ? $signed(_T_958) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_959 = mix_val_11[7:0]; // @[Control.scala 255:37]
  assign _T_960 = $unsigned(src_0_11); // @[Control.scala 256:30]
  assign _T_961 = $unsigned(src_1_11); // @[Control.scala 256:59]
  assign _T_962 = _T_960 + _T_961; // @[Control.scala 256:49]
  assign _T_963 = _T_960 + _T_961; // @[Control.scala 256:49]
  assign _T_964 = $signed(_T_963); // @[Control.scala 256:79]
  assign add_val_11 = _T_674 ? $signed(_T_964) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_11 = _T_674 ? $signed(add_val_11) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_965 = add_res_11[7:0]; // @[Control.scala 258:37]
  assign _T_967 = src_1_11[4:0]; // @[Control.scala 259:60]
  assign _T_968 = _T_960 >> _T_967; // @[Control.scala 259:49]
  assign _T_969 = $signed(_T_968); // @[Control.scala 259:84]
  assign shr_val_11 = _T_674 ? $signed(_T_969) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_11 = _T_674 ? $signed(shr_val_11) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_970 = shr_res_11[7:0]; // @[Control.scala 261:37]
  assign src_0_12 = _T_674 ? $signed(_GEN_41) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_12 = _T_674 ? $signed(_GEN_61) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_971 = $signed(src_0_12) < $signed(src_1_12); // @[Control.scala 253:34]
  assign _T_972 = _T_971 ? $signed(src_0_12) : $signed(src_1_12); // @[Control.scala 253:24]
  assign mix_val_12 = _T_674 ? $signed(_T_972) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_973 = mix_val_12[7:0]; // @[Control.scala 255:37]
  assign _T_974 = $unsigned(src_0_12); // @[Control.scala 256:30]
  assign _T_975 = $unsigned(src_1_12); // @[Control.scala 256:59]
  assign _T_976 = _T_974 + _T_975; // @[Control.scala 256:49]
  assign _T_977 = _T_974 + _T_975; // @[Control.scala 256:49]
  assign _T_978 = $signed(_T_977); // @[Control.scala 256:79]
  assign add_val_12 = _T_674 ? $signed(_T_978) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_12 = _T_674 ? $signed(add_val_12) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_979 = add_res_12[7:0]; // @[Control.scala 258:37]
  assign _T_981 = src_1_12[4:0]; // @[Control.scala 259:60]
  assign _T_982 = _T_974 >> _T_981; // @[Control.scala 259:49]
  assign _T_983 = $signed(_T_982); // @[Control.scala 259:84]
  assign shr_val_12 = _T_674 ? $signed(_T_983) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_12 = _T_674 ? $signed(shr_val_12) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_984 = shr_res_12[7:0]; // @[Control.scala 261:37]
  assign src_0_13 = _T_674 ? $signed(_GEN_43) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_13 = _T_674 ? $signed(_GEN_62) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_985 = $signed(src_0_13) < $signed(src_1_13); // @[Control.scala 253:34]
  assign _T_986 = _T_985 ? $signed(src_0_13) : $signed(src_1_13); // @[Control.scala 253:24]
  assign mix_val_13 = _T_674 ? $signed(_T_986) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_987 = mix_val_13[7:0]; // @[Control.scala 255:37]
  assign _T_988 = $unsigned(src_0_13); // @[Control.scala 256:30]
  assign _T_989 = $unsigned(src_1_13); // @[Control.scala 256:59]
  assign _T_990 = _T_988 + _T_989; // @[Control.scala 256:49]
  assign _T_991 = _T_988 + _T_989; // @[Control.scala 256:49]
  assign _T_992 = $signed(_T_991); // @[Control.scala 256:79]
  assign add_val_13 = _T_674 ? $signed(_T_992) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_13 = _T_674 ? $signed(add_val_13) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_993 = add_res_13[7:0]; // @[Control.scala 258:37]
  assign _T_995 = src_1_13[4:0]; // @[Control.scala 259:60]
  assign _T_996 = _T_988 >> _T_995; // @[Control.scala 259:49]
  assign _T_997 = $signed(_T_996); // @[Control.scala 259:84]
  assign shr_val_13 = _T_674 ? $signed(_T_997) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_13 = _T_674 ? $signed(shr_val_13) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_998 = shr_res_13[7:0]; // @[Control.scala 261:37]
  assign src_0_14 = _T_674 ? $signed(_GEN_45) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_14 = _T_674 ? $signed(_GEN_63) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_999 = $signed(src_0_14) < $signed(src_1_14); // @[Control.scala 253:34]
  assign _T_1000 = _T_999 ? $signed(src_0_14) : $signed(src_1_14); // @[Control.scala 253:24]
  assign mix_val_14 = _T_674 ? $signed(_T_1000) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1001 = mix_val_14[7:0]; // @[Control.scala 255:37]
  assign _T_1002 = $unsigned(src_0_14); // @[Control.scala 256:30]
  assign _T_1003 = $unsigned(src_1_14); // @[Control.scala 256:59]
  assign _T_1004 = _T_1002 + _T_1003; // @[Control.scala 256:49]
  assign _T_1005 = _T_1002 + _T_1003; // @[Control.scala 256:49]
  assign _T_1006 = $signed(_T_1005); // @[Control.scala 256:79]
  assign add_val_14 = _T_674 ? $signed(_T_1006) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_14 = _T_674 ? $signed(add_val_14) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1007 = add_res_14[7:0]; // @[Control.scala 258:37]
  assign _T_1009 = src_1_14[4:0]; // @[Control.scala 259:60]
  assign _T_1010 = _T_1002 >> _T_1009; // @[Control.scala 259:49]
  assign _T_1011 = $signed(_T_1010); // @[Control.scala 259:84]
  assign shr_val_14 = _T_674 ? $signed(_T_1011) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_14 = _T_674 ? $signed(shr_val_14) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1012 = shr_res_14[7:0]; // @[Control.scala 261:37]
  assign src_0_15 = _T_674 ? $signed(_GEN_47) : $signed(32'sh0); // @[Control.scala 219:40]
  assign src_1_15 = _T_674 ? $signed(_GEN_64) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1013 = $signed(src_0_15) < $signed(src_1_15); // @[Control.scala 253:34]
  assign _T_1014 = _T_1013 ? $signed(src_0_15) : $signed(src_1_15); // @[Control.scala 253:24]
  assign mix_val_15 = _T_674 ? $signed(_T_1014) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1015 = mix_val_15[7:0]; // @[Control.scala 255:37]
  assign _T_1016 = $unsigned(src_0_15); // @[Control.scala 256:30]
  assign _T_1017 = $unsigned(src_1_15); // @[Control.scala 256:59]
  assign _T_1018 = _T_1016 + _T_1017; // @[Control.scala 256:49]
  assign _T_1019 = _T_1016 + _T_1017; // @[Control.scala 256:49]
  assign _T_1020 = $signed(_T_1019); // @[Control.scala 256:79]
  assign add_val_15 = _T_674 ? $signed(_T_1020) : $signed(32'sh0); // @[Control.scala 219:40]
  assign add_res_15 = _T_674 ? $signed(add_val_15) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1021 = add_res_15[7:0]; // @[Control.scala 258:37]
  assign _T_1023 = src_1_15[4:0]; // @[Control.scala 259:60]
  assign _T_1024 = _T_1016 >> _T_1023; // @[Control.scala 259:49]
  assign _T_1025 = $signed(_T_1024); // @[Control.scala 259:84]
  assign shr_val_15 = _T_674 ? $signed(_T_1025) : $signed(32'sh0); // @[Control.scala 219:40]
  assign shr_res_15 = _T_674 ? $signed(shr_val_15) : $signed(32'sh0); // @[Control.scala 219:40]
  assign _T_1026 = shr_res_15[7:0]; // @[Control.scala 261:37]
  assign short_cmp_res_0 = _T_674 ? _T_805 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_0 = _T_674 ? _T_811 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_0 = _T_674 ? _T_816 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_1 = _T_674 ? _T_819 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_1 = _T_674 ? _T_825 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_1 = _T_674 ? _T_830 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_2 = _T_674 ? _T_833 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_2 = _T_674 ? _T_839 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_2 = _T_674 ? _T_844 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_3 = _T_674 ? _T_847 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_3 = _T_674 ? _T_853 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_3 = _T_674 ? _T_858 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_4 = _T_674 ? _T_861 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_4 = _T_674 ? _T_867 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_4 = _T_674 ? _T_872 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_5 = _T_674 ? _T_875 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_5 = _T_674 ? _T_881 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_5 = _T_674 ? _T_886 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_6 = _T_674 ? _T_889 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_6 = _T_674 ? _T_895 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_6 = _T_674 ? _T_900 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_7 = _T_674 ? _T_903 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_7 = _T_674 ? _T_909 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_7 = _T_674 ? _T_914 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_8 = _T_674 ? _T_917 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_8 = _T_674 ? _T_923 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_8 = _T_674 ? _T_928 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_9 = _T_674 ? _T_931 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_9 = _T_674 ? _T_937 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_9 = _T_674 ? _T_942 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_10 = _T_674 ? _T_945 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_10 = _T_674 ? _T_951 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_10 = _T_674 ? _T_956 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_11 = _T_674 ? _T_959 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_11 = _T_674 ? _T_965 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_11 = _T_674 ? _T_970 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_12 = _T_674 ? _T_973 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_12 = _T_674 ? _T_979 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_12 = _T_674 ? _T_984 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_13 = _T_674 ? _T_987 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_13 = _T_674 ? _T_993 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_13 = _T_674 ? _T_998 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_14 = _T_674 ? _T_1001 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_14 = _T_674 ? _T_1007 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_14 = _T_674 ? _T_1012 : 8'h0; // @[Control.scala 219:40]
  assign short_cmp_res_15 = _T_674 ? _T_1015 : 8'h0; // @[Control.scala 219:40]
  assign short_add_res_15 = _T_674 ? _T_1021 : 8'h0; // @[Control.scala 219:40]
  assign short_shr_res_15 = _T_674 ? _T_1026 : 8'h0; // @[Control.scala 219:40]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Control.scala 275:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Control.scala 276:39]
  assign _T_1035 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1043 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1035}; // @[Cat.scala 30:58]
  assign _T_1050 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1058 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1050}; // @[Cat.scala 30:58]
  assign _T_1065 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1073 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1065}; // @[Cat.scala 30:58]
  assign _T_1074 = alu_opcode_add_en ? _T_1058 : _T_1073; // @[Control.scala 278:30]
  assign io_done_readdata = opcode == 3'h3; // @[Control.scala 136:20]
  assign io_uops_ready = _T_193 & insn_valid; // @[Control.scala 121:19 Control.scala 124:19]
  assign io_biases_ready = opcode_load_en & memory_type_acc_en; // @[Control.scala 150:19]
  assign io_gemm_queue_ready = io_gemm_queue_valid & _T_189; // @[Control.scala 112:25 Control.scala 115:25]
  assign io_g2l_dep_queue_valid = g2l_queue_io_deq_valid; // @[Control.scala 308:26]
  assign io_g2l_dep_queue_data = g2l_queue_io_deq_bits; // @[Control.scala 310:26]
  assign io_g2s_dep_queue_valid = g2s_queue_io_deq_valid; // @[Control.scala 311:26]
  assign io_g2s_dep_queue_data = g2s_queue_io_deq_bits; // @[Control.scala 313:26]
  assign io_out_mem_address = {{6'd0}, out_mem_addr}; // @[Control.scala 272:22]
  assign io_out_mem_write = out_mem_write_en; // @[Control.scala 274:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1043 : _T_1074; // @[Control.scala 277:24]
  assign io_uop_mem_write = opcode_load_en & memory_type_uop_en; // @[Control.scala 141:20]
  assign io_uop_mem_writedata = uops_data; // @[Control.scala 142:24]
  assign g2l_queue_clock = clock;
  assign g2l_queue_reset = reset;
  assign g2l_queue_io_enq_valid = push_prev_dep & in_loop_cntr_wrap; // @[Control.scala 317:26]
  assign g2l_queue_io_deq_ready = io_g2l_dep_queue_ready; // @[Control.scala 309:26]
  assign g2s_queue_clock = clock;
  assign g2s_queue_reset = reset;
  assign g2s_queue_io_enq_valid = push_next_dep & in_loop_cntr_wrap; // @[Control.scala 318:26]
  assign g2s_queue_io_deq_ready = io_g2s_dep_queue_ready; // @[Control.scala 312:26]
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
    if(acc_mem__T_221_en & acc_mem__T_221_mask) begin
      acc_mem[acc_mem__T_221_addr] <= acc_mem__T_221_data; // @[Control.scala 26:20]
    end
    if (_T_190) begin
      insn <= io_gemm_queue_data;
    end
    if (_T_194) begin
      uops_data <= io_uops_data;
    end
    if (reset) begin
      acc_x_cntr_val <= 8'h0;
    end else begin
      if (_T_150) begin
        if (_T_152) begin
          acc_x_cntr_val <= 8'h0;
        end else begin
          acc_x_cntr_val <= _T_157;
        end
      end
    end
    if (reset) begin
      dst_offset_in <= 8'h0;
    end else begin
      if (_T_165) begin
        if (_T_167) begin
          dst_offset_in <= 8'h0;
        end else begin
          dst_offset_in <= _T_172;
        end
      end
    end
    dst_vector <= acc_mem__T_236_data;
    src_vector <= acc_mem__T_239_data;
    out_mem_addr <= _T_231 + _GEN_257;
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
  input          io_g2l_dep_queue_ready,
  output         io_g2l_dep_queue_valid,
  output         io_g2l_dep_queue_data,
  input          io_g2s_dep_queue_ready,
  output         io_g2s_dep_queue_valid,
  output         io_g2s_dep_queue_data,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata
);
  wire  ctrl_clock; // @[Core.scala 161:21]
  wire  ctrl_reset; // @[Core.scala 161:21]
  wire  ctrl_io_done_readdata; // @[Core.scala 161:21]
  wire  ctrl_io_uops_ready; // @[Core.scala 161:21]
  wire  ctrl_io_uops_valid; // @[Core.scala 161:21]
  wire [31:0] ctrl_io_uops_data; // @[Core.scala 161:21]
  wire  ctrl_io_biases_ready; // @[Core.scala 161:21]
  wire  ctrl_io_biases_valid; // @[Core.scala 161:21]
  wire [511:0] ctrl_io_biases_data; // @[Core.scala 161:21]
  wire  ctrl_io_gemm_queue_ready; // @[Core.scala 161:21]
  wire  ctrl_io_gemm_queue_valid; // @[Core.scala 161:21]
  wire [127:0] ctrl_io_gemm_queue_data; // @[Core.scala 161:21]
  wire  ctrl_io_g2l_dep_queue_ready; // @[Core.scala 161:21]
  wire  ctrl_io_g2l_dep_queue_valid; // @[Core.scala 161:21]
  wire  ctrl_io_g2l_dep_queue_data; // @[Core.scala 161:21]
  wire  ctrl_io_g2s_dep_queue_ready; // @[Core.scala 161:21]
  wire  ctrl_io_g2s_dep_queue_valid; // @[Core.scala 161:21]
  wire  ctrl_io_g2s_dep_queue_data; // @[Core.scala 161:21]
  wire  ctrl_io_out_mem_waitrequest; // @[Core.scala 161:21]
  wire [16:0] ctrl_io_out_mem_address; // @[Core.scala 161:21]
  wire  ctrl_io_out_mem_write; // @[Core.scala 161:21]
  wire [127:0] ctrl_io_out_mem_writedata; // @[Core.scala 161:21]
  wire [31:0] ctrl_io_uop_mem_readdata; // @[Core.scala 161:21]
  wire  ctrl_io_uop_mem_write; // @[Core.scala 161:21]
  wire [31:0] ctrl_io_uop_mem_writedata; // @[Core.scala 161:21]
  wire  uop_mem_clock; // @[Core.scala 162:23]
  wire [31:0] uop_mem_io_readdata; // @[Core.scala 162:23]
  wire  uop_mem_io_write; // @[Core.scala 162:23]
  wire [31:0] uop_mem_io_writedata; // @[Core.scala 162:23]
  Control ctrl ( // @[Core.scala 161:21]
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
    .io_g2l_dep_queue_ready(ctrl_io_g2l_dep_queue_ready),
    .io_g2l_dep_queue_valid(ctrl_io_g2l_dep_queue_valid),
    .io_g2l_dep_queue_data(ctrl_io_g2l_dep_queue_data),
    .io_g2s_dep_queue_ready(ctrl_io_g2s_dep_queue_ready),
    .io_g2s_dep_queue_valid(ctrl_io_g2s_dep_queue_valid),
    .io_g2s_dep_queue_data(ctrl_io_g2s_dep_queue_data),
    .io_out_mem_waitrequest(ctrl_io_out_mem_waitrequest),
    .io_out_mem_address(ctrl_io_out_mem_address),
    .io_out_mem_write(ctrl_io_out_mem_write),
    .io_out_mem_writedata(ctrl_io_out_mem_writedata),
    .io_uop_mem_readdata(ctrl_io_uop_mem_readdata),
    .io_uop_mem_write(ctrl_io_uop_mem_write),
    .io_uop_mem_writedata(ctrl_io_uop_mem_writedata)
  );
  MemBlock uop_mem ( // @[Core.scala 162:23]
    .clock(uop_mem_clock),
    .io_readdata(uop_mem_io_readdata),
    .io_write(uop_mem_io_write),
    .io_writedata(uop_mem_io_writedata)
  );
  assign io_done_readdata = ctrl_io_done_readdata; // @[Core.scala 166:16]
  assign io_uops_ready = ctrl_io_uops_ready; // @[Core.scala 167:16]
  assign io_biases_ready = ctrl_io_biases_ready; // @[Core.scala 168:18]
  assign io_gemm_queue_ready = ctrl_io_gemm_queue_ready; // @[Core.scala 169:22]
  assign io_g2l_dep_queue_valid = ctrl_io_g2l_dep_queue_valid; // @[Core.scala 172:25]
  assign io_g2l_dep_queue_data = ctrl_io_g2l_dep_queue_data; // @[Core.scala 172:25]
  assign io_g2s_dep_queue_valid = ctrl_io_g2s_dep_queue_valid; // @[Core.scala 173:25]
  assign io_g2s_dep_queue_data = ctrl_io_g2s_dep_queue_data; // @[Core.scala 173:25]
  assign io_out_mem_address = ctrl_io_out_mem_address; // @[Core.scala 174:19]
  assign io_out_mem_write = ctrl_io_out_mem_write; // @[Core.scala 174:19]
  assign io_out_mem_writedata = ctrl_io_out_mem_writedata; // @[Core.scala 174:19]
  assign ctrl_clock = clock;
  assign ctrl_reset = reset;
  assign ctrl_io_uops_valid = io_uops_valid; // @[Core.scala 167:16]
  assign ctrl_io_uops_data = io_uops_data; // @[Core.scala 167:16]
  assign ctrl_io_biases_valid = io_biases_valid; // @[Core.scala 168:18]
  assign ctrl_io_biases_data = io_biases_data; // @[Core.scala 168:18]
  assign ctrl_io_gemm_queue_valid = io_gemm_queue_valid; // @[Core.scala 169:22]
  assign ctrl_io_gemm_queue_data = io_gemm_queue_data; // @[Core.scala 169:22]
  assign ctrl_io_g2l_dep_queue_ready = io_g2l_dep_queue_ready; // @[Core.scala 172:25]
  assign ctrl_io_g2s_dep_queue_ready = io_g2s_dep_queue_ready; // @[Core.scala 173:25]
  assign ctrl_io_out_mem_waitrequest = io_out_mem_waitrequest; // @[Core.scala 174:19]
  assign ctrl_io_uop_mem_readdata = uop_mem_io_readdata; // @[Core.scala 177:19]
  assign uop_mem_clock = clock;
  assign uop_mem_io_write = ctrl_io_uop_mem_write; // @[Core.scala 177:19]
  assign uop_mem_io_writedata = ctrl_io_uop_mem_writedata; // @[Core.scala 177:19]
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
  wire  core_io_g2l_dep_queue_ready; // @[Compute.scala 40:20]
  wire  core_io_g2l_dep_queue_valid; // @[Compute.scala 40:20]
  wire  core_io_g2l_dep_queue_data; // @[Compute.scala 40:20]
  wire  core_io_g2s_dep_queue_ready; // @[Compute.scala 40:20]
  wire  core_io_g2s_dep_queue_valid; // @[Compute.scala 40:20]
  wire  core_io_g2s_dep_queue_data; // @[Compute.scala 40:20]
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
    .io_g2l_dep_queue_ready(core_io_g2l_dep_queue_ready),
    .io_g2l_dep_queue_valid(core_io_g2l_dep_queue_valid),
    .io_g2l_dep_queue_data(core_io_g2l_dep_queue_data),
    .io_g2s_dep_queue_ready(core_io_g2s_dep_queue_ready),
    .io_g2s_dep_queue_valid(core_io_g2s_dep_queue_valid),
    .io_g2s_dep_queue_data(core_io_g2s_dep_queue_data),
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
  assign io_l2g_dep_queue_ready = 1'h0; // @[Compute.scala 46:20]
  assign io_s2g_dep_queue_ready = 1'h0; // @[Compute.scala 47:20]
  assign io_g2l_dep_queue_valid = core_io_g2l_dep_queue_valid; // @[Compute.scala 48:20]
  assign io_g2l_dep_queue_data = core_io_g2l_dep_queue_data; // @[Compute.scala 48:20]
  assign io_g2s_dep_queue_valid = core_io_g2s_dep_queue_valid; // @[Compute.scala 49:20]
  assign io_g2s_dep_queue_data = core_io_g2s_dep_queue_data; // @[Compute.scala 49:20]
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
  assign core_io_g2l_dep_queue_ready = io_g2l_dep_queue_ready; // @[Compute.scala 48:20]
  assign core_io_g2s_dep_queue_ready = io_g2s_dep_queue_ready; // @[Compute.scala 49:20]
  assign core_io_out_mem_waitrequest = io_out_mem_waitrequest; // @[Compute.scala 52:14]
endmodule
