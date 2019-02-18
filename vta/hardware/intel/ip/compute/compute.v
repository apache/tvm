module OutQueue(
  input          clock,
  input          reset,
  output         io_enq_ready,
  input          io_enq_valid,
  input  [159:0] io_enq_bits,
  input          io_deq_ready,
  output         io_deq_valid,
  output [159:0] io_deq_bits
);
  reg [159:0] _T_35 [0:31]; // @[Decoupled.scala 215:24]
  reg [159:0] _RAND_0;
  wire [159:0] _T_35__T_68_data; // @[Decoupled.scala 215:24]
  wire [4:0] _T_35__T_68_addr; // @[Decoupled.scala 215:24]
  wire [159:0] _T_35__T_54_data; // @[Decoupled.scala 215:24]
  wire [4:0] _T_35__T_54_addr; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_mask; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_en; // @[Decoupled.scala 215:24]
  reg [4:0] value; // @[Counter.scala 26:33]
  reg [31:0] _RAND_1;
  reg [4:0] value_1; // @[Counter.scala 26:33]
  reg [31:0] _RAND_2;
  reg  _T_42; // @[Decoupled.scala 218:35]
  reg [31:0] _RAND_3;
  wire  _T_43; // @[Decoupled.scala 220:41]
  wire  _T_45; // @[Decoupled.scala 221:36]
  wire  _T_46; // @[Decoupled.scala 221:33]
  wire  _T_47; // @[Decoupled.scala 222:32]
  wire  _T_48; // @[Decoupled.scala 37:37]
  wire  _T_51; // @[Decoupled.scala 37:37]
  wire [5:0] _T_57; // @[Counter.scala 35:22]
  wire [4:0] _T_58; // @[Counter.scala 35:22]
  wire [4:0] _GEN_5; // @[Decoupled.scala 226:17]
  wire [5:0] _T_61; // @[Counter.scala 35:22]
  wire [4:0] _T_62; // @[Counter.scala 35:22]
  wire [4:0] _GEN_6; // @[Decoupled.scala 230:17]
  wire  _T_63; // @[Decoupled.scala 233:16]
  wire  _GEN_7; // @[Decoupled.scala 233:28]
  assign _T_35__T_68_addr = value_1;
  assign _T_35__T_68_data = _T_35[_T_35__T_68_addr]; // @[Decoupled.scala 215:24]
  assign _T_35__T_54_data = io_enq_bits;
  assign _T_35__T_54_addr = value;
  assign _T_35__T_54_mask = 1'h1;
  assign _T_35__T_54_en = io_enq_ready & io_enq_valid;
  assign _T_43 = value == value_1; // @[Decoupled.scala 220:41]
  assign _T_45 = _T_42 == 1'h0; // @[Decoupled.scala 221:36]
  assign _T_46 = _T_43 & _T_45; // @[Decoupled.scala 221:33]
  assign _T_47 = _T_43 & _T_42; // @[Decoupled.scala 222:32]
  assign _T_48 = io_enq_ready & io_enq_valid; // @[Decoupled.scala 37:37]
  assign _T_51 = io_deq_ready & io_deq_valid; // @[Decoupled.scala 37:37]
  assign _T_57 = value + 5'h1; // @[Counter.scala 35:22]
  assign _T_58 = value + 5'h1; // @[Counter.scala 35:22]
  assign _GEN_5 = _T_48 ? _T_58 : value; // @[Decoupled.scala 226:17]
  assign _T_61 = value_1 + 5'h1; // @[Counter.scala 35:22]
  assign _T_62 = value_1 + 5'h1; // @[Counter.scala 35:22]
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
  _RAND_0 = {5{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 32; initvar = initvar+1)
    _T_35[initvar] = _RAND_0[159:0];
  `endif // RANDOMIZE_MEM_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  value = _RAND_1[4:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  value_1 = _RAND_2[4:0];
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
      value <= 5'h0;
    end else begin
      if (_T_48) begin
        value <= _T_58;
      end
    end
    if (reset) begin
      value_1 <= 5'h0;
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
  reg [511:0] acc_mem [0:255] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [511:0] _RAND_0;
  wire [511:0] acc_mem__T_468_data; // @[Compute.scala 34:20]
  wire [7:0] acc_mem__T_468_addr; // @[Compute.scala 34:20]
  wire [511:0] acc_mem__T_470_data; // @[Compute.scala 34:20]
  wire [7:0] acc_mem__T_470_addr; // @[Compute.scala 34:20]
  wire [511:0] acc_mem__T_445_data; // @[Compute.scala 34:20]
  wire [7:0] acc_mem__T_445_addr; // @[Compute.scala 34:20]
  wire  acc_mem__T_445_mask; // @[Compute.scala 34:20]
  wire  acc_mem__T_445_en; // @[Compute.scala 34:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 35:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem__T_450_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_450_addr; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_378_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_378_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_378_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_378_en; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_384_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_384_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_384_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_384_en; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_390_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_390_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_390_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_390_en; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_396_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_396_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_396_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_396_en; // @[Compute.scala 35:20]
  wire  out_mem_fifo_clock; // @[Compute.scala 346:28]
  wire  out_mem_fifo_reset; // @[Compute.scala 346:28]
  wire  out_mem_fifo_io_enq_ready; // @[Compute.scala 346:28]
  wire  out_mem_fifo_io_enq_valid; // @[Compute.scala 346:28]
  wire [159:0] out_mem_fifo_io_enq_bits; // @[Compute.scala 346:28]
  wire  out_mem_fifo_io_deq_ready; // @[Compute.scala 346:28]
  wire  out_mem_fifo_io_deq_valid; // @[Compute.scala 346:28]
  wire [159:0] out_mem_fifo_io_deq_bits; // @[Compute.scala 346:28]
  wire  started; // @[Compute.scala 32:17]
  reg [127:0] insn; // @[Compute.scala 37:28]
  reg [127:0] _RAND_2;
  wire  _T_201; // @[Compute.scala 38:31]
  wire  insn_valid; // @[Compute.scala 38:40]
  wire [2:0] opcode; // @[Compute.scala 40:29]
  wire  pop_prev_dep; // @[Compute.scala 41:29]
  wire  pop_next_dep; // @[Compute.scala 42:29]
  wire  push_prev_dep; // @[Compute.scala 43:29]
  wire  push_next_dep; // @[Compute.scala 44:29]
  wire [1:0] memory_type; // @[Compute.scala 46:25]
  wire [15:0] sram_base; // @[Compute.scala 47:25]
  wire [31:0] dram_base; // @[Compute.scala 48:25]
  wire [15:0] x_size; // @[Compute.scala 50:25]
  wire [3:0] y_pad_0; // @[Compute.scala 52:25]
  wire [3:0] x_pad_0; // @[Compute.scala 54:25]
  wire [3:0] x_pad_1; // @[Compute.scala 55:25]
  reg [15:0] uop_bgn; // @[Compute.scala 58:20]
  reg [31:0] _RAND_3;
  wire [12:0] _T_203; // @[Compute.scala 59:18]
  reg [15:0] uop_end; // @[Compute.scala 60:20]
  reg [31:0] _RAND_4;
  wire [13:0] _T_205; // @[Compute.scala 61:18]
  wire [13:0] iter_out; // @[Compute.scala 62:22]
  wire [13:0] iter_in; // @[Compute.scala 63:21]
  wire [1:0] alu_opcode; // @[Compute.scala 70:24]
  wire  use_imm; // @[Compute.scala 71:21]
  wire [15:0] imm_raw; // @[Compute.scala 72:21]
  wire [15:0] _T_206; // @[Compute.scala 73:25]
  wire  _T_208; // @[Compute.scala 73:32]
  wire [31:0] _T_210; // @[Cat.scala 30:58]
  wire [16:0] _T_212; // @[Cat.scala 30:58]
  wire [31:0] _T_213; // @[Compute.scala 73:16]
  wire [31:0] imm; // @[Compute.scala 73:89]
  wire [15:0] _GEN_316; // @[Compute.scala 77:30]
  wire [15:0] _GEN_318; // @[Compute.scala 78:30]
  wire [16:0] _T_217; // @[Compute.scala 78:30]
  wire [15:0] _T_218; // @[Compute.scala 78:30]
  wire [15:0] _GEN_319; // @[Compute.scala 78:39]
  wire [16:0] _T_219; // @[Compute.scala 78:39]
  wire [15:0] x_size_total; // @[Compute.scala 78:39]
  wire [19:0] y_offset; // @[Compute.scala 79:31]
  wire  opcode_finish_en; // @[Compute.scala 82:34]
  wire  _T_222; // @[Compute.scala 83:32]
  wire  _T_224; // @[Compute.scala 83:60]
  wire  opcode_load_en; // @[Compute.scala 83:50]
  wire  opcode_gemm_en; // @[Compute.scala 84:32]
  wire  opcode_alu_en; // @[Compute.scala 85:31]
  wire  memory_type_uop_en; // @[Compute.scala 87:40]
  wire  memory_type_acc_en; // @[Compute.scala 88:40]
  reg [2:0] state; // @[Compute.scala 91:22]
  reg [31:0] _RAND_5;
  wire  idle; // @[Compute.scala 93:20]
  wire  dump; // @[Compute.scala 94:20]
  wire  busy; // @[Compute.scala 95:20]
  wire  push; // @[Compute.scala 96:20]
  wire  done; // @[Compute.scala 97:20]
  reg  uops_read; // @[Compute.scala 100:24]
  reg [31:0] _RAND_6;
  reg  biases_read; // @[Compute.scala 103:24]
  reg [31:0] _RAND_7;
  reg [127:0] biases_data_0; // @[Compute.scala 106:24]
  reg [127:0] _RAND_8;
  reg [127:0] biases_data_1; // @[Compute.scala 106:24]
  reg [127:0] _RAND_9;
  reg [127:0] biases_data_2; // @[Compute.scala 106:24]
  reg [127:0] _RAND_10;
  reg [127:0] biases_data_3; // @[Compute.scala 106:24]
  reg [127:0] _RAND_11;
  reg  out_mem_write; // @[Compute.scala 108:31]
  reg [31:0] _RAND_12;
  wire [15:0] uop_cntr_max_val; // @[Compute.scala 111:33]
  wire  _T_248; // @[Compute.scala 112:43]
  wire [15:0] uop_cntr_max; // @[Compute.scala 112:25]
  wire  _T_250; // @[Compute.scala 113:37]
  wire  uop_cntr_en; // @[Compute.scala 113:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 115:25]
  reg [31:0] _RAND_13;
  wire  _T_252; // @[Compute.scala 116:38]
  wire  _T_253; // @[Compute.scala 116:56]
  wire  uop_cntr_wrap; // @[Compute.scala 116:71]
  wire [18:0] _T_255; // @[Compute.scala 118:29]
  wire [19:0] _T_257; // @[Compute.scala 118:46]
  wire [18:0] acc_cntr_max; // @[Compute.scala 118:46]
  wire  _T_258; // @[Compute.scala 119:37]
  wire  acc_cntr_en; // @[Compute.scala 119:59]
  reg [15:0] acc_cntr_val; // @[Compute.scala 121:25]
  reg [31:0] _RAND_14;
  wire [18:0] _GEN_321; // @[Compute.scala 122:38]
  wire  _T_260; // @[Compute.scala 122:38]
  wire  _T_261; // @[Compute.scala 122:56]
  wire  acc_cntr_wrap; // @[Compute.scala 122:71]
  wire [16:0] _T_262; // @[Compute.scala 124:34]
  wire [16:0] _T_263; // @[Compute.scala 124:34]
  wire [15:0] upc_cntr_max_val; // @[Compute.scala 124:34]
  wire  _T_265; // @[Compute.scala 125:43]
  wire [15:0] upc_cntr_max; // @[Compute.scala 125:25]
  wire [27:0] _T_267; // @[Compute.scala 126:34]
  wire [27:0] _GEN_322; // @[Compute.scala 126:45]
  wire [43:0] out_cntr_max_val; // @[Compute.scala 126:45]
  wire [44:0] _T_269; // @[Compute.scala 127:39]
  wire [43:0] out_cntr_max; // @[Compute.scala 127:39]
  wire  _T_270; // @[Compute.scala 128:37]
  wire  out_cntr_en; // @[Compute.scala 128:56]
  reg [15:0] out_cntr_val; // @[Compute.scala 130:25]
  reg [31:0] _RAND_15;
  wire [43:0] _GEN_323; // @[Compute.scala 131:38]
  wire  _T_272; // @[Compute.scala 131:38]
  wire  _T_273; // @[Compute.scala 131:56]
  wire  out_cntr_wrap; // @[Compute.scala 131:71]
  reg  pop_prev_dep_ready; // @[Compute.scala 134:35]
  reg [31:0] _RAND_16;
  reg  pop_next_dep_ready; // @[Compute.scala 135:35]
  reg [31:0] _RAND_17;
  wire  push_prev_dep_valid; // @[Compute.scala 136:43]
  wire  push_next_dep_valid; // @[Compute.scala 137:43]
  reg  push_prev_dep_ready; // @[Compute.scala 138:36]
  reg [31:0] _RAND_18;
  reg  push_next_dep_ready; // @[Compute.scala 139:36]
  reg [31:0] _RAND_19;
  reg  gemm_queue_ready; // @[Compute.scala 141:33]
  reg [31:0] _RAND_20;
  reg  finish_wrap; // @[Compute.scala 144:28]
  reg [31:0] _RAND_21;
  wire  _T_286; // @[Compute.scala 146:68]
  wire  _T_287; // @[Compute.scala 147:68]
  wire  _T_288; // @[Compute.scala 148:68]
  wire  _T_289; // @[Compute.scala 149:68]
  wire  _GEN_0; // @[Compute.scala 149:31]
  wire  _GEN_1; // @[Compute.scala 148:31]
  wire  _GEN_2; // @[Compute.scala 147:31]
  wire  _GEN_3; // @[Compute.scala 146:31]
  wire  _GEN_4; // @[Compute.scala 145:27]
  wire  _T_292; // @[Compute.scala 152:23]
  wire  _T_293; // @[Compute.scala 152:40]
  wire  _T_294; // @[Compute.scala 152:57]
  wire  _T_295; // @[Compute.scala 153:25]
  wire [2:0] _GEN_5; // @[Compute.scala 153:43]
  wire [2:0] _GEN_6; // @[Compute.scala 152:73]
  wire  _T_297; // @[Compute.scala 161:18]
  wire  _T_299; // @[Compute.scala 161:41]
  wire  _T_300; // @[Compute.scala 161:38]
  wire  _T_301; // @[Compute.scala 161:14]
  wire  _T_302; // @[Compute.scala 161:79]
  wire  _T_303; // @[Compute.scala 161:62]
  wire [2:0] _GEN_7; // @[Compute.scala 161:97]
  wire  _T_304; // @[Compute.scala 162:38]
  wire  _T_305; // @[Compute.scala 162:14]
  wire [2:0] _GEN_8; // @[Compute.scala 162:63]
  wire  _T_306; // @[Compute.scala 163:38]
  wire  _T_307; // @[Compute.scala 163:14]
  wire [2:0] _GEN_9; // @[Compute.scala 163:63]
  wire  _T_310; // @[Compute.scala 170:22]
  wire  _T_311; // @[Compute.scala 170:30]
  wire  _GEN_10; // @[Compute.scala 170:57]
  wire  _T_313; // @[Compute.scala 173:22]
  wire  _T_314; // @[Compute.scala 173:30]
  wire  _GEN_11; // @[Compute.scala 173:57]
  wire  _T_318; // @[Compute.scala 180:29]
  wire  _T_319; // @[Compute.scala 180:55]
  wire  _GEN_12; // @[Compute.scala 180:64]
  wire  _T_321; // @[Compute.scala 183:29]
  wire  _T_322; // @[Compute.scala 183:55]
  wire  _GEN_13; // @[Compute.scala 183:64]
  wire  _T_325; // @[Compute.scala 188:22]
  wire  _T_326; // @[Compute.scala 188:19]
  wire  _T_327; // @[Compute.scala 188:37]
  wire  _T_328; // @[Compute.scala 188:61]
  wire  _T_329; // @[Compute.scala 188:45]
  wire [16:0] _T_331; // @[Compute.scala 189:34]
  wire [15:0] _T_332; // @[Compute.scala 189:34]
  wire [15:0] _GEN_14; // @[Compute.scala 188:77]
  wire  _T_334; // @[Compute.scala 191:24]
  wire  _T_335; // @[Compute.scala 191:21]
  wire  _T_336; // @[Compute.scala 191:39]
  wire  _T_337; // @[Compute.scala 191:63]
  wire  _T_338; // @[Compute.scala 191:47]
  wire [16:0] _T_340; // @[Compute.scala 192:34]
  wire [15:0] _T_341; // @[Compute.scala 192:34]
  wire [15:0] _GEN_15; // @[Compute.scala 191:79]
  wire  _T_342; // @[Compute.scala 197:23]
  wire  _T_343; // @[Compute.scala 197:47]
  wire  _T_344; // @[Compute.scala 197:31]
  wire [16:0] _T_346; // @[Compute.scala 198:34]
  wire [15:0] _T_347; // @[Compute.scala 198:34]
  wire [15:0] _GEN_16; // @[Compute.scala 197:63]
  wire  _GEN_21; // @[Compute.scala 202:27]
  wire  _GEN_22; // @[Compute.scala 202:27]
  wire  _GEN_23; // @[Compute.scala 202:27]
  wire  _GEN_24; // @[Compute.scala 202:27]
  wire [2:0] _GEN_25; // @[Compute.scala 202:27]
  wire  _T_355; // @[Compute.scala 215:52]
  wire  _T_356; // @[Compute.scala 215:43]
  wire  _GEN_26; // @[Compute.scala 217:27]
  wire [31:0] _GEN_326; // @[Compute.scala 227:33]
  wire [32:0] _T_361; // @[Compute.scala 227:33]
  wire [31:0] _T_362; // @[Compute.scala 227:33]
  wire [38:0] _GEN_327; // @[Compute.scala 227:49]
  wire [38:0] uop_dram_addr; // @[Compute.scala 227:49]
  wire [16:0] _T_364; // @[Compute.scala 228:33]
  wire [15:0] _T_365; // @[Compute.scala 228:33]
  wire [18:0] _GEN_328; // @[Compute.scala 228:49]
  wire [18:0] uop_sram_addr; // @[Compute.scala 228:49]
  wire  _T_368; // @[Compute.scala 229:31]
  wire  _T_369; // @[Compute.scala 229:28]
  wire  _T_370; // @[Compute.scala 229:46]
  wire [19:0] _T_375; // @[Compute.scala 237:29]
  wire [18:0] _T_376; // @[Compute.scala 237:29]
  wire [19:0] _T_381; // @[Compute.scala 237:29]
  wire [18:0] _T_382; // @[Compute.scala 237:29]
  wire [19:0] _T_387; // @[Compute.scala 237:29]
  wire [18:0] _T_388; // @[Compute.scala 237:29]
  wire [19:0] _T_393; // @[Compute.scala 237:29]
  wire [18:0] _T_394; // @[Compute.scala 237:29]
  wire [16:0] _T_399; // @[Compute.scala 239:42]
  wire [16:0] _T_400; // @[Compute.scala 239:42]
  wire [15:0] _T_401; // @[Compute.scala 239:42]
  wire  _T_402; // @[Compute.scala 239:24]
  wire  _GEN_27; // @[Compute.scala 239:50]
  wire [31:0] _GEN_329; // @[Compute.scala 243:36]
  wire [32:0] _T_404; // @[Compute.scala 243:36]
  wire [31:0] _T_405; // @[Compute.scala 243:36]
  wire [31:0] _GEN_330; // @[Compute.scala 243:47]
  wire [32:0] _T_406; // @[Compute.scala 243:47]
  wire [31:0] _T_407; // @[Compute.scala 243:47]
  wire [34:0] _GEN_331; // @[Compute.scala 243:58]
  wire [34:0] _T_409; // @[Compute.scala 243:58]
  wire [35:0] _T_411; // @[Compute.scala 243:66]
  wire [35:0] _GEN_332; // @[Compute.scala 243:76]
  wire [36:0] _T_412; // @[Compute.scala 243:76]
  wire [35:0] _T_413; // @[Compute.scala 243:76]
  wire [42:0] _GEN_333; // @[Compute.scala 243:92]
  wire [42:0] acc_dram_addr; // @[Compute.scala 243:92]
  wire [19:0] _GEN_334; // @[Compute.scala 244:36]
  wire [20:0] _T_415; // @[Compute.scala 244:36]
  wire [19:0] _T_416; // @[Compute.scala 244:36]
  wire [19:0] _GEN_335; // @[Compute.scala 244:47]
  wire [20:0] _T_417; // @[Compute.scala 244:47]
  wire [19:0] _T_418; // @[Compute.scala 244:47]
  wire [22:0] _GEN_336; // @[Compute.scala 244:58]
  wire [22:0] _T_420; // @[Compute.scala 244:58]
  wire [23:0] _T_422; // @[Compute.scala 244:66]
  wire [23:0] _GEN_337; // @[Compute.scala 244:76]
  wire [24:0] _T_423; // @[Compute.scala 244:76]
  wire [23:0] _T_424; // @[Compute.scala 244:76]
  wire [23:0] _T_426; // @[Compute.scala 244:92]
  wire [24:0] _T_428; // @[Compute.scala 244:121]
  wire [24:0] _T_429; // @[Compute.scala 244:121]
  wire [23:0] acc_sram_addr; // @[Compute.scala 244:121]
  wire  _T_431; // @[Compute.scala 245:33]
  wire [15:0] _GEN_17; // @[Compute.scala 251:30]
  wire [2:0] _T_437; // @[Compute.scala 251:30]
  wire [127:0] _GEN_40; // @[Compute.scala 251:48]
  wire [127:0] _GEN_41; // @[Compute.scala 251:48]
  wire [127:0] _GEN_42; // @[Compute.scala 251:48]
  wire [127:0] _GEN_43; // @[Compute.scala 251:48]
  wire  _T_443; // @[Compute.scala 255:43]
  wire [255:0] _T_446; // @[Cat.scala 30:58]
  wire [255:0] _T_447; // @[Cat.scala 30:58]
  wire [15:0] _GEN_18; // @[Compute.scala 261:26]
  wire [15:0] upc; // @[Compute.scala 261:26]
  reg [31:0] uop; // @[Compute.scala 262:20]
  reg [31:0] _RAND_22;
  reg [15:0] _T_453; // @[Compute.scala 265:22]
  reg [31:0] _RAND_23;
  wire [27:0] _GEN_338; // @[Compute.scala 265:37]
  wire [27:0] _GEN_19; // @[Compute.scala 265:37]
  wire [15:0] it_in; // @[Compute.scala 265:37]
  wire [31:0] _T_455; // @[Compute.scala 266:47]
  wire [15:0] _T_456; // @[Compute.scala 266:63]
  wire [16:0] _T_457; // @[Compute.scala 266:38]
  wire [15:0] dst_offset_in; // @[Compute.scala 266:38]
  wire [10:0] _T_461; // @[Compute.scala 268:20]
  wire [15:0] _GEN_339; // @[Compute.scala 268:47]
  wire [16:0] _T_462; // @[Compute.scala 268:47]
  wire [15:0] dst_idx; // @[Compute.scala 268:47]
  wire [10:0] _T_463; // @[Compute.scala 269:20]
  wire [15:0] _GEN_340; // @[Compute.scala 269:47]
  wire [16:0] _T_464; // @[Compute.scala 269:47]
  wire [15:0] src_idx; // @[Compute.scala 269:47]
  reg [511:0] dst_vector; // @[Compute.scala 272:23]
  reg [511:0] _RAND_24;
  reg [511:0] src_vector; // @[Compute.scala 273:23]
  reg [511:0] _RAND_25;
  wire  alu_opcode_min_en; // @[Compute.scala 292:38]
  wire  alu_opcode_max_en; // @[Compute.scala 293:38]
  wire  _T_902; // @[Compute.scala 312:20]
  wire [31:0] _T_903; // @[Compute.scala 315:31]
  wire [31:0] _T_904; // @[Compute.scala 315:72]
  wire [31:0] _T_905; // @[Compute.scala 316:31]
  wire [31:0] _T_906; // @[Compute.scala 316:72]
  wire [31:0] _T_907; // @[Compute.scala 315:31]
  wire [31:0] _T_908; // @[Compute.scala 315:72]
  wire [31:0] _T_909; // @[Compute.scala 316:31]
  wire [31:0] _T_910; // @[Compute.scala 316:72]
  wire [31:0] _T_911; // @[Compute.scala 315:31]
  wire [31:0] _T_912; // @[Compute.scala 315:72]
  wire [31:0] _T_913; // @[Compute.scala 316:31]
  wire [31:0] _T_914; // @[Compute.scala 316:72]
  wire [31:0] _T_915; // @[Compute.scala 315:31]
  wire [31:0] _T_916; // @[Compute.scala 315:72]
  wire [31:0] _T_917; // @[Compute.scala 316:31]
  wire [31:0] _T_918; // @[Compute.scala 316:72]
  wire [31:0] _T_919; // @[Compute.scala 315:31]
  wire [31:0] _T_920; // @[Compute.scala 315:72]
  wire [31:0] _T_921; // @[Compute.scala 316:31]
  wire [31:0] _T_922; // @[Compute.scala 316:72]
  wire [31:0] _T_923; // @[Compute.scala 315:31]
  wire [31:0] _T_924; // @[Compute.scala 315:72]
  wire [31:0] _T_925; // @[Compute.scala 316:31]
  wire [31:0] _T_926; // @[Compute.scala 316:72]
  wire [31:0] _T_927; // @[Compute.scala 315:31]
  wire [31:0] _T_928; // @[Compute.scala 315:72]
  wire [31:0] _T_929; // @[Compute.scala 316:31]
  wire [31:0] _T_930; // @[Compute.scala 316:72]
  wire [31:0] _T_931; // @[Compute.scala 315:31]
  wire [31:0] _T_932; // @[Compute.scala 315:72]
  wire [31:0] _T_933; // @[Compute.scala 316:31]
  wire [31:0] _T_934; // @[Compute.scala 316:72]
  wire [31:0] _T_935; // @[Compute.scala 315:31]
  wire [31:0] _T_936; // @[Compute.scala 315:72]
  wire [31:0] _T_937; // @[Compute.scala 316:31]
  wire [31:0] _T_938; // @[Compute.scala 316:72]
  wire [31:0] _T_939; // @[Compute.scala 315:31]
  wire [31:0] _T_940; // @[Compute.scala 315:72]
  wire [31:0] _T_941; // @[Compute.scala 316:31]
  wire [31:0] _T_942; // @[Compute.scala 316:72]
  wire [31:0] _T_943; // @[Compute.scala 315:31]
  wire [31:0] _T_944; // @[Compute.scala 315:72]
  wire [31:0] _T_945; // @[Compute.scala 316:31]
  wire [31:0] _T_946; // @[Compute.scala 316:72]
  wire [31:0] _T_947; // @[Compute.scala 315:31]
  wire [31:0] _T_948; // @[Compute.scala 315:72]
  wire [31:0] _T_949; // @[Compute.scala 316:31]
  wire [31:0] _T_950; // @[Compute.scala 316:72]
  wire [31:0] _T_951; // @[Compute.scala 315:31]
  wire [31:0] _T_952; // @[Compute.scala 315:72]
  wire [31:0] _T_953; // @[Compute.scala 316:31]
  wire [31:0] _T_954; // @[Compute.scala 316:72]
  wire [31:0] _T_955; // @[Compute.scala 315:31]
  wire [31:0] _T_956; // @[Compute.scala 315:72]
  wire [31:0] _T_957; // @[Compute.scala 316:31]
  wire [31:0] _T_958; // @[Compute.scala 316:72]
  wire [31:0] _T_959; // @[Compute.scala 315:31]
  wire [31:0] _T_960; // @[Compute.scala 315:72]
  wire [31:0] _T_961; // @[Compute.scala 316:31]
  wire [31:0] _T_962; // @[Compute.scala 316:72]
  wire [31:0] _T_963; // @[Compute.scala 315:31]
  wire [31:0] _T_964; // @[Compute.scala 315:72]
  wire [31:0] _T_965; // @[Compute.scala 316:31]
  wire [31:0] _T_966; // @[Compute.scala 316:72]
  wire [31:0] _GEN_66; // @[Compute.scala 313:30]
  wire [31:0] _GEN_67; // @[Compute.scala 313:30]
  wire [31:0] _GEN_68; // @[Compute.scala 313:30]
  wire [31:0] _GEN_69; // @[Compute.scala 313:30]
  wire [31:0] _GEN_70; // @[Compute.scala 313:30]
  wire [31:0] _GEN_71; // @[Compute.scala 313:30]
  wire [31:0] _GEN_72; // @[Compute.scala 313:30]
  wire [31:0] _GEN_73; // @[Compute.scala 313:30]
  wire [31:0] _GEN_74; // @[Compute.scala 313:30]
  wire [31:0] _GEN_75; // @[Compute.scala 313:30]
  wire [31:0] _GEN_76; // @[Compute.scala 313:30]
  wire [31:0] _GEN_77; // @[Compute.scala 313:30]
  wire [31:0] _GEN_78; // @[Compute.scala 313:30]
  wire [31:0] _GEN_79; // @[Compute.scala 313:30]
  wire [31:0] _GEN_80; // @[Compute.scala 313:30]
  wire [31:0] _GEN_81; // @[Compute.scala 313:30]
  wire [31:0] _GEN_82; // @[Compute.scala 313:30]
  wire [31:0] _GEN_83; // @[Compute.scala 313:30]
  wire [31:0] _GEN_84; // @[Compute.scala 313:30]
  wire [31:0] _GEN_85; // @[Compute.scala 313:30]
  wire [31:0] _GEN_86; // @[Compute.scala 313:30]
  wire [31:0] _GEN_87; // @[Compute.scala 313:30]
  wire [31:0] _GEN_88; // @[Compute.scala 313:30]
  wire [31:0] _GEN_89; // @[Compute.scala 313:30]
  wire [31:0] _GEN_90; // @[Compute.scala 313:30]
  wire [31:0] _GEN_91; // @[Compute.scala 313:30]
  wire [31:0] _GEN_92; // @[Compute.scala 313:30]
  wire [31:0] _GEN_93; // @[Compute.scala 313:30]
  wire [31:0] _GEN_94; // @[Compute.scala 313:30]
  wire [31:0] _GEN_95; // @[Compute.scala 313:30]
  wire [31:0] _GEN_96; // @[Compute.scala 313:30]
  wire [31:0] _GEN_97; // @[Compute.scala 313:30]
  wire [31:0] _GEN_98; // @[Compute.scala 324:20]
  wire [31:0] _GEN_99; // @[Compute.scala 324:20]
  wire [31:0] _GEN_100; // @[Compute.scala 324:20]
  wire [31:0] _GEN_101; // @[Compute.scala 324:20]
  wire [31:0] _GEN_102; // @[Compute.scala 324:20]
  wire [31:0] _GEN_103; // @[Compute.scala 324:20]
  wire [31:0] _GEN_104; // @[Compute.scala 324:20]
  wire [31:0] _GEN_105; // @[Compute.scala 324:20]
  wire [31:0] _GEN_106; // @[Compute.scala 324:20]
  wire [31:0] _GEN_107; // @[Compute.scala 324:20]
  wire [31:0] _GEN_108; // @[Compute.scala 324:20]
  wire [31:0] _GEN_109; // @[Compute.scala 324:20]
  wire [31:0] _GEN_110; // @[Compute.scala 324:20]
  wire [31:0] _GEN_111; // @[Compute.scala 324:20]
  wire [31:0] _GEN_112; // @[Compute.scala 324:20]
  wire [31:0] _GEN_113; // @[Compute.scala 324:20]
  wire [31:0] src_0_0; // @[Compute.scala 312:36]
  wire [31:0] src_1_0; // @[Compute.scala 312:36]
  wire  _T_1031; // @[Compute.scala 329:34]
  wire [31:0] _T_1032; // @[Compute.scala 329:24]
  wire [31:0] mix_val_0; // @[Compute.scala 312:36]
  wire [7:0] _T_1033; // @[Compute.scala 331:37]
  wire [31:0] _T_1034; // @[Compute.scala 332:30]
  wire [31:0] _T_1035; // @[Compute.scala 332:59]
  wire [32:0] _T_1036; // @[Compute.scala 332:49]
  wire [31:0] _T_1037; // @[Compute.scala 332:49]
  wire [31:0] _T_1038; // @[Compute.scala 332:79]
  wire [31:0] add_val_0; // @[Compute.scala 312:36]
  wire [31:0] add_res_0; // @[Compute.scala 312:36]
  wire [7:0] _T_1039; // @[Compute.scala 334:37]
  wire [4:0] _T_1041; // @[Compute.scala 335:60]
  wire [31:0] _T_1042; // @[Compute.scala 335:49]
  wire [31:0] _T_1043; // @[Compute.scala 335:84]
  wire [31:0] shr_val_0; // @[Compute.scala 312:36]
  wire [31:0] shr_res_0; // @[Compute.scala 312:36]
  wire [7:0] _T_1044; // @[Compute.scala 337:37]
  wire [31:0] src_0_1; // @[Compute.scala 312:36]
  wire [31:0] src_1_1; // @[Compute.scala 312:36]
  wire  _T_1045; // @[Compute.scala 329:34]
  wire [31:0] _T_1046; // @[Compute.scala 329:24]
  wire [31:0] mix_val_1; // @[Compute.scala 312:36]
  wire [7:0] _T_1047; // @[Compute.scala 331:37]
  wire [31:0] _T_1048; // @[Compute.scala 332:30]
  wire [31:0] _T_1049; // @[Compute.scala 332:59]
  wire [32:0] _T_1050; // @[Compute.scala 332:49]
  wire [31:0] _T_1051; // @[Compute.scala 332:49]
  wire [31:0] _T_1052; // @[Compute.scala 332:79]
  wire [31:0] add_val_1; // @[Compute.scala 312:36]
  wire [31:0] add_res_1; // @[Compute.scala 312:36]
  wire [7:0] _T_1053; // @[Compute.scala 334:37]
  wire [4:0] _T_1055; // @[Compute.scala 335:60]
  wire [31:0] _T_1056; // @[Compute.scala 335:49]
  wire [31:0] _T_1057; // @[Compute.scala 335:84]
  wire [31:0] shr_val_1; // @[Compute.scala 312:36]
  wire [31:0] shr_res_1; // @[Compute.scala 312:36]
  wire [7:0] _T_1058; // @[Compute.scala 337:37]
  wire [31:0] src_0_2; // @[Compute.scala 312:36]
  wire [31:0] src_1_2; // @[Compute.scala 312:36]
  wire  _T_1059; // @[Compute.scala 329:34]
  wire [31:0] _T_1060; // @[Compute.scala 329:24]
  wire [31:0] mix_val_2; // @[Compute.scala 312:36]
  wire [7:0] _T_1061; // @[Compute.scala 331:37]
  wire [31:0] _T_1062; // @[Compute.scala 332:30]
  wire [31:0] _T_1063; // @[Compute.scala 332:59]
  wire [32:0] _T_1064; // @[Compute.scala 332:49]
  wire [31:0] _T_1065; // @[Compute.scala 332:49]
  wire [31:0] _T_1066; // @[Compute.scala 332:79]
  wire [31:0] add_val_2; // @[Compute.scala 312:36]
  wire [31:0] add_res_2; // @[Compute.scala 312:36]
  wire [7:0] _T_1067; // @[Compute.scala 334:37]
  wire [4:0] _T_1069; // @[Compute.scala 335:60]
  wire [31:0] _T_1070; // @[Compute.scala 335:49]
  wire [31:0] _T_1071; // @[Compute.scala 335:84]
  wire [31:0] shr_val_2; // @[Compute.scala 312:36]
  wire [31:0] shr_res_2; // @[Compute.scala 312:36]
  wire [7:0] _T_1072; // @[Compute.scala 337:37]
  wire [31:0] src_0_3; // @[Compute.scala 312:36]
  wire [31:0] src_1_3; // @[Compute.scala 312:36]
  wire  _T_1073; // @[Compute.scala 329:34]
  wire [31:0] _T_1074; // @[Compute.scala 329:24]
  wire [31:0] mix_val_3; // @[Compute.scala 312:36]
  wire [7:0] _T_1075; // @[Compute.scala 331:37]
  wire [31:0] _T_1076; // @[Compute.scala 332:30]
  wire [31:0] _T_1077; // @[Compute.scala 332:59]
  wire [32:0] _T_1078; // @[Compute.scala 332:49]
  wire [31:0] _T_1079; // @[Compute.scala 332:49]
  wire [31:0] _T_1080; // @[Compute.scala 332:79]
  wire [31:0] add_val_3; // @[Compute.scala 312:36]
  wire [31:0] add_res_3; // @[Compute.scala 312:36]
  wire [7:0] _T_1081; // @[Compute.scala 334:37]
  wire [4:0] _T_1083; // @[Compute.scala 335:60]
  wire [31:0] _T_1084; // @[Compute.scala 335:49]
  wire [31:0] _T_1085; // @[Compute.scala 335:84]
  wire [31:0] shr_val_3; // @[Compute.scala 312:36]
  wire [31:0] shr_res_3; // @[Compute.scala 312:36]
  wire [7:0] _T_1086; // @[Compute.scala 337:37]
  wire [31:0] src_0_4; // @[Compute.scala 312:36]
  wire [31:0] src_1_4; // @[Compute.scala 312:36]
  wire  _T_1087; // @[Compute.scala 329:34]
  wire [31:0] _T_1088; // @[Compute.scala 329:24]
  wire [31:0] mix_val_4; // @[Compute.scala 312:36]
  wire [7:0] _T_1089; // @[Compute.scala 331:37]
  wire [31:0] _T_1090; // @[Compute.scala 332:30]
  wire [31:0] _T_1091; // @[Compute.scala 332:59]
  wire [32:0] _T_1092; // @[Compute.scala 332:49]
  wire [31:0] _T_1093; // @[Compute.scala 332:49]
  wire [31:0] _T_1094; // @[Compute.scala 332:79]
  wire [31:0] add_val_4; // @[Compute.scala 312:36]
  wire [31:0] add_res_4; // @[Compute.scala 312:36]
  wire [7:0] _T_1095; // @[Compute.scala 334:37]
  wire [4:0] _T_1097; // @[Compute.scala 335:60]
  wire [31:0] _T_1098; // @[Compute.scala 335:49]
  wire [31:0] _T_1099; // @[Compute.scala 335:84]
  wire [31:0] shr_val_4; // @[Compute.scala 312:36]
  wire [31:0] shr_res_4; // @[Compute.scala 312:36]
  wire [7:0] _T_1100; // @[Compute.scala 337:37]
  wire [31:0] src_0_5; // @[Compute.scala 312:36]
  wire [31:0] src_1_5; // @[Compute.scala 312:36]
  wire  _T_1101; // @[Compute.scala 329:34]
  wire [31:0] _T_1102; // @[Compute.scala 329:24]
  wire [31:0] mix_val_5; // @[Compute.scala 312:36]
  wire [7:0] _T_1103; // @[Compute.scala 331:37]
  wire [31:0] _T_1104; // @[Compute.scala 332:30]
  wire [31:0] _T_1105; // @[Compute.scala 332:59]
  wire [32:0] _T_1106; // @[Compute.scala 332:49]
  wire [31:0] _T_1107; // @[Compute.scala 332:49]
  wire [31:0] _T_1108; // @[Compute.scala 332:79]
  wire [31:0] add_val_5; // @[Compute.scala 312:36]
  wire [31:0] add_res_5; // @[Compute.scala 312:36]
  wire [7:0] _T_1109; // @[Compute.scala 334:37]
  wire [4:0] _T_1111; // @[Compute.scala 335:60]
  wire [31:0] _T_1112; // @[Compute.scala 335:49]
  wire [31:0] _T_1113; // @[Compute.scala 335:84]
  wire [31:0] shr_val_5; // @[Compute.scala 312:36]
  wire [31:0] shr_res_5; // @[Compute.scala 312:36]
  wire [7:0] _T_1114; // @[Compute.scala 337:37]
  wire [31:0] src_0_6; // @[Compute.scala 312:36]
  wire [31:0] src_1_6; // @[Compute.scala 312:36]
  wire  _T_1115; // @[Compute.scala 329:34]
  wire [31:0] _T_1116; // @[Compute.scala 329:24]
  wire [31:0] mix_val_6; // @[Compute.scala 312:36]
  wire [7:0] _T_1117; // @[Compute.scala 331:37]
  wire [31:0] _T_1118; // @[Compute.scala 332:30]
  wire [31:0] _T_1119; // @[Compute.scala 332:59]
  wire [32:0] _T_1120; // @[Compute.scala 332:49]
  wire [31:0] _T_1121; // @[Compute.scala 332:49]
  wire [31:0] _T_1122; // @[Compute.scala 332:79]
  wire [31:0] add_val_6; // @[Compute.scala 312:36]
  wire [31:0] add_res_6; // @[Compute.scala 312:36]
  wire [7:0] _T_1123; // @[Compute.scala 334:37]
  wire [4:0] _T_1125; // @[Compute.scala 335:60]
  wire [31:0] _T_1126; // @[Compute.scala 335:49]
  wire [31:0] _T_1127; // @[Compute.scala 335:84]
  wire [31:0] shr_val_6; // @[Compute.scala 312:36]
  wire [31:0] shr_res_6; // @[Compute.scala 312:36]
  wire [7:0] _T_1128; // @[Compute.scala 337:37]
  wire [31:0] src_0_7; // @[Compute.scala 312:36]
  wire [31:0] src_1_7; // @[Compute.scala 312:36]
  wire  _T_1129; // @[Compute.scala 329:34]
  wire [31:0] _T_1130; // @[Compute.scala 329:24]
  wire [31:0] mix_val_7; // @[Compute.scala 312:36]
  wire [7:0] _T_1131; // @[Compute.scala 331:37]
  wire [31:0] _T_1132; // @[Compute.scala 332:30]
  wire [31:0] _T_1133; // @[Compute.scala 332:59]
  wire [32:0] _T_1134; // @[Compute.scala 332:49]
  wire [31:0] _T_1135; // @[Compute.scala 332:49]
  wire [31:0] _T_1136; // @[Compute.scala 332:79]
  wire [31:0] add_val_7; // @[Compute.scala 312:36]
  wire [31:0] add_res_7; // @[Compute.scala 312:36]
  wire [7:0] _T_1137; // @[Compute.scala 334:37]
  wire [4:0] _T_1139; // @[Compute.scala 335:60]
  wire [31:0] _T_1140; // @[Compute.scala 335:49]
  wire [31:0] _T_1141; // @[Compute.scala 335:84]
  wire [31:0] shr_val_7; // @[Compute.scala 312:36]
  wire [31:0] shr_res_7; // @[Compute.scala 312:36]
  wire [7:0] _T_1142; // @[Compute.scala 337:37]
  wire [31:0] src_0_8; // @[Compute.scala 312:36]
  wire [31:0] src_1_8; // @[Compute.scala 312:36]
  wire  _T_1143; // @[Compute.scala 329:34]
  wire [31:0] _T_1144; // @[Compute.scala 329:24]
  wire [31:0] mix_val_8; // @[Compute.scala 312:36]
  wire [7:0] _T_1145; // @[Compute.scala 331:37]
  wire [31:0] _T_1146; // @[Compute.scala 332:30]
  wire [31:0] _T_1147; // @[Compute.scala 332:59]
  wire [32:0] _T_1148; // @[Compute.scala 332:49]
  wire [31:0] _T_1149; // @[Compute.scala 332:49]
  wire [31:0] _T_1150; // @[Compute.scala 332:79]
  wire [31:0] add_val_8; // @[Compute.scala 312:36]
  wire [31:0] add_res_8; // @[Compute.scala 312:36]
  wire [7:0] _T_1151; // @[Compute.scala 334:37]
  wire [4:0] _T_1153; // @[Compute.scala 335:60]
  wire [31:0] _T_1154; // @[Compute.scala 335:49]
  wire [31:0] _T_1155; // @[Compute.scala 335:84]
  wire [31:0] shr_val_8; // @[Compute.scala 312:36]
  wire [31:0] shr_res_8; // @[Compute.scala 312:36]
  wire [7:0] _T_1156; // @[Compute.scala 337:37]
  wire [31:0] src_0_9; // @[Compute.scala 312:36]
  wire [31:0] src_1_9; // @[Compute.scala 312:36]
  wire  _T_1157; // @[Compute.scala 329:34]
  wire [31:0] _T_1158; // @[Compute.scala 329:24]
  wire [31:0] mix_val_9; // @[Compute.scala 312:36]
  wire [7:0] _T_1159; // @[Compute.scala 331:37]
  wire [31:0] _T_1160; // @[Compute.scala 332:30]
  wire [31:0] _T_1161; // @[Compute.scala 332:59]
  wire [32:0] _T_1162; // @[Compute.scala 332:49]
  wire [31:0] _T_1163; // @[Compute.scala 332:49]
  wire [31:0] _T_1164; // @[Compute.scala 332:79]
  wire [31:0] add_val_9; // @[Compute.scala 312:36]
  wire [31:0] add_res_9; // @[Compute.scala 312:36]
  wire [7:0] _T_1165; // @[Compute.scala 334:37]
  wire [4:0] _T_1167; // @[Compute.scala 335:60]
  wire [31:0] _T_1168; // @[Compute.scala 335:49]
  wire [31:0] _T_1169; // @[Compute.scala 335:84]
  wire [31:0] shr_val_9; // @[Compute.scala 312:36]
  wire [31:0] shr_res_9; // @[Compute.scala 312:36]
  wire [7:0] _T_1170; // @[Compute.scala 337:37]
  wire [31:0] src_0_10; // @[Compute.scala 312:36]
  wire [31:0] src_1_10; // @[Compute.scala 312:36]
  wire  _T_1171; // @[Compute.scala 329:34]
  wire [31:0] _T_1172; // @[Compute.scala 329:24]
  wire [31:0] mix_val_10; // @[Compute.scala 312:36]
  wire [7:0] _T_1173; // @[Compute.scala 331:37]
  wire [31:0] _T_1174; // @[Compute.scala 332:30]
  wire [31:0] _T_1175; // @[Compute.scala 332:59]
  wire [32:0] _T_1176; // @[Compute.scala 332:49]
  wire [31:0] _T_1177; // @[Compute.scala 332:49]
  wire [31:0] _T_1178; // @[Compute.scala 332:79]
  wire [31:0] add_val_10; // @[Compute.scala 312:36]
  wire [31:0] add_res_10; // @[Compute.scala 312:36]
  wire [7:0] _T_1179; // @[Compute.scala 334:37]
  wire [4:0] _T_1181; // @[Compute.scala 335:60]
  wire [31:0] _T_1182; // @[Compute.scala 335:49]
  wire [31:0] _T_1183; // @[Compute.scala 335:84]
  wire [31:0] shr_val_10; // @[Compute.scala 312:36]
  wire [31:0] shr_res_10; // @[Compute.scala 312:36]
  wire [7:0] _T_1184; // @[Compute.scala 337:37]
  wire [31:0] src_0_11; // @[Compute.scala 312:36]
  wire [31:0] src_1_11; // @[Compute.scala 312:36]
  wire  _T_1185; // @[Compute.scala 329:34]
  wire [31:0] _T_1186; // @[Compute.scala 329:24]
  wire [31:0] mix_val_11; // @[Compute.scala 312:36]
  wire [7:0] _T_1187; // @[Compute.scala 331:37]
  wire [31:0] _T_1188; // @[Compute.scala 332:30]
  wire [31:0] _T_1189; // @[Compute.scala 332:59]
  wire [32:0] _T_1190; // @[Compute.scala 332:49]
  wire [31:0] _T_1191; // @[Compute.scala 332:49]
  wire [31:0] _T_1192; // @[Compute.scala 332:79]
  wire [31:0] add_val_11; // @[Compute.scala 312:36]
  wire [31:0] add_res_11; // @[Compute.scala 312:36]
  wire [7:0] _T_1193; // @[Compute.scala 334:37]
  wire [4:0] _T_1195; // @[Compute.scala 335:60]
  wire [31:0] _T_1196; // @[Compute.scala 335:49]
  wire [31:0] _T_1197; // @[Compute.scala 335:84]
  wire [31:0] shr_val_11; // @[Compute.scala 312:36]
  wire [31:0] shr_res_11; // @[Compute.scala 312:36]
  wire [7:0] _T_1198; // @[Compute.scala 337:37]
  wire [31:0] src_0_12; // @[Compute.scala 312:36]
  wire [31:0] src_1_12; // @[Compute.scala 312:36]
  wire  _T_1199; // @[Compute.scala 329:34]
  wire [31:0] _T_1200; // @[Compute.scala 329:24]
  wire [31:0] mix_val_12; // @[Compute.scala 312:36]
  wire [7:0] _T_1201; // @[Compute.scala 331:37]
  wire [31:0] _T_1202; // @[Compute.scala 332:30]
  wire [31:0] _T_1203; // @[Compute.scala 332:59]
  wire [32:0] _T_1204; // @[Compute.scala 332:49]
  wire [31:0] _T_1205; // @[Compute.scala 332:49]
  wire [31:0] _T_1206; // @[Compute.scala 332:79]
  wire [31:0] add_val_12; // @[Compute.scala 312:36]
  wire [31:0] add_res_12; // @[Compute.scala 312:36]
  wire [7:0] _T_1207; // @[Compute.scala 334:37]
  wire [4:0] _T_1209; // @[Compute.scala 335:60]
  wire [31:0] _T_1210; // @[Compute.scala 335:49]
  wire [31:0] _T_1211; // @[Compute.scala 335:84]
  wire [31:0] shr_val_12; // @[Compute.scala 312:36]
  wire [31:0] shr_res_12; // @[Compute.scala 312:36]
  wire [7:0] _T_1212; // @[Compute.scala 337:37]
  wire [31:0] src_0_13; // @[Compute.scala 312:36]
  wire [31:0] src_1_13; // @[Compute.scala 312:36]
  wire  _T_1213; // @[Compute.scala 329:34]
  wire [31:0] _T_1214; // @[Compute.scala 329:24]
  wire [31:0] mix_val_13; // @[Compute.scala 312:36]
  wire [7:0] _T_1215; // @[Compute.scala 331:37]
  wire [31:0] _T_1216; // @[Compute.scala 332:30]
  wire [31:0] _T_1217; // @[Compute.scala 332:59]
  wire [32:0] _T_1218; // @[Compute.scala 332:49]
  wire [31:0] _T_1219; // @[Compute.scala 332:49]
  wire [31:0] _T_1220; // @[Compute.scala 332:79]
  wire [31:0] add_val_13; // @[Compute.scala 312:36]
  wire [31:0] add_res_13; // @[Compute.scala 312:36]
  wire [7:0] _T_1221; // @[Compute.scala 334:37]
  wire [4:0] _T_1223; // @[Compute.scala 335:60]
  wire [31:0] _T_1224; // @[Compute.scala 335:49]
  wire [31:0] _T_1225; // @[Compute.scala 335:84]
  wire [31:0] shr_val_13; // @[Compute.scala 312:36]
  wire [31:0] shr_res_13; // @[Compute.scala 312:36]
  wire [7:0] _T_1226; // @[Compute.scala 337:37]
  wire [31:0] src_0_14; // @[Compute.scala 312:36]
  wire [31:0] src_1_14; // @[Compute.scala 312:36]
  wire  _T_1227; // @[Compute.scala 329:34]
  wire [31:0] _T_1228; // @[Compute.scala 329:24]
  wire [31:0] mix_val_14; // @[Compute.scala 312:36]
  wire [7:0] _T_1229; // @[Compute.scala 331:37]
  wire [31:0] _T_1230; // @[Compute.scala 332:30]
  wire [31:0] _T_1231; // @[Compute.scala 332:59]
  wire [32:0] _T_1232; // @[Compute.scala 332:49]
  wire [31:0] _T_1233; // @[Compute.scala 332:49]
  wire [31:0] _T_1234; // @[Compute.scala 332:79]
  wire [31:0] add_val_14; // @[Compute.scala 312:36]
  wire [31:0] add_res_14; // @[Compute.scala 312:36]
  wire [7:0] _T_1235; // @[Compute.scala 334:37]
  wire [4:0] _T_1237; // @[Compute.scala 335:60]
  wire [31:0] _T_1238; // @[Compute.scala 335:49]
  wire [31:0] _T_1239; // @[Compute.scala 335:84]
  wire [31:0] shr_val_14; // @[Compute.scala 312:36]
  wire [31:0] shr_res_14; // @[Compute.scala 312:36]
  wire [7:0] _T_1240; // @[Compute.scala 337:37]
  wire [31:0] src_0_15; // @[Compute.scala 312:36]
  wire [31:0] src_1_15; // @[Compute.scala 312:36]
  wire  _T_1241; // @[Compute.scala 329:34]
  wire [31:0] _T_1242; // @[Compute.scala 329:24]
  wire [31:0] mix_val_15; // @[Compute.scala 312:36]
  wire [7:0] _T_1243; // @[Compute.scala 331:37]
  wire [31:0] _T_1244; // @[Compute.scala 332:30]
  wire [31:0] _T_1245; // @[Compute.scala 332:59]
  wire [32:0] _T_1246; // @[Compute.scala 332:49]
  wire [31:0] _T_1247; // @[Compute.scala 332:49]
  wire [31:0] _T_1248; // @[Compute.scala 332:79]
  wire [31:0] add_val_15; // @[Compute.scala 312:36]
  wire [31:0] add_res_15; // @[Compute.scala 312:36]
  wire [7:0] _T_1249; // @[Compute.scala 334:37]
  wire [4:0] _T_1251; // @[Compute.scala 335:60]
  wire [31:0] _T_1252; // @[Compute.scala 335:49]
  wire [31:0] _T_1253; // @[Compute.scala 335:84]
  wire [31:0] shr_val_15; // @[Compute.scala 312:36]
  wire [31:0] shr_res_15; // @[Compute.scala 312:36]
  wire [7:0] _T_1254; // @[Compute.scala 337:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 312:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 312:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 312:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 312:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 342:48]
  wire  alu_opcode_add_en; // @[Compute.scala 343:39]
  wire [63:0] _T_1264; // @[Cat.scala 30:58]
  wire [127:0] _T_1272; // @[Cat.scala 30:58]
  wire [63:0] _T_1279; // @[Cat.scala 30:58]
  wire [127:0] _T_1287; // @[Cat.scala 30:58]
  wire [63:0] _T_1294; // @[Cat.scala 30:58]
  wire [127:0] _T_1302; // @[Cat.scala 30:58]
  wire [127:0] _T_1303; // @[Compute.scala 348:29]
  wire [127:0] out_mem_enq_bits; // @[Compute.scala 347:29]
  wire  _T_1304; // @[Compute.scala 349:34]
  wire  _T_1305; // @[Compute.scala 349:59]
  wire  _T_1306; // @[Compute.scala 349:42]
  wire  _T_1308; // @[Compute.scala 351:63]
  wire  _T_1309; // @[Compute.scala 351:46]
  wire [44:0] _T_1311; // @[Compute.scala 351:105]
  wire [44:0] _T_1312; // @[Compute.scala 351:105]
  wire [43:0] _T_1313; // @[Compute.scala 351:105]
  wire  _T_1314; // @[Compute.scala 351:88]
  reg [15:0] _T_1317; // @[Compute.scala 352:42]
  reg [31:0] _RAND_26;
  wire [143:0] _T_1318; // @[Cat.scala 30:58]
  wire [31:0] _T_1319; // @[Compute.scala 360:49]
  wire [38:0] _GEN_343; // @[Compute.scala 360:66]
  wire [38:0] _T_1321; // @[Compute.scala 360:66]
  OutQueue out_mem_fifo ( // @[Compute.scala 346:28]
    .clock(out_mem_fifo_clock),
    .reset(out_mem_fifo_reset),
    .io_enq_ready(out_mem_fifo_io_enq_ready),
    .io_enq_valid(out_mem_fifo_io_enq_valid),
    .io_enq_bits(out_mem_fifo_io_enq_bits),
    .io_deq_ready(out_mem_fifo_io_deq_ready),
    .io_deq_valid(out_mem_fifo_io_deq_valid),
    .io_deq_bits(out_mem_fifo_io_deq_bits)
  );
  assign acc_mem__T_468_addr = dst_idx[7:0];
  assign acc_mem__T_468_data = acc_mem[acc_mem__T_468_addr]; // @[Compute.scala 34:20]
  assign acc_mem__T_470_addr = src_idx[7:0];
  assign acc_mem__T_470_data = acc_mem[acc_mem__T_470_addr]; // @[Compute.scala 34:20]
  assign acc_mem__T_445_data = {_T_447,_T_446};
  assign acc_mem__T_445_addr = acc_sram_addr[7:0];
  assign acc_mem__T_445_mask = 1'h1;
  assign acc_mem__T_445_en = _T_335 ? _T_443 : 1'h0;
  assign uop_mem__T_450_addr = upc[9:0];
  assign uop_mem__T_450_data = uop_mem[uop_mem__T_450_addr]; // @[Compute.scala 35:20]
  assign uop_mem__T_378_data = io_uops_readdata[31:0];
  assign uop_mem__T_378_addr = _T_376[9:0];
  assign uop_mem__T_378_mask = 1'h1;
  assign uop_mem__T_378_en = uops_read & _T_325;
  assign uop_mem__T_384_data = io_uops_readdata[63:32];
  assign uop_mem__T_384_addr = _T_382[9:0];
  assign uop_mem__T_384_mask = 1'h1;
  assign uop_mem__T_384_en = uops_read & _T_325;
  assign uop_mem__T_390_data = io_uops_readdata[95:64];
  assign uop_mem__T_390_addr = _T_388[9:0];
  assign uop_mem__T_390_mask = 1'h1;
  assign uop_mem__T_390_en = uops_read & _T_325;
  assign uop_mem__T_396_data = io_uops_readdata[127:96];
  assign uop_mem__T_396_addr = _T_394[9:0];
  assign uop_mem__T_396_mask = 1'h1;
  assign uop_mem__T_396_en = uops_read & _T_325;
  assign started = reset == 1'h0; // @[Compute.scala 32:17]
  assign _T_201 = insn != 128'h0; // @[Compute.scala 38:31]
  assign insn_valid = _T_201 & started; // @[Compute.scala 38:40]
  assign opcode = insn[2:0]; // @[Compute.scala 40:29]
  assign pop_prev_dep = insn[3]; // @[Compute.scala 41:29]
  assign pop_next_dep = insn[4]; // @[Compute.scala 42:29]
  assign push_prev_dep = insn[5]; // @[Compute.scala 43:29]
  assign push_next_dep = insn[6]; // @[Compute.scala 44:29]
  assign memory_type = insn[8:7]; // @[Compute.scala 46:25]
  assign sram_base = insn[24:9]; // @[Compute.scala 47:25]
  assign dram_base = insn[56:25]; // @[Compute.scala 48:25]
  assign x_size = insn[95:80]; // @[Compute.scala 50:25]
  assign y_pad_0 = insn[115:112]; // @[Compute.scala 52:25]
  assign x_pad_0 = insn[123:120]; // @[Compute.scala 54:25]
  assign x_pad_1 = insn[127:124]; // @[Compute.scala 55:25]
  assign _T_203 = insn[20:8]; // @[Compute.scala 59:18]
  assign _T_205 = insn[34:21]; // @[Compute.scala 61:18]
  assign iter_out = insn[48:35]; // @[Compute.scala 62:22]
  assign iter_in = insn[62:49]; // @[Compute.scala 63:21]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 70:24]
  assign use_imm = insn[110]; // @[Compute.scala 71:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 72:21]
  assign _T_206 = $signed(imm_raw); // @[Compute.scala 73:25]
  assign _T_208 = $signed(_T_206) < $signed(16'sh0); // @[Compute.scala 73:32]
  assign _T_210 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_212 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_213 = _T_208 ? _T_210 : {{15'd0}, _T_212}; // @[Compute.scala 73:16]
  assign imm = $signed(_T_213); // @[Compute.scala 73:89]
  assign _GEN_316 = {{12'd0}, y_pad_0}; // @[Compute.scala 77:30]
  assign _GEN_318 = {{12'd0}, x_pad_0}; // @[Compute.scala 78:30]
  assign _T_217 = _GEN_318 + x_size; // @[Compute.scala 78:30]
  assign _T_218 = _GEN_318 + x_size; // @[Compute.scala 78:30]
  assign _GEN_319 = {{12'd0}, x_pad_1}; // @[Compute.scala 78:39]
  assign _T_219 = _T_218 + _GEN_319; // @[Compute.scala 78:39]
  assign x_size_total = _T_218 + _GEN_319; // @[Compute.scala 78:39]
  assign y_offset = x_size_total * _GEN_316; // @[Compute.scala 79:31]
  assign opcode_finish_en = opcode == 3'h3; // @[Compute.scala 82:34]
  assign _T_222 = opcode == 3'h0; // @[Compute.scala 83:32]
  assign _T_224 = opcode == 3'h1; // @[Compute.scala 83:60]
  assign opcode_load_en = _T_222 | _T_224; // @[Compute.scala 83:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Compute.scala 84:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Compute.scala 85:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Compute.scala 87:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Compute.scala 88:40]
  assign idle = state == 3'h0; // @[Compute.scala 93:20]
  assign dump = state == 3'h1; // @[Compute.scala 94:20]
  assign busy = state == 3'h2; // @[Compute.scala 95:20]
  assign push = state == 3'h3; // @[Compute.scala 96:20]
  assign done = state == 3'h4; // @[Compute.scala 97:20]
  assign uop_cntr_max_val = x_size >> 2'h2; // @[Compute.scala 111:33]
  assign _T_248 = uop_cntr_max_val == 16'h0; // @[Compute.scala 112:43]
  assign uop_cntr_max = _T_248 ? 16'h1 : uop_cntr_max_val; // @[Compute.scala 112:25]
  assign _T_250 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 113:37]
  assign uop_cntr_en = _T_250 & insn_valid; // @[Compute.scala 113:59]
  assign _T_252 = uop_cntr_val == uop_cntr_max; // @[Compute.scala 116:38]
  assign _T_253 = _T_252 & uop_cntr_en; // @[Compute.scala 116:56]
  assign uop_cntr_wrap = _T_253 & busy; // @[Compute.scala 116:71]
  assign _T_255 = x_size * 16'h4; // @[Compute.scala 118:29]
  assign _T_257 = _T_255 + 19'h1; // @[Compute.scala 118:46]
  assign acc_cntr_max = _T_255 + 19'h1; // @[Compute.scala 118:46]
  assign _T_258 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 119:37]
  assign acc_cntr_en = _T_258 & insn_valid; // @[Compute.scala 119:59]
  assign _GEN_321 = {{3'd0}, acc_cntr_val}; // @[Compute.scala 122:38]
  assign _T_260 = _GEN_321 == acc_cntr_max; // @[Compute.scala 122:38]
  assign _T_261 = _T_260 & acc_cntr_en; // @[Compute.scala 122:56]
  assign acc_cntr_wrap = _T_261 & busy; // @[Compute.scala 122:71]
  assign _T_262 = uop_end - uop_bgn; // @[Compute.scala 124:34]
  assign _T_263 = $unsigned(_T_262); // @[Compute.scala 124:34]
  assign upc_cntr_max_val = _T_263[15:0]; // @[Compute.scala 124:34]
  assign _T_265 = upc_cntr_max_val <= 16'h0; // @[Compute.scala 125:43]
  assign upc_cntr_max = _T_265 ? 16'h1 : upc_cntr_max_val; // @[Compute.scala 125:25]
  assign _T_267 = iter_in * iter_out; // @[Compute.scala 126:34]
  assign _GEN_322 = {{12'd0}, upc_cntr_max}; // @[Compute.scala 126:45]
  assign out_cntr_max_val = _T_267 * _GEN_322; // @[Compute.scala 126:45]
  assign _T_269 = out_cntr_max_val + 44'h2; // @[Compute.scala 127:39]
  assign out_cntr_max = out_cntr_max_val + 44'h2; // @[Compute.scala 127:39]
  assign _T_270 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 128:37]
  assign out_cntr_en = _T_270 & insn_valid; // @[Compute.scala 128:56]
  assign _GEN_323 = {{28'd0}, out_cntr_val}; // @[Compute.scala 131:38]
  assign _T_272 = _GEN_323 == out_cntr_max; // @[Compute.scala 131:38]
  assign _T_273 = _T_272 & out_cntr_en; // @[Compute.scala 131:56]
  assign out_cntr_wrap = _T_273 & busy; // @[Compute.scala 131:71]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Compute.scala 136:43]
  assign push_next_dep_valid = push_next_dep & push; // @[Compute.scala 137:43]
  assign _T_286 = pop_prev_dep_ready & busy; // @[Compute.scala 146:68]
  assign _T_287 = pop_next_dep_ready & busy; // @[Compute.scala 147:68]
  assign _T_288 = push_prev_dep_ready & busy; // @[Compute.scala 148:68]
  assign _T_289 = push_next_dep_ready & busy; // @[Compute.scala 149:68]
  assign _GEN_0 = push_next_dep ? _T_289 : 1'h0; // @[Compute.scala 149:31]
  assign _GEN_1 = push_prev_dep ? _T_288 : _GEN_0; // @[Compute.scala 148:31]
  assign _GEN_2 = pop_next_dep ? _T_287 : _GEN_1; // @[Compute.scala 147:31]
  assign _GEN_3 = pop_prev_dep ? _T_286 : _GEN_2; // @[Compute.scala 146:31]
  assign _GEN_4 = opcode_finish_en ? _GEN_3 : 1'h0; // @[Compute.scala 145:27]
  assign _T_292 = uop_cntr_wrap | acc_cntr_wrap; // @[Compute.scala 152:23]
  assign _T_293 = _T_292 | out_cntr_wrap; // @[Compute.scala 152:40]
  assign _T_294 = _T_293 | finish_wrap; // @[Compute.scala 152:57]
  assign _T_295 = push_prev_dep | push_next_dep; // @[Compute.scala 153:25]
  assign _GEN_5 = _T_295 ? 3'h3 : 3'h4; // @[Compute.scala 153:43]
  assign _GEN_6 = _T_294 ? _GEN_5 : state; // @[Compute.scala 152:73]
  assign _T_297 = pop_prev_dep_ready == 1'h0; // @[Compute.scala 161:18]
  assign _T_299 = pop_next_dep_ready == 1'h0; // @[Compute.scala 161:41]
  assign _T_300 = _T_297 & _T_299; // @[Compute.scala 161:38]
  assign _T_301 = busy & _T_300; // @[Compute.scala 161:14]
  assign _T_302 = pop_prev_dep | pop_next_dep; // @[Compute.scala 161:79]
  assign _T_303 = _T_301 & _T_302; // @[Compute.scala 161:62]
  assign _GEN_7 = _T_303 ? 3'h1 : _GEN_6; // @[Compute.scala 161:97]
  assign _T_304 = pop_prev_dep_ready | pop_next_dep_ready; // @[Compute.scala 162:38]
  assign _T_305 = dump & _T_304; // @[Compute.scala 162:14]
  assign _GEN_8 = _T_305 ? 3'h2 : _GEN_7; // @[Compute.scala 162:63]
  assign _T_306 = push_prev_dep_ready | push_next_dep_ready; // @[Compute.scala 163:38]
  assign _T_307 = push & _T_306; // @[Compute.scala 163:14]
  assign _GEN_9 = _T_307 ? 3'h4 : _GEN_8; // @[Compute.scala 163:63]
  assign _T_310 = pop_prev_dep & dump; // @[Compute.scala 170:22]
  assign _T_311 = _T_310 & io_l2g_dep_queue_valid; // @[Compute.scala 170:30]
  assign _GEN_10 = _T_311 ? 1'h1 : pop_prev_dep_ready; // @[Compute.scala 170:57]
  assign _T_313 = pop_next_dep & dump; // @[Compute.scala 173:22]
  assign _T_314 = _T_313 & io_s2g_dep_queue_valid; // @[Compute.scala 173:30]
  assign _GEN_11 = _T_314 ? 1'h1 : pop_next_dep_ready; // @[Compute.scala 173:57]
  assign _T_318 = push_prev_dep_valid & io_g2l_dep_queue_ready; // @[Compute.scala 180:29]
  assign _T_319 = _T_318 & push; // @[Compute.scala 180:55]
  assign _GEN_12 = _T_319 ? 1'h1 : push_prev_dep_ready; // @[Compute.scala 180:64]
  assign _T_321 = push_next_dep_valid & io_g2s_dep_queue_ready; // @[Compute.scala 183:29]
  assign _T_322 = _T_321 & push; // @[Compute.scala 183:55]
  assign _GEN_13 = _T_322 ? 1'h1 : push_next_dep_ready; // @[Compute.scala 183:64]
  assign _T_325 = io_uops_waitrequest == 1'h0; // @[Compute.scala 188:22]
  assign _T_326 = uops_read & _T_325; // @[Compute.scala 188:19]
  assign _T_327 = _T_326 & busy; // @[Compute.scala 188:37]
  assign _T_328 = uop_cntr_val < uop_cntr_max; // @[Compute.scala 188:61]
  assign _T_329 = _T_327 & _T_328; // @[Compute.scala 188:45]
  assign _T_331 = uop_cntr_val + 16'h1; // @[Compute.scala 189:34]
  assign _T_332 = uop_cntr_val + 16'h1; // @[Compute.scala 189:34]
  assign _GEN_14 = _T_329 ? _T_332 : uop_cntr_val; // @[Compute.scala 188:77]
  assign _T_334 = io_biases_waitrequest == 1'h0; // @[Compute.scala 191:24]
  assign _T_335 = biases_read & _T_334; // @[Compute.scala 191:21]
  assign _T_336 = _T_335 & busy; // @[Compute.scala 191:39]
  assign _T_337 = _GEN_321 < acc_cntr_max; // @[Compute.scala 191:63]
  assign _T_338 = _T_336 & _T_337; // @[Compute.scala 191:47]
  assign _T_340 = acc_cntr_val + 16'h1; // @[Compute.scala 192:34]
  assign _T_341 = acc_cntr_val + 16'h1; // @[Compute.scala 192:34]
  assign _GEN_15 = _T_338 ? _T_341 : acc_cntr_val; // @[Compute.scala 191:79]
  assign _T_342 = out_mem_write & busy; // @[Compute.scala 197:23]
  assign _T_343 = _GEN_323 < out_cntr_max; // @[Compute.scala 197:47]
  assign _T_344 = _T_342 & _T_343; // @[Compute.scala 197:31]
  assign _T_346 = out_cntr_val + 16'h1; // @[Compute.scala 198:34]
  assign _T_347 = out_cntr_val + 16'h1; // @[Compute.scala 198:34]
  assign _GEN_16 = _T_344 ? _T_347 : out_cntr_val; // @[Compute.scala 197:63]
  assign _GEN_21 = gemm_queue_ready ? 1'h0 : _GEN_10; // @[Compute.scala 202:27]
  assign _GEN_22 = gemm_queue_ready ? 1'h0 : _GEN_11; // @[Compute.scala 202:27]
  assign _GEN_23 = gemm_queue_ready ? 1'h0 : _GEN_12; // @[Compute.scala 202:27]
  assign _GEN_24 = gemm_queue_ready ? 1'h0 : _GEN_13; // @[Compute.scala 202:27]
  assign _GEN_25 = gemm_queue_ready ? 3'h2 : _GEN_9; // @[Compute.scala 202:27]
  assign _T_355 = idle | done; // @[Compute.scala 215:52]
  assign _T_356 = io_gemm_queue_valid & _T_355; // @[Compute.scala 215:43]
  assign _GEN_26 = gemm_queue_ready ? 1'h0 : _T_356; // @[Compute.scala 217:27]
  assign _GEN_326 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 227:33]
  assign _T_361 = dram_base + _GEN_326; // @[Compute.scala 227:33]
  assign _T_362 = dram_base + _GEN_326; // @[Compute.scala 227:33]
  assign _GEN_327 = {{7'd0}, _T_362}; // @[Compute.scala 227:49]
  assign uop_dram_addr = _GEN_327 << 3'h4; // @[Compute.scala 227:49]
  assign _T_364 = sram_base + uop_cntr_val; // @[Compute.scala 228:33]
  assign _T_365 = sram_base + uop_cntr_val; // @[Compute.scala 228:33]
  assign _GEN_328 = {{3'd0}, _T_365}; // @[Compute.scala 228:49]
  assign uop_sram_addr = _GEN_328 << 2'h2; // @[Compute.scala 228:49]
  assign _T_368 = uop_cntr_wrap == 1'h0; // @[Compute.scala 229:31]
  assign _T_369 = uop_cntr_en & _T_368; // @[Compute.scala 229:28]
  assign _T_370 = _T_369 & busy; // @[Compute.scala 229:46]
  assign _T_375 = {{1'd0}, uop_sram_addr}; // @[Compute.scala 237:29]
  assign _T_376 = _T_375[18:0]; // @[Compute.scala 237:29]
  assign _T_381 = uop_sram_addr + 19'h1; // @[Compute.scala 237:29]
  assign _T_382 = uop_sram_addr + 19'h1; // @[Compute.scala 237:29]
  assign _T_387 = uop_sram_addr + 19'h2; // @[Compute.scala 237:29]
  assign _T_388 = uop_sram_addr + 19'h2; // @[Compute.scala 237:29]
  assign _T_393 = uop_sram_addr + 19'h3; // @[Compute.scala 237:29]
  assign _T_394 = uop_sram_addr + 19'h3; // @[Compute.scala 237:29]
  assign _T_399 = uop_cntr_max - 16'h1; // @[Compute.scala 239:42]
  assign _T_400 = $unsigned(_T_399); // @[Compute.scala 239:42]
  assign _T_401 = _T_400[15:0]; // @[Compute.scala 239:42]
  assign _T_402 = uop_cntr_val == _T_401; // @[Compute.scala 239:24]
  assign _GEN_27 = _T_402 ? 1'h0 : _T_370; // @[Compute.scala 239:50]
  assign _GEN_329 = {{12'd0}, y_offset}; // @[Compute.scala 243:36]
  assign _T_404 = dram_base + _GEN_329; // @[Compute.scala 243:36]
  assign _T_405 = dram_base + _GEN_329; // @[Compute.scala 243:36]
  assign _GEN_330 = {{28'd0}, x_pad_0}; // @[Compute.scala 243:47]
  assign _T_406 = _T_405 + _GEN_330; // @[Compute.scala 243:47]
  assign _T_407 = _T_405 + _GEN_330; // @[Compute.scala 243:47]
  assign _GEN_331 = {{3'd0}, _T_407}; // @[Compute.scala 243:58]
  assign _T_409 = _GEN_331 << 2'h2; // @[Compute.scala 243:58]
  assign _T_411 = _T_409 * 35'h1; // @[Compute.scala 243:66]
  assign _GEN_332 = {{20'd0}, acc_cntr_val}; // @[Compute.scala 243:76]
  assign _T_412 = _T_411 + _GEN_332; // @[Compute.scala 243:76]
  assign _T_413 = _T_411 + _GEN_332; // @[Compute.scala 243:76]
  assign _GEN_333 = {{7'd0}, _T_413}; // @[Compute.scala 243:92]
  assign acc_dram_addr = _GEN_333 << 3'h4; // @[Compute.scala 243:92]
  assign _GEN_334 = {{4'd0}, sram_base}; // @[Compute.scala 244:36]
  assign _T_415 = _GEN_334 + y_offset; // @[Compute.scala 244:36]
  assign _T_416 = _GEN_334 + y_offset; // @[Compute.scala 244:36]
  assign _GEN_335 = {{16'd0}, x_pad_0}; // @[Compute.scala 244:47]
  assign _T_417 = _T_416 + _GEN_335; // @[Compute.scala 244:47]
  assign _T_418 = _T_416 + _GEN_335; // @[Compute.scala 244:47]
  assign _GEN_336 = {{3'd0}, _T_418}; // @[Compute.scala 244:58]
  assign _T_420 = _GEN_336 << 2'h2; // @[Compute.scala 244:58]
  assign _T_422 = _T_420 * 23'h1; // @[Compute.scala 244:66]
  assign _GEN_337 = {{8'd0}, acc_cntr_val}; // @[Compute.scala 244:76]
  assign _T_423 = _T_422 + _GEN_337; // @[Compute.scala 244:76]
  assign _T_424 = _T_422 + _GEN_337; // @[Compute.scala 244:76]
  assign _T_426 = _T_424 >> 2'h2; // @[Compute.scala 244:92]
  assign _T_428 = _T_426 - 24'h1; // @[Compute.scala 244:121]
  assign _T_429 = $unsigned(_T_428); // @[Compute.scala 244:121]
  assign acc_sram_addr = _T_429[23:0]; // @[Compute.scala 244:121]
  assign _T_431 = done == 1'h0; // @[Compute.scala 245:33]
  assign _GEN_17 = acc_cntr_val % 16'h4; // @[Compute.scala 251:30]
  assign _T_437 = _GEN_17[2:0]; // @[Compute.scala 251:30]
  assign _GEN_40 = 3'h0 == _T_437 ? io_biases_readdata : biases_data_0; // @[Compute.scala 251:48]
  assign _GEN_41 = 3'h1 == _T_437 ? io_biases_readdata : biases_data_1; // @[Compute.scala 251:48]
  assign _GEN_42 = 3'h2 == _T_437 ? io_biases_readdata : biases_data_2; // @[Compute.scala 251:48]
  assign _GEN_43 = 3'h3 == _T_437 ? io_biases_readdata : biases_data_3; // @[Compute.scala 251:48]
  assign _T_443 = _T_437 == 3'h0; // @[Compute.scala 255:43]
  assign _T_446 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_447 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign _GEN_18 = out_cntr_val % upc_cntr_max; // @[Compute.scala 261:26]
  assign upc = _GEN_18[15:0]; // @[Compute.scala 261:26]
  assign _GEN_338 = {{12'd0}, _T_453}; // @[Compute.scala 265:37]
  assign _GEN_19 = _GEN_338 % _T_267; // @[Compute.scala 265:37]
  assign it_in = _GEN_19[15:0]; // @[Compute.scala 265:37]
  assign _T_455 = it_in * 16'h1; // @[Compute.scala 266:47]
  assign _T_456 = _T_455[15:0]; // @[Compute.scala 266:63]
  assign _T_457 = {{1'd0}, _T_456}; // @[Compute.scala 266:38]
  assign dst_offset_in = _T_457[15:0]; // @[Compute.scala 266:38]
  assign _T_461 = uop[10:0]; // @[Compute.scala 268:20]
  assign _GEN_339 = {{5'd0}, _T_461}; // @[Compute.scala 268:47]
  assign _T_462 = _GEN_339 + dst_offset_in; // @[Compute.scala 268:47]
  assign dst_idx = _GEN_339 + dst_offset_in; // @[Compute.scala 268:47]
  assign _T_463 = uop[21:11]; // @[Compute.scala 269:20]
  assign _GEN_340 = {{5'd0}, _T_463}; // @[Compute.scala 269:47]
  assign _T_464 = _GEN_340 + dst_offset_in; // @[Compute.scala 269:47]
  assign src_idx = _GEN_340 + dst_offset_in; // @[Compute.scala 269:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 292:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 293:38]
  assign _T_902 = insn_valid & out_cntr_en; // @[Compute.scala 312:20]
  assign _T_903 = src_vector[31:0]; // @[Compute.scala 315:31]
  assign _T_904 = $signed(_T_903); // @[Compute.scala 315:72]
  assign _T_905 = dst_vector[31:0]; // @[Compute.scala 316:31]
  assign _T_906 = $signed(_T_905); // @[Compute.scala 316:72]
  assign _T_907 = src_vector[63:32]; // @[Compute.scala 315:31]
  assign _T_908 = $signed(_T_907); // @[Compute.scala 315:72]
  assign _T_909 = dst_vector[63:32]; // @[Compute.scala 316:31]
  assign _T_910 = $signed(_T_909); // @[Compute.scala 316:72]
  assign _T_911 = src_vector[95:64]; // @[Compute.scala 315:31]
  assign _T_912 = $signed(_T_911); // @[Compute.scala 315:72]
  assign _T_913 = dst_vector[95:64]; // @[Compute.scala 316:31]
  assign _T_914 = $signed(_T_913); // @[Compute.scala 316:72]
  assign _T_915 = src_vector[127:96]; // @[Compute.scala 315:31]
  assign _T_916 = $signed(_T_915); // @[Compute.scala 315:72]
  assign _T_917 = dst_vector[127:96]; // @[Compute.scala 316:31]
  assign _T_918 = $signed(_T_917); // @[Compute.scala 316:72]
  assign _T_919 = src_vector[159:128]; // @[Compute.scala 315:31]
  assign _T_920 = $signed(_T_919); // @[Compute.scala 315:72]
  assign _T_921 = dst_vector[159:128]; // @[Compute.scala 316:31]
  assign _T_922 = $signed(_T_921); // @[Compute.scala 316:72]
  assign _T_923 = src_vector[191:160]; // @[Compute.scala 315:31]
  assign _T_924 = $signed(_T_923); // @[Compute.scala 315:72]
  assign _T_925 = dst_vector[191:160]; // @[Compute.scala 316:31]
  assign _T_926 = $signed(_T_925); // @[Compute.scala 316:72]
  assign _T_927 = src_vector[223:192]; // @[Compute.scala 315:31]
  assign _T_928 = $signed(_T_927); // @[Compute.scala 315:72]
  assign _T_929 = dst_vector[223:192]; // @[Compute.scala 316:31]
  assign _T_930 = $signed(_T_929); // @[Compute.scala 316:72]
  assign _T_931 = src_vector[255:224]; // @[Compute.scala 315:31]
  assign _T_932 = $signed(_T_931); // @[Compute.scala 315:72]
  assign _T_933 = dst_vector[255:224]; // @[Compute.scala 316:31]
  assign _T_934 = $signed(_T_933); // @[Compute.scala 316:72]
  assign _T_935 = src_vector[287:256]; // @[Compute.scala 315:31]
  assign _T_936 = $signed(_T_935); // @[Compute.scala 315:72]
  assign _T_937 = dst_vector[287:256]; // @[Compute.scala 316:31]
  assign _T_938 = $signed(_T_937); // @[Compute.scala 316:72]
  assign _T_939 = src_vector[319:288]; // @[Compute.scala 315:31]
  assign _T_940 = $signed(_T_939); // @[Compute.scala 315:72]
  assign _T_941 = dst_vector[319:288]; // @[Compute.scala 316:31]
  assign _T_942 = $signed(_T_941); // @[Compute.scala 316:72]
  assign _T_943 = src_vector[351:320]; // @[Compute.scala 315:31]
  assign _T_944 = $signed(_T_943); // @[Compute.scala 315:72]
  assign _T_945 = dst_vector[351:320]; // @[Compute.scala 316:31]
  assign _T_946 = $signed(_T_945); // @[Compute.scala 316:72]
  assign _T_947 = src_vector[383:352]; // @[Compute.scala 315:31]
  assign _T_948 = $signed(_T_947); // @[Compute.scala 315:72]
  assign _T_949 = dst_vector[383:352]; // @[Compute.scala 316:31]
  assign _T_950 = $signed(_T_949); // @[Compute.scala 316:72]
  assign _T_951 = src_vector[415:384]; // @[Compute.scala 315:31]
  assign _T_952 = $signed(_T_951); // @[Compute.scala 315:72]
  assign _T_953 = dst_vector[415:384]; // @[Compute.scala 316:31]
  assign _T_954 = $signed(_T_953); // @[Compute.scala 316:72]
  assign _T_955 = src_vector[447:416]; // @[Compute.scala 315:31]
  assign _T_956 = $signed(_T_955); // @[Compute.scala 315:72]
  assign _T_957 = dst_vector[447:416]; // @[Compute.scala 316:31]
  assign _T_958 = $signed(_T_957); // @[Compute.scala 316:72]
  assign _T_959 = src_vector[479:448]; // @[Compute.scala 315:31]
  assign _T_960 = $signed(_T_959); // @[Compute.scala 315:72]
  assign _T_961 = dst_vector[479:448]; // @[Compute.scala 316:31]
  assign _T_962 = $signed(_T_961); // @[Compute.scala 316:72]
  assign _T_963 = src_vector[511:480]; // @[Compute.scala 315:31]
  assign _T_964 = $signed(_T_963); // @[Compute.scala 315:72]
  assign _T_965 = dst_vector[511:480]; // @[Compute.scala 316:31]
  assign _T_966 = $signed(_T_965); // @[Compute.scala 316:72]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_904) : $signed(_T_906); // @[Compute.scala 313:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_906) : $signed(_T_904); // @[Compute.scala 313:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_908) : $signed(_T_910); // @[Compute.scala 313:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_910) : $signed(_T_908); // @[Compute.scala 313:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_912) : $signed(_T_914); // @[Compute.scala 313:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_914) : $signed(_T_912); // @[Compute.scala 313:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_916) : $signed(_T_918); // @[Compute.scala 313:30]
  assign _GEN_73 = alu_opcode_max_en ? $signed(_T_918) : $signed(_T_916); // @[Compute.scala 313:30]
  assign _GEN_74 = alu_opcode_max_en ? $signed(_T_920) : $signed(_T_922); // @[Compute.scala 313:30]
  assign _GEN_75 = alu_opcode_max_en ? $signed(_T_922) : $signed(_T_920); // @[Compute.scala 313:30]
  assign _GEN_76 = alu_opcode_max_en ? $signed(_T_924) : $signed(_T_926); // @[Compute.scala 313:30]
  assign _GEN_77 = alu_opcode_max_en ? $signed(_T_926) : $signed(_T_924); // @[Compute.scala 313:30]
  assign _GEN_78 = alu_opcode_max_en ? $signed(_T_928) : $signed(_T_930); // @[Compute.scala 313:30]
  assign _GEN_79 = alu_opcode_max_en ? $signed(_T_930) : $signed(_T_928); // @[Compute.scala 313:30]
  assign _GEN_80 = alu_opcode_max_en ? $signed(_T_932) : $signed(_T_934); // @[Compute.scala 313:30]
  assign _GEN_81 = alu_opcode_max_en ? $signed(_T_934) : $signed(_T_932); // @[Compute.scala 313:30]
  assign _GEN_82 = alu_opcode_max_en ? $signed(_T_936) : $signed(_T_938); // @[Compute.scala 313:30]
  assign _GEN_83 = alu_opcode_max_en ? $signed(_T_938) : $signed(_T_936); // @[Compute.scala 313:30]
  assign _GEN_84 = alu_opcode_max_en ? $signed(_T_940) : $signed(_T_942); // @[Compute.scala 313:30]
  assign _GEN_85 = alu_opcode_max_en ? $signed(_T_942) : $signed(_T_940); // @[Compute.scala 313:30]
  assign _GEN_86 = alu_opcode_max_en ? $signed(_T_944) : $signed(_T_946); // @[Compute.scala 313:30]
  assign _GEN_87 = alu_opcode_max_en ? $signed(_T_946) : $signed(_T_944); // @[Compute.scala 313:30]
  assign _GEN_88 = alu_opcode_max_en ? $signed(_T_948) : $signed(_T_950); // @[Compute.scala 313:30]
  assign _GEN_89 = alu_opcode_max_en ? $signed(_T_950) : $signed(_T_948); // @[Compute.scala 313:30]
  assign _GEN_90 = alu_opcode_max_en ? $signed(_T_952) : $signed(_T_954); // @[Compute.scala 313:30]
  assign _GEN_91 = alu_opcode_max_en ? $signed(_T_954) : $signed(_T_952); // @[Compute.scala 313:30]
  assign _GEN_92 = alu_opcode_max_en ? $signed(_T_956) : $signed(_T_958); // @[Compute.scala 313:30]
  assign _GEN_93 = alu_opcode_max_en ? $signed(_T_958) : $signed(_T_956); // @[Compute.scala 313:30]
  assign _GEN_94 = alu_opcode_max_en ? $signed(_T_960) : $signed(_T_962); // @[Compute.scala 313:30]
  assign _GEN_95 = alu_opcode_max_en ? $signed(_T_962) : $signed(_T_960); // @[Compute.scala 313:30]
  assign _GEN_96 = alu_opcode_max_en ? $signed(_T_964) : $signed(_T_966); // @[Compute.scala 313:30]
  assign _GEN_97 = alu_opcode_max_en ? $signed(_T_966) : $signed(_T_964); // @[Compute.scala 313:30]
  assign _GEN_98 = use_imm ? $signed(imm) : $signed(_GEN_67); // @[Compute.scala 324:20]
  assign _GEN_99 = use_imm ? $signed(imm) : $signed(_GEN_69); // @[Compute.scala 324:20]
  assign _GEN_100 = use_imm ? $signed(imm) : $signed(_GEN_71); // @[Compute.scala 324:20]
  assign _GEN_101 = use_imm ? $signed(imm) : $signed(_GEN_73); // @[Compute.scala 324:20]
  assign _GEN_102 = use_imm ? $signed(imm) : $signed(_GEN_75); // @[Compute.scala 324:20]
  assign _GEN_103 = use_imm ? $signed(imm) : $signed(_GEN_77); // @[Compute.scala 324:20]
  assign _GEN_104 = use_imm ? $signed(imm) : $signed(_GEN_79); // @[Compute.scala 324:20]
  assign _GEN_105 = use_imm ? $signed(imm) : $signed(_GEN_81); // @[Compute.scala 324:20]
  assign _GEN_106 = use_imm ? $signed(imm) : $signed(_GEN_83); // @[Compute.scala 324:20]
  assign _GEN_107 = use_imm ? $signed(imm) : $signed(_GEN_85); // @[Compute.scala 324:20]
  assign _GEN_108 = use_imm ? $signed(imm) : $signed(_GEN_87); // @[Compute.scala 324:20]
  assign _GEN_109 = use_imm ? $signed(imm) : $signed(_GEN_89); // @[Compute.scala 324:20]
  assign _GEN_110 = use_imm ? $signed(imm) : $signed(_GEN_91); // @[Compute.scala 324:20]
  assign _GEN_111 = use_imm ? $signed(imm) : $signed(_GEN_93); // @[Compute.scala 324:20]
  assign _GEN_112 = use_imm ? $signed(imm) : $signed(_GEN_95); // @[Compute.scala 324:20]
  assign _GEN_113 = use_imm ? $signed(imm) : $signed(_GEN_97); // @[Compute.scala 324:20]
  assign src_0_0 = _T_902 ? $signed(_GEN_66) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_0 = _T_902 ? $signed(_GEN_98) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1031 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 329:34]
  assign _T_1032 = _T_1031 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 329:24]
  assign mix_val_0 = _T_902 ? $signed(_T_1032) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1033 = mix_val_0[7:0]; // @[Compute.scala 331:37]
  assign _T_1034 = $unsigned(src_0_0); // @[Compute.scala 332:30]
  assign _T_1035 = $unsigned(src_1_0); // @[Compute.scala 332:59]
  assign _T_1036 = _T_1034 + _T_1035; // @[Compute.scala 332:49]
  assign _T_1037 = _T_1034 + _T_1035; // @[Compute.scala 332:49]
  assign _T_1038 = $signed(_T_1037); // @[Compute.scala 332:79]
  assign add_val_0 = _T_902 ? $signed(_T_1038) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_0 = _T_902 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1039 = add_res_0[7:0]; // @[Compute.scala 334:37]
  assign _T_1041 = src_1_0[4:0]; // @[Compute.scala 335:60]
  assign _T_1042 = _T_1034 >> _T_1041; // @[Compute.scala 335:49]
  assign _T_1043 = $signed(_T_1042); // @[Compute.scala 335:84]
  assign shr_val_0 = _T_902 ? $signed(_T_1043) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_0 = _T_902 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1044 = shr_res_0[7:0]; // @[Compute.scala 337:37]
  assign src_0_1 = _T_902 ? $signed(_GEN_68) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_1 = _T_902 ? $signed(_GEN_99) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1045 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 329:34]
  assign _T_1046 = _T_1045 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 329:24]
  assign mix_val_1 = _T_902 ? $signed(_T_1046) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1047 = mix_val_1[7:0]; // @[Compute.scala 331:37]
  assign _T_1048 = $unsigned(src_0_1); // @[Compute.scala 332:30]
  assign _T_1049 = $unsigned(src_1_1); // @[Compute.scala 332:59]
  assign _T_1050 = _T_1048 + _T_1049; // @[Compute.scala 332:49]
  assign _T_1051 = _T_1048 + _T_1049; // @[Compute.scala 332:49]
  assign _T_1052 = $signed(_T_1051); // @[Compute.scala 332:79]
  assign add_val_1 = _T_902 ? $signed(_T_1052) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_1 = _T_902 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1053 = add_res_1[7:0]; // @[Compute.scala 334:37]
  assign _T_1055 = src_1_1[4:0]; // @[Compute.scala 335:60]
  assign _T_1056 = _T_1048 >> _T_1055; // @[Compute.scala 335:49]
  assign _T_1057 = $signed(_T_1056); // @[Compute.scala 335:84]
  assign shr_val_1 = _T_902 ? $signed(_T_1057) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_1 = _T_902 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1058 = shr_res_1[7:0]; // @[Compute.scala 337:37]
  assign src_0_2 = _T_902 ? $signed(_GEN_70) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_2 = _T_902 ? $signed(_GEN_100) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1059 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 329:34]
  assign _T_1060 = _T_1059 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 329:24]
  assign mix_val_2 = _T_902 ? $signed(_T_1060) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1061 = mix_val_2[7:0]; // @[Compute.scala 331:37]
  assign _T_1062 = $unsigned(src_0_2); // @[Compute.scala 332:30]
  assign _T_1063 = $unsigned(src_1_2); // @[Compute.scala 332:59]
  assign _T_1064 = _T_1062 + _T_1063; // @[Compute.scala 332:49]
  assign _T_1065 = _T_1062 + _T_1063; // @[Compute.scala 332:49]
  assign _T_1066 = $signed(_T_1065); // @[Compute.scala 332:79]
  assign add_val_2 = _T_902 ? $signed(_T_1066) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_2 = _T_902 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1067 = add_res_2[7:0]; // @[Compute.scala 334:37]
  assign _T_1069 = src_1_2[4:0]; // @[Compute.scala 335:60]
  assign _T_1070 = _T_1062 >> _T_1069; // @[Compute.scala 335:49]
  assign _T_1071 = $signed(_T_1070); // @[Compute.scala 335:84]
  assign shr_val_2 = _T_902 ? $signed(_T_1071) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_2 = _T_902 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1072 = shr_res_2[7:0]; // @[Compute.scala 337:37]
  assign src_0_3 = _T_902 ? $signed(_GEN_72) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_3 = _T_902 ? $signed(_GEN_101) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1073 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 329:34]
  assign _T_1074 = _T_1073 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 329:24]
  assign mix_val_3 = _T_902 ? $signed(_T_1074) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1075 = mix_val_3[7:0]; // @[Compute.scala 331:37]
  assign _T_1076 = $unsigned(src_0_3); // @[Compute.scala 332:30]
  assign _T_1077 = $unsigned(src_1_3); // @[Compute.scala 332:59]
  assign _T_1078 = _T_1076 + _T_1077; // @[Compute.scala 332:49]
  assign _T_1079 = _T_1076 + _T_1077; // @[Compute.scala 332:49]
  assign _T_1080 = $signed(_T_1079); // @[Compute.scala 332:79]
  assign add_val_3 = _T_902 ? $signed(_T_1080) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_3 = _T_902 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1081 = add_res_3[7:0]; // @[Compute.scala 334:37]
  assign _T_1083 = src_1_3[4:0]; // @[Compute.scala 335:60]
  assign _T_1084 = _T_1076 >> _T_1083; // @[Compute.scala 335:49]
  assign _T_1085 = $signed(_T_1084); // @[Compute.scala 335:84]
  assign shr_val_3 = _T_902 ? $signed(_T_1085) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_3 = _T_902 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1086 = shr_res_3[7:0]; // @[Compute.scala 337:37]
  assign src_0_4 = _T_902 ? $signed(_GEN_74) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_4 = _T_902 ? $signed(_GEN_102) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1087 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 329:34]
  assign _T_1088 = _T_1087 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 329:24]
  assign mix_val_4 = _T_902 ? $signed(_T_1088) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1089 = mix_val_4[7:0]; // @[Compute.scala 331:37]
  assign _T_1090 = $unsigned(src_0_4); // @[Compute.scala 332:30]
  assign _T_1091 = $unsigned(src_1_4); // @[Compute.scala 332:59]
  assign _T_1092 = _T_1090 + _T_1091; // @[Compute.scala 332:49]
  assign _T_1093 = _T_1090 + _T_1091; // @[Compute.scala 332:49]
  assign _T_1094 = $signed(_T_1093); // @[Compute.scala 332:79]
  assign add_val_4 = _T_902 ? $signed(_T_1094) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_4 = _T_902 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1095 = add_res_4[7:0]; // @[Compute.scala 334:37]
  assign _T_1097 = src_1_4[4:0]; // @[Compute.scala 335:60]
  assign _T_1098 = _T_1090 >> _T_1097; // @[Compute.scala 335:49]
  assign _T_1099 = $signed(_T_1098); // @[Compute.scala 335:84]
  assign shr_val_4 = _T_902 ? $signed(_T_1099) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_4 = _T_902 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1100 = shr_res_4[7:0]; // @[Compute.scala 337:37]
  assign src_0_5 = _T_902 ? $signed(_GEN_76) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_5 = _T_902 ? $signed(_GEN_103) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1101 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 329:34]
  assign _T_1102 = _T_1101 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 329:24]
  assign mix_val_5 = _T_902 ? $signed(_T_1102) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1103 = mix_val_5[7:0]; // @[Compute.scala 331:37]
  assign _T_1104 = $unsigned(src_0_5); // @[Compute.scala 332:30]
  assign _T_1105 = $unsigned(src_1_5); // @[Compute.scala 332:59]
  assign _T_1106 = _T_1104 + _T_1105; // @[Compute.scala 332:49]
  assign _T_1107 = _T_1104 + _T_1105; // @[Compute.scala 332:49]
  assign _T_1108 = $signed(_T_1107); // @[Compute.scala 332:79]
  assign add_val_5 = _T_902 ? $signed(_T_1108) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_5 = _T_902 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1109 = add_res_5[7:0]; // @[Compute.scala 334:37]
  assign _T_1111 = src_1_5[4:0]; // @[Compute.scala 335:60]
  assign _T_1112 = _T_1104 >> _T_1111; // @[Compute.scala 335:49]
  assign _T_1113 = $signed(_T_1112); // @[Compute.scala 335:84]
  assign shr_val_5 = _T_902 ? $signed(_T_1113) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_5 = _T_902 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1114 = shr_res_5[7:0]; // @[Compute.scala 337:37]
  assign src_0_6 = _T_902 ? $signed(_GEN_78) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_6 = _T_902 ? $signed(_GEN_104) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1115 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 329:34]
  assign _T_1116 = _T_1115 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 329:24]
  assign mix_val_6 = _T_902 ? $signed(_T_1116) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1117 = mix_val_6[7:0]; // @[Compute.scala 331:37]
  assign _T_1118 = $unsigned(src_0_6); // @[Compute.scala 332:30]
  assign _T_1119 = $unsigned(src_1_6); // @[Compute.scala 332:59]
  assign _T_1120 = _T_1118 + _T_1119; // @[Compute.scala 332:49]
  assign _T_1121 = _T_1118 + _T_1119; // @[Compute.scala 332:49]
  assign _T_1122 = $signed(_T_1121); // @[Compute.scala 332:79]
  assign add_val_6 = _T_902 ? $signed(_T_1122) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_6 = _T_902 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1123 = add_res_6[7:0]; // @[Compute.scala 334:37]
  assign _T_1125 = src_1_6[4:0]; // @[Compute.scala 335:60]
  assign _T_1126 = _T_1118 >> _T_1125; // @[Compute.scala 335:49]
  assign _T_1127 = $signed(_T_1126); // @[Compute.scala 335:84]
  assign shr_val_6 = _T_902 ? $signed(_T_1127) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_6 = _T_902 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1128 = shr_res_6[7:0]; // @[Compute.scala 337:37]
  assign src_0_7 = _T_902 ? $signed(_GEN_80) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_7 = _T_902 ? $signed(_GEN_105) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1129 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 329:34]
  assign _T_1130 = _T_1129 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 329:24]
  assign mix_val_7 = _T_902 ? $signed(_T_1130) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1131 = mix_val_7[7:0]; // @[Compute.scala 331:37]
  assign _T_1132 = $unsigned(src_0_7); // @[Compute.scala 332:30]
  assign _T_1133 = $unsigned(src_1_7); // @[Compute.scala 332:59]
  assign _T_1134 = _T_1132 + _T_1133; // @[Compute.scala 332:49]
  assign _T_1135 = _T_1132 + _T_1133; // @[Compute.scala 332:49]
  assign _T_1136 = $signed(_T_1135); // @[Compute.scala 332:79]
  assign add_val_7 = _T_902 ? $signed(_T_1136) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_7 = _T_902 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1137 = add_res_7[7:0]; // @[Compute.scala 334:37]
  assign _T_1139 = src_1_7[4:0]; // @[Compute.scala 335:60]
  assign _T_1140 = _T_1132 >> _T_1139; // @[Compute.scala 335:49]
  assign _T_1141 = $signed(_T_1140); // @[Compute.scala 335:84]
  assign shr_val_7 = _T_902 ? $signed(_T_1141) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_7 = _T_902 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1142 = shr_res_7[7:0]; // @[Compute.scala 337:37]
  assign src_0_8 = _T_902 ? $signed(_GEN_82) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_8 = _T_902 ? $signed(_GEN_106) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1143 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 329:34]
  assign _T_1144 = _T_1143 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 329:24]
  assign mix_val_8 = _T_902 ? $signed(_T_1144) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1145 = mix_val_8[7:0]; // @[Compute.scala 331:37]
  assign _T_1146 = $unsigned(src_0_8); // @[Compute.scala 332:30]
  assign _T_1147 = $unsigned(src_1_8); // @[Compute.scala 332:59]
  assign _T_1148 = _T_1146 + _T_1147; // @[Compute.scala 332:49]
  assign _T_1149 = _T_1146 + _T_1147; // @[Compute.scala 332:49]
  assign _T_1150 = $signed(_T_1149); // @[Compute.scala 332:79]
  assign add_val_8 = _T_902 ? $signed(_T_1150) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_8 = _T_902 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1151 = add_res_8[7:0]; // @[Compute.scala 334:37]
  assign _T_1153 = src_1_8[4:0]; // @[Compute.scala 335:60]
  assign _T_1154 = _T_1146 >> _T_1153; // @[Compute.scala 335:49]
  assign _T_1155 = $signed(_T_1154); // @[Compute.scala 335:84]
  assign shr_val_8 = _T_902 ? $signed(_T_1155) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_8 = _T_902 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1156 = shr_res_8[7:0]; // @[Compute.scala 337:37]
  assign src_0_9 = _T_902 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_9 = _T_902 ? $signed(_GEN_107) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1157 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 329:34]
  assign _T_1158 = _T_1157 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 329:24]
  assign mix_val_9 = _T_902 ? $signed(_T_1158) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1159 = mix_val_9[7:0]; // @[Compute.scala 331:37]
  assign _T_1160 = $unsigned(src_0_9); // @[Compute.scala 332:30]
  assign _T_1161 = $unsigned(src_1_9); // @[Compute.scala 332:59]
  assign _T_1162 = _T_1160 + _T_1161; // @[Compute.scala 332:49]
  assign _T_1163 = _T_1160 + _T_1161; // @[Compute.scala 332:49]
  assign _T_1164 = $signed(_T_1163); // @[Compute.scala 332:79]
  assign add_val_9 = _T_902 ? $signed(_T_1164) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_9 = _T_902 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1165 = add_res_9[7:0]; // @[Compute.scala 334:37]
  assign _T_1167 = src_1_9[4:0]; // @[Compute.scala 335:60]
  assign _T_1168 = _T_1160 >> _T_1167; // @[Compute.scala 335:49]
  assign _T_1169 = $signed(_T_1168); // @[Compute.scala 335:84]
  assign shr_val_9 = _T_902 ? $signed(_T_1169) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_9 = _T_902 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1170 = shr_res_9[7:0]; // @[Compute.scala 337:37]
  assign src_0_10 = _T_902 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_10 = _T_902 ? $signed(_GEN_108) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1171 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 329:34]
  assign _T_1172 = _T_1171 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 329:24]
  assign mix_val_10 = _T_902 ? $signed(_T_1172) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1173 = mix_val_10[7:0]; // @[Compute.scala 331:37]
  assign _T_1174 = $unsigned(src_0_10); // @[Compute.scala 332:30]
  assign _T_1175 = $unsigned(src_1_10); // @[Compute.scala 332:59]
  assign _T_1176 = _T_1174 + _T_1175; // @[Compute.scala 332:49]
  assign _T_1177 = _T_1174 + _T_1175; // @[Compute.scala 332:49]
  assign _T_1178 = $signed(_T_1177); // @[Compute.scala 332:79]
  assign add_val_10 = _T_902 ? $signed(_T_1178) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_10 = _T_902 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1179 = add_res_10[7:0]; // @[Compute.scala 334:37]
  assign _T_1181 = src_1_10[4:0]; // @[Compute.scala 335:60]
  assign _T_1182 = _T_1174 >> _T_1181; // @[Compute.scala 335:49]
  assign _T_1183 = $signed(_T_1182); // @[Compute.scala 335:84]
  assign shr_val_10 = _T_902 ? $signed(_T_1183) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_10 = _T_902 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1184 = shr_res_10[7:0]; // @[Compute.scala 337:37]
  assign src_0_11 = _T_902 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_11 = _T_902 ? $signed(_GEN_109) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1185 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 329:34]
  assign _T_1186 = _T_1185 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 329:24]
  assign mix_val_11 = _T_902 ? $signed(_T_1186) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1187 = mix_val_11[7:0]; // @[Compute.scala 331:37]
  assign _T_1188 = $unsigned(src_0_11); // @[Compute.scala 332:30]
  assign _T_1189 = $unsigned(src_1_11); // @[Compute.scala 332:59]
  assign _T_1190 = _T_1188 + _T_1189; // @[Compute.scala 332:49]
  assign _T_1191 = _T_1188 + _T_1189; // @[Compute.scala 332:49]
  assign _T_1192 = $signed(_T_1191); // @[Compute.scala 332:79]
  assign add_val_11 = _T_902 ? $signed(_T_1192) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_11 = _T_902 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1193 = add_res_11[7:0]; // @[Compute.scala 334:37]
  assign _T_1195 = src_1_11[4:0]; // @[Compute.scala 335:60]
  assign _T_1196 = _T_1188 >> _T_1195; // @[Compute.scala 335:49]
  assign _T_1197 = $signed(_T_1196); // @[Compute.scala 335:84]
  assign shr_val_11 = _T_902 ? $signed(_T_1197) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_11 = _T_902 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1198 = shr_res_11[7:0]; // @[Compute.scala 337:37]
  assign src_0_12 = _T_902 ? $signed(_GEN_90) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_12 = _T_902 ? $signed(_GEN_110) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1199 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 329:34]
  assign _T_1200 = _T_1199 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 329:24]
  assign mix_val_12 = _T_902 ? $signed(_T_1200) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1201 = mix_val_12[7:0]; // @[Compute.scala 331:37]
  assign _T_1202 = $unsigned(src_0_12); // @[Compute.scala 332:30]
  assign _T_1203 = $unsigned(src_1_12); // @[Compute.scala 332:59]
  assign _T_1204 = _T_1202 + _T_1203; // @[Compute.scala 332:49]
  assign _T_1205 = _T_1202 + _T_1203; // @[Compute.scala 332:49]
  assign _T_1206 = $signed(_T_1205); // @[Compute.scala 332:79]
  assign add_val_12 = _T_902 ? $signed(_T_1206) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_12 = _T_902 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1207 = add_res_12[7:0]; // @[Compute.scala 334:37]
  assign _T_1209 = src_1_12[4:0]; // @[Compute.scala 335:60]
  assign _T_1210 = _T_1202 >> _T_1209; // @[Compute.scala 335:49]
  assign _T_1211 = $signed(_T_1210); // @[Compute.scala 335:84]
  assign shr_val_12 = _T_902 ? $signed(_T_1211) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_12 = _T_902 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1212 = shr_res_12[7:0]; // @[Compute.scala 337:37]
  assign src_0_13 = _T_902 ? $signed(_GEN_92) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_13 = _T_902 ? $signed(_GEN_111) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1213 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 329:34]
  assign _T_1214 = _T_1213 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 329:24]
  assign mix_val_13 = _T_902 ? $signed(_T_1214) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1215 = mix_val_13[7:0]; // @[Compute.scala 331:37]
  assign _T_1216 = $unsigned(src_0_13); // @[Compute.scala 332:30]
  assign _T_1217 = $unsigned(src_1_13); // @[Compute.scala 332:59]
  assign _T_1218 = _T_1216 + _T_1217; // @[Compute.scala 332:49]
  assign _T_1219 = _T_1216 + _T_1217; // @[Compute.scala 332:49]
  assign _T_1220 = $signed(_T_1219); // @[Compute.scala 332:79]
  assign add_val_13 = _T_902 ? $signed(_T_1220) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_13 = _T_902 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1221 = add_res_13[7:0]; // @[Compute.scala 334:37]
  assign _T_1223 = src_1_13[4:0]; // @[Compute.scala 335:60]
  assign _T_1224 = _T_1216 >> _T_1223; // @[Compute.scala 335:49]
  assign _T_1225 = $signed(_T_1224); // @[Compute.scala 335:84]
  assign shr_val_13 = _T_902 ? $signed(_T_1225) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_13 = _T_902 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1226 = shr_res_13[7:0]; // @[Compute.scala 337:37]
  assign src_0_14 = _T_902 ? $signed(_GEN_94) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_14 = _T_902 ? $signed(_GEN_112) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1227 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 329:34]
  assign _T_1228 = _T_1227 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 329:24]
  assign mix_val_14 = _T_902 ? $signed(_T_1228) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1229 = mix_val_14[7:0]; // @[Compute.scala 331:37]
  assign _T_1230 = $unsigned(src_0_14); // @[Compute.scala 332:30]
  assign _T_1231 = $unsigned(src_1_14); // @[Compute.scala 332:59]
  assign _T_1232 = _T_1230 + _T_1231; // @[Compute.scala 332:49]
  assign _T_1233 = _T_1230 + _T_1231; // @[Compute.scala 332:49]
  assign _T_1234 = $signed(_T_1233); // @[Compute.scala 332:79]
  assign add_val_14 = _T_902 ? $signed(_T_1234) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_14 = _T_902 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1235 = add_res_14[7:0]; // @[Compute.scala 334:37]
  assign _T_1237 = src_1_14[4:0]; // @[Compute.scala 335:60]
  assign _T_1238 = _T_1230 >> _T_1237; // @[Compute.scala 335:49]
  assign _T_1239 = $signed(_T_1238); // @[Compute.scala 335:84]
  assign shr_val_14 = _T_902 ? $signed(_T_1239) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_14 = _T_902 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1240 = shr_res_14[7:0]; // @[Compute.scala 337:37]
  assign src_0_15 = _T_902 ? $signed(_GEN_96) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign src_1_15 = _T_902 ? $signed(_GEN_113) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1241 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 329:34]
  assign _T_1242 = _T_1241 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 329:24]
  assign mix_val_15 = _T_902 ? $signed(_T_1242) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1243 = mix_val_15[7:0]; // @[Compute.scala 331:37]
  assign _T_1244 = $unsigned(src_0_15); // @[Compute.scala 332:30]
  assign _T_1245 = $unsigned(src_1_15); // @[Compute.scala 332:59]
  assign _T_1246 = _T_1244 + _T_1245; // @[Compute.scala 332:49]
  assign _T_1247 = _T_1244 + _T_1245; // @[Compute.scala 332:49]
  assign _T_1248 = $signed(_T_1247); // @[Compute.scala 332:79]
  assign add_val_15 = _T_902 ? $signed(_T_1248) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign add_res_15 = _T_902 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1249 = add_res_15[7:0]; // @[Compute.scala 334:37]
  assign _T_1251 = src_1_15[4:0]; // @[Compute.scala 335:60]
  assign _T_1252 = _T_1244 >> _T_1251; // @[Compute.scala 335:49]
  assign _T_1253 = $signed(_T_1252); // @[Compute.scala 335:84]
  assign shr_val_15 = _T_902 ? $signed(_T_1253) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign shr_res_15 = _T_902 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 312:36]
  assign _T_1254 = shr_res_15[7:0]; // @[Compute.scala 337:37]
  assign short_cmp_res_0 = _T_902 ? _T_1033 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_0 = _T_902 ? _T_1039 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_0 = _T_902 ? _T_1044 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_1 = _T_902 ? _T_1047 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_1 = _T_902 ? _T_1053 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_1 = _T_902 ? _T_1058 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_2 = _T_902 ? _T_1061 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_2 = _T_902 ? _T_1067 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_2 = _T_902 ? _T_1072 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_3 = _T_902 ? _T_1075 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_3 = _T_902 ? _T_1081 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_3 = _T_902 ? _T_1086 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_4 = _T_902 ? _T_1089 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_4 = _T_902 ? _T_1095 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_4 = _T_902 ? _T_1100 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_5 = _T_902 ? _T_1103 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_5 = _T_902 ? _T_1109 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_5 = _T_902 ? _T_1114 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_6 = _T_902 ? _T_1117 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_6 = _T_902 ? _T_1123 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_6 = _T_902 ? _T_1128 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_7 = _T_902 ? _T_1131 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_7 = _T_902 ? _T_1137 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_7 = _T_902 ? _T_1142 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_8 = _T_902 ? _T_1145 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_8 = _T_902 ? _T_1151 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_8 = _T_902 ? _T_1156 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_9 = _T_902 ? _T_1159 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_9 = _T_902 ? _T_1165 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_9 = _T_902 ? _T_1170 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_10 = _T_902 ? _T_1173 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_10 = _T_902 ? _T_1179 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_10 = _T_902 ? _T_1184 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_11 = _T_902 ? _T_1187 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_11 = _T_902 ? _T_1193 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_11 = _T_902 ? _T_1198 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_12 = _T_902 ? _T_1201 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_12 = _T_902 ? _T_1207 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_12 = _T_902 ? _T_1212 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_13 = _T_902 ? _T_1215 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_13 = _T_902 ? _T_1221 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_13 = _T_902 ? _T_1226 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_14 = _T_902 ? _T_1229 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_14 = _T_902 ? _T_1235 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_14 = _T_902 ? _T_1240 : 8'h0; // @[Compute.scala 312:36]
  assign short_cmp_res_15 = _T_902 ? _T_1243 : 8'h0; // @[Compute.scala 312:36]
  assign short_add_res_15 = _T_902 ? _T_1249 : 8'h0; // @[Compute.scala 312:36]
  assign short_shr_res_15 = _T_902 ? _T_1254 : 8'h0; // @[Compute.scala 312:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 342:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 343:39]
  assign _T_1264 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1272 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1264}; // @[Cat.scala 30:58]
  assign _T_1279 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1287 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1279}; // @[Cat.scala 30:58]
  assign _T_1294 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1302 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1294}; // @[Cat.scala 30:58]
  assign _T_1303 = alu_opcode_add_en ? _T_1287 : _T_1302; // @[Compute.scala 348:29]
  assign out_mem_enq_bits = alu_opcode_minmax_en ? _T_1272 : _T_1303; // @[Compute.scala 347:29]
  assign _T_1304 = opcode_alu_en & busy; // @[Compute.scala 349:34]
  assign _T_1305 = _GEN_323 <= out_cntr_max_val; // @[Compute.scala 349:59]
  assign _T_1306 = _T_1304 & _T_1305; // @[Compute.scala 349:42]
  assign _T_1308 = out_cntr_val >= 16'h2; // @[Compute.scala 351:63]
  assign _T_1309 = out_mem_write & _T_1308; // @[Compute.scala 351:46]
  assign _T_1311 = out_cntr_max - 44'h1; // @[Compute.scala 351:105]
  assign _T_1312 = $unsigned(_T_1311); // @[Compute.scala 351:105]
  assign _T_1313 = _T_1312[43:0]; // @[Compute.scala 351:105]
  assign _T_1314 = _GEN_323 <= _T_1313; // @[Compute.scala 351:88]
  assign _T_1318 = {_T_1317,out_mem_enq_bits}; // @[Cat.scala 30:58]
  assign _T_1319 = out_mem_fifo_io_deq_bits[159:128]; // @[Compute.scala 360:49]
  assign _GEN_343 = {{7'd0}, _T_1319}; // @[Compute.scala 360:66]
  assign _T_1321 = _GEN_343 << 3'h4; // @[Compute.scala 360:66]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 220:23]
  assign io_done_readdata = opcode == 3'h3; // @[Compute.scala 223:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 231:19]
  assign io_uops_read = uops_read; // @[Compute.scala 230:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 128'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 246:21]
  assign io_biases_read = biases_read; // @[Compute.scala 247:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = gemm_queue_ready; // @[Compute.scala 216:23]
  assign io_l2g_dep_queue_ready = pop_prev_dep_ready & dump; // @[Compute.scala 166:26]
  assign io_s2g_dep_queue_ready = pop_next_dep_ready & dump; // @[Compute.scala 167:26]
  assign io_g2l_dep_queue_valid = push_prev_dep & push; // @[Compute.scala 178:26]
  assign io_g2l_dep_queue_data = 1'h1; // @[Compute.scala 176:25]
  assign io_g2s_dep_queue_valid = push_next_dep & push; // @[Compute.scala 179:26]
  assign io_g2s_dep_queue_data = 1'h1; // @[Compute.scala 177:25]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = _T_1321[16:0]; // @[Compute.scala 360:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_fifo_io_deq_valid; // @[Compute.scala 361:20]
  assign io_out_mem_writedata = out_mem_fifo_io_deq_bits[127:0]; // @[Compute.scala 364:24]
  assign out_mem_fifo_clock = clock;
  assign out_mem_fifo_reset = reset;
  assign out_mem_fifo_io_enq_valid = _T_1309 & _T_1314; // @[Compute.scala 351:29]
  assign out_mem_fifo_io_enq_bits = {{16'd0}, _T_1318}; // @[Compute.scala 352:28]
  assign out_mem_fifo_io_deq_ready = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 362:29]
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
  _RAND_22 = {1{`RANDOM}};
  uop = _RAND_22[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_23 = {1{`RANDOM}};
  _T_453 = _RAND_23[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_24 = {16{`RANDOM}};
  dst_vector = _RAND_24[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_25 = {16{`RANDOM}};
  src_vector = _RAND_25[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_26 = {1{`RANDOM}};
  _T_1317 = _RAND_26[15:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_445_en & acc_mem__T_445_mask) begin
      acc_mem[acc_mem__T_445_addr] <= acc_mem__T_445_data; // @[Compute.scala 34:20]
    end
    if(uop_mem__T_378_en & uop_mem__T_378_mask) begin
      uop_mem[uop_mem__T_378_addr] <= uop_mem__T_378_data; // @[Compute.scala 35:20]
    end
    if(uop_mem__T_384_en & uop_mem__T_384_mask) begin
      uop_mem[uop_mem__T_384_addr] <= uop_mem__T_384_data; // @[Compute.scala 35:20]
    end
    if(uop_mem__T_390_en & uop_mem__T_390_mask) begin
      uop_mem[uop_mem__T_390_addr] <= uop_mem__T_390_data; // @[Compute.scala 35:20]
    end
    if(uop_mem__T_396_en & uop_mem__T_396_mask) begin
      uop_mem[uop_mem__T_396_addr] <= uop_mem__T_396_data; // @[Compute.scala 35:20]
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
        if (_T_307) begin
          state <= 3'h4;
        end else begin
          if (_T_305) begin
            state <= 3'h2;
          end else begin
            if (_T_303) begin
              state <= 3'h1;
            end else begin
              if (_T_294) begin
                if (_T_295) begin
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
    if (_T_326) begin
      if (_T_402) begin
        uops_read <= 1'h0;
      end else begin
        uops_read <= _T_370;
      end
    end else begin
      uops_read <= _T_370;
    end
    biases_read <= acc_cntr_en & _T_431;
    if (_T_335) begin
      if (3'h0 == _T_437) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_335) begin
      if (3'h1 == _T_437) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_335) begin
      if (3'h2 == _T_437) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_335) begin
      if (3'h3 == _T_437) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    if (reset) begin
      out_mem_write <= 1'h0;
    end else begin
      out_mem_write <= _T_1306;
    end
    if (gemm_queue_ready) begin
      uop_cntr_val <= 16'h0;
    end else begin
      if (_T_329) begin
        uop_cntr_val <= _T_332;
      end
    end
    if (gemm_queue_ready) begin
      acc_cntr_val <= 16'h0;
    end else begin
      if (_T_338) begin
        acc_cntr_val <= _T_341;
      end
    end
    if (gemm_queue_ready) begin
      out_cntr_val <= 16'h0;
    end else begin
      if (_T_344) begin
        out_cntr_val <= _T_347;
      end
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_311) begin
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
        if (_T_314) begin
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
        if (_T_319) begin
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
        if (_T_322) begin
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
        gemm_queue_ready <= _T_356;
      end
    end
    if (reset) begin
      finish_wrap <= 1'h0;
    end else begin
      if (opcode_finish_en) begin
        if (pop_prev_dep) begin
          finish_wrap <= _T_286;
        end else begin
          if (pop_next_dep) begin
            finish_wrap <= _T_287;
          end else begin
            if (push_prev_dep) begin
              finish_wrap <= _T_288;
            end else begin
              if (push_next_dep) begin
                finish_wrap <= _T_289;
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
    uop <= uop_mem__T_450_data;
    _T_453 <= out_cntr_val;
    if (out_mem_write) begin
      dst_vector <= acc_mem__T_468_data;
    end
    if (out_mem_write) begin
      src_vector <= acc_mem__T_470_data;
    end
    _T_1317 <= _GEN_339 + dst_offset_in;
  end
endmodule
