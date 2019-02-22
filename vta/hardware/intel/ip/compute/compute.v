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
  wire [511:0] acc_mem__T_726_data; // @[Compute.scala 34:20]
  wire [7:0] acc_mem__T_726_addr; // @[Compute.scala 34:20]
  wire [511:0] acc_mem__T_728_data; // @[Compute.scala 34:20]
  wire [7:0] acc_mem__T_728_addr; // @[Compute.scala 34:20]
  wire [511:0] acc_mem__T_458_data; // @[Compute.scala 34:20]
  wire [7:0] acc_mem__T_458_addr; // @[Compute.scala 34:20]
  wire  acc_mem__T_458_mask; // @[Compute.scala 34:20]
  wire  acc_mem__T_458_en; // @[Compute.scala 34:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 35:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem__T_463_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_463_addr; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_397_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_397_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_397_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_397_en; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_403_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_403_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_403_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_403_en; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_409_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_409_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_409_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_409_en; // @[Compute.scala 35:20]
  wire [31:0] uop_mem__T_415_data; // @[Compute.scala 35:20]
  wire [9:0] uop_mem__T_415_addr; // @[Compute.scala 35:20]
  wire  uop_mem__T_415_mask; // @[Compute.scala 35:20]
  wire  uop_mem__T_415_en; // @[Compute.scala 35:20]
  wire  out_mem_fifo_clock; // @[Compute.scala 348:28]
  wire  out_mem_fifo_reset; // @[Compute.scala 348:28]
  wire  out_mem_fifo_io_enq_ready; // @[Compute.scala 348:28]
  wire  out_mem_fifo_io_enq_valid; // @[Compute.scala 348:28]
  wire [159:0] out_mem_fifo_io_enq_bits; // @[Compute.scala 348:28]
  wire  out_mem_fifo_io_deq_ready; // @[Compute.scala 348:28]
  wire  out_mem_fifo_io_deq_valid; // @[Compute.scala 348:28]
  wire [159:0] out_mem_fifo_io_deq_bits; // @[Compute.scala 348:28]
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
  reg [15:0] imm_raw; // @[Compute.scala 72:24]
  reg [31:0] _RAND_5;
  wire [15:0] _T_208; // @[Compute.scala 73:33]
  wire  _T_210; // @[Compute.scala 73:40]
  wire [31:0] _T_212; // @[Cat.scala 30:58]
  wire [16:0] _T_214; // @[Cat.scala 30:58]
  wire [31:0] _T_215; // @[Compute.scala 73:24]
  reg [31:0] imm; // @[Compute.scala 73:20]
  reg [31:0] _RAND_6;
  wire [15:0] _GEN_286; // @[Compute.scala 77:30]
  wire [15:0] _GEN_288; // @[Compute.scala 78:30]
  wire [16:0] _T_221; // @[Compute.scala 78:30]
  wire [15:0] _T_222; // @[Compute.scala 78:30]
  wire [15:0] _GEN_289; // @[Compute.scala 78:39]
  wire [16:0] _T_223; // @[Compute.scala 78:39]
  wire [15:0] x_size_total; // @[Compute.scala 78:39]
  wire [19:0] y_offset; // @[Compute.scala 79:31]
  wire  opcode_finish_en; // @[Compute.scala 82:34]
  wire  _T_226; // @[Compute.scala 83:32]
  wire  _T_228; // @[Compute.scala 83:60]
  wire  opcode_load_en; // @[Compute.scala 83:50]
  wire  opcode_gemm_en; // @[Compute.scala 84:32]
  wire  opcode_alu_en; // @[Compute.scala 85:31]
  wire  memory_type_uop_en; // @[Compute.scala 87:40]
  wire  memory_type_acc_en; // @[Compute.scala 88:40]
  reg [2:0] state; // @[Compute.scala 91:22]
  reg [31:0] _RAND_7;
  wire  idle; // @[Compute.scala 93:20]
  wire  dump; // @[Compute.scala 94:20]
  wire  busy; // @[Compute.scala 95:20]
  wire  push; // @[Compute.scala 96:20]
  wire  done; // @[Compute.scala 97:20]
  reg  uops_read; // @[Compute.scala 100:24]
  reg [31:0] _RAND_8;
  reg [159:0] uops_data; // @[Compute.scala 101:24]
  reg [159:0] _RAND_9;
  reg  biases_read; // @[Compute.scala 103:24]
  reg [31:0] _RAND_10;
  reg [127:0] biases_data_0; // @[Compute.scala 106:24]
  reg [127:0] _RAND_11;
  reg [127:0] biases_data_1; // @[Compute.scala 106:24]
  reg [127:0] _RAND_12;
  reg [127:0] biases_data_2; // @[Compute.scala 106:24]
  reg [127:0] _RAND_13;
  reg [127:0] biases_data_3; // @[Compute.scala 106:24]
  reg [127:0] _RAND_14;
  reg  out_mem_write; // @[Compute.scala 108:31]
  reg [31:0] _RAND_15;
  wire [15:0] uop_cntr_max_val; // @[Compute.scala 111:33]
  wire  _T_252; // @[Compute.scala 112:43]
  wire [15:0] uop_cntr_max; // @[Compute.scala 112:25]
  wire  _T_254; // @[Compute.scala 113:37]
  wire  uop_cntr_en; // @[Compute.scala 113:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 115:25]
  reg [31:0] _RAND_16;
  wire  _T_256; // @[Compute.scala 116:38]
  wire  _T_257; // @[Compute.scala 116:56]
  wire  uop_cntr_wrap; // @[Compute.scala 116:71]
  wire [18:0] _T_259; // @[Compute.scala 118:29]
  wire [19:0] _T_261; // @[Compute.scala 118:46]
  wire [18:0] acc_cntr_max; // @[Compute.scala 118:46]
  wire  _T_262; // @[Compute.scala 119:37]
  wire  acc_cntr_en; // @[Compute.scala 119:59]
  reg [15:0] acc_cntr_val; // @[Compute.scala 121:25]
  reg [31:0] _RAND_17;
  wire [18:0] _GEN_291; // @[Compute.scala 122:38]
  wire  _T_264; // @[Compute.scala 122:38]
  wire  _T_265; // @[Compute.scala 122:56]
  wire  acc_cntr_wrap; // @[Compute.scala 122:71]
  wire [16:0] _T_266; // @[Compute.scala 124:34]
  wire [16:0] _T_267; // @[Compute.scala 124:34]
  wire [15:0] upc_cntr_max_val; // @[Compute.scala 124:34]
  wire  _T_269; // @[Compute.scala 125:43]
  wire [15:0] upc_cntr_max; // @[Compute.scala 125:25]
  wire [27:0] _T_271; // @[Compute.scala 126:35]
  wire [15:0] _T_272; // @[Compute.scala 126:46]
  wire [31:0] out_cntr_max_val; // @[Compute.scala 126:54]
  wire [32:0] _T_274; // @[Compute.scala 127:39]
  wire [31:0] out_cntr_max; // @[Compute.scala 127:39]
  wire  _T_275; // @[Compute.scala 128:37]
  wire  out_cntr_en; // @[Compute.scala 128:56]
  reg [15:0] out_cntr_val; // @[Compute.scala 130:25]
  reg [31:0] _RAND_18;
  wire [31:0] _GEN_292; // @[Compute.scala 131:38]
  wire  _T_277; // @[Compute.scala 131:38]
  wire  _T_278; // @[Compute.scala 131:56]
  wire  out_cntr_wrap; // @[Compute.scala 131:71]
  reg  pop_prev_dep_ready; // @[Compute.scala 134:35]
  reg [31:0] _RAND_19;
  reg  pop_next_dep_ready; // @[Compute.scala 135:35]
  reg [31:0] _RAND_20;
  wire  push_prev_dep_valid; // @[Compute.scala 136:43]
  wire  push_next_dep_valid; // @[Compute.scala 137:43]
  reg  push_prev_dep_ready; // @[Compute.scala 138:36]
  reg [31:0] _RAND_21;
  reg  push_next_dep_ready; // @[Compute.scala 139:36]
  reg [31:0] _RAND_22;
  reg  gemm_queue_ready; // @[Compute.scala 141:33]
  reg [31:0] _RAND_23;
  reg  finish_wrap; // @[Compute.scala 144:28]
  reg [31:0] _RAND_24;
  wire  _T_291; // @[Compute.scala 146:68]
  wire  _T_292; // @[Compute.scala 147:68]
  wire  _T_293; // @[Compute.scala 148:68]
  wire  _T_294; // @[Compute.scala 149:68]
  wire  _GEN_0; // @[Compute.scala 149:31]
  wire  _GEN_1; // @[Compute.scala 148:31]
  wire  _GEN_2; // @[Compute.scala 147:31]
  wire  _GEN_3; // @[Compute.scala 146:31]
  wire  _GEN_4; // @[Compute.scala 145:27]
  wire  _T_297; // @[Compute.scala 152:23]
  wire  _T_298; // @[Compute.scala 152:40]
  wire  _T_299; // @[Compute.scala 152:57]
  wire  _T_300; // @[Compute.scala 153:25]
  wire [2:0] _GEN_5; // @[Compute.scala 153:43]
  wire [2:0] _GEN_6; // @[Compute.scala 152:73]
  wire  _T_302; // @[Compute.scala 161:18]
  wire  _T_304; // @[Compute.scala 161:41]
  wire  _T_305; // @[Compute.scala 161:38]
  wire  _T_306; // @[Compute.scala 161:14]
  wire  _T_307; // @[Compute.scala 161:79]
  wire  _T_308; // @[Compute.scala 161:62]
  wire [2:0] _GEN_7; // @[Compute.scala 161:97]
  wire  _T_309; // @[Compute.scala 162:38]
  wire  _T_310; // @[Compute.scala 162:14]
  wire [2:0] _GEN_8; // @[Compute.scala 162:63]
  wire  _T_311; // @[Compute.scala 163:38]
  wire  _T_312; // @[Compute.scala 163:14]
  wire [2:0] _GEN_9; // @[Compute.scala 163:63]
  wire  _T_315; // @[Compute.scala 170:22]
  wire  _T_316; // @[Compute.scala 170:30]
  wire  _GEN_10; // @[Compute.scala 170:57]
  wire  _T_318; // @[Compute.scala 173:22]
  wire  _T_319; // @[Compute.scala 173:30]
  wire  _GEN_11; // @[Compute.scala 173:57]
  wire  _T_323; // @[Compute.scala 180:29]
  wire  _T_324; // @[Compute.scala 180:55]
  wire  _GEN_12; // @[Compute.scala 180:64]
  wire  _T_326; // @[Compute.scala 183:29]
  wire  _T_327; // @[Compute.scala 183:55]
  wire  _GEN_13; // @[Compute.scala 183:64]
  wire  _T_330; // @[Compute.scala 188:22]
  wire  _T_331; // @[Compute.scala 188:19]
  wire  _T_332; // @[Compute.scala 188:37]
  wire  _T_333; // @[Compute.scala 188:61]
  wire  _T_334; // @[Compute.scala 188:45]
  wire [16:0] _T_336; // @[Compute.scala 189:34]
  wire [15:0] _T_337; // @[Compute.scala 189:34]
  wire [15:0] _GEN_14; // @[Compute.scala 188:77]
  wire  _T_339; // @[Compute.scala 191:24]
  wire  _T_340; // @[Compute.scala 191:21]
  wire  _T_341; // @[Compute.scala 191:39]
  wire  _T_342; // @[Compute.scala 191:63]
  wire  _T_343; // @[Compute.scala 191:47]
  wire [16:0] _T_345; // @[Compute.scala 192:34]
  wire [15:0] _T_346; // @[Compute.scala 192:34]
  wire [15:0] _GEN_15; // @[Compute.scala 191:79]
  wire  _T_347; // @[Compute.scala 197:23]
  wire  _T_348; // @[Compute.scala 197:47]
  wire  _T_349; // @[Compute.scala 197:31]
  wire [16:0] _T_351; // @[Compute.scala 198:34]
  wire [15:0] _T_352; // @[Compute.scala 198:34]
  wire [15:0] _GEN_16; // @[Compute.scala 197:63]
  wire  _GEN_21; // @[Compute.scala 202:27]
  wire  _GEN_22; // @[Compute.scala 202:27]
  wire  _GEN_23; // @[Compute.scala 202:27]
  wire  _GEN_24; // @[Compute.scala 202:27]
  wire [2:0] _GEN_25; // @[Compute.scala 202:27]
  wire  _T_360; // @[Compute.scala 215:52]
  wire  _T_361; // @[Compute.scala 215:43]
  wire  _GEN_26; // @[Compute.scala 217:27]
  wire [31:0] _GEN_295; // @[Compute.scala 227:33]
  wire [32:0] _T_366; // @[Compute.scala 227:33]
  wire [31:0] _T_367; // @[Compute.scala 227:33]
  wire [38:0] _GEN_296; // @[Compute.scala 227:49]
  wire [38:0] uop_dram_addr; // @[Compute.scala 227:49]
  wire [16:0] _T_371; // @[Compute.scala 230:30]
  wire [15:0] _T_372; // @[Compute.scala 230:30]
  wire [18:0] _GEN_297; // @[Compute.scala 230:46]
  wire [18:0] _T_374; // @[Compute.scala 230:46]
  wire  _T_376; // @[Compute.scala 231:31]
  wire  _T_377; // @[Compute.scala 231:28]
  wire  _T_378; // @[Compute.scala 231:46]
  reg  uops_read_en; // @[Compute.scala 236:25]
  reg [31:0] _RAND_25;
  wire [31:0] uop_sram_addr; // @[Compute.scala 228:27 Compute.scala 229:17 Compute.scala 230:17]
  wire [159:0] _T_383; // @[Cat.scala 30:58]
  wire [16:0] _T_386; // @[Compute.scala 240:42]
  wire [16:0] _T_387; // @[Compute.scala 240:42]
  wire [15:0] _T_388; // @[Compute.scala 240:42]
  wire  _T_389; // @[Compute.scala 240:24]
  wire  _GEN_27; // @[Compute.scala 240:50]
  wire [31:0] _T_392; // @[Compute.scala 243:35]
  wire [32:0] _T_394; // @[Compute.scala 245:30]
  wire [31:0] _T_395; // @[Compute.scala 245:30]
  wire [32:0] _T_400; // @[Compute.scala 245:30]
  wire [31:0] _T_401; // @[Compute.scala 245:30]
  wire [32:0] _T_406; // @[Compute.scala 245:30]
  wire [31:0] _T_407; // @[Compute.scala 245:30]
  wire [32:0] _T_412; // @[Compute.scala 245:30]
  wire [31:0] _T_413; // @[Compute.scala 245:30]
  wire [31:0] _GEN_298; // @[Compute.scala 250:36]
  wire [32:0] _T_417; // @[Compute.scala 250:36]
  wire [31:0] _T_418; // @[Compute.scala 250:36]
  wire [31:0] _GEN_299; // @[Compute.scala 250:47]
  wire [32:0] _T_419; // @[Compute.scala 250:47]
  wire [31:0] _T_420; // @[Compute.scala 250:47]
  wire [34:0] _GEN_300; // @[Compute.scala 250:58]
  wire [34:0] _T_422; // @[Compute.scala 250:58]
  wire [35:0] _T_424; // @[Compute.scala 250:66]
  wire [35:0] _GEN_301; // @[Compute.scala 250:76]
  wire [36:0] _T_425; // @[Compute.scala 250:76]
  wire [35:0] _T_426; // @[Compute.scala 250:76]
  wire [42:0] _GEN_302; // @[Compute.scala 250:92]
  wire [42:0] acc_dram_addr; // @[Compute.scala 250:92]
  wire [19:0] _GEN_303; // @[Compute.scala 251:36]
  wire [20:0] _T_428; // @[Compute.scala 251:36]
  wire [19:0] _T_429; // @[Compute.scala 251:36]
  wire [19:0] _GEN_304; // @[Compute.scala 251:47]
  wire [20:0] _T_430; // @[Compute.scala 251:47]
  wire [19:0] _T_431; // @[Compute.scala 251:47]
  wire [22:0] _GEN_305; // @[Compute.scala 251:58]
  wire [22:0] _T_433; // @[Compute.scala 251:58]
  wire [23:0] _T_435; // @[Compute.scala 251:66]
  wire [23:0] _GEN_306; // @[Compute.scala 251:76]
  wire [24:0] _T_436; // @[Compute.scala 251:76]
  wire [23:0] _T_437; // @[Compute.scala 251:76]
  wire [23:0] _T_439; // @[Compute.scala 251:92]
  wire [24:0] _T_441; // @[Compute.scala 251:121]
  wire [24:0] _T_442; // @[Compute.scala 251:121]
  wire [23:0] acc_sram_addr; // @[Compute.scala 251:121]
  wire  _T_444; // @[Compute.scala 252:33]
  wire [15:0] _GEN_17; // @[Compute.scala 258:30]
  wire [2:0] _T_450; // @[Compute.scala 258:30]
  wire [127:0] _GEN_42; // @[Compute.scala 258:48]
  wire [127:0] _GEN_43; // @[Compute.scala 258:48]
  wire [127:0] _GEN_44; // @[Compute.scala 258:48]
  wire [127:0] _GEN_45; // @[Compute.scala 258:48]
  wire  _T_456; // @[Compute.scala 262:43]
  wire [255:0] _T_459; // @[Cat.scala 30:58]
  wire [255:0] _T_460; // @[Cat.scala 30:58]
  wire [15:0] _GEN_18; // @[Compute.scala 268:26]
  wire [15:0] upc; // @[Compute.scala 268:26]
  reg [31:0] uop; // @[Compute.scala 269:20]
  reg [31:0] _RAND_26;
  reg [15:0] _T_466; // @[Compute.scala 272:22]
  reg [31:0] _RAND_27;
  wire [15:0] _GEN_19; // @[Compute.scala 272:37]
  wire [15:0] it_in; // @[Compute.scala 272:37]
  wire [31:0] _T_469; // @[Compute.scala 273:47]
  wire [15:0] _T_470; // @[Compute.scala 273:63]
  wire [16:0] _T_471; // @[Compute.scala 273:38]
  wire [15:0] dst_offset_in; // @[Compute.scala 273:38]
  wire [10:0] _T_475; // @[Compute.scala 275:20]
  wire [15:0] _GEN_307; // @[Compute.scala 275:47]
  wire [16:0] _T_476; // @[Compute.scala 275:47]
  wire [15:0] dst_idx; // @[Compute.scala 275:47]
  wire [10:0] _T_477; // @[Compute.scala 276:20]
  wire [15:0] _GEN_308; // @[Compute.scala 276:47]
  wire [16:0] _T_478; // @[Compute.scala 276:47]
  wire [15:0] src_idx; // @[Compute.scala 276:47]
  reg [511:0] dst_vector; // @[Compute.scala 279:23]
  reg [511:0] _RAND_28;
  reg [511:0] src_vector; // @[Compute.scala 280:23]
  reg [511:0] _RAND_29;
  wire  alu_opcode_min_en; // @[Compute.scala 295:38]
  wire  alu_opcode_max_en; // @[Compute.scala 296:38]
  wire  _T_916; // @[Compute.scala 320:20]
  wire [31:0] _T_917; // @[Compute.scala 322:29]
  wire [31:0] _T_918; // @[Compute.scala 322:70]
  wire [31:0] _T_919; // @[Compute.scala 323:29]
  wire [31:0] _T_920; // @[Compute.scala 323:70]
  wire [31:0] _T_921; // @[Compute.scala 322:29]
  wire [31:0] _T_922; // @[Compute.scala 322:70]
  wire [31:0] _T_923; // @[Compute.scala 323:29]
  wire [31:0] _T_924; // @[Compute.scala 323:70]
  wire [31:0] _T_925; // @[Compute.scala 322:29]
  wire [31:0] _T_926; // @[Compute.scala 322:70]
  wire [31:0] _T_927; // @[Compute.scala 323:29]
  wire [31:0] _T_928; // @[Compute.scala 323:70]
  wire [31:0] _T_929; // @[Compute.scala 322:29]
  wire [31:0] _T_930; // @[Compute.scala 322:70]
  wire [31:0] _T_931; // @[Compute.scala 323:29]
  wire [31:0] _T_932; // @[Compute.scala 323:70]
  wire [31:0] _T_933; // @[Compute.scala 322:29]
  wire [31:0] _T_934; // @[Compute.scala 322:70]
  wire [31:0] _T_935; // @[Compute.scala 323:29]
  wire [31:0] _T_936; // @[Compute.scala 323:70]
  wire [31:0] _T_937; // @[Compute.scala 322:29]
  wire [31:0] _T_938; // @[Compute.scala 322:70]
  wire [31:0] _T_939; // @[Compute.scala 323:29]
  wire [31:0] _T_940; // @[Compute.scala 323:70]
  wire [31:0] _T_941; // @[Compute.scala 322:29]
  wire [31:0] _T_942; // @[Compute.scala 322:70]
  wire [31:0] _T_943; // @[Compute.scala 323:29]
  wire [31:0] _T_944; // @[Compute.scala 323:70]
  wire [31:0] _T_945; // @[Compute.scala 322:29]
  wire [31:0] _T_946; // @[Compute.scala 322:70]
  wire [31:0] _T_947; // @[Compute.scala 323:29]
  wire [31:0] _T_948; // @[Compute.scala 323:70]
  wire [31:0] _T_949; // @[Compute.scala 322:29]
  wire [31:0] _T_950; // @[Compute.scala 322:70]
  wire [31:0] _T_951; // @[Compute.scala 323:29]
  wire [31:0] _T_952; // @[Compute.scala 323:70]
  wire [31:0] _T_953; // @[Compute.scala 322:29]
  wire [31:0] _T_954; // @[Compute.scala 322:70]
  wire [31:0] _T_955; // @[Compute.scala 323:29]
  wire [31:0] _T_956; // @[Compute.scala 323:70]
  wire [31:0] _T_957; // @[Compute.scala 322:29]
  wire [31:0] _T_958; // @[Compute.scala 322:70]
  wire [31:0] _T_959; // @[Compute.scala 323:29]
  wire [31:0] _T_960; // @[Compute.scala 323:70]
  wire [31:0] _T_961; // @[Compute.scala 322:29]
  wire [31:0] _T_962; // @[Compute.scala 322:70]
  wire [31:0] _T_963; // @[Compute.scala 323:29]
  wire [31:0] _T_964; // @[Compute.scala 323:70]
  wire [31:0] _T_965; // @[Compute.scala 322:29]
  wire [31:0] _T_966; // @[Compute.scala 322:70]
  wire [31:0] _T_967; // @[Compute.scala 323:29]
  wire [31:0] _T_968; // @[Compute.scala 323:70]
  wire [31:0] _T_969; // @[Compute.scala 322:29]
  wire [31:0] _T_970; // @[Compute.scala 322:70]
  wire [31:0] _T_971; // @[Compute.scala 323:29]
  wire [31:0] _T_972; // @[Compute.scala 323:70]
  wire [31:0] _T_973; // @[Compute.scala 322:29]
  wire [31:0] _T_974; // @[Compute.scala 322:70]
  wire [31:0] _T_975; // @[Compute.scala 323:29]
  wire [31:0] _T_976; // @[Compute.scala 323:70]
  wire [31:0] _T_977; // @[Compute.scala 322:29]
  wire [31:0] _T_978; // @[Compute.scala 322:70]
  wire [31:0] _T_979; // @[Compute.scala 323:29]
  wire [31:0] _T_980; // @[Compute.scala 323:70]
  wire [31:0] _GEN_68; // @[Compute.scala 325:20]
  wire [31:0] _GEN_69; // @[Compute.scala 325:20]
  wire [31:0] _GEN_70; // @[Compute.scala 325:20]
  wire [31:0] _GEN_71; // @[Compute.scala 325:20]
  wire [31:0] _GEN_72; // @[Compute.scala 325:20]
  wire [31:0] _GEN_73; // @[Compute.scala 325:20]
  wire [31:0] _GEN_74; // @[Compute.scala 325:20]
  wire [31:0] _GEN_75; // @[Compute.scala 325:20]
  wire [31:0] _GEN_76; // @[Compute.scala 325:20]
  wire [31:0] _GEN_77; // @[Compute.scala 325:20]
  wire [31:0] _GEN_78; // @[Compute.scala 325:20]
  wire [31:0] _GEN_79; // @[Compute.scala 325:20]
  wire [31:0] _GEN_80; // @[Compute.scala 325:20]
  wire [31:0] _GEN_81; // @[Compute.scala 325:20]
  wire [31:0] _GEN_82; // @[Compute.scala 325:20]
  wire [31:0] _GEN_83; // @[Compute.scala 325:20]
  wire [31:0] src_0_0; // @[Compute.scala 320:36]
  wire [31:0] src_1_0; // @[Compute.scala 320:36]
  wire  _T_981; // @[Compute.scala 330:57]
  wire [31:0] _T_982; // @[Compute.scala 330:47]
  wire  _T_983; // @[Compute.scala 331:57]
  wire [31:0] _T_984; // @[Compute.scala 331:47]
  wire [31:0] _T_985; // @[Compute.scala 330:24]
  wire [31:0] mix_val_0; // @[Compute.scala 320:36]
  wire [7:0] _T_986; // @[Compute.scala 333:37]
  wire [31:0] _T_987; // @[Compute.scala 334:30]
  wire [31:0] _T_988; // @[Compute.scala 334:59]
  wire [32:0] _T_989; // @[Compute.scala 334:49]
  wire [31:0] _T_990; // @[Compute.scala 334:49]
  wire [31:0] _T_991; // @[Compute.scala 334:79]
  wire [31:0] add_val_0; // @[Compute.scala 320:36]
  wire [31:0] add_res_0; // @[Compute.scala 320:36]
  wire [7:0] _T_992; // @[Compute.scala 336:37]
  wire [4:0] _T_994; // @[Compute.scala 337:60]
  wire [31:0] _T_995; // @[Compute.scala 337:49]
  wire [31:0] _T_996; // @[Compute.scala 337:84]
  wire [31:0] shr_val_0; // @[Compute.scala 320:36]
  wire [31:0] shr_res_0; // @[Compute.scala 320:36]
  wire [7:0] _T_997; // @[Compute.scala 339:37]
  wire [31:0] src_0_1; // @[Compute.scala 320:36]
  wire [31:0] src_1_1; // @[Compute.scala 320:36]
  wire  _T_998; // @[Compute.scala 330:57]
  wire [31:0] _T_999; // @[Compute.scala 330:47]
  wire  _T_1000; // @[Compute.scala 331:57]
  wire [31:0] _T_1001; // @[Compute.scala 331:47]
  wire [31:0] _T_1002; // @[Compute.scala 330:24]
  wire [31:0] mix_val_1; // @[Compute.scala 320:36]
  wire [7:0] _T_1003; // @[Compute.scala 333:37]
  wire [31:0] _T_1004; // @[Compute.scala 334:30]
  wire [31:0] _T_1005; // @[Compute.scala 334:59]
  wire [32:0] _T_1006; // @[Compute.scala 334:49]
  wire [31:0] _T_1007; // @[Compute.scala 334:49]
  wire [31:0] _T_1008; // @[Compute.scala 334:79]
  wire [31:0] add_val_1; // @[Compute.scala 320:36]
  wire [31:0] add_res_1; // @[Compute.scala 320:36]
  wire [7:0] _T_1009; // @[Compute.scala 336:37]
  wire [4:0] _T_1011; // @[Compute.scala 337:60]
  wire [31:0] _T_1012; // @[Compute.scala 337:49]
  wire [31:0] _T_1013; // @[Compute.scala 337:84]
  wire [31:0] shr_val_1; // @[Compute.scala 320:36]
  wire [31:0] shr_res_1; // @[Compute.scala 320:36]
  wire [7:0] _T_1014; // @[Compute.scala 339:37]
  wire [31:0] src_0_2; // @[Compute.scala 320:36]
  wire [31:0] src_1_2; // @[Compute.scala 320:36]
  wire  _T_1015; // @[Compute.scala 330:57]
  wire [31:0] _T_1016; // @[Compute.scala 330:47]
  wire  _T_1017; // @[Compute.scala 331:57]
  wire [31:0] _T_1018; // @[Compute.scala 331:47]
  wire [31:0] _T_1019; // @[Compute.scala 330:24]
  wire [31:0] mix_val_2; // @[Compute.scala 320:36]
  wire [7:0] _T_1020; // @[Compute.scala 333:37]
  wire [31:0] _T_1021; // @[Compute.scala 334:30]
  wire [31:0] _T_1022; // @[Compute.scala 334:59]
  wire [32:0] _T_1023; // @[Compute.scala 334:49]
  wire [31:0] _T_1024; // @[Compute.scala 334:49]
  wire [31:0] _T_1025; // @[Compute.scala 334:79]
  wire [31:0] add_val_2; // @[Compute.scala 320:36]
  wire [31:0] add_res_2; // @[Compute.scala 320:36]
  wire [7:0] _T_1026; // @[Compute.scala 336:37]
  wire [4:0] _T_1028; // @[Compute.scala 337:60]
  wire [31:0] _T_1029; // @[Compute.scala 337:49]
  wire [31:0] _T_1030; // @[Compute.scala 337:84]
  wire [31:0] shr_val_2; // @[Compute.scala 320:36]
  wire [31:0] shr_res_2; // @[Compute.scala 320:36]
  wire [7:0] _T_1031; // @[Compute.scala 339:37]
  wire [31:0] src_0_3; // @[Compute.scala 320:36]
  wire [31:0] src_1_3; // @[Compute.scala 320:36]
  wire  _T_1032; // @[Compute.scala 330:57]
  wire [31:0] _T_1033; // @[Compute.scala 330:47]
  wire  _T_1034; // @[Compute.scala 331:57]
  wire [31:0] _T_1035; // @[Compute.scala 331:47]
  wire [31:0] _T_1036; // @[Compute.scala 330:24]
  wire [31:0] mix_val_3; // @[Compute.scala 320:36]
  wire [7:0] _T_1037; // @[Compute.scala 333:37]
  wire [31:0] _T_1038; // @[Compute.scala 334:30]
  wire [31:0] _T_1039; // @[Compute.scala 334:59]
  wire [32:0] _T_1040; // @[Compute.scala 334:49]
  wire [31:0] _T_1041; // @[Compute.scala 334:49]
  wire [31:0] _T_1042; // @[Compute.scala 334:79]
  wire [31:0] add_val_3; // @[Compute.scala 320:36]
  wire [31:0] add_res_3; // @[Compute.scala 320:36]
  wire [7:0] _T_1043; // @[Compute.scala 336:37]
  wire [4:0] _T_1045; // @[Compute.scala 337:60]
  wire [31:0] _T_1046; // @[Compute.scala 337:49]
  wire [31:0] _T_1047; // @[Compute.scala 337:84]
  wire [31:0] shr_val_3; // @[Compute.scala 320:36]
  wire [31:0] shr_res_3; // @[Compute.scala 320:36]
  wire [7:0] _T_1048; // @[Compute.scala 339:37]
  wire [31:0] src_0_4; // @[Compute.scala 320:36]
  wire [31:0] src_1_4; // @[Compute.scala 320:36]
  wire  _T_1049; // @[Compute.scala 330:57]
  wire [31:0] _T_1050; // @[Compute.scala 330:47]
  wire  _T_1051; // @[Compute.scala 331:57]
  wire [31:0] _T_1052; // @[Compute.scala 331:47]
  wire [31:0] _T_1053; // @[Compute.scala 330:24]
  wire [31:0] mix_val_4; // @[Compute.scala 320:36]
  wire [7:0] _T_1054; // @[Compute.scala 333:37]
  wire [31:0] _T_1055; // @[Compute.scala 334:30]
  wire [31:0] _T_1056; // @[Compute.scala 334:59]
  wire [32:0] _T_1057; // @[Compute.scala 334:49]
  wire [31:0] _T_1058; // @[Compute.scala 334:49]
  wire [31:0] _T_1059; // @[Compute.scala 334:79]
  wire [31:0] add_val_4; // @[Compute.scala 320:36]
  wire [31:0] add_res_4; // @[Compute.scala 320:36]
  wire [7:0] _T_1060; // @[Compute.scala 336:37]
  wire [4:0] _T_1062; // @[Compute.scala 337:60]
  wire [31:0] _T_1063; // @[Compute.scala 337:49]
  wire [31:0] _T_1064; // @[Compute.scala 337:84]
  wire [31:0] shr_val_4; // @[Compute.scala 320:36]
  wire [31:0] shr_res_4; // @[Compute.scala 320:36]
  wire [7:0] _T_1065; // @[Compute.scala 339:37]
  wire [31:0] src_0_5; // @[Compute.scala 320:36]
  wire [31:0] src_1_5; // @[Compute.scala 320:36]
  wire  _T_1066; // @[Compute.scala 330:57]
  wire [31:0] _T_1067; // @[Compute.scala 330:47]
  wire  _T_1068; // @[Compute.scala 331:57]
  wire [31:0] _T_1069; // @[Compute.scala 331:47]
  wire [31:0] _T_1070; // @[Compute.scala 330:24]
  wire [31:0] mix_val_5; // @[Compute.scala 320:36]
  wire [7:0] _T_1071; // @[Compute.scala 333:37]
  wire [31:0] _T_1072; // @[Compute.scala 334:30]
  wire [31:0] _T_1073; // @[Compute.scala 334:59]
  wire [32:0] _T_1074; // @[Compute.scala 334:49]
  wire [31:0] _T_1075; // @[Compute.scala 334:49]
  wire [31:0] _T_1076; // @[Compute.scala 334:79]
  wire [31:0] add_val_5; // @[Compute.scala 320:36]
  wire [31:0] add_res_5; // @[Compute.scala 320:36]
  wire [7:0] _T_1077; // @[Compute.scala 336:37]
  wire [4:0] _T_1079; // @[Compute.scala 337:60]
  wire [31:0] _T_1080; // @[Compute.scala 337:49]
  wire [31:0] _T_1081; // @[Compute.scala 337:84]
  wire [31:0] shr_val_5; // @[Compute.scala 320:36]
  wire [31:0] shr_res_5; // @[Compute.scala 320:36]
  wire [7:0] _T_1082; // @[Compute.scala 339:37]
  wire [31:0] src_0_6; // @[Compute.scala 320:36]
  wire [31:0] src_1_6; // @[Compute.scala 320:36]
  wire  _T_1083; // @[Compute.scala 330:57]
  wire [31:0] _T_1084; // @[Compute.scala 330:47]
  wire  _T_1085; // @[Compute.scala 331:57]
  wire [31:0] _T_1086; // @[Compute.scala 331:47]
  wire [31:0] _T_1087; // @[Compute.scala 330:24]
  wire [31:0] mix_val_6; // @[Compute.scala 320:36]
  wire [7:0] _T_1088; // @[Compute.scala 333:37]
  wire [31:0] _T_1089; // @[Compute.scala 334:30]
  wire [31:0] _T_1090; // @[Compute.scala 334:59]
  wire [32:0] _T_1091; // @[Compute.scala 334:49]
  wire [31:0] _T_1092; // @[Compute.scala 334:49]
  wire [31:0] _T_1093; // @[Compute.scala 334:79]
  wire [31:0] add_val_6; // @[Compute.scala 320:36]
  wire [31:0] add_res_6; // @[Compute.scala 320:36]
  wire [7:0] _T_1094; // @[Compute.scala 336:37]
  wire [4:0] _T_1096; // @[Compute.scala 337:60]
  wire [31:0] _T_1097; // @[Compute.scala 337:49]
  wire [31:0] _T_1098; // @[Compute.scala 337:84]
  wire [31:0] shr_val_6; // @[Compute.scala 320:36]
  wire [31:0] shr_res_6; // @[Compute.scala 320:36]
  wire [7:0] _T_1099; // @[Compute.scala 339:37]
  wire [31:0] src_0_7; // @[Compute.scala 320:36]
  wire [31:0] src_1_7; // @[Compute.scala 320:36]
  wire  _T_1100; // @[Compute.scala 330:57]
  wire [31:0] _T_1101; // @[Compute.scala 330:47]
  wire  _T_1102; // @[Compute.scala 331:57]
  wire [31:0] _T_1103; // @[Compute.scala 331:47]
  wire [31:0] _T_1104; // @[Compute.scala 330:24]
  wire [31:0] mix_val_7; // @[Compute.scala 320:36]
  wire [7:0] _T_1105; // @[Compute.scala 333:37]
  wire [31:0] _T_1106; // @[Compute.scala 334:30]
  wire [31:0] _T_1107; // @[Compute.scala 334:59]
  wire [32:0] _T_1108; // @[Compute.scala 334:49]
  wire [31:0] _T_1109; // @[Compute.scala 334:49]
  wire [31:0] _T_1110; // @[Compute.scala 334:79]
  wire [31:0] add_val_7; // @[Compute.scala 320:36]
  wire [31:0] add_res_7; // @[Compute.scala 320:36]
  wire [7:0] _T_1111; // @[Compute.scala 336:37]
  wire [4:0] _T_1113; // @[Compute.scala 337:60]
  wire [31:0] _T_1114; // @[Compute.scala 337:49]
  wire [31:0] _T_1115; // @[Compute.scala 337:84]
  wire [31:0] shr_val_7; // @[Compute.scala 320:36]
  wire [31:0] shr_res_7; // @[Compute.scala 320:36]
  wire [7:0] _T_1116; // @[Compute.scala 339:37]
  wire [31:0] src_0_8; // @[Compute.scala 320:36]
  wire [31:0] src_1_8; // @[Compute.scala 320:36]
  wire  _T_1117; // @[Compute.scala 330:57]
  wire [31:0] _T_1118; // @[Compute.scala 330:47]
  wire  _T_1119; // @[Compute.scala 331:57]
  wire [31:0] _T_1120; // @[Compute.scala 331:47]
  wire [31:0] _T_1121; // @[Compute.scala 330:24]
  wire [31:0] mix_val_8; // @[Compute.scala 320:36]
  wire [7:0] _T_1122; // @[Compute.scala 333:37]
  wire [31:0] _T_1123; // @[Compute.scala 334:30]
  wire [31:0] _T_1124; // @[Compute.scala 334:59]
  wire [32:0] _T_1125; // @[Compute.scala 334:49]
  wire [31:0] _T_1126; // @[Compute.scala 334:49]
  wire [31:0] _T_1127; // @[Compute.scala 334:79]
  wire [31:0] add_val_8; // @[Compute.scala 320:36]
  wire [31:0] add_res_8; // @[Compute.scala 320:36]
  wire [7:0] _T_1128; // @[Compute.scala 336:37]
  wire [4:0] _T_1130; // @[Compute.scala 337:60]
  wire [31:0] _T_1131; // @[Compute.scala 337:49]
  wire [31:0] _T_1132; // @[Compute.scala 337:84]
  wire [31:0] shr_val_8; // @[Compute.scala 320:36]
  wire [31:0] shr_res_8; // @[Compute.scala 320:36]
  wire [7:0] _T_1133; // @[Compute.scala 339:37]
  wire [31:0] src_0_9; // @[Compute.scala 320:36]
  wire [31:0] src_1_9; // @[Compute.scala 320:36]
  wire  _T_1134; // @[Compute.scala 330:57]
  wire [31:0] _T_1135; // @[Compute.scala 330:47]
  wire  _T_1136; // @[Compute.scala 331:57]
  wire [31:0] _T_1137; // @[Compute.scala 331:47]
  wire [31:0] _T_1138; // @[Compute.scala 330:24]
  wire [31:0] mix_val_9; // @[Compute.scala 320:36]
  wire [7:0] _T_1139; // @[Compute.scala 333:37]
  wire [31:0] _T_1140; // @[Compute.scala 334:30]
  wire [31:0] _T_1141; // @[Compute.scala 334:59]
  wire [32:0] _T_1142; // @[Compute.scala 334:49]
  wire [31:0] _T_1143; // @[Compute.scala 334:49]
  wire [31:0] _T_1144; // @[Compute.scala 334:79]
  wire [31:0] add_val_9; // @[Compute.scala 320:36]
  wire [31:0] add_res_9; // @[Compute.scala 320:36]
  wire [7:0] _T_1145; // @[Compute.scala 336:37]
  wire [4:0] _T_1147; // @[Compute.scala 337:60]
  wire [31:0] _T_1148; // @[Compute.scala 337:49]
  wire [31:0] _T_1149; // @[Compute.scala 337:84]
  wire [31:0] shr_val_9; // @[Compute.scala 320:36]
  wire [31:0] shr_res_9; // @[Compute.scala 320:36]
  wire [7:0] _T_1150; // @[Compute.scala 339:37]
  wire [31:0] src_0_10; // @[Compute.scala 320:36]
  wire [31:0] src_1_10; // @[Compute.scala 320:36]
  wire  _T_1151; // @[Compute.scala 330:57]
  wire [31:0] _T_1152; // @[Compute.scala 330:47]
  wire  _T_1153; // @[Compute.scala 331:57]
  wire [31:0] _T_1154; // @[Compute.scala 331:47]
  wire [31:0] _T_1155; // @[Compute.scala 330:24]
  wire [31:0] mix_val_10; // @[Compute.scala 320:36]
  wire [7:0] _T_1156; // @[Compute.scala 333:37]
  wire [31:0] _T_1157; // @[Compute.scala 334:30]
  wire [31:0] _T_1158; // @[Compute.scala 334:59]
  wire [32:0] _T_1159; // @[Compute.scala 334:49]
  wire [31:0] _T_1160; // @[Compute.scala 334:49]
  wire [31:0] _T_1161; // @[Compute.scala 334:79]
  wire [31:0] add_val_10; // @[Compute.scala 320:36]
  wire [31:0] add_res_10; // @[Compute.scala 320:36]
  wire [7:0] _T_1162; // @[Compute.scala 336:37]
  wire [4:0] _T_1164; // @[Compute.scala 337:60]
  wire [31:0] _T_1165; // @[Compute.scala 337:49]
  wire [31:0] _T_1166; // @[Compute.scala 337:84]
  wire [31:0] shr_val_10; // @[Compute.scala 320:36]
  wire [31:0] shr_res_10; // @[Compute.scala 320:36]
  wire [7:0] _T_1167; // @[Compute.scala 339:37]
  wire [31:0] src_0_11; // @[Compute.scala 320:36]
  wire [31:0] src_1_11; // @[Compute.scala 320:36]
  wire  _T_1168; // @[Compute.scala 330:57]
  wire [31:0] _T_1169; // @[Compute.scala 330:47]
  wire  _T_1170; // @[Compute.scala 331:57]
  wire [31:0] _T_1171; // @[Compute.scala 331:47]
  wire [31:0] _T_1172; // @[Compute.scala 330:24]
  wire [31:0] mix_val_11; // @[Compute.scala 320:36]
  wire [7:0] _T_1173; // @[Compute.scala 333:37]
  wire [31:0] _T_1174; // @[Compute.scala 334:30]
  wire [31:0] _T_1175; // @[Compute.scala 334:59]
  wire [32:0] _T_1176; // @[Compute.scala 334:49]
  wire [31:0] _T_1177; // @[Compute.scala 334:49]
  wire [31:0] _T_1178; // @[Compute.scala 334:79]
  wire [31:0] add_val_11; // @[Compute.scala 320:36]
  wire [31:0] add_res_11; // @[Compute.scala 320:36]
  wire [7:0] _T_1179; // @[Compute.scala 336:37]
  wire [4:0] _T_1181; // @[Compute.scala 337:60]
  wire [31:0] _T_1182; // @[Compute.scala 337:49]
  wire [31:0] _T_1183; // @[Compute.scala 337:84]
  wire [31:0] shr_val_11; // @[Compute.scala 320:36]
  wire [31:0] shr_res_11; // @[Compute.scala 320:36]
  wire [7:0] _T_1184; // @[Compute.scala 339:37]
  wire [31:0] src_0_12; // @[Compute.scala 320:36]
  wire [31:0] src_1_12; // @[Compute.scala 320:36]
  wire  _T_1185; // @[Compute.scala 330:57]
  wire [31:0] _T_1186; // @[Compute.scala 330:47]
  wire  _T_1187; // @[Compute.scala 331:57]
  wire [31:0] _T_1188; // @[Compute.scala 331:47]
  wire [31:0] _T_1189; // @[Compute.scala 330:24]
  wire [31:0] mix_val_12; // @[Compute.scala 320:36]
  wire [7:0] _T_1190; // @[Compute.scala 333:37]
  wire [31:0] _T_1191; // @[Compute.scala 334:30]
  wire [31:0] _T_1192; // @[Compute.scala 334:59]
  wire [32:0] _T_1193; // @[Compute.scala 334:49]
  wire [31:0] _T_1194; // @[Compute.scala 334:49]
  wire [31:0] _T_1195; // @[Compute.scala 334:79]
  wire [31:0] add_val_12; // @[Compute.scala 320:36]
  wire [31:0] add_res_12; // @[Compute.scala 320:36]
  wire [7:0] _T_1196; // @[Compute.scala 336:37]
  wire [4:0] _T_1198; // @[Compute.scala 337:60]
  wire [31:0] _T_1199; // @[Compute.scala 337:49]
  wire [31:0] _T_1200; // @[Compute.scala 337:84]
  wire [31:0] shr_val_12; // @[Compute.scala 320:36]
  wire [31:0] shr_res_12; // @[Compute.scala 320:36]
  wire [7:0] _T_1201; // @[Compute.scala 339:37]
  wire [31:0] src_0_13; // @[Compute.scala 320:36]
  wire [31:0] src_1_13; // @[Compute.scala 320:36]
  wire  _T_1202; // @[Compute.scala 330:57]
  wire [31:0] _T_1203; // @[Compute.scala 330:47]
  wire  _T_1204; // @[Compute.scala 331:57]
  wire [31:0] _T_1205; // @[Compute.scala 331:47]
  wire [31:0] _T_1206; // @[Compute.scala 330:24]
  wire [31:0] mix_val_13; // @[Compute.scala 320:36]
  wire [7:0] _T_1207; // @[Compute.scala 333:37]
  wire [31:0] _T_1208; // @[Compute.scala 334:30]
  wire [31:0] _T_1209; // @[Compute.scala 334:59]
  wire [32:0] _T_1210; // @[Compute.scala 334:49]
  wire [31:0] _T_1211; // @[Compute.scala 334:49]
  wire [31:0] _T_1212; // @[Compute.scala 334:79]
  wire [31:0] add_val_13; // @[Compute.scala 320:36]
  wire [31:0] add_res_13; // @[Compute.scala 320:36]
  wire [7:0] _T_1213; // @[Compute.scala 336:37]
  wire [4:0] _T_1215; // @[Compute.scala 337:60]
  wire [31:0] _T_1216; // @[Compute.scala 337:49]
  wire [31:0] _T_1217; // @[Compute.scala 337:84]
  wire [31:0] shr_val_13; // @[Compute.scala 320:36]
  wire [31:0] shr_res_13; // @[Compute.scala 320:36]
  wire [7:0] _T_1218; // @[Compute.scala 339:37]
  wire [31:0] src_0_14; // @[Compute.scala 320:36]
  wire [31:0] src_1_14; // @[Compute.scala 320:36]
  wire  _T_1219; // @[Compute.scala 330:57]
  wire [31:0] _T_1220; // @[Compute.scala 330:47]
  wire  _T_1221; // @[Compute.scala 331:57]
  wire [31:0] _T_1222; // @[Compute.scala 331:47]
  wire [31:0] _T_1223; // @[Compute.scala 330:24]
  wire [31:0] mix_val_14; // @[Compute.scala 320:36]
  wire [7:0] _T_1224; // @[Compute.scala 333:37]
  wire [31:0] _T_1225; // @[Compute.scala 334:30]
  wire [31:0] _T_1226; // @[Compute.scala 334:59]
  wire [32:0] _T_1227; // @[Compute.scala 334:49]
  wire [31:0] _T_1228; // @[Compute.scala 334:49]
  wire [31:0] _T_1229; // @[Compute.scala 334:79]
  wire [31:0] add_val_14; // @[Compute.scala 320:36]
  wire [31:0] add_res_14; // @[Compute.scala 320:36]
  wire [7:0] _T_1230; // @[Compute.scala 336:37]
  wire [4:0] _T_1232; // @[Compute.scala 337:60]
  wire [31:0] _T_1233; // @[Compute.scala 337:49]
  wire [31:0] _T_1234; // @[Compute.scala 337:84]
  wire [31:0] shr_val_14; // @[Compute.scala 320:36]
  wire [31:0] shr_res_14; // @[Compute.scala 320:36]
  wire [7:0] _T_1235; // @[Compute.scala 339:37]
  wire [31:0] src_0_15; // @[Compute.scala 320:36]
  wire [31:0] src_1_15; // @[Compute.scala 320:36]
  wire  _T_1236; // @[Compute.scala 330:57]
  wire [31:0] _T_1237; // @[Compute.scala 330:47]
  wire  _T_1238; // @[Compute.scala 331:57]
  wire [31:0] _T_1239; // @[Compute.scala 331:47]
  wire [31:0] _T_1240; // @[Compute.scala 330:24]
  wire [31:0] mix_val_15; // @[Compute.scala 320:36]
  wire [7:0] _T_1241; // @[Compute.scala 333:37]
  wire [31:0] _T_1242; // @[Compute.scala 334:30]
  wire [31:0] _T_1243; // @[Compute.scala 334:59]
  wire [32:0] _T_1244; // @[Compute.scala 334:49]
  wire [31:0] _T_1245; // @[Compute.scala 334:49]
  wire [31:0] _T_1246; // @[Compute.scala 334:79]
  wire [31:0] add_val_15; // @[Compute.scala 320:36]
  wire [31:0] add_res_15; // @[Compute.scala 320:36]
  wire [7:0] _T_1247; // @[Compute.scala 336:37]
  wire [4:0] _T_1249; // @[Compute.scala 337:60]
  wire [31:0] _T_1250; // @[Compute.scala 337:49]
  wire [31:0] _T_1251; // @[Compute.scala 337:84]
  wire [31:0] shr_val_15; // @[Compute.scala 320:36]
  wire [31:0] shr_res_15; // @[Compute.scala 320:36]
  wire [7:0] _T_1252; // @[Compute.scala 339:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 320:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 320:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 320:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 320:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 344:48]
  wire  alu_opcode_add_en; // @[Compute.scala 345:39]
  wire [63:0] _T_1262; // @[Cat.scala 30:58]
  wire [127:0] _T_1270; // @[Cat.scala 30:58]
  wire [63:0] _T_1277; // @[Cat.scala 30:58]
  wire [127:0] _T_1285; // @[Cat.scala 30:58]
  wire [63:0] _T_1292; // @[Cat.scala 30:58]
  wire [127:0] _T_1300; // @[Cat.scala 30:58]
  wire [127:0] _T_1301; // @[Compute.scala 350:29]
  wire [127:0] out_mem_enq_bits; // @[Compute.scala 349:29]
  wire  _T_1302; // @[Compute.scala 351:34]
  wire  _T_1303; // @[Compute.scala 351:59]
  wire  _T_1304; // @[Compute.scala 351:42]
  wire  _T_1306; // @[Compute.scala 353:63]
  wire  _T_1307; // @[Compute.scala 353:46]
  wire [32:0] _T_1309; // @[Compute.scala 353:105]
  wire [32:0] _T_1310; // @[Compute.scala 353:105]
  wire [31:0] _T_1311; // @[Compute.scala 353:105]
  wire  _T_1312; // @[Compute.scala 353:88]
  reg [15:0] _T_1315; // @[Compute.scala 354:42]
  reg [31:0] _RAND_30;
  wire [143:0] _T_1316; // @[Cat.scala 30:58]
  wire [31:0] _T_1317; // @[Compute.scala 357:49]
  wire [38:0] _GEN_311; // @[Compute.scala 357:66]
  wire [38:0] _T_1319; // @[Compute.scala 357:66]
  OutQueue out_mem_fifo ( // @[Compute.scala 348:28]
    .clock(out_mem_fifo_clock),
    .reset(out_mem_fifo_reset),
    .io_enq_ready(out_mem_fifo_io_enq_ready),
    .io_enq_valid(out_mem_fifo_io_enq_valid),
    .io_enq_bits(out_mem_fifo_io_enq_bits),
    .io_deq_ready(out_mem_fifo_io_deq_ready),
    .io_deq_valid(out_mem_fifo_io_deq_valid),
    .io_deq_bits(out_mem_fifo_io_deq_bits)
  );
  assign acc_mem__T_726_addr = dst_idx[7:0];
  assign acc_mem__T_726_data = acc_mem[acc_mem__T_726_addr]; // @[Compute.scala 34:20]
  assign acc_mem__T_728_addr = src_idx[7:0];
  assign acc_mem__T_728_data = acc_mem[acc_mem__T_728_addr]; // @[Compute.scala 34:20]
  assign acc_mem__T_458_data = {_T_460,_T_459};
  assign acc_mem__T_458_addr = acc_sram_addr[7:0];
  assign acc_mem__T_458_mask = 1'h1;
  assign acc_mem__T_458_en = _T_340 ? _T_456 : 1'h0;
  assign uop_mem__T_463_addr = upc[9:0];
  assign uop_mem__T_463_data = uop_mem[uop_mem__T_463_addr]; // @[Compute.scala 35:20]
  assign uop_mem__T_397_data = uops_data[31:0];
  assign uop_mem__T_397_addr = _T_395[9:0];
  assign uop_mem__T_397_mask = 1'h1;
  assign uop_mem__T_397_en = uops_read_en;
  assign uop_mem__T_403_data = uops_data[63:32];
  assign uop_mem__T_403_addr = _T_401[9:0];
  assign uop_mem__T_403_mask = 1'h1;
  assign uop_mem__T_403_en = uops_read_en;
  assign uop_mem__T_409_data = uops_data[95:64];
  assign uop_mem__T_409_addr = _T_407[9:0];
  assign uop_mem__T_409_mask = 1'h1;
  assign uop_mem__T_409_en = uops_read_en;
  assign uop_mem__T_415_data = uops_data[127:96];
  assign uop_mem__T_415_addr = _T_413[9:0];
  assign uop_mem__T_415_mask = 1'h1;
  assign uop_mem__T_415_en = uops_read_en;
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
  assign _T_208 = $signed(imm_raw); // @[Compute.scala 73:33]
  assign _T_210 = $signed(_T_208) < $signed(16'sh0); // @[Compute.scala 73:40]
  assign _T_212 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_214 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_215 = _T_210 ? _T_212 : {{15'd0}, _T_214}; // @[Compute.scala 73:24]
  assign _GEN_286 = {{12'd0}, y_pad_0}; // @[Compute.scala 77:30]
  assign _GEN_288 = {{12'd0}, x_pad_0}; // @[Compute.scala 78:30]
  assign _T_221 = _GEN_288 + x_size; // @[Compute.scala 78:30]
  assign _T_222 = _GEN_288 + x_size; // @[Compute.scala 78:30]
  assign _GEN_289 = {{12'd0}, x_pad_1}; // @[Compute.scala 78:39]
  assign _T_223 = _T_222 + _GEN_289; // @[Compute.scala 78:39]
  assign x_size_total = _T_222 + _GEN_289; // @[Compute.scala 78:39]
  assign y_offset = x_size_total * _GEN_286; // @[Compute.scala 79:31]
  assign opcode_finish_en = opcode == 3'h3; // @[Compute.scala 82:34]
  assign _T_226 = opcode == 3'h0; // @[Compute.scala 83:32]
  assign _T_228 = opcode == 3'h1; // @[Compute.scala 83:60]
  assign opcode_load_en = _T_226 | _T_228; // @[Compute.scala 83:50]
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
  assign _T_252 = uop_cntr_max_val == 16'h0; // @[Compute.scala 112:43]
  assign uop_cntr_max = _T_252 ? 16'h1 : uop_cntr_max_val; // @[Compute.scala 112:25]
  assign _T_254 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 113:37]
  assign uop_cntr_en = _T_254 & insn_valid; // @[Compute.scala 113:59]
  assign _T_256 = uop_cntr_val == uop_cntr_max; // @[Compute.scala 116:38]
  assign _T_257 = _T_256 & uop_cntr_en; // @[Compute.scala 116:56]
  assign uop_cntr_wrap = _T_257 & busy; // @[Compute.scala 116:71]
  assign _T_259 = x_size * 16'h4; // @[Compute.scala 118:29]
  assign _T_261 = _T_259 + 19'h1; // @[Compute.scala 118:46]
  assign acc_cntr_max = _T_259 + 19'h1; // @[Compute.scala 118:46]
  assign _T_262 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 119:37]
  assign acc_cntr_en = _T_262 & insn_valid; // @[Compute.scala 119:59]
  assign _GEN_291 = {{3'd0}, acc_cntr_val}; // @[Compute.scala 122:38]
  assign _T_264 = _GEN_291 == acc_cntr_max; // @[Compute.scala 122:38]
  assign _T_265 = _T_264 & acc_cntr_en; // @[Compute.scala 122:56]
  assign acc_cntr_wrap = _T_265 & busy; // @[Compute.scala 122:71]
  assign _T_266 = uop_end - uop_bgn; // @[Compute.scala 124:34]
  assign _T_267 = $unsigned(_T_266); // @[Compute.scala 124:34]
  assign upc_cntr_max_val = _T_267[15:0]; // @[Compute.scala 124:34]
  assign _T_269 = upc_cntr_max_val <= 16'h0; // @[Compute.scala 125:43]
  assign upc_cntr_max = _T_269 ? 16'h1 : upc_cntr_max_val; // @[Compute.scala 125:25]
  assign _T_271 = iter_in * iter_out; // @[Compute.scala 126:35]
  assign _T_272 = _T_271[15:0]; // @[Compute.scala 126:46]
  assign out_cntr_max_val = _T_272 * upc_cntr_max; // @[Compute.scala 126:54]
  assign _T_274 = out_cntr_max_val + 32'h2; // @[Compute.scala 127:39]
  assign out_cntr_max = out_cntr_max_val + 32'h2; // @[Compute.scala 127:39]
  assign _T_275 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 128:37]
  assign out_cntr_en = _T_275 & insn_valid; // @[Compute.scala 128:56]
  assign _GEN_292 = {{16'd0}, out_cntr_val}; // @[Compute.scala 131:38]
  assign _T_277 = _GEN_292 == out_cntr_max; // @[Compute.scala 131:38]
  assign _T_278 = _T_277 & out_cntr_en; // @[Compute.scala 131:56]
  assign out_cntr_wrap = _T_278 & busy; // @[Compute.scala 131:71]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Compute.scala 136:43]
  assign push_next_dep_valid = push_next_dep & push; // @[Compute.scala 137:43]
  assign _T_291 = pop_prev_dep_ready & busy; // @[Compute.scala 146:68]
  assign _T_292 = pop_next_dep_ready & busy; // @[Compute.scala 147:68]
  assign _T_293 = push_prev_dep_ready & busy; // @[Compute.scala 148:68]
  assign _T_294 = push_next_dep_ready & busy; // @[Compute.scala 149:68]
  assign _GEN_0 = push_next_dep ? _T_294 : 1'h0; // @[Compute.scala 149:31]
  assign _GEN_1 = push_prev_dep ? _T_293 : _GEN_0; // @[Compute.scala 148:31]
  assign _GEN_2 = pop_next_dep ? _T_292 : _GEN_1; // @[Compute.scala 147:31]
  assign _GEN_3 = pop_prev_dep ? _T_291 : _GEN_2; // @[Compute.scala 146:31]
  assign _GEN_4 = opcode_finish_en ? _GEN_3 : 1'h0; // @[Compute.scala 145:27]
  assign _T_297 = uop_cntr_wrap | acc_cntr_wrap; // @[Compute.scala 152:23]
  assign _T_298 = _T_297 | out_cntr_wrap; // @[Compute.scala 152:40]
  assign _T_299 = _T_298 | finish_wrap; // @[Compute.scala 152:57]
  assign _T_300 = push_prev_dep | push_next_dep; // @[Compute.scala 153:25]
  assign _GEN_5 = _T_300 ? 3'h3 : 3'h4; // @[Compute.scala 153:43]
  assign _GEN_6 = _T_299 ? _GEN_5 : state; // @[Compute.scala 152:73]
  assign _T_302 = pop_prev_dep_ready == 1'h0; // @[Compute.scala 161:18]
  assign _T_304 = pop_next_dep_ready == 1'h0; // @[Compute.scala 161:41]
  assign _T_305 = _T_302 & _T_304; // @[Compute.scala 161:38]
  assign _T_306 = busy & _T_305; // @[Compute.scala 161:14]
  assign _T_307 = pop_prev_dep | pop_next_dep; // @[Compute.scala 161:79]
  assign _T_308 = _T_306 & _T_307; // @[Compute.scala 161:62]
  assign _GEN_7 = _T_308 ? 3'h1 : _GEN_6; // @[Compute.scala 161:97]
  assign _T_309 = pop_prev_dep_ready | pop_next_dep_ready; // @[Compute.scala 162:38]
  assign _T_310 = dump & _T_309; // @[Compute.scala 162:14]
  assign _GEN_8 = _T_310 ? 3'h2 : _GEN_7; // @[Compute.scala 162:63]
  assign _T_311 = push_prev_dep_ready | push_next_dep_ready; // @[Compute.scala 163:38]
  assign _T_312 = push & _T_311; // @[Compute.scala 163:14]
  assign _GEN_9 = _T_312 ? 3'h4 : _GEN_8; // @[Compute.scala 163:63]
  assign _T_315 = pop_prev_dep & dump; // @[Compute.scala 170:22]
  assign _T_316 = _T_315 & io_l2g_dep_queue_valid; // @[Compute.scala 170:30]
  assign _GEN_10 = _T_316 ? 1'h1 : pop_prev_dep_ready; // @[Compute.scala 170:57]
  assign _T_318 = pop_next_dep & dump; // @[Compute.scala 173:22]
  assign _T_319 = _T_318 & io_s2g_dep_queue_valid; // @[Compute.scala 173:30]
  assign _GEN_11 = _T_319 ? 1'h1 : pop_next_dep_ready; // @[Compute.scala 173:57]
  assign _T_323 = push_prev_dep_valid & io_g2l_dep_queue_ready; // @[Compute.scala 180:29]
  assign _T_324 = _T_323 & push; // @[Compute.scala 180:55]
  assign _GEN_12 = _T_324 ? 1'h1 : push_prev_dep_ready; // @[Compute.scala 180:64]
  assign _T_326 = push_next_dep_valid & io_g2s_dep_queue_ready; // @[Compute.scala 183:29]
  assign _T_327 = _T_326 & push; // @[Compute.scala 183:55]
  assign _GEN_13 = _T_327 ? 1'h1 : push_next_dep_ready; // @[Compute.scala 183:64]
  assign _T_330 = io_uops_waitrequest == 1'h0; // @[Compute.scala 188:22]
  assign _T_331 = uops_read & _T_330; // @[Compute.scala 188:19]
  assign _T_332 = _T_331 & busy; // @[Compute.scala 188:37]
  assign _T_333 = uop_cntr_val < uop_cntr_max; // @[Compute.scala 188:61]
  assign _T_334 = _T_332 & _T_333; // @[Compute.scala 188:45]
  assign _T_336 = uop_cntr_val + 16'h1; // @[Compute.scala 189:34]
  assign _T_337 = uop_cntr_val + 16'h1; // @[Compute.scala 189:34]
  assign _GEN_14 = _T_334 ? _T_337 : uop_cntr_val; // @[Compute.scala 188:77]
  assign _T_339 = io_biases_waitrequest == 1'h0; // @[Compute.scala 191:24]
  assign _T_340 = biases_read & _T_339; // @[Compute.scala 191:21]
  assign _T_341 = _T_340 & busy; // @[Compute.scala 191:39]
  assign _T_342 = _GEN_291 < acc_cntr_max; // @[Compute.scala 191:63]
  assign _T_343 = _T_341 & _T_342; // @[Compute.scala 191:47]
  assign _T_345 = acc_cntr_val + 16'h1; // @[Compute.scala 192:34]
  assign _T_346 = acc_cntr_val + 16'h1; // @[Compute.scala 192:34]
  assign _GEN_15 = _T_343 ? _T_346 : acc_cntr_val; // @[Compute.scala 191:79]
  assign _T_347 = out_mem_write & busy; // @[Compute.scala 197:23]
  assign _T_348 = _GEN_292 < out_cntr_max; // @[Compute.scala 197:47]
  assign _T_349 = _T_347 & _T_348; // @[Compute.scala 197:31]
  assign _T_351 = out_cntr_val + 16'h1; // @[Compute.scala 198:34]
  assign _T_352 = out_cntr_val + 16'h1; // @[Compute.scala 198:34]
  assign _GEN_16 = _T_349 ? _T_352 : out_cntr_val; // @[Compute.scala 197:63]
  assign _GEN_21 = gemm_queue_ready ? 1'h0 : _GEN_10; // @[Compute.scala 202:27]
  assign _GEN_22 = gemm_queue_ready ? 1'h0 : _GEN_11; // @[Compute.scala 202:27]
  assign _GEN_23 = gemm_queue_ready ? 1'h0 : _GEN_12; // @[Compute.scala 202:27]
  assign _GEN_24 = gemm_queue_ready ? 1'h0 : _GEN_13; // @[Compute.scala 202:27]
  assign _GEN_25 = gemm_queue_ready ? 3'h2 : _GEN_9; // @[Compute.scala 202:27]
  assign _T_360 = idle | done; // @[Compute.scala 215:52]
  assign _T_361 = io_gemm_queue_valid & _T_360; // @[Compute.scala 215:43]
  assign _GEN_26 = gemm_queue_ready ? 1'h0 : _T_361; // @[Compute.scala 217:27]
  assign _GEN_295 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 227:33]
  assign _T_366 = dram_base + _GEN_295; // @[Compute.scala 227:33]
  assign _T_367 = dram_base + _GEN_295; // @[Compute.scala 227:33]
  assign _GEN_296 = {{7'd0}, _T_367}; // @[Compute.scala 227:49]
  assign uop_dram_addr = _GEN_296 << 3'h4; // @[Compute.scala 227:49]
  assign _T_371 = sram_base + uop_cntr_val; // @[Compute.scala 230:30]
  assign _T_372 = sram_base + uop_cntr_val; // @[Compute.scala 230:30]
  assign _GEN_297 = {{3'd0}, _T_372}; // @[Compute.scala 230:46]
  assign _T_374 = _GEN_297 << 2'h2; // @[Compute.scala 230:46]
  assign _T_376 = uop_cntr_wrap == 1'h0; // @[Compute.scala 231:31]
  assign _T_377 = uop_cntr_en & _T_376; // @[Compute.scala 231:28]
  assign _T_378 = _T_377 & busy; // @[Compute.scala 231:46]
  assign uop_sram_addr = {{13'd0}, _T_374}; // @[Compute.scala 228:27 Compute.scala 229:17 Compute.scala 230:17]
  assign _T_383 = {uop_sram_addr,io_uops_readdata}; // @[Cat.scala 30:58]
  assign _T_386 = uop_cntr_max - 16'h1; // @[Compute.scala 240:42]
  assign _T_387 = $unsigned(_T_386); // @[Compute.scala 240:42]
  assign _T_388 = _T_387[15:0]; // @[Compute.scala 240:42]
  assign _T_389 = uop_cntr_val == _T_388; // @[Compute.scala 240:24]
  assign _GEN_27 = _T_389 ? 1'h0 : _T_378; // @[Compute.scala 240:50]
  assign _T_392 = uops_data[159:128]; // @[Compute.scala 243:35]
  assign _T_394 = {{1'd0}, _T_392}; // @[Compute.scala 245:30]
  assign _T_395 = _T_394[31:0]; // @[Compute.scala 245:30]
  assign _T_400 = _T_392 + 32'h1; // @[Compute.scala 245:30]
  assign _T_401 = _T_392 + 32'h1; // @[Compute.scala 245:30]
  assign _T_406 = _T_392 + 32'h2; // @[Compute.scala 245:30]
  assign _T_407 = _T_392 + 32'h2; // @[Compute.scala 245:30]
  assign _T_412 = _T_392 + 32'h3; // @[Compute.scala 245:30]
  assign _T_413 = _T_392 + 32'h3; // @[Compute.scala 245:30]
  assign _GEN_298 = {{12'd0}, y_offset}; // @[Compute.scala 250:36]
  assign _T_417 = dram_base + _GEN_298; // @[Compute.scala 250:36]
  assign _T_418 = dram_base + _GEN_298; // @[Compute.scala 250:36]
  assign _GEN_299 = {{28'd0}, x_pad_0}; // @[Compute.scala 250:47]
  assign _T_419 = _T_418 + _GEN_299; // @[Compute.scala 250:47]
  assign _T_420 = _T_418 + _GEN_299; // @[Compute.scala 250:47]
  assign _GEN_300 = {{3'd0}, _T_420}; // @[Compute.scala 250:58]
  assign _T_422 = _GEN_300 << 2'h2; // @[Compute.scala 250:58]
  assign _T_424 = _T_422 * 35'h1; // @[Compute.scala 250:66]
  assign _GEN_301 = {{20'd0}, acc_cntr_val}; // @[Compute.scala 250:76]
  assign _T_425 = _T_424 + _GEN_301; // @[Compute.scala 250:76]
  assign _T_426 = _T_424 + _GEN_301; // @[Compute.scala 250:76]
  assign _GEN_302 = {{7'd0}, _T_426}; // @[Compute.scala 250:92]
  assign acc_dram_addr = _GEN_302 << 3'h4; // @[Compute.scala 250:92]
  assign _GEN_303 = {{4'd0}, sram_base}; // @[Compute.scala 251:36]
  assign _T_428 = _GEN_303 + y_offset; // @[Compute.scala 251:36]
  assign _T_429 = _GEN_303 + y_offset; // @[Compute.scala 251:36]
  assign _GEN_304 = {{16'd0}, x_pad_0}; // @[Compute.scala 251:47]
  assign _T_430 = _T_429 + _GEN_304; // @[Compute.scala 251:47]
  assign _T_431 = _T_429 + _GEN_304; // @[Compute.scala 251:47]
  assign _GEN_305 = {{3'd0}, _T_431}; // @[Compute.scala 251:58]
  assign _T_433 = _GEN_305 << 2'h2; // @[Compute.scala 251:58]
  assign _T_435 = _T_433 * 23'h1; // @[Compute.scala 251:66]
  assign _GEN_306 = {{8'd0}, acc_cntr_val}; // @[Compute.scala 251:76]
  assign _T_436 = _T_435 + _GEN_306; // @[Compute.scala 251:76]
  assign _T_437 = _T_435 + _GEN_306; // @[Compute.scala 251:76]
  assign _T_439 = _T_437 >> 2'h2; // @[Compute.scala 251:92]
  assign _T_441 = _T_439 - 24'h1; // @[Compute.scala 251:121]
  assign _T_442 = $unsigned(_T_441); // @[Compute.scala 251:121]
  assign acc_sram_addr = _T_442[23:0]; // @[Compute.scala 251:121]
  assign _T_444 = done == 1'h0; // @[Compute.scala 252:33]
  assign _GEN_17 = acc_cntr_val % 16'h4; // @[Compute.scala 258:30]
  assign _T_450 = _GEN_17[2:0]; // @[Compute.scala 258:30]
  assign _GEN_42 = 3'h0 == _T_450 ? io_biases_readdata : biases_data_0; // @[Compute.scala 258:48]
  assign _GEN_43 = 3'h1 == _T_450 ? io_biases_readdata : biases_data_1; // @[Compute.scala 258:48]
  assign _GEN_44 = 3'h2 == _T_450 ? io_biases_readdata : biases_data_2; // @[Compute.scala 258:48]
  assign _GEN_45 = 3'h3 == _T_450 ? io_biases_readdata : biases_data_3; // @[Compute.scala 258:48]
  assign _T_456 = _T_450 == 3'h0; // @[Compute.scala 262:43]
  assign _T_459 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_460 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign _GEN_18 = out_cntr_val % upc_cntr_max; // @[Compute.scala 268:26]
  assign upc = _GEN_18[15:0]; // @[Compute.scala 268:26]
  assign _GEN_19 = _T_466 % _T_272; // @[Compute.scala 272:37]
  assign it_in = _GEN_19[15:0]; // @[Compute.scala 272:37]
  assign _T_469 = it_in * 16'h1; // @[Compute.scala 273:47]
  assign _T_470 = _T_469[15:0]; // @[Compute.scala 273:63]
  assign _T_471 = {{1'd0}, _T_470}; // @[Compute.scala 273:38]
  assign dst_offset_in = _T_471[15:0]; // @[Compute.scala 273:38]
  assign _T_475 = uop[10:0]; // @[Compute.scala 275:20]
  assign _GEN_307 = {{5'd0}, _T_475}; // @[Compute.scala 275:47]
  assign _T_476 = _GEN_307 + dst_offset_in; // @[Compute.scala 275:47]
  assign dst_idx = _GEN_307 + dst_offset_in; // @[Compute.scala 275:47]
  assign _T_477 = uop[21:11]; // @[Compute.scala 276:20]
  assign _GEN_308 = {{5'd0}, _T_477}; // @[Compute.scala 276:47]
  assign _T_478 = _GEN_308 + dst_offset_in; // @[Compute.scala 276:47]
  assign src_idx = _GEN_308 + dst_offset_in; // @[Compute.scala 276:47]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 295:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 296:38]
  assign _T_916 = insn_valid & out_cntr_en; // @[Compute.scala 320:20]
  assign _T_917 = dst_vector[31:0]; // @[Compute.scala 322:29]
  assign _T_918 = $signed(_T_917); // @[Compute.scala 322:70]
  assign _T_919 = src_vector[31:0]; // @[Compute.scala 323:29]
  assign _T_920 = $signed(_T_919); // @[Compute.scala 323:70]
  assign _T_921 = dst_vector[63:32]; // @[Compute.scala 322:29]
  assign _T_922 = $signed(_T_921); // @[Compute.scala 322:70]
  assign _T_923 = src_vector[63:32]; // @[Compute.scala 323:29]
  assign _T_924 = $signed(_T_923); // @[Compute.scala 323:70]
  assign _T_925 = dst_vector[95:64]; // @[Compute.scala 322:29]
  assign _T_926 = $signed(_T_925); // @[Compute.scala 322:70]
  assign _T_927 = src_vector[95:64]; // @[Compute.scala 323:29]
  assign _T_928 = $signed(_T_927); // @[Compute.scala 323:70]
  assign _T_929 = dst_vector[127:96]; // @[Compute.scala 322:29]
  assign _T_930 = $signed(_T_929); // @[Compute.scala 322:70]
  assign _T_931 = src_vector[127:96]; // @[Compute.scala 323:29]
  assign _T_932 = $signed(_T_931); // @[Compute.scala 323:70]
  assign _T_933 = dst_vector[159:128]; // @[Compute.scala 322:29]
  assign _T_934 = $signed(_T_933); // @[Compute.scala 322:70]
  assign _T_935 = src_vector[159:128]; // @[Compute.scala 323:29]
  assign _T_936 = $signed(_T_935); // @[Compute.scala 323:70]
  assign _T_937 = dst_vector[191:160]; // @[Compute.scala 322:29]
  assign _T_938 = $signed(_T_937); // @[Compute.scala 322:70]
  assign _T_939 = src_vector[191:160]; // @[Compute.scala 323:29]
  assign _T_940 = $signed(_T_939); // @[Compute.scala 323:70]
  assign _T_941 = dst_vector[223:192]; // @[Compute.scala 322:29]
  assign _T_942 = $signed(_T_941); // @[Compute.scala 322:70]
  assign _T_943 = src_vector[223:192]; // @[Compute.scala 323:29]
  assign _T_944 = $signed(_T_943); // @[Compute.scala 323:70]
  assign _T_945 = dst_vector[255:224]; // @[Compute.scala 322:29]
  assign _T_946 = $signed(_T_945); // @[Compute.scala 322:70]
  assign _T_947 = src_vector[255:224]; // @[Compute.scala 323:29]
  assign _T_948 = $signed(_T_947); // @[Compute.scala 323:70]
  assign _T_949 = dst_vector[287:256]; // @[Compute.scala 322:29]
  assign _T_950 = $signed(_T_949); // @[Compute.scala 322:70]
  assign _T_951 = src_vector[287:256]; // @[Compute.scala 323:29]
  assign _T_952 = $signed(_T_951); // @[Compute.scala 323:70]
  assign _T_953 = dst_vector[319:288]; // @[Compute.scala 322:29]
  assign _T_954 = $signed(_T_953); // @[Compute.scala 322:70]
  assign _T_955 = src_vector[319:288]; // @[Compute.scala 323:29]
  assign _T_956 = $signed(_T_955); // @[Compute.scala 323:70]
  assign _T_957 = dst_vector[351:320]; // @[Compute.scala 322:29]
  assign _T_958 = $signed(_T_957); // @[Compute.scala 322:70]
  assign _T_959 = src_vector[351:320]; // @[Compute.scala 323:29]
  assign _T_960 = $signed(_T_959); // @[Compute.scala 323:70]
  assign _T_961 = dst_vector[383:352]; // @[Compute.scala 322:29]
  assign _T_962 = $signed(_T_961); // @[Compute.scala 322:70]
  assign _T_963 = src_vector[383:352]; // @[Compute.scala 323:29]
  assign _T_964 = $signed(_T_963); // @[Compute.scala 323:70]
  assign _T_965 = dst_vector[415:384]; // @[Compute.scala 322:29]
  assign _T_966 = $signed(_T_965); // @[Compute.scala 322:70]
  assign _T_967 = src_vector[415:384]; // @[Compute.scala 323:29]
  assign _T_968 = $signed(_T_967); // @[Compute.scala 323:70]
  assign _T_969 = dst_vector[447:416]; // @[Compute.scala 322:29]
  assign _T_970 = $signed(_T_969); // @[Compute.scala 322:70]
  assign _T_971 = src_vector[447:416]; // @[Compute.scala 323:29]
  assign _T_972 = $signed(_T_971); // @[Compute.scala 323:70]
  assign _T_973 = dst_vector[479:448]; // @[Compute.scala 322:29]
  assign _T_974 = $signed(_T_973); // @[Compute.scala 322:70]
  assign _T_975 = src_vector[479:448]; // @[Compute.scala 323:29]
  assign _T_976 = $signed(_T_975); // @[Compute.scala 323:70]
  assign _T_977 = dst_vector[511:480]; // @[Compute.scala 322:29]
  assign _T_978 = $signed(_T_977); // @[Compute.scala 322:70]
  assign _T_979 = src_vector[511:480]; // @[Compute.scala 323:29]
  assign _T_980 = $signed(_T_979); // @[Compute.scala 323:70]
  assign _GEN_68 = use_imm ? $signed(imm) : $signed(_T_920); // @[Compute.scala 325:20]
  assign _GEN_69 = use_imm ? $signed(imm) : $signed(_T_924); // @[Compute.scala 325:20]
  assign _GEN_70 = use_imm ? $signed(imm) : $signed(_T_928); // @[Compute.scala 325:20]
  assign _GEN_71 = use_imm ? $signed(imm) : $signed(_T_932); // @[Compute.scala 325:20]
  assign _GEN_72 = use_imm ? $signed(imm) : $signed(_T_936); // @[Compute.scala 325:20]
  assign _GEN_73 = use_imm ? $signed(imm) : $signed(_T_940); // @[Compute.scala 325:20]
  assign _GEN_74 = use_imm ? $signed(imm) : $signed(_T_944); // @[Compute.scala 325:20]
  assign _GEN_75 = use_imm ? $signed(imm) : $signed(_T_948); // @[Compute.scala 325:20]
  assign _GEN_76 = use_imm ? $signed(imm) : $signed(_T_952); // @[Compute.scala 325:20]
  assign _GEN_77 = use_imm ? $signed(imm) : $signed(_T_956); // @[Compute.scala 325:20]
  assign _GEN_78 = use_imm ? $signed(imm) : $signed(_T_960); // @[Compute.scala 325:20]
  assign _GEN_79 = use_imm ? $signed(imm) : $signed(_T_964); // @[Compute.scala 325:20]
  assign _GEN_80 = use_imm ? $signed(imm) : $signed(_T_968); // @[Compute.scala 325:20]
  assign _GEN_81 = use_imm ? $signed(imm) : $signed(_T_972); // @[Compute.scala 325:20]
  assign _GEN_82 = use_imm ? $signed(imm) : $signed(_T_976); // @[Compute.scala 325:20]
  assign _GEN_83 = use_imm ? $signed(imm) : $signed(_T_980); // @[Compute.scala 325:20]
  assign src_0_0 = _T_916 ? $signed(_T_918) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_0 = _T_916 ? $signed(_GEN_68) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_981 = $signed(src_0_0) > $signed(src_1_0); // @[Compute.scala 330:57]
  assign _T_982 = _T_981 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 330:47]
  assign _T_983 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 331:57]
  assign _T_984 = _T_983 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 331:47]
  assign _T_985 = alu_opcode_max_en ? $signed(_T_982) : $signed(_T_984); // @[Compute.scala 330:24]
  assign mix_val_0 = _T_916 ? $signed(_T_985) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_986 = mix_val_0[7:0]; // @[Compute.scala 333:37]
  assign _T_987 = $unsigned(src_0_0); // @[Compute.scala 334:30]
  assign _T_988 = $unsigned(src_1_0); // @[Compute.scala 334:59]
  assign _T_989 = _T_987 + _T_988; // @[Compute.scala 334:49]
  assign _T_990 = _T_987 + _T_988; // @[Compute.scala 334:49]
  assign _T_991 = $signed(_T_990); // @[Compute.scala 334:79]
  assign add_val_0 = _T_916 ? $signed(_T_991) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_0 = _T_916 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_992 = add_res_0[7:0]; // @[Compute.scala 336:37]
  assign _T_994 = src_1_0[4:0]; // @[Compute.scala 337:60]
  assign _T_995 = _T_987 >> _T_994; // @[Compute.scala 337:49]
  assign _T_996 = $signed(_T_995); // @[Compute.scala 337:84]
  assign shr_val_0 = _T_916 ? $signed(_T_996) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_0 = _T_916 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_997 = shr_res_0[7:0]; // @[Compute.scala 339:37]
  assign src_0_1 = _T_916 ? $signed(_T_922) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_1 = _T_916 ? $signed(_GEN_69) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_998 = $signed(src_0_1) > $signed(src_1_1); // @[Compute.scala 330:57]
  assign _T_999 = _T_998 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 330:47]
  assign _T_1000 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 331:57]
  assign _T_1001 = _T_1000 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 331:47]
  assign _T_1002 = alu_opcode_max_en ? $signed(_T_999) : $signed(_T_1001); // @[Compute.scala 330:24]
  assign mix_val_1 = _T_916 ? $signed(_T_1002) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1003 = mix_val_1[7:0]; // @[Compute.scala 333:37]
  assign _T_1004 = $unsigned(src_0_1); // @[Compute.scala 334:30]
  assign _T_1005 = $unsigned(src_1_1); // @[Compute.scala 334:59]
  assign _T_1006 = _T_1004 + _T_1005; // @[Compute.scala 334:49]
  assign _T_1007 = _T_1004 + _T_1005; // @[Compute.scala 334:49]
  assign _T_1008 = $signed(_T_1007); // @[Compute.scala 334:79]
  assign add_val_1 = _T_916 ? $signed(_T_1008) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_1 = _T_916 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1009 = add_res_1[7:0]; // @[Compute.scala 336:37]
  assign _T_1011 = src_1_1[4:0]; // @[Compute.scala 337:60]
  assign _T_1012 = _T_1004 >> _T_1011; // @[Compute.scala 337:49]
  assign _T_1013 = $signed(_T_1012); // @[Compute.scala 337:84]
  assign shr_val_1 = _T_916 ? $signed(_T_1013) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_1 = _T_916 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1014 = shr_res_1[7:0]; // @[Compute.scala 339:37]
  assign src_0_2 = _T_916 ? $signed(_T_926) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_2 = _T_916 ? $signed(_GEN_70) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1015 = $signed(src_0_2) > $signed(src_1_2); // @[Compute.scala 330:57]
  assign _T_1016 = _T_1015 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 330:47]
  assign _T_1017 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 331:57]
  assign _T_1018 = _T_1017 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 331:47]
  assign _T_1019 = alu_opcode_max_en ? $signed(_T_1016) : $signed(_T_1018); // @[Compute.scala 330:24]
  assign mix_val_2 = _T_916 ? $signed(_T_1019) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1020 = mix_val_2[7:0]; // @[Compute.scala 333:37]
  assign _T_1021 = $unsigned(src_0_2); // @[Compute.scala 334:30]
  assign _T_1022 = $unsigned(src_1_2); // @[Compute.scala 334:59]
  assign _T_1023 = _T_1021 + _T_1022; // @[Compute.scala 334:49]
  assign _T_1024 = _T_1021 + _T_1022; // @[Compute.scala 334:49]
  assign _T_1025 = $signed(_T_1024); // @[Compute.scala 334:79]
  assign add_val_2 = _T_916 ? $signed(_T_1025) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_2 = _T_916 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1026 = add_res_2[7:0]; // @[Compute.scala 336:37]
  assign _T_1028 = src_1_2[4:0]; // @[Compute.scala 337:60]
  assign _T_1029 = _T_1021 >> _T_1028; // @[Compute.scala 337:49]
  assign _T_1030 = $signed(_T_1029); // @[Compute.scala 337:84]
  assign shr_val_2 = _T_916 ? $signed(_T_1030) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_2 = _T_916 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1031 = shr_res_2[7:0]; // @[Compute.scala 339:37]
  assign src_0_3 = _T_916 ? $signed(_T_930) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_3 = _T_916 ? $signed(_GEN_71) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1032 = $signed(src_0_3) > $signed(src_1_3); // @[Compute.scala 330:57]
  assign _T_1033 = _T_1032 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 330:47]
  assign _T_1034 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 331:57]
  assign _T_1035 = _T_1034 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 331:47]
  assign _T_1036 = alu_opcode_max_en ? $signed(_T_1033) : $signed(_T_1035); // @[Compute.scala 330:24]
  assign mix_val_3 = _T_916 ? $signed(_T_1036) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1037 = mix_val_3[7:0]; // @[Compute.scala 333:37]
  assign _T_1038 = $unsigned(src_0_3); // @[Compute.scala 334:30]
  assign _T_1039 = $unsigned(src_1_3); // @[Compute.scala 334:59]
  assign _T_1040 = _T_1038 + _T_1039; // @[Compute.scala 334:49]
  assign _T_1041 = _T_1038 + _T_1039; // @[Compute.scala 334:49]
  assign _T_1042 = $signed(_T_1041); // @[Compute.scala 334:79]
  assign add_val_3 = _T_916 ? $signed(_T_1042) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_3 = _T_916 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1043 = add_res_3[7:0]; // @[Compute.scala 336:37]
  assign _T_1045 = src_1_3[4:0]; // @[Compute.scala 337:60]
  assign _T_1046 = _T_1038 >> _T_1045; // @[Compute.scala 337:49]
  assign _T_1047 = $signed(_T_1046); // @[Compute.scala 337:84]
  assign shr_val_3 = _T_916 ? $signed(_T_1047) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_3 = _T_916 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1048 = shr_res_3[7:0]; // @[Compute.scala 339:37]
  assign src_0_4 = _T_916 ? $signed(_T_934) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_4 = _T_916 ? $signed(_GEN_72) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1049 = $signed(src_0_4) > $signed(src_1_4); // @[Compute.scala 330:57]
  assign _T_1050 = _T_1049 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 330:47]
  assign _T_1051 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 331:57]
  assign _T_1052 = _T_1051 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 331:47]
  assign _T_1053 = alu_opcode_max_en ? $signed(_T_1050) : $signed(_T_1052); // @[Compute.scala 330:24]
  assign mix_val_4 = _T_916 ? $signed(_T_1053) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1054 = mix_val_4[7:0]; // @[Compute.scala 333:37]
  assign _T_1055 = $unsigned(src_0_4); // @[Compute.scala 334:30]
  assign _T_1056 = $unsigned(src_1_4); // @[Compute.scala 334:59]
  assign _T_1057 = _T_1055 + _T_1056; // @[Compute.scala 334:49]
  assign _T_1058 = _T_1055 + _T_1056; // @[Compute.scala 334:49]
  assign _T_1059 = $signed(_T_1058); // @[Compute.scala 334:79]
  assign add_val_4 = _T_916 ? $signed(_T_1059) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_4 = _T_916 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1060 = add_res_4[7:0]; // @[Compute.scala 336:37]
  assign _T_1062 = src_1_4[4:0]; // @[Compute.scala 337:60]
  assign _T_1063 = _T_1055 >> _T_1062; // @[Compute.scala 337:49]
  assign _T_1064 = $signed(_T_1063); // @[Compute.scala 337:84]
  assign shr_val_4 = _T_916 ? $signed(_T_1064) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_4 = _T_916 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1065 = shr_res_4[7:0]; // @[Compute.scala 339:37]
  assign src_0_5 = _T_916 ? $signed(_T_938) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_5 = _T_916 ? $signed(_GEN_73) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1066 = $signed(src_0_5) > $signed(src_1_5); // @[Compute.scala 330:57]
  assign _T_1067 = _T_1066 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 330:47]
  assign _T_1068 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 331:57]
  assign _T_1069 = _T_1068 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 331:47]
  assign _T_1070 = alu_opcode_max_en ? $signed(_T_1067) : $signed(_T_1069); // @[Compute.scala 330:24]
  assign mix_val_5 = _T_916 ? $signed(_T_1070) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1071 = mix_val_5[7:0]; // @[Compute.scala 333:37]
  assign _T_1072 = $unsigned(src_0_5); // @[Compute.scala 334:30]
  assign _T_1073 = $unsigned(src_1_5); // @[Compute.scala 334:59]
  assign _T_1074 = _T_1072 + _T_1073; // @[Compute.scala 334:49]
  assign _T_1075 = _T_1072 + _T_1073; // @[Compute.scala 334:49]
  assign _T_1076 = $signed(_T_1075); // @[Compute.scala 334:79]
  assign add_val_5 = _T_916 ? $signed(_T_1076) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_5 = _T_916 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1077 = add_res_5[7:0]; // @[Compute.scala 336:37]
  assign _T_1079 = src_1_5[4:0]; // @[Compute.scala 337:60]
  assign _T_1080 = _T_1072 >> _T_1079; // @[Compute.scala 337:49]
  assign _T_1081 = $signed(_T_1080); // @[Compute.scala 337:84]
  assign shr_val_5 = _T_916 ? $signed(_T_1081) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_5 = _T_916 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1082 = shr_res_5[7:0]; // @[Compute.scala 339:37]
  assign src_0_6 = _T_916 ? $signed(_T_942) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_6 = _T_916 ? $signed(_GEN_74) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1083 = $signed(src_0_6) > $signed(src_1_6); // @[Compute.scala 330:57]
  assign _T_1084 = _T_1083 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 330:47]
  assign _T_1085 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 331:57]
  assign _T_1086 = _T_1085 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 331:47]
  assign _T_1087 = alu_opcode_max_en ? $signed(_T_1084) : $signed(_T_1086); // @[Compute.scala 330:24]
  assign mix_val_6 = _T_916 ? $signed(_T_1087) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1088 = mix_val_6[7:0]; // @[Compute.scala 333:37]
  assign _T_1089 = $unsigned(src_0_6); // @[Compute.scala 334:30]
  assign _T_1090 = $unsigned(src_1_6); // @[Compute.scala 334:59]
  assign _T_1091 = _T_1089 + _T_1090; // @[Compute.scala 334:49]
  assign _T_1092 = _T_1089 + _T_1090; // @[Compute.scala 334:49]
  assign _T_1093 = $signed(_T_1092); // @[Compute.scala 334:79]
  assign add_val_6 = _T_916 ? $signed(_T_1093) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_6 = _T_916 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1094 = add_res_6[7:0]; // @[Compute.scala 336:37]
  assign _T_1096 = src_1_6[4:0]; // @[Compute.scala 337:60]
  assign _T_1097 = _T_1089 >> _T_1096; // @[Compute.scala 337:49]
  assign _T_1098 = $signed(_T_1097); // @[Compute.scala 337:84]
  assign shr_val_6 = _T_916 ? $signed(_T_1098) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_6 = _T_916 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1099 = shr_res_6[7:0]; // @[Compute.scala 339:37]
  assign src_0_7 = _T_916 ? $signed(_T_946) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_7 = _T_916 ? $signed(_GEN_75) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1100 = $signed(src_0_7) > $signed(src_1_7); // @[Compute.scala 330:57]
  assign _T_1101 = _T_1100 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 330:47]
  assign _T_1102 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 331:57]
  assign _T_1103 = _T_1102 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 331:47]
  assign _T_1104 = alu_opcode_max_en ? $signed(_T_1101) : $signed(_T_1103); // @[Compute.scala 330:24]
  assign mix_val_7 = _T_916 ? $signed(_T_1104) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1105 = mix_val_7[7:0]; // @[Compute.scala 333:37]
  assign _T_1106 = $unsigned(src_0_7); // @[Compute.scala 334:30]
  assign _T_1107 = $unsigned(src_1_7); // @[Compute.scala 334:59]
  assign _T_1108 = _T_1106 + _T_1107; // @[Compute.scala 334:49]
  assign _T_1109 = _T_1106 + _T_1107; // @[Compute.scala 334:49]
  assign _T_1110 = $signed(_T_1109); // @[Compute.scala 334:79]
  assign add_val_7 = _T_916 ? $signed(_T_1110) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_7 = _T_916 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1111 = add_res_7[7:0]; // @[Compute.scala 336:37]
  assign _T_1113 = src_1_7[4:0]; // @[Compute.scala 337:60]
  assign _T_1114 = _T_1106 >> _T_1113; // @[Compute.scala 337:49]
  assign _T_1115 = $signed(_T_1114); // @[Compute.scala 337:84]
  assign shr_val_7 = _T_916 ? $signed(_T_1115) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_7 = _T_916 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1116 = shr_res_7[7:0]; // @[Compute.scala 339:37]
  assign src_0_8 = _T_916 ? $signed(_T_950) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_8 = _T_916 ? $signed(_GEN_76) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1117 = $signed(src_0_8) > $signed(src_1_8); // @[Compute.scala 330:57]
  assign _T_1118 = _T_1117 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 330:47]
  assign _T_1119 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 331:57]
  assign _T_1120 = _T_1119 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 331:47]
  assign _T_1121 = alu_opcode_max_en ? $signed(_T_1118) : $signed(_T_1120); // @[Compute.scala 330:24]
  assign mix_val_8 = _T_916 ? $signed(_T_1121) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1122 = mix_val_8[7:0]; // @[Compute.scala 333:37]
  assign _T_1123 = $unsigned(src_0_8); // @[Compute.scala 334:30]
  assign _T_1124 = $unsigned(src_1_8); // @[Compute.scala 334:59]
  assign _T_1125 = _T_1123 + _T_1124; // @[Compute.scala 334:49]
  assign _T_1126 = _T_1123 + _T_1124; // @[Compute.scala 334:49]
  assign _T_1127 = $signed(_T_1126); // @[Compute.scala 334:79]
  assign add_val_8 = _T_916 ? $signed(_T_1127) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_8 = _T_916 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1128 = add_res_8[7:0]; // @[Compute.scala 336:37]
  assign _T_1130 = src_1_8[4:0]; // @[Compute.scala 337:60]
  assign _T_1131 = _T_1123 >> _T_1130; // @[Compute.scala 337:49]
  assign _T_1132 = $signed(_T_1131); // @[Compute.scala 337:84]
  assign shr_val_8 = _T_916 ? $signed(_T_1132) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_8 = _T_916 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1133 = shr_res_8[7:0]; // @[Compute.scala 339:37]
  assign src_0_9 = _T_916 ? $signed(_T_954) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_9 = _T_916 ? $signed(_GEN_77) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1134 = $signed(src_0_9) > $signed(src_1_9); // @[Compute.scala 330:57]
  assign _T_1135 = _T_1134 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 330:47]
  assign _T_1136 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 331:57]
  assign _T_1137 = _T_1136 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 331:47]
  assign _T_1138 = alu_opcode_max_en ? $signed(_T_1135) : $signed(_T_1137); // @[Compute.scala 330:24]
  assign mix_val_9 = _T_916 ? $signed(_T_1138) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1139 = mix_val_9[7:0]; // @[Compute.scala 333:37]
  assign _T_1140 = $unsigned(src_0_9); // @[Compute.scala 334:30]
  assign _T_1141 = $unsigned(src_1_9); // @[Compute.scala 334:59]
  assign _T_1142 = _T_1140 + _T_1141; // @[Compute.scala 334:49]
  assign _T_1143 = _T_1140 + _T_1141; // @[Compute.scala 334:49]
  assign _T_1144 = $signed(_T_1143); // @[Compute.scala 334:79]
  assign add_val_9 = _T_916 ? $signed(_T_1144) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_9 = _T_916 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1145 = add_res_9[7:0]; // @[Compute.scala 336:37]
  assign _T_1147 = src_1_9[4:0]; // @[Compute.scala 337:60]
  assign _T_1148 = _T_1140 >> _T_1147; // @[Compute.scala 337:49]
  assign _T_1149 = $signed(_T_1148); // @[Compute.scala 337:84]
  assign shr_val_9 = _T_916 ? $signed(_T_1149) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_9 = _T_916 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1150 = shr_res_9[7:0]; // @[Compute.scala 339:37]
  assign src_0_10 = _T_916 ? $signed(_T_958) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_10 = _T_916 ? $signed(_GEN_78) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1151 = $signed(src_0_10) > $signed(src_1_10); // @[Compute.scala 330:57]
  assign _T_1152 = _T_1151 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 330:47]
  assign _T_1153 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 331:57]
  assign _T_1154 = _T_1153 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 331:47]
  assign _T_1155 = alu_opcode_max_en ? $signed(_T_1152) : $signed(_T_1154); // @[Compute.scala 330:24]
  assign mix_val_10 = _T_916 ? $signed(_T_1155) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1156 = mix_val_10[7:0]; // @[Compute.scala 333:37]
  assign _T_1157 = $unsigned(src_0_10); // @[Compute.scala 334:30]
  assign _T_1158 = $unsigned(src_1_10); // @[Compute.scala 334:59]
  assign _T_1159 = _T_1157 + _T_1158; // @[Compute.scala 334:49]
  assign _T_1160 = _T_1157 + _T_1158; // @[Compute.scala 334:49]
  assign _T_1161 = $signed(_T_1160); // @[Compute.scala 334:79]
  assign add_val_10 = _T_916 ? $signed(_T_1161) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_10 = _T_916 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1162 = add_res_10[7:0]; // @[Compute.scala 336:37]
  assign _T_1164 = src_1_10[4:0]; // @[Compute.scala 337:60]
  assign _T_1165 = _T_1157 >> _T_1164; // @[Compute.scala 337:49]
  assign _T_1166 = $signed(_T_1165); // @[Compute.scala 337:84]
  assign shr_val_10 = _T_916 ? $signed(_T_1166) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_10 = _T_916 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1167 = shr_res_10[7:0]; // @[Compute.scala 339:37]
  assign src_0_11 = _T_916 ? $signed(_T_962) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_11 = _T_916 ? $signed(_GEN_79) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1168 = $signed(src_0_11) > $signed(src_1_11); // @[Compute.scala 330:57]
  assign _T_1169 = _T_1168 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 330:47]
  assign _T_1170 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 331:57]
  assign _T_1171 = _T_1170 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 331:47]
  assign _T_1172 = alu_opcode_max_en ? $signed(_T_1169) : $signed(_T_1171); // @[Compute.scala 330:24]
  assign mix_val_11 = _T_916 ? $signed(_T_1172) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1173 = mix_val_11[7:0]; // @[Compute.scala 333:37]
  assign _T_1174 = $unsigned(src_0_11); // @[Compute.scala 334:30]
  assign _T_1175 = $unsigned(src_1_11); // @[Compute.scala 334:59]
  assign _T_1176 = _T_1174 + _T_1175; // @[Compute.scala 334:49]
  assign _T_1177 = _T_1174 + _T_1175; // @[Compute.scala 334:49]
  assign _T_1178 = $signed(_T_1177); // @[Compute.scala 334:79]
  assign add_val_11 = _T_916 ? $signed(_T_1178) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_11 = _T_916 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1179 = add_res_11[7:0]; // @[Compute.scala 336:37]
  assign _T_1181 = src_1_11[4:0]; // @[Compute.scala 337:60]
  assign _T_1182 = _T_1174 >> _T_1181; // @[Compute.scala 337:49]
  assign _T_1183 = $signed(_T_1182); // @[Compute.scala 337:84]
  assign shr_val_11 = _T_916 ? $signed(_T_1183) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_11 = _T_916 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1184 = shr_res_11[7:0]; // @[Compute.scala 339:37]
  assign src_0_12 = _T_916 ? $signed(_T_966) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_12 = _T_916 ? $signed(_GEN_80) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1185 = $signed(src_0_12) > $signed(src_1_12); // @[Compute.scala 330:57]
  assign _T_1186 = _T_1185 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 330:47]
  assign _T_1187 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 331:57]
  assign _T_1188 = _T_1187 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 331:47]
  assign _T_1189 = alu_opcode_max_en ? $signed(_T_1186) : $signed(_T_1188); // @[Compute.scala 330:24]
  assign mix_val_12 = _T_916 ? $signed(_T_1189) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1190 = mix_val_12[7:0]; // @[Compute.scala 333:37]
  assign _T_1191 = $unsigned(src_0_12); // @[Compute.scala 334:30]
  assign _T_1192 = $unsigned(src_1_12); // @[Compute.scala 334:59]
  assign _T_1193 = _T_1191 + _T_1192; // @[Compute.scala 334:49]
  assign _T_1194 = _T_1191 + _T_1192; // @[Compute.scala 334:49]
  assign _T_1195 = $signed(_T_1194); // @[Compute.scala 334:79]
  assign add_val_12 = _T_916 ? $signed(_T_1195) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_12 = _T_916 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1196 = add_res_12[7:0]; // @[Compute.scala 336:37]
  assign _T_1198 = src_1_12[4:0]; // @[Compute.scala 337:60]
  assign _T_1199 = _T_1191 >> _T_1198; // @[Compute.scala 337:49]
  assign _T_1200 = $signed(_T_1199); // @[Compute.scala 337:84]
  assign shr_val_12 = _T_916 ? $signed(_T_1200) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_12 = _T_916 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1201 = shr_res_12[7:0]; // @[Compute.scala 339:37]
  assign src_0_13 = _T_916 ? $signed(_T_970) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_13 = _T_916 ? $signed(_GEN_81) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1202 = $signed(src_0_13) > $signed(src_1_13); // @[Compute.scala 330:57]
  assign _T_1203 = _T_1202 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 330:47]
  assign _T_1204 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 331:57]
  assign _T_1205 = _T_1204 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 331:47]
  assign _T_1206 = alu_opcode_max_en ? $signed(_T_1203) : $signed(_T_1205); // @[Compute.scala 330:24]
  assign mix_val_13 = _T_916 ? $signed(_T_1206) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1207 = mix_val_13[7:0]; // @[Compute.scala 333:37]
  assign _T_1208 = $unsigned(src_0_13); // @[Compute.scala 334:30]
  assign _T_1209 = $unsigned(src_1_13); // @[Compute.scala 334:59]
  assign _T_1210 = _T_1208 + _T_1209; // @[Compute.scala 334:49]
  assign _T_1211 = _T_1208 + _T_1209; // @[Compute.scala 334:49]
  assign _T_1212 = $signed(_T_1211); // @[Compute.scala 334:79]
  assign add_val_13 = _T_916 ? $signed(_T_1212) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_13 = _T_916 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1213 = add_res_13[7:0]; // @[Compute.scala 336:37]
  assign _T_1215 = src_1_13[4:0]; // @[Compute.scala 337:60]
  assign _T_1216 = _T_1208 >> _T_1215; // @[Compute.scala 337:49]
  assign _T_1217 = $signed(_T_1216); // @[Compute.scala 337:84]
  assign shr_val_13 = _T_916 ? $signed(_T_1217) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_13 = _T_916 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1218 = shr_res_13[7:0]; // @[Compute.scala 339:37]
  assign src_0_14 = _T_916 ? $signed(_T_974) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_14 = _T_916 ? $signed(_GEN_82) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1219 = $signed(src_0_14) > $signed(src_1_14); // @[Compute.scala 330:57]
  assign _T_1220 = _T_1219 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 330:47]
  assign _T_1221 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 331:57]
  assign _T_1222 = _T_1221 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 331:47]
  assign _T_1223 = alu_opcode_max_en ? $signed(_T_1220) : $signed(_T_1222); // @[Compute.scala 330:24]
  assign mix_val_14 = _T_916 ? $signed(_T_1223) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1224 = mix_val_14[7:0]; // @[Compute.scala 333:37]
  assign _T_1225 = $unsigned(src_0_14); // @[Compute.scala 334:30]
  assign _T_1226 = $unsigned(src_1_14); // @[Compute.scala 334:59]
  assign _T_1227 = _T_1225 + _T_1226; // @[Compute.scala 334:49]
  assign _T_1228 = _T_1225 + _T_1226; // @[Compute.scala 334:49]
  assign _T_1229 = $signed(_T_1228); // @[Compute.scala 334:79]
  assign add_val_14 = _T_916 ? $signed(_T_1229) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_14 = _T_916 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1230 = add_res_14[7:0]; // @[Compute.scala 336:37]
  assign _T_1232 = src_1_14[4:0]; // @[Compute.scala 337:60]
  assign _T_1233 = _T_1225 >> _T_1232; // @[Compute.scala 337:49]
  assign _T_1234 = $signed(_T_1233); // @[Compute.scala 337:84]
  assign shr_val_14 = _T_916 ? $signed(_T_1234) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_14 = _T_916 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1235 = shr_res_14[7:0]; // @[Compute.scala 339:37]
  assign src_0_15 = _T_916 ? $signed(_T_978) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign src_1_15 = _T_916 ? $signed(_GEN_83) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1236 = $signed(src_0_15) > $signed(src_1_15); // @[Compute.scala 330:57]
  assign _T_1237 = _T_1236 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 330:47]
  assign _T_1238 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 331:57]
  assign _T_1239 = _T_1238 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 331:47]
  assign _T_1240 = alu_opcode_max_en ? $signed(_T_1237) : $signed(_T_1239); // @[Compute.scala 330:24]
  assign mix_val_15 = _T_916 ? $signed(_T_1240) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1241 = mix_val_15[7:0]; // @[Compute.scala 333:37]
  assign _T_1242 = $unsigned(src_0_15); // @[Compute.scala 334:30]
  assign _T_1243 = $unsigned(src_1_15); // @[Compute.scala 334:59]
  assign _T_1244 = _T_1242 + _T_1243; // @[Compute.scala 334:49]
  assign _T_1245 = _T_1242 + _T_1243; // @[Compute.scala 334:49]
  assign _T_1246 = $signed(_T_1245); // @[Compute.scala 334:79]
  assign add_val_15 = _T_916 ? $signed(_T_1246) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign add_res_15 = _T_916 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1247 = add_res_15[7:0]; // @[Compute.scala 336:37]
  assign _T_1249 = src_1_15[4:0]; // @[Compute.scala 337:60]
  assign _T_1250 = _T_1242 >> _T_1249; // @[Compute.scala 337:49]
  assign _T_1251 = $signed(_T_1250); // @[Compute.scala 337:84]
  assign shr_val_15 = _T_916 ? $signed(_T_1251) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign shr_res_15 = _T_916 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 320:36]
  assign _T_1252 = shr_res_15[7:0]; // @[Compute.scala 339:37]
  assign short_cmp_res_0 = _T_916 ? _T_986 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_0 = _T_916 ? _T_992 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_0 = _T_916 ? _T_997 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_1 = _T_916 ? _T_1003 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_1 = _T_916 ? _T_1009 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_1 = _T_916 ? _T_1014 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_2 = _T_916 ? _T_1020 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_2 = _T_916 ? _T_1026 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_2 = _T_916 ? _T_1031 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_3 = _T_916 ? _T_1037 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_3 = _T_916 ? _T_1043 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_3 = _T_916 ? _T_1048 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_4 = _T_916 ? _T_1054 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_4 = _T_916 ? _T_1060 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_4 = _T_916 ? _T_1065 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_5 = _T_916 ? _T_1071 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_5 = _T_916 ? _T_1077 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_5 = _T_916 ? _T_1082 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_6 = _T_916 ? _T_1088 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_6 = _T_916 ? _T_1094 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_6 = _T_916 ? _T_1099 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_7 = _T_916 ? _T_1105 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_7 = _T_916 ? _T_1111 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_7 = _T_916 ? _T_1116 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_8 = _T_916 ? _T_1122 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_8 = _T_916 ? _T_1128 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_8 = _T_916 ? _T_1133 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_9 = _T_916 ? _T_1139 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_9 = _T_916 ? _T_1145 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_9 = _T_916 ? _T_1150 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_10 = _T_916 ? _T_1156 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_10 = _T_916 ? _T_1162 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_10 = _T_916 ? _T_1167 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_11 = _T_916 ? _T_1173 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_11 = _T_916 ? _T_1179 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_11 = _T_916 ? _T_1184 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_12 = _T_916 ? _T_1190 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_12 = _T_916 ? _T_1196 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_12 = _T_916 ? _T_1201 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_13 = _T_916 ? _T_1207 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_13 = _T_916 ? _T_1213 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_13 = _T_916 ? _T_1218 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_14 = _T_916 ? _T_1224 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_14 = _T_916 ? _T_1230 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_14 = _T_916 ? _T_1235 : 8'h0; // @[Compute.scala 320:36]
  assign short_cmp_res_15 = _T_916 ? _T_1241 : 8'h0; // @[Compute.scala 320:36]
  assign short_add_res_15 = _T_916 ? _T_1247 : 8'h0; // @[Compute.scala 320:36]
  assign short_shr_res_15 = _T_916 ? _T_1252 : 8'h0; // @[Compute.scala 320:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 344:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 345:39]
  assign _T_1262 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1270 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1262}; // @[Cat.scala 30:58]
  assign _T_1277 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1285 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1277}; // @[Cat.scala 30:58]
  assign _T_1292 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1300 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1292}; // @[Cat.scala 30:58]
  assign _T_1301 = alu_opcode_add_en ? _T_1285 : _T_1300; // @[Compute.scala 350:29]
  assign out_mem_enq_bits = alu_opcode_minmax_en ? _T_1270 : _T_1301; // @[Compute.scala 349:29]
  assign _T_1302 = opcode_alu_en & busy; // @[Compute.scala 351:34]
  assign _T_1303 = _GEN_292 <= out_cntr_max_val; // @[Compute.scala 351:59]
  assign _T_1304 = _T_1302 & _T_1303; // @[Compute.scala 351:42]
  assign _T_1306 = out_cntr_val >= 16'h2; // @[Compute.scala 353:63]
  assign _T_1307 = out_mem_write & _T_1306; // @[Compute.scala 353:46]
  assign _T_1309 = out_cntr_max - 32'h1; // @[Compute.scala 353:105]
  assign _T_1310 = $unsigned(_T_1309); // @[Compute.scala 353:105]
  assign _T_1311 = _T_1310[31:0]; // @[Compute.scala 353:105]
  assign _T_1312 = _GEN_292 <= _T_1311; // @[Compute.scala 353:88]
  assign _T_1316 = {_T_1315,out_mem_enq_bits}; // @[Cat.scala 30:58]
  assign _T_1317 = out_mem_fifo_io_deq_bits[159:128]; // @[Compute.scala 357:49]
  assign _GEN_311 = {{7'd0}, _T_1317}; // @[Compute.scala 357:66]
  assign _T_1319 = _GEN_311 << 3'h4; // @[Compute.scala 357:66]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 220:23]
  assign io_done_readdata = opcode == 3'h3; // @[Compute.scala 223:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 233:19]
  assign io_uops_read = uops_read; // @[Compute.scala 232:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 128'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 253:21]
  assign io_biases_read = biases_read; // @[Compute.scala 254:18]
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
  assign io_out_mem_address = _T_1319[16:0]; // @[Compute.scala 357:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_fifo_io_deq_valid; // @[Compute.scala 358:20]
  assign io_out_mem_writedata = out_mem_fifo_io_deq_bits[127:0]; // @[Compute.scala 361:24]
  assign out_mem_fifo_clock = clock;
  assign out_mem_fifo_reset = reset;
  assign out_mem_fifo_io_enq_valid = _T_1307 & _T_1312; // @[Compute.scala 353:29]
  assign out_mem_fifo_io_enq_bits = {{16'd0}, _T_1316}; // @[Compute.scala 354:28]
  assign out_mem_fifo_io_deq_ready = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 359:29]
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
  imm_raw = _RAND_5[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  imm = _RAND_6[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {1{`RANDOM}};
  state = _RAND_7[2:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {1{`RANDOM}};
  uops_read = _RAND_8[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {5{`RANDOM}};
  uops_data = _RAND_9[159:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {1{`RANDOM}};
  biases_read = _RAND_10[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_11 = {4{`RANDOM}};
  biases_data_0 = _RAND_11[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_12 = {4{`RANDOM}};
  biases_data_1 = _RAND_12[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_13 = {4{`RANDOM}};
  biases_data_2 = _RAND_13[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_14 = {4{`RANDOM}};
  biases_data_3 = _RAND_14[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_15 = {1{`RANDOM}};
  out_mem_write = _RAND_15[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_16 = {1{`RANDOM}};
  uop_cntr_val = _RAND_16[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_17 = {1{`RANDOM}};
  acc_cntr_val = _RAND_17[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_18 = {1{`RANDOM}};
  out_cntr_val = _RAND_18[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_19 = {1{`RANDOM}};
  pop_prev_dep_ready = _RAND_19[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_20 = {1{`RANDOM}};
  pop_next_dep_ready = _RAND_20[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_21 = {1{`RANDOM}};
  push_prev_dep_ready = _RAND_21[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_22 = {1{`RANDOM}};
  push_next_dep_ready = _RAND_22[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_23 = {1{`RANDOM}};
  gemm_queue_ready = _RAND_23[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_24 = {1{`RANDOM}};
  finish_wrap = _RAND_24[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_25 = {1{`RANDOM}};
  uops_read_en = _RAND_25[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_26 = {1{`RANDOM}};
  uop = _RAND_26[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_27 = {1{`RANDOM}};
  _T_466 = _RAND_27[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_28 = {16{`RANDOM}};
  dst_vector = _RAND_28[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_29 = {16{`RANDOM}};
  src_vector = _RAND_29[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_30 = {1{`RANDOM}};
  _T_1315 = _RAND_30[15:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_458_en & acc_mem__T_458_mask) begin
      acc_mem[acc_mem__T_458_addr] <= acc_mem__T_458_data; // @[Compute.scala 34:20]
    end
    if(uop_mem__T_397_en & uop_mem__T_397_mask) begin
      uop_mem[uop_mem__T_397_addr] <= uop_mem__T_397_data; // @[Compute.scala 35:20]
    end
    if(uop_mem__T_403_en & uop_mem__T_403_mask) begin
      uop_mem[uop_mem__T_403_addr] <= uop_mem__T_403_data; // @[Compute.scala 35:20]
    end
    if(uop_mem__T_409_en & uop_mem__T_409_mask) begin
      uop_mem[uop_mem__T_409_addr] <= uop_mem__T_409_data; // @[Compute.scala 35:20]
    end
    if(uop_mem__T_415_en & uop_mem__T_415_mask) begin
      uop_mem[uop_mem__T_415_addr] <= uop_mem__T_415_data; // @[Compute.scala 35:20]
    end
    if (gemm_queue_ready) begin
      insn <= io_gemm_queue_data;
    end
    uop_bgn <= {{3'd0}, _T_203};
    uop_end <= {{2'd0}, _T_205};
    imm_raw <= insn[126:111];
    imm <= $signed(_T_215);
    if (reset) begin
      state <= 3'h0;
    end else begin
      if (gemm_queue_ready) begin
        state <= 3'h2;
      end else begin
        if (_T_312) begin
          state <= 3'h4;
        end else begin
          if (_T_310) begin
            state <= 3'h2;
          end else begin
            if (_T_308) begin
              state <= 3'h1;
            end else begin
              if (_T_299) begin
                if (_T_300) begin
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
    if (_T_331) begin
      if (_T_389) begin
        uops_read <= 1'h0;
      end else begin
        uops_read <= _T_378;
      end
    end else begin
      uops_read <= _T_378;
    end
    if (_T_331) begin
      uops_data <= _T_383;
    end
    biases_read <= acc_cntr_en & _T_444;
    if (_T_340) begin
      if (3'h0 == _T_450) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_340) begin
      if (3'h1 == _T_450) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_340) begin
      if (3'h2 == _T_450) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_340) begin
      if (3'h3 == _T_450) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    if (reset) begin
      out_mem_write <= 1'h0;
    end else begin
      out_mem_write <= _T_1304;
    end
    if (gemm_queue_ready) begin
      uop_cntr_val <= 16'h0;
    end else begin
      if (_T_334) begin
        uop_cntr_val <= _T_337;
      end
    end
    if (gemm_queue_ready) begin
      acc_cntr_val <= 16'h0;
    end else begin
      if (_T_343) begin
        acc_cntr_val <= _T_346;
      end
    end
    if (gemm_queue_ready) begin
      out_cntr_val <= 16'h0;
    end else begin
      if (_T_349) begin
        out_cntr_val <= _T_352;
      end
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (gemm_queue_ready) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_316) begin
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
        if (_T_319) begin
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
        if (_T_324) begin
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
        if (_T_327) begin
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
        gemm_queue_ready <= _T_361;
      end
    end
    if (reset) begin
      finish_wrap <= 1'h0;
    end else begin
      if (opcode_finish_en) begin
        if (pop_prev_dep) begin
          finish_wrap <= _T_291;
        end else begin
          if (pop_next_dep) begin
            finish_wrap <= _T_292;
          end else begin
            if (push_prev_dep) begin
              finish_wrap <= _T_293;
            end else begin
              if (push_next_dep) begin
                finish_wrap <= _T_294;
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
    uops_read_en <= uops_read & _T_330;
    uop <= uop_mem__T_463_data;
    _T_466 <= out_cntr_val;
    if (out_mem_write) begin
      dst_vector <= acc_mem__T_726_data;
    end
    if (out_mem_write) begin
      src_vector <= acc_mem__T_728_data;
    end
    _T_1315 <= _GEN_307 + dst_offset_in;
  end
endmodule
