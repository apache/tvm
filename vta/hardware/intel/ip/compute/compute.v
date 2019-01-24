module BinaryQueue(
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
  wire [511:0] acc_mem__T_382_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_382_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_385_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_385_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_365_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_365_addr; // @[Compute.scala 33:20]
  wire  acc_mem__T_365_mask; // @[Compute.scala 33:20]
  wire  acc_mem__T_365_en; // @[Compute.scala 33:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem_uop_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem_uop_addr; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_327_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_327_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_327_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_327_en; // @[Compute.scala 34:20]
  wire  g2l_queue_clock; // @[Compute.scala 285:25]
  wire  g2l_queue_reset; // @[Compute.scala 285:25]
  wire  g2l_queue_io_enq_ready; // @[Compute.scala 285:25]
  wire  g2l_queue_io_enq_valid; // @[Compute.scala 285:25]
  wire  g2l_queue_io_deq_ready; // @[Compute.scala 285:25]
  wire  g2l_queue_io_deq_valid; // @[Compute.scala 285:25]
  wire  g2l_queue_io_deq_bits; // @[Compute.scala 285:25]
  wire  g2s_queue_clock; // @[Compute.scala 286:25]
  wire  g2s_queue_reset; // @[Compute.scala 286:25]
  wire  g2s_queue_io_enq_ready; // @[Compute.scala 286:25]
  wire  g2s_queue_io_enq_valid; // @[Compute.scala 286:25]
  wire  g2s_queue_io_deq_ready; // @[Compute.scala 286:25]
  wire  g2s_queue_io_deq_valid; // @[Compute.scala 286:25]
  wire  g2s_queue_io_deq_bits; // @[Compute.scala 286:25]
  wire  started; // @[Compute.scala 31:17]
  reg [127:0] insn; // @[Compute.scala 36:28]
  reg [127:0] _RAND_2;
  wire  insn_valid; // @[Compute.scala 37:30]
  wire [2:0] opcode; // @[Compute.scala 39:29]
  wire  push_prev_dep; // @[Compute.scala 42:29]
  wire  push_next_dep; // @[Compute.scala 43:29]
  wire [1:0] memory_type; // @[Compute.scala 45:25]
  wire [15:0] sram_base; // @[Compute.scala 46:25]
  wire [31:0] dram_base; // @[Compute.scala 47:25]
  wire [15:0] x_size; // @[Compute.scala 49:25]
  wire [3:0] y_pad_0; // @[Compute.scala 51:25]
  wire [3:0] x_pad_0; // @[Compute.scala 53:25]
  wire [3:0] x_pad_1; // @[Compute.scala 54:25]
  wire [15:0] _GEN_281; // @[Compute.scala 58:30]
  wire [15:0] _GEN_283; // @[Compute.scala 59:30]
  wire [16:0] _T_204; // @[Compute.scala 59:30]
  wire [15:0] _T_205; // @[Compute.scala 59:30]
  wire [15:0] _GEN_284; // @[Compute.scala 59:39]
  wire [16:0] _T_206; // @[Compute.scala 59:39]
  wire [15:0] x_size_total; // @[Compute.scala 59:39]
  wire [19:0] y_offset; // @[Compute.scala 60:31]
  wire  opcode_finish_en; // @[Compute.scala 63:34]
  wire  _T_209; // @[Compute.scala 64:32]
  wire  _T_211; // @[Compute.scala 64:60]
  wire  opcode_load_en; // @[Compute.scala 64:50]
  wire  opcode_gemm_en; // @[Compute.scala 65:32]
  wire  opcode_alu_en; // @[Compute.scala 66:31]
  wire  memory_type_uop_en; // @[Compute.scala 68:40]
  wire  memory_type_acc_en; // @[Compute.scala 69:40]
  wire  _T_216; // @[Compute.scala 73:37]
  wire  _T_217; // @[Compute.scala 73:59]
  wire  uop_cntr_en; // @[Compute.scala 73:70]
  reg [15:0] uop_cntr_val; // @[Compute.scala 75:25]
  reg [31:0] _RAND_3;
  wire  _T_220; // @[Compute.scala 76:38]
  wire  uop_cntr_wrap; // @[Compute.scala 76:58]
  wire  _T_221; // @[Compute.scala 79:37]
  wire  _T_222; // @[Compute.scala 79:59]
  wire  acc_cntr_en; // @[Compute.scala 79:70]
  reg [15:0] acc_cntr_val; // @[Compute.scala 81:25]
  reg [31:0] _RAND_4;
  wire  _T_225; // @[Compute.scala 82:38]
  wire  acc_cntr_wrap; // @[Compute.scala 82:58]
  wire  _T_226; // @[Compute.scala 85:37]
  wire  _T_227; // @[Compute.scala 85:56]
  wire  out_cntr_en; // @[Compute.scala 85:67]
  reg [15:0] dst_offset_in; // @[Compute.scala 87:25]
  reg [31:0] _RAND_5;
  wire  _T_230; // @[Compute.scala 88:38]
  wire  out_cntr_wrap; // @[Compute.scala 88:58]
  reg  uops_read; // @[Compute.scala 91:24]
  reg [31:0] _RAND_6;
  reg [31:0] uops_data; // @[Compute.scala 92:24]
  reg [31:0] _RAND_7;
  reg  biases_read; // @[Compute.scala 95:24]
  reg [31:0] _RAND_8;
  reg [127:0] biases_data_0; // @[Compute.scala 96:24]
  reg [127:0] _RAND_9;
  reg [127:0] biases_data_1; // @[Compute.scala 96:24]
  reg [127:0] _RAND_10;
  reg [127:0] biases_data_2; // @[Compute.scala 96:24]
  reg [127:0] _RAND_11;
  reg [127:0] biases_data_3; // @[Compute.scala 96:24]
  reg [127:0] _RAND_12;
  reg [3:0] state; // @[Compute.scala 100:18]
  reg [31:0] _RAND_13;
  wire  idle; // @[Compute.scala 101:20]
  wire  busy; // @[Compute.scala 102:20]
  wire  done; // @[Compute.scala 103:20]
  wire  _T_251; // @[Compute.scala 106:31]
  wire  _T_252; // @[Compute.scala 106:28]
  wire  _T_255; // @[Compute.scala 107:31]
  wire  _T_256; // @[Compute.scala 107:28]
  wire  _T_259; // @[Compute.scala 108:31]
  wire  _T_260; // @[Compute.scala 108:28]
  wire  _T_264; // @[Compute.scala 107:15]
  wire  _T_265; // @[Compute.scala 106:15]
  wire  _T_267; // @[Compute.scala 111:22]
  wire  _T_268; // @[Compute.scala 111:19]
  wire  _T_269; // @[Compute.scala 111:37]
  wire  _T_271; // @[Compute.scala 112:24]
  wire [16:0] _T_273; // @[Compute.scala 113:36]
  wire [15:0] _T_274; // @[Compute.scala 113:36]
  wire [15:0] _GEN_0; // @[Compute.scala 112:42]
  wire [1:0] _GEN_1; // @[Compute.scala 112:42]
  wire [15:0] _GEN_2; // @[Compute.scala 111:46]
  wire [1:0] _GEN_3; // @[Compute.scala 111:46]
  wire  _T_277; // @[Compute.scala 119:24]
  wire  _T_278; // @[Compute.scala 119:21]
  wire  _T_279; // @[Compute.scala 119:39]
  wire  _T_281; // @[Compute.scala 120:24]
  wire [16:0] _T_283; // @[Compute.scala 121:36]
  wire [15:0] _T_284; // @[Compute.scala 121:36]
  wire [15:0] _GEN_4; // @[Compute.scala 120:42]
  wire [1:0] _GEN_5; // @[Compute.scala 120:42]
  wire [15:0] _GEN_6; // @[Compute.scala 119:48]
  wire [1:0] _GEN_7; // @[Compute.scala 119:48]
  wire  _T_287; // @[Compute.scala 127:24]
  wire  _T_288; // @[Compute.scala 127:21]
  wire  _T_289; // @[Compute.scala 127:39]
  wire  _T_291; // @[Compute.scala 128:24]
  wire [16:0] _T_293; // @[Compute.scala 129:36]
  wire [15:0] _T_294; // @[Compute.scala 129:36]
  wire [15:0] _GEN_8; // @[Compute.scala 128:42]
  wire [1:0] _GEN_9; // @[Compute.scala 128:42]
  wire [15:0] _GEN_10; // @[Compute.scala 127:48]
  wire [1:0] _GEN_11; // @[Compute.scala 127:48]
  wire  _T_297; // @[Compute.scala 139:25]
  wire [15:0] _GEN_12; // @[Compute.scala 139:41]
  wire  _T_299; // @[Compute.scala 140:25]
  wire [15:0] _GEN_13; // @[Compute.scala 140:41]
  wire  _T_301; // @[Compute.scala 141:25]
  wire [15:0] _GEN_14; // @[Compute.scala 141:41]
  wire [1:0] _GEN_15; // @[Compute.scala 137:15]
  wire  _T_303; // @[Compute.scala 145:29]
  reg  _T_313; // @[Compute.scala 159:30]
  reg [31:0] _RAND_14;
  wire [31:0] _GEN_286; // @[Compute.scala 163:33]
  wire [32:0] _T_316; // @[Compute.scala 163:33]
  wire [31:0] _T_317; // @[Compute.scala 163:33]
  wire [34:0] _GEN_287; // @[Compute.scala 163:49]
  wire [34:0] uop_dram_addr; // @[Compute.scala 163:49]
  wire [16:0] _T_319; // @[Compute.scala 164:33]
  wire [15:0] uop_sram_addr; // @[Compute.scala 164:33]
  wire [31:0] _GEN_288; // @[Compute.scala 176:35]
  wire [32:0] _T_328; // @[Compute.scala 176:35]
  wire [31:0] _T_329; // @[Compute.scala 176:35]
  wire [31:0] _GEN_289; // @[Compute.scala 176:46]
  wire [32:0] _T_330; // @[Compute.scala 176:46]
  wire [31:0] _T_331; // @[Compute.scala 176:46]
  wire [32:0] _T_333; // @[Compute.scala 176:57]
  wire [32:0] _GEN_290; // @[Compute.scala 176:67]
  wire [33:0] _T_334; // @[Compute.scala 176:67]
  wire [32:0] _T_335; // @[Compute.scala 176:67]
  wire [39:0] _GEN_291; // @[Compute.scala 176:83]
  wire [39:0] acc_dram_addr; // @[Compute.scala 176:83]
  wire [19:0] _GEN_292; // @[Compute.scala 177:35]
  wire [20:0] _T_337; // @[Compute.scala 177:35]
  wire [19:0] _T_338; // @[Compute.scala 177:35]
  wire [19:0] _GEN_293; // @[Compute.scala 177:46]
  wire [20:0] _T_339; // @[Compute.scala 177:46]
  wire [19:0] _T_340; // @[Compute.scala 177:46]
  wire [20:0] _T_342; // @[Compute.scala 177:57]
  wire [20:0] _GEN_294; // @[Compute.scala 177:67]
  wire [21:0] _T_343; // @[Compute.scala 177:67]
  wire [20:0] _T_344; // @[Compute.scala 177:67]
  wire [20:0] _T_346; // @[Compute.scala 177:83]
  wire [21:0] _T_348; // @[Compute.scala 177:91]
  wire [21:0] _T_349; // @[Compute.scala 177:91]
  wire [20:0] acc_sram_addr; // @[Compute.scala 177:91]
  wire  _T_351; // @[Compute.scala 179:33]
  wire [15:0] _GEN_16; // @[Compute.scala 185:30]
  wire [2:0] _T_357; // @[Compute.scala 185:30]
  wire [127:0] _GEN_23; // @[Compute.scala 185:67]
  wire [127:0] _GEN_24; // @[Compute.scala 185:67]
  wire [127:0] _GEN_25; // @[Compute.scala 185:67]
  wire [127:0] _GEN_26; // @[Compute.scala 185:67]
  wire  _T_363; // @[Compute.scala 186:64]
  wire [255:0] _T_366; // @[Cat.scala 30:58]
  wire [255:0] _T_367; // @[Cat.scala 30:58]
  wire [1:0] alu_opcode; // @[Compute.scala 196:24]
  wire  use_imm; // @[Compute.scala 197:21]
  wire [15:0] imm_raw; // @[Compute.scala 198:21]
  wire [15:0] _T_369; // @[Compute.scala 199:25]
  wire  _T_371; // @[Compute.scala 199:32]
  wire [31:0] _T_373; // @[Cat.scala 30:58]
  wire [16:0] _T_375; // @[Cat.scala 30:58]
  wire [31:0] _T_376; // @[Compute.scala 199:16]
  wire [31:0] imm; // @[Compute.scala 199:89]
  wire [10:0] _T_377; // @[Compute.scala 207:20]
  wire [15:0] _GEN_295; // @[Compute.scala 207:47]
  wire [16:0] _T_378; // @[Compute.scala 207:47]
  wire [15:0] dst_idx; // @[Compute.scala 207:47]
  wire [10:0] _T_379; // @[Compute.scala 208:20]
  wire [15:0] _GEN_296; // @[Compute.scala 208:47]
  wire [16:0] _T_380; // @[Compute.scala 208:47]
  wire [15:0] src_idx; // @[Compute.scala 208:47]
  reg [511:0] dst_vector; // @[Compute.scala 211:27]
  reg [511:0] _RAND_15;
  reg [511:0] src_vector; // @[Compute.scala 212:27]
  reg [511:0] _RAND_16;
  wire [22:0] _GEN_297; // @[Compute.scala 225:39]
  reg [22:0] out_mem_addr; // @[Compute.scala 225:30]
  reg [31:0] _RAND_17;
  reg  out_mem_write_en; // @[Compute.scala 226:34]
  reg [31:0] _RAND_18;
  wire  alu_opcode_min_en; // @[Compute.scala 228:38]
  wire  alu_opcode_max_en; // @[Compute.scala 229:38]
  wire  _T_822; // @[Compute.scala 248:20]
  wire [31:0] _T_823; // @[Compute.scala 251:31]
  wire [31:0] _T_824; // @[Compute.scala 251:72]
  wire [31:0] _T_825; // @[Compute.scala 252:31]
  wire [31:0] _T_826; // @[Compute.scala 252:72]
  wire [31:0] _T_827; // @[Compute.scala 251:31]
  wire [31:0] _T_828; // @[Compute.scala 251:72]
  wire [31:0] _T_829; // @[Compute.scala 252:31]
  wire [31:0] _T_830; // @[Compute.scala 252:72]
  wire [31:0] _T_831; // @[Compute.scala 251:31]
  wire [31:0] _T_832; // @[Compute.scala 251:72]
  wire [31:0] _T_833; // @[Compute.scala 252:31]
  wire [31:0] _T_834; // @[Compute.scala 252:72]
  wire [31:0] _T_835; // @[Compute.scala 251:31]
  wire [31:0] _T_836; // @[Compute.scala 251:72]
  wire [31:0] _T_837; // @[Compute.scala 252:31]
  wire [31:0] _T_838; // @[Compute.scala 252:72]
  wire [31:0] _T_839; // @[Compute.scala 251:31]
  wire [31:0] _T_840; // @[Compute.scala 251:72]
  wire [31:0] _T_841; // @[Compute.scala 252:31]
  wire [31:0] _T_842; // @[Compute.scala 252:72]
  wire [31:0] _T_843; // @[Compute.scala 251:31]
  wire [31:0] _T_844; // @[Compute.scala 251:72]
  wire [31:0] _T_845; // @[Compute.scala 252:31]
  wire [31:0] _T_846; // @[Compute.scala 252:72]
  wire [31:0] _T_847; // @[Compute.scala 251:31]
  wire [31:0] _T_848; // @[Compute.scala 251:72]
  wire [31:0] _T_849; // @[Compute.scala 252:31]
  wire [31:0] _T_850; // @[Compute.scala 252:72]
  wire [31:0] _T_851; // @[Compute.scala 251:31]
  wire [31:0] _T_852; // @[Compute.scala 251:72]
  wire [31:0] _T_853; // @[Compute.scala 252:31]
  wire [31:0] _T_854; // @[Compute.scala 252:72]
  wire [31:0] _T_855; // @[Compute.scala 251:31]
  wire [31:0] _T_856; // @[Compute.scala 251:72]
  wire [31:0] _T_857; // @[Compute.scala 252:31]
  wire [31:0] _T_858; // @[Compute.scala 252:72]
  wire [31:0] _T_859; // @[Compute.scala 251:31]
  wire [31:0] _T_860; // @[Compute.scala 251:72]
  wire [31:0] _T_861; // @[Compute.scala 252:31]
  wire [31:0] _T_862; // @[Compute.scala 252:72]
  wire [31:0] _T_863; // @[Compute.scala 251:31]
  wire [31:0] _T_864; // @[Compute.scala 251:72]
  wire [31:0] _T_865; // @[Compute.scala 252:31]
  wire [31:0] _T_866; // @[Compute.scala 252:72]
  wire [31:0] _T_867; // @[Compute.scala 251:31]
  wire [31:0] _T_868; // @[Compute.scala 251:72]
  wire [31:0] _T_869; // @[Compute.scala 252:31]
  wire [31:0] _T_870; // @[Compute.scala 252:72]
  wire [31:0] _T_871; // @[Compute.scala 251:31]
  wire [31:0] _T_872; // @[Compute.scala 251:72]
  wire [31:0] _T_873; // @[Compute.scala 252:31]
  wire [31:0] _T_874; // @[Compute.scala 252:72]
  wire [31:0] _T_875; // @[Compute.scala 251:31]
  wire [31:0] _T_876; // @[Compute.scala 251:72]
  wire [31:0] _T_877; // @[Compute.scala 252:31]
  wire [31:0] _T_878; // @[Compute.scala 252:72]
  wire [31:0] _T_879; // @[Compute.scala 251:31]
  wire [31:0] _T_880; // @[Compute.scala 251:72]
  wire [31:0] _T_881; // @[Compute.scala 252:31]
  wire [31:0] _T_882; // @[Compute.scala 252:72]
  wire [31:0] _T_883; // @[Compute.scala 251:31]
  wire [31:0] _T_884; // @[Compute.scala 251:72]
  wire [31:0] _T_885; // @[Compute.scala 252:31]
  wire [31:0] _T_886; // @[Compute.scala 252:72]
  wire [31:0] _GEN_43; // @[Compute.scala 249:30]
  wire [31:0] _GEN_44; // @[Compute.scala 249:30]
  wire [31:0] _GEN_45; // @[Compute.scala 249:30]
  wire [31:0] _GEN_46; // @[Compute.scala 249:30]
  wire [31:0] _GEN_47; // @[Compute.scala 249:30]
  wire [31:0] _GEN_48; // @[Compute.scala 249:30]
  wire [31:0] _GEN_49; // @[Compute.scala 249:30]
  wire [31:0] _GEN_50; // @[Compute.scala 249:30]
  wire [31:0] _GEN_51; // @[Compute.scala 249:30]
  wire [31:0] _GEN_52; // @[Compute.scala 249:30]
  wire [31:0] _GEN_53; // @[Compute.scala 249:30]
  wire [31:0] _GEN_54; // @[Compute.scala 249:30]
  wire [31:0] _GEN_55; // @[Compute.scala 249:30]
  wire [31:0] _GEN_56; // @[Compute.scala 249:30]
  wire [31:0] _GEN_57; // @[Compute.scala 249:30]
  wire [31:0] _GEN_58; // @[Compute.scala 249:30]
  wire [31:0] _GEN_59; // @[Compute.scala 249:30]
  wire [31:0] _GEN_60; // @[Compute.scala 249:30]
  wire [31:0] _GEN_61; // @[Compute.scala 249:30]
  wire [31:0] _GEN_62; // @[Compute.scala 249:30]
  wire [31:0] _GEN_63; // @[Compute.scala 249:30]
  wire [31:0] _GEN_64; // @[Compute.scala 249:30]
  wire [31:0] _GEN_65; // @[Compute.scala 249:30]
  wire [31:0] _GEN_66; // @[Compute.scala 249:30]
  wire [31:0] _GEN_67; // @[Compute.scala 249:30]
  wire [31:0] _GEN_68; // @[Compute.scala 249:30]
  wire [31:0] _GEN_69; // @[Compute.scala 249:30]
  wire [31:0] _GEN_70; // @[Compute.scala 249:30]
  wire [31:0] _GEN_71; // @[Compute.scala 249:30]
  wire [31:0] _GEN_72; // @[Compute.scala 249:30]
  wire [31:0] _GEN_73; // @[Compute.scala 249:30]
  wire [31:0] _GEN_74; // @[Compute.scala 249:30]
  wire [31:0] _GEN_75; // @[Compute.scala 260:20]
  wire [31:0] _GEN_76; // @[Compute.scala 260:20]
  wire [31:0] _GEN_77; // @[Compute.scala 260:20]
  wire [31:0] _GEN_78; // @[Compute.scala 260:20]
  wire [31:0] _GEN_79; // @[Compute.scala 260:20]
  wire [31:0] _GEN_80; // @[Compute.scala 260:20]
  wire [31:0] _GEN_81; // @[Compute.scala 260:20]
  wire [31:0] _GEN_82; // @[Compute.scala 260:20]
  wire [31:0] _GEN_83; // @[Compute.scala 260:20]
  wire [31:0] _GEN_84; // @[Compute.scala 260:20]
  wire [31:0] _GEN_85; // @[Compute.scala 260:20]
  wire [31:0] _GEN_86; // @[Compute.scala 260:20]
  wire [31:0] _GEN_87; // @[Compute.scala 260:20]
  wire [31:0] _GEN_88; // @[Compute.scala 260:20]
  wire [31:0] _GEN_89; // @[Compute.scala 260:20]
  wire [31:0] _GEN_90; // @[Compute.scala 260:20]
  wire [31:0] src_0_0; // @[Compute.scala 248:36]
  wire [31:0] src_1_0; // @[Compute.scala 248:36]
  wire  _T_951; // @[Compute.scala 265:34]
  wire [31:0] _T_952; // @[Compute.scala 265:24]
  wire [31:0] mix_val_0; // @[Compute.scala 248:36]
  wire [7:0] _T_953; // @[Compute.scala 267:37]
  wire [31:0] _T_954; // @[Compute.scala 268:30]
  wire [31:0] _T_955; // @[Compute.scala 268:59]
  wire [32:0] _T_956; // @[Compute.scala 268:49]
  wire [31:0] _T_957; // @[Compute.scala 268:49]
  wire [31:0] _T_958; // @[Compute.scala 268:79]
  wire [31:0] add_val_0; // @[Compute.scala 248:36]
  wire [31:0] add_res_0; // @[Compute.scala 248:36]
  wire [7:0] _T_959; // @[Compute.scala 270:37]
  wire [4:0] _T_961; // @[Compute.scala 271:60]
  wire [31:0] _T_962; // @[Compute.scala 271:49]
  wire [31:0] _T_963; // @[Compute.scala 271:84]
  wire [31:0] shr_val_0; // @[Compute.scala 248:36]
  wire [31:0] shr_res_0; // @[Compute.scala 248:36]
  wire [7:0] _T_964; // @[Compute.scala 273:37]
  wire [31:0] src_0_1; // @[Compute.scala 248:36]
  wire [31:0] src_1_1; // @[Compute.scala 248:36]
  wire  _T_965; // @[Compute.scala 265:34]
  wire [31:0] _T_966; // @[Compute.scala 265:24]
  wire [31:0] mix_val_1; // @[Compute.scala 248:36]
  wire [7:0] _T_967; // @[Compute.scala 267:37]
  wire [31:0] _T_968; // @[Compute.scala 268:30]
  wire [31:0] _T_969; // @[Compute.scala 268:59]
  wire [32:0] _T_970; // @[Compute.scala 268:49]
  wire [31:0] _T_971; // @[Compute.scala 268:49]
  wire [31:0] _T_972; // @[Compute.scala 268:79]
  wire [31:0] add_val_1; // @[Compute.scala 248:36]
  wire [31:0] add_res_1; // @[Compute.scala 248:36]
  wire [7:0] _T_973; // @[Compute.scala 270:37]
  wire [4:0] _T_975; // @[Compute.scala 271:60]
  wire [31:0] _T_976; // @[Compute.scala 271:49]
  wire [31:0] _T_977; // @[Compute.scala 271:84]
  wire [31:0] shr_val_1; // @[Compute.scala 248:36]
  wire [31:0] shr_res_1; // @[Compute.scala 248:36]
  wire [7:0] _T_978; // @[Compute.scala 273:37]
  wire [31:0] src_0_2; // @[Compute.scala 248:36]
  wire [31:0] src_1_2; // @[Compute.scala 248:36]
  wire  _T_979; // @[Compute.scala 265:34]
  wire [31:0] _T_980; // @[Compute.scala 265:24]
  wire [31:0] mix_val_2; // @[Compute.scala 248:36]
  wire [7:0] _T_981; // @[Compute.scala 267:37]
  wire [31:0] _T_982; // @[Compute.scala 268:30]
  wire [31:0] _T_983; // @[Compute.scala 268:59]
  wire [32:0] _T_984; // @[Compute.scala 268:49]
  wire [31:0] _T_985; // @[Compute.scala 268:49]
  wire [31:0] _T_986; // @[Compute.scala 268:79]
  wire [31:0] add_val_2; // @[Compute.scala 248:36]
  wire [31:0] add_res_2; // @[Compute.scala 248:36]
  wire [7:0] _T_987; // @[Compute.scala 270:37]
  wire [4:0] _T_989; // @[Compute.scala 271:60]
  wire [31:0] _T_990; // @[Compute.scala 271:49]
  wire [31:0] _T_991; // @[Compute.scala 271:84]
  wire [31:0] shr_val_2; // @[Compute.scala 248:36]
  wire [31:0] shr_res_2; // @[Compute.scala 248:36]
  wire [7:0] _T_992; // @[Compute.scala 273:37]
  wire [31:0] src_0_3; // @[Compute.scala 248:36]
  wire [31:0] src_1_3; // @[Compute.scala 248:36]
  wire  _T_993; // @[Compute.scala 265:34]
  wire [31:0] _T_994; // @[Compute.scala 265:24]
  wire [31:0] mix_val_3; // @[Compute.scala 248:36]
  wire [7:0] _T_995; // @[Compute.scala 267:37]
  wire [31:0] _T_996; // @[Compute.scala 268:30]
  wire [31:0] _T_997; // @[Compute.scala 268:59]
  wire [32:0] _T_998; // @[Compute.scala 268:49]
  wire [31:0] _T_999; // @[Compute.scala 268:49]
  wire [31:0] _T_1000; // @[Compute.scala 268:79]
  wire [31:0] add_val_3; // @[Compute.scala 248:36]
  wire [31:0] add_res_3; // @[Compute.scala 248:36]
  wire [7:0] _T_1001; // @[Compute.scala 270:37]
  wire [4:0] _T_1003; // @[Compute.scala 271:60]
  wire [31:0] _T_1004; // @[Compute.scala 271:49]
  wire [31:0] _T_1005; // @[Compute.scala 271:84]
  wire [31:0] shr_val_3; // @[Compute.scala 248:36]
  wire [31:0] shr_res_3; // @[Compute.scala 248:36]
  wire [7:0] _T_1006; // @[Compute.scala 273:37]
  wire [31:0] src_0_4; // @[Compute.scala 248:36]
  wire [31:0] src_1_4; // @[Compute.scala 248:36]
  wire  _T_1007; // @[Compute.scala 265:34]
  wire [31:0] _T_1008; // @[Compute.scala 265:24]
  wire [31:0] mix_val_4; // @[Compute.scala 248:36]
  wire [7:0] _T_1009; // @[Compute.scala 267:37]
  wire [31:0] _T_1010; // @[Compute.scala 268:30]
  wire [31:0] _T_1011; // @[Compute.scala 268:59]
  wire [32:0] _T_1012; // @[Compute.scala 268:49]
  wire [31:0] _T_1013; // @[Compute.scala 268:49]
  wire [31:0] _T_1014; // @[Compute.scala 268:79]
  wire [31:0] add_val_4; // @[Compute.scala 248:36]
  wire [31:0] add_res_4; // @[Compute.scala 248:36]
  wire [7:0] _T_1015; // @[Compute.scala 270:37]
  wire [4:0] _T_1017; // @[Compute.scala 271:60]
  wire [31:0] _T_1018; // @[Compute.scala 271:49]
  wire [31:0] _T_1019; // @[Compute.scala 271:84]
  wire [31:0] shr_val_4; // @[Compute.scala 248:36]
  wire [31:0] shr_res_4; // @[Compute.scala 248:36]
  wire [7:0] _T_1020; // @[Compute.scala 273:37]
  wire [31:0] src_0_5; // @[Compute.scala 248:36]
  wire [31:0] src_1_5; // @[Compute.scala 248:36]
  wire  _T_1021; // @[Compute.scala 265:34]
  wire [31:0] _T_1022; // @[Compute.scala 265:24]
  wire [31:0] mix_val_5; // @[Compute.scala 248:36]
  wire [7:0] _T_1023; // @[Compute.scala 267:37]
  wire [31:0] _T_1024; // @[Compute.scala 268:30]
  wire [31:0] _T_1025; // @[Compute.scala 268:59]
  wire [32:0] _T_1026; // @[Compute.scala 268:49]
  wire [31:0] _T_1027; // @[Compute.scala 268:49]
  wire [31:0] _T_1028; // @[Compute.scala 268:79]
  wire [31:0] add_val_5; // @[Compute.scala 248:36]
  wire [31:0] add_res_5; // @[Compute.scala 248:36]
  wire [7:0] _T_1029; // @[Compute.scala 270:37]
  wire [4:0] _T_1031; // @[Compute.scala 271:60]
  wire [31:0] _T_1032; // @[Compute.scala 271:49]
  wire [31:0] _T_1033; // @[Compute.scala 271:84]
  wire [31:0] shr_val_5; // @[Compute.scala 248:36]
  wire [31:0] shr_res_5; // @[Compute.scala 248:36]
  wire [7:0] _T_1034; // @[Compute.scala 273:37]
  wire [31:0] src_0_6; // @[Compute.scala 248:36]
  wire [31:0] src_1_6; // @[Compute.scala 248:36]
  wire  _T_1035; // @[Compute.scala 265:34]
  wire [31:0] _T_1036; // @[Compute.scala 265:24]
  wire [31:0] mix_val_6; // @[Compute.scala 248:36]
  wire [7:0] _T_1037; // @[Compute.scala 267:37]
  wire [31:0] _T_1038; // @[Compute.scala 268:30]
  wire [31:0] _T_1039; // @[Compute.scala 268:59]
  wire [32:0] _T_1040; // @[Compute.scala 268:49]
  wire [31:0] _T_1041; // @[Compute.scala 268:49]
  wire [31:0] _T_1042; // @[Compute.scala 268:79]
  wire [31:0] add_val_6; // @[Compute.scala 248:36]
  wire [31:0] add_res_6; // @[Compute.scala 248:36]
  wire [7:0] _T_1043; // @[Compute.scala 270:37]
  wire [4:0] _T_1045; // @[Compute.scala 271:60]
  wire [31:0] _T_1046; // @[Compute.scala 271:49]
  wire [31:0] _T_1047; // @[Compute.scala 271:84]
  wire [31:0] shr_val_6; // @[Compute.scala 248:36]
  wire [31:0] shr_res_6; // @[Compute.scala 248:36]
  wire [7:0] _T_1048; // @[Compute.scala 273:37]
  wire [31:0] src_0_7; // @[Compute.scala 248:36]
  wire [31:0] src_1_7; // @[Compute.scala 248:36]
  wire  _T_1049; // @[Compute.scala 265:34]
  wire [31:0] _T_1050; // @[Compute.scala 265:24]
  wire [31:0] mix_val_7; // @[Compute.scala 248:36]
  wire [7:0] _T_1051; // @[Compute.scala 267:37]
  wire [31:0] _T_1052; // @[Compute.scala 268:30]
  wire [31:0] _T_1053; // @[Compute.scala 268:59]
  wire [32:0] _T_1054; // @[Compute.scala 268:49]
  wire [31:0] _T_1055; // @[Compute.scala 268:49]
  wire [31:0] _T_1056; // @[Compute.scala 268:79]
  wire [31:0] add_val_7; // @[Compute.scala 248:36]
  wire [31:0] add_res_7; // @[Compute.scala 248:36]
  wire [7:0] _T_1057; // @[Compute.scala 270:37]
  wire [4:0] _T_1059; // @[Compute.scala 271:60]
  wire [31:0] _T_1060; // @[Compute.scala 271:49]
  wire [31:0] _T_1061; // @[Compute.scala 271:84]
  wire [31:0] shr_val_7; // @[Compute.scala 248:36]
  wire [31:0] shr_res_7; // @[Compute.scala 248:36]
  wire [7:0] _T_1062; // @[Compute.scala 273:37]
  wire [31:0] src_0_8; // @[Compute.scala 248:36]
  wire [31:0] src_1_8; // @[Compute.scala 248:36]
  wire  _T_1063; // @[Compute.scala 265:34]
  wire [31:0] _T_1064; // @[Compute.scala 265:24]
  wire [31:0] mix_val_8; // @[Compute.scala 248:36]
  wire [7:0] _T_1065; // @[Compute.scala 267:37]
  wire [31:0] _T_1066; // @[Compute.scala 268:30]
  wire [31:0] _T_1067; // @[Compute.scala 268:59]
  wire [32:0] _T_1068; // @[Compute.scala 268:49]
  wire [31:0] _T_1069; // @[Compute.scala 268:49]
  wire [31:0] _T_1070; // @[Compute.scala 268:79]
  wire [31:0] add_val_8; // @[Compute.scala 248:36]
  wire [31:0] add_res_8; // @[Compute.scala 248:36]
  wire [7:0] _T_1071; // @[Compute.scala 270:37]
  wire [4:0] _T_1073; // @[Compute.scala 271:60]
  wire [31:0] _T_1074; // @[Compute.scala 271:49]
  wire [31:0] _T_1075; // @[Compute.scala 271:84]
  wire [31:0] shr_val_8; // @[Compute.scala 248:36]
  wire [31:0] shr_res_8; // @[Compute.scala 248:36]
  wire [7:0] _T_1076; // @[Compute.scala 273:37]
  wire [31:0] src_0_9; // @[Compute.scala 248:36]
  wire [31:0] src_1_9; // @[Compute.scala 248:36]
  wire  _T_1077; // @[Compute.scala 265:34]
  wire [31:0] _T_1078; // @[Compute.scala 265:24]
  wire [31:0] mix_val_9; // @[Compute.scala 248:36]
  wire [7:0] _T_1079; // @[Compute.scala 267:37]
  wire [31:0] _T_1080; // @[Compute.scala 268:30]
  wire [31:0] _T_1081; // @[Compute.scala 268:59]
  wire [32:0] _T_1082; // @[Compute.scala 268:49]
  wire [31:0] _T_1083; // @[Compute.scala 268:49]
  wire [31:0] _T_1084; // @[Compute.scala 268:79]
  wire [31:0] add_val_9; // @[Compute.scala 248:36]
  wire [31:0] add_res_9; // @[Compute.scala 248:36]
  wire [7:0] _T_1085; // @[Compute.scala 270:37]
  wire [4:0] _T_1087; // @[Compute.scala 271:60]
  wire [31:0] _T_1088; // @[Compute.scala 271:49]
  wire [31:0] _T_1089; // @[Compute.scala 271:84]
  wire [31:0] shr_val_9; // @[Compute.scala 248:36]
  wire [31:0] shr_res_9; // @[Compute.scala 248:36]
  wire [7:0] _T_1090; // @[Compute.scala 273:37]
  wire [31:0] src_0_10; // @[Compute.scala 248:36]
  wire [31:0] src_1_10; // @[Compute.scala 248:36]
  wire  _T_1091; // @[Compute.scala 265:34]
  wire [31:0] _T_1092; // @[Compute.scala 265:24]
  wire [31:0] mix_val_10; // @[Compute.scala 248:36]
  wire [7:0] _T_1093; // @[Compute.scala 267:37]
  wire [31:0] _T_1094; // @[Compute.scala 268:30]
  wire [31:0] _T_1095; // @[Compute.scala 268:59]
  wire [32:0] _T_1096; // @[Compute.scala 268:49]
  wire [31:0] _T_1097; // @[Compute.scala 268:49]
  wire [31:0] _T_1098; // @[Compute.scala 268:79]
  wire [31:0] add_val_10; // @[Compute.scala 248:36]
  wire [31:0] add_res_10; // @[Compute.scala 248:36]
  wire [7:0] _T_1099; // @[Compute.scala 270:37]
  wire [4:0] _T_1101; // @[Compute.scala 271:60]
  wire [31:0] _T_1102; // @[Compute.scala 271:49]
  wire [31:0] _T_1103; // @[Compute.scala 271:84]
  wire [31:0] shr_val_10; // @[Compute.scala 248:36]
  wire [31:0] shr_res_10; // @[Compute.scala 248:36]
  wire [7:0] _T_1104; // @[Compute.scala 273:37]
  wire [31:0] src_0_11; // @[Compute.scala 248:36]
  wire [31:0] src_1_11; // @[Compute.scala 248:36]
  wire  _T_1105; // @[Compute.scala 265:34]
  wire [31:0] _T_1106; // @[Compute.scala 265:24]
  wire [31:0] mix_val_11; // @[Compute.scala 248:36]
  wire [7:0] _T_1107; // @[Compute.scala 267:37]
  wire [31:0] _T_1108; // @[Compute.scala 268:30]
  wire [31:0] _T_1109; // @[Compute.scala 268:59]
  wire [32:0] _T_1110; // @[Compute.scala 268:49]
  wire [31:0] _T_1111; // @[Compute.scala 268:49]
  wire [31:0] _T_1112; // @[Compute.scala 268:79]
  wire [31:0] add_val_11; // @[Compute.scala 248:36]
  wire [31:0] add_res_11; // @[Compute.scala 248:36]
  wire [7:0] _T_1113; // @[Compute.scala 270:37]
  wire [4:0] _T_1115; // @[Compute.scala 271:60]
  wire [31:0] _T_1116; // @[Compute.scala 271:49]
  wire [31:0] _T_1117; // @[Compute.scala 271:84]
  wire [31:0] shr_val_11; // @[Compute.scala 248:36]
  wire [31:0] shr_res_11; // @[Compute.scala 248:36]
  wire [7:0] _T_1118; // @[Compute.scala 273:37]
  wire [31:0] src_0_12; // @[Compute.scala 248:36]
  wire [31:0] src_1_12; // @[Compute.scala 248:36]
  wire  _T_1119; // @[Compute.scala 265:34]
  wire [31:0] _T_1120; // @[Compute.scala 265:24]
  wire [31:0] mix_val_12; // @[Compute.scala 248:36]
  wire [7:0] _T_1121; // @[Compute.scala 267:37]
  wire [31:0] _T_1122; // @[Compute.scala 268:30]
  wire [31:0] _T_1123; // @[Compute.scala 268:59]
  wire [32:0] _T_1124; // @[Compute.scala 268:49]
  wire [31:0] _T_1125; // @[Compute.scala 268:49]
  wire [31:0] _T_1126; // @[Compute.scala 268:79]
  wire [31:0] add_val_12; // @[Compute.scala 248:36]
  wire [31:0] add_res_12; // @[Compute.scala 248:36]
  wire [7:0] _T_1127; // @[Compute.scala 270:37]
  wire [4:0] _T_1129; // @[Compute.scala 271:60]
  wire [31:0] _T_1130; // @[Compute.scala 271:49]
  wire [31:0] _T_1131; // @[Compute.scala 271:84]
  wire [31:0] shr_val_12; // @[Compute.scala 248:36]
  wire [31:0] shr_res_12; // @[Compute.scala 248:36]
  wire [7:0] _T_1132; // @[Compute.scala 273:37]
  wire [31:0] src_0_13; // @[Compute.scala 248:36]
  wire [31:0] src_1_13; // @[Compute.scala 248:36]
  wire  _T_1133; // @[Compute.scala 265:34]
  wire [31:0] _T_1134; // @[Compute.scala 265:24]
  wire [31:0] mix_val_13; // @[Compute.scala 248:36]
  wire [7:0] _T_1135; // @[Compute.scala 267:37]
  wire [31:0] _T_1136; // @[Compute.scala 268:30]
  wire [31:0] _T_1137; // @[Compute.scala 268:59]
  wire [32:0] _T_1138; // @[Compute.scala 268:49]
  wire [31:0] _T_1139; // @[Compute.scala 268:49]
  wire [31:0] _T_1140; // @[Compute.scala 268:79]
  wire [31:0] add_val_13; // @[Compute.scala 248:36]
  wire [31:0] add_res_13; // @[Compute.scala 248:36]
  wire [7:0] _T_1141; // @[Compute.scala 270:37]
  wire [4:0] _T_1143; // @[Compute.scala 271:60]
  wire [31:0] _T_1144; // @[Compute.scala 271:49]
  wire [31:0] _T_1145; // @[Compute.scala 271:84]
  wire [31:0] shr_val_13; // @[Compute.scala 248:36]
  wire [31:0] shr_res_13; // @[Compute.scala 248:36]
  wire [7:0] _T_1146; // @[Compute.scala 273:37]
  wire [31:0] src_0_14; // @[Compute.scala 248:36]
  wire [31:0] src_1_14; // @[Compute.scala 248:36]
  wire  _T_1147; // @[Compute.scala 265:34]
  wire [31:0] _T_1148; // @[Compute.scala 265:24]
  wire [31:0] mix_val_14; // @[Compute.scala 248:36]
  wire [7:0] _T_1149; // @[Compute.scala 267:37]
  wire [31:0] _T_1150; // @[Compute.scala 268:30]
  wire [31:0] _T_1151; // @[Compute.scala 268:59]
  wire [32:0] _T_1152; // @[Compute.scala 268:49]
  wire [31:0] _T_1153; // @[Compute.scala 268:49]
  wire [31:0] _T_1154; // @[Compute.scala 268:79]
  wire [31:0] add_val_14; // @[Compute.scala 248:36]
  wire [31:0] add_res_14; // @[Compute.scala 248:36]
  wire [7:0] _T_1155; // @[Compute.scala 270:37]
  wire [4:0] _T_1157; // @[Compute.scala 271:60]
  wire [31:0] _T_1158; // @[Compute.scala 271:49]
  wire [31:0] _T_1159; // @[Compute.scala 271:84]
  wire [31:0] shr_val_14; // @[Compute.scala 248:36]
  wire [31:0] shr_res_14; // @[Compute.scala 248:36]
  wire [7:0] _T_1160; // @[Compute.scala 273:37]
  wire [31:0] src_0_15; // @[Compute.scala 248:36]
  wire [31:0] src_1_15; // @[Compute.scala 248:36]
  wire  _T_1161; // @[Compute.scala 265:34]
  wire [31:0] _T_1162; // @[Compute.scala 265:24]
  wire [31:0] mix_val_15; // @[Compute.scala 248:36]
  wire [7:0] _T_1163; // @[Compute.scala 267:37]
  wire [31:0] _T_1164; // @[Compute.scala 268:30]
  wire [31:0] _T_1165; // @[Compute.scala 268:59]
  wire [32:0] _T_1166; // @[Compute.scala 268:49]
  wire [31:0] _T_1167; // @[Compute.scala 268:49]
  wire [31:0] _T_1168; // @[Compute.scala 268:79]
  wire [31:0] add_val_15; // @[Compute.scala 248:36]
  wire [31:0] add_res_15; // @[Compute.scala 248:36]
  wire [7:0] _T_1169; // @[Compute.scala 270:37]
  wire [4:0] _T_1171; // @[Compute.scala 271:60]
  wire [31:0] _T_1172; // @[Compute.scala 271:49]
  wire [31:0] _T_1173; // @[Compute.scala 271:84]
  wire [31:0] shr_val_15; // @[Compute.scala 248:36]
  wire [31:0] shr_res_15; // @[Compute.scala 248:36]
  wire [7:0] _T_1174; // @[Compute.scala 273:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 248:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 248:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 248:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 248:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 280:48]
  wire  alu_opcode_add_en; // @[Compute.scala 281:39]
  wire [63:0] _T_1182; // @[Cat.scala 30:58]
  wire [127:0] _T_1190; // @[Cat.scala 30:58]
  wire [63:0] _T_1197; // @[Cat.scala 30:58]
  wire [127:0] _T_1205; // @[Cat.scala 30:58]
  wire [63:0] _T_1212; // @[Cat.scala 30:58]
  wire [127:0] _T_1220; // @[Cat.scala 30:58]
  wire [127:0] _T_1221; // @[Compute.scala 283:30]
  BinaryQueue g2l_queue ( // @[Compute.scala 285:25]
    .clock(g2l_queue_clock),
    .reset(g2l_queue_reset),
    .io_enq_ready(g2l_queue_io_enq_ready),
    .io_enq_valid(g2l_queue_io_enq_valid),
    .io_deq_ready(g2l_queue_io_deq_ready),
    .io_deq_valid(g2l_queue_io_deq_valid),
    .io_deq_bits(g2l_queue_io_deq_bits)
  );
  BinaryQueue g2s_queue ( // @[Compute.scala 286:25]
    .clock(g2s_queue_clock),
    .reset(g2s_queue_reset),
    .io_enq_ready(g2s_queue_io_enq_ready),
    .io_enq_valid(g2s_queue_io_enq_valid),
    .io_deq_ready(g2s_queue_io_deq_ready),
    .io_deq_valid(g2s_queue_io_deq_valid),
    .io_deq_bits(g2s_queue_io_deq_bits)
  );
  assign acc_mem__T_382_addr = dst_idx[7:0];
  assign acc_mem__T_382_data = acc_mem[acc_mem__T_382_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_385_addr = src_idx[7:0];
  assign acc_mem__T_385_data = acc_mem[acc_mem__T_385_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_365_data = {_T_367,_T_366};
  assign acc_mem__T_365_addr = acc_sram_addr[7:0];
  assign acc_mem__T_365_mask = 1'h1;
  assign acc_mem__T_365_en = _T_278 ? _T_363 : 1'h0;
  assign uop_mem_uop_addr = 10'h0;
  assign uop_mem_uop_data = uop_mem[uop_mem_uop_addr]; // @[Compute.scala 34:20]
  assign uop_mem__T_327_data = uops_data;
  assign uop_mem__T_327_addr = uop_sram_addr[9:0];
  assign uop_mem__T_327_mask = 1'h1;
  assign uop_mem__T_327_en = 1'h1;
  assign started = reset == 1'h0; // @[Compute.scala 31:17]
  assign insn_valid = insn != 128'h0; // @[Compute.scala 37:30]
  assign opcode = insn[2:0]; // @[Compute.scala 39:29]
  assign push_prev_dep = insn[5]; // @[Compute.scala 42:29]
  assign push_next_dep = insn[6]; // @[Compute.scala 43:29]
  assign memory_type = insn[8:7]; // @[Compute.scala 45:25]
  assign sram_base = insn[24:9]; // @[Compute.scala 46:25]
  assign dram_base = insn[56:25]; // @[Compute.scala 47:25]
  assign x_size = insn[95:80]; // @[Compute.scala 49:25]
  assign y_pad_0 = insn[115:112]; // @[Compute.scala 51:25]
  assign x_pad_0 = insn[123:120]; // @[Compute.scala 53:25]
  assign x_pad_1 = insn[127:124]; // @[Compute.scala 54:25]
  assign _GEN_281 = {{12'd0}, y_pad_0}; // @[Compute.scala 58:30]
  assign _GEN_283 = {{12'd0}, x_pad_0}; // @[Compute.scala 59:30]
  assign _T_204 = _GEN_283 + x_size; // @[Compute.scala 59:30]
  assign _T_205 = _GEN_283 + x_size; // @[Compute.scala 59:30]
  assign _GEN_284 = {{12'd0}, x_pad_1}; // @[Compute.scala 59:39]
  assign _T_206 = _T_205 + _GEN_284; // @[Compute.scala 59:39]
  assign x_size_total = _T_205 + _GEN_284; // @[Compute.scala 59:39]
  assign y_offset = x_size_total * _GEN_281; // @[Compute.scala 60:31]
  assign opcode_finish_en = opcode == 3'h3; // @[Compute.scala 63:34]
  assign _T_209 = opcode == 3'h0; // @[Compute.scala 64:32]
  assign _T_211 = opcode == 3'h1; // @[Compute.scala 64:60]
  assign opcode_load_en = _T_209 | _T_211; // @[Compute.scala 64:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Compute.scala 65:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Compute.scala 66:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Compute.scala 68:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Compute.scala 69:40]
  assign _T_216 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 73:37]
  assign _T_217 = _T_216 & started; // @[Compute.scala 73:59]
  assign uop_cntr_en = _T_217 & insn_valid; // @[Compute.scala 73:70]
  assign _T_220 = uop_cntr_val == 16'h1; // @[Compute.scala 76:38]
  assign uop_cntr_wrap = _T_220 & uop_cntr_en; // @[Compute.scala 76:58]
  assign _T_221 = opcode_load_en & memory_type_acc_en; // @[Compute.scala 79:37]
  assign _T_222 = _T_221 & started; // @[Compute.scala 79:59]
  assign acc_cntr_en = _T_222 & insn_valid; // @[Compute.scala 79:70]
  assign _T_225 = acc_cntr_val == 16'h20; // @[Compute.scala 82:38]
  assign acc_cntr_wrap = _T_225 & acc_cntr_en; // @[Compute.scala 82:58]
  assign _T_226 = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 85:37]
  assign _T_227 = _T_226 & started; // @[Compute.scala 85:56]
  assign out_cntr_en = _T_227 & insn_valid; // @[Compute.scala 85:67]
  assign _T_230 = dst_offset_in == 16'h8; // @[Compute.scala 88:38]
  assign out_cntr_wrap = _T_230 & out_cntr_en; // @[Compute.scala 88:58]
  assign idle = state == 4'h0; // @[Compute.scala 101:20]
  assign busy = state == 4'h1; // @[Compute.scala 102:20]
  assign done = state == 4'h2; // @[Compute.scala 103:20]
  assign _T_251 = uop_cntr_wrap == 1'h0; // @[Compute.scala 106:31]
  assign _T_252 = uop_cntr_en & _T_251; // @[Compute.scala 106:28]
  assign _T_255 = acc_cntr_wrap == 1'h0; // @[Compute.scala 107:31]
  assign _T_256 = acc_cntr_en & _T_255; // @[Compute.scala 107:28]
  assign _T_259 = out_cntr_wrap == 1'h0; // @[Compute.scala 108:31]
  assign _T_260 = out_cntr_en & _T_259; // @[Compute.scala 108:28]
  assign _T_264 = _T_256 ? 1'h1 : _T_260; // @[Compute.scala 107:15]
  assign _T_265 = _T_252 ? 1'h1 : _T_264; // @[Compute.scala 106:15]
  assign _T_267 = io_uops_waitrequest == 1'h0; // @[Compute.scala 111:22]
  assign _T_268 = uops_read & _T_267; // @[Compute.scala 111:19]
  assign _T_269 = _T_268 & busy; // @[Compute.scala 111:37]
  assign _T_271 = uop_cntr_val < 16'h1; // @[Compute.scala 112:24]
  assign _T_273 = uop_cntr_val + 16'h1; // @[Compute.scala 113:36]
  assign _T_274 = uop_cntr_val + 16'h1; // @[Compute.scala 113:36]
  assign _GEN_0 = _T_271 ? _T_274 : uop_cntr_val; // @[Compute.scala 112:42]
  assign _GEN_1 = _T_271 ? {{1'd0}, _T_265} : 2'h2; // @[Compute.scala 112:42]
  assign _GEN_2 = _T_269 ? _GEN_0 : uop_cntr_val; // @[Compute.scala 111:46]
  assign _GEN_3 = _T_269 ? _GEN_1 : {{1'd0}, _T_265}; // @[Compute.scala 111:46]
  assign _T_277 = io_biases_waitrequest == 1'h0; // @[Compute.scala 119:24]
  assign _T_278 = biases_read & _T_277; // @[Compute.scala 119:21]
  assign _T_279 = _T_278 & busy; // @[Compute.scala 119:39]
  assign _T_281 = acc_cntr_val < 16'h20; // @[Compute.scala 120:24]
  assign _T_283 = acc_cntr_val + 16'h1; // @[Compute.scala 121:36]
  assign _T_284 = acc_cntr_val + 16'h1; // @[Compute.scala 121:36]
  assign _GEN_4 = _T_281 ? _T_284 : acc_cntr_val; // @[Compute.scala 120:42]
  assign _GEN_5 = _T_281 ? _GEN_3 : 2'h2; // @[Compute.scala 120:42]
  assign _GEN_6 = _T_279 ? _GEN_4 : acc_cntr_val; // @[Compute.scala 119:48]
  assign _GEN_7 = _T_279 ? _GEN_5 : _GEN_3; // @[Compute.scala 119:48]
  assign _T_287 = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 127:24]
  assign _T_288 = out_cntr_en & _T_287; // @[Compute.scala 127:21]
  assign _T_289 = _T_288 & busy; // @[Compute.scala 127:39]
  assign _T_291 = dst_offset_in < 16'h8; // @[Compute.scala 128:24]
  assign _T_293 = dst_offset_in + 16'h1; // @[Compute.scala 129:36]
  assign _T_294 = dst_offset_in + 16'h1; // @[Compute.scala 129:36]
  assign _GEN_8 = _T_291 ? _T_294 : dst_offset_in; // @[Compute.scala 128:42]
  assign _GEN_9 = _T_291 ? _GEN_7 : 2'h2; // @[Compute.scala 128:42]
  assign _GEN_10 = _T_289 ? _GEN_8 : dst_offset_in; // @[Compute.scala 127:48]
  assign _GEN_11 = _T_289 ? _GEN_9 : _GEN_7; // @[Compute.scala 127:48]
  assign _T_297 = uop_cntr_wrap & uop_cntr_en; // @[Compute.scala 139:25]
  assign _GEN_12 = _T_297 ? 16'h0 : _GEN_2; // @[Compute.scala 139:41]
  assign _T_299 = acc_cntr_wrap & acc_cntr_en; // @[Compute.scala 140:25]
  assign _GEN_13 = _T_299 ? 16'h0 : _GEN_6; // @[Compute.scala 140:41]
  assign _T_301 = out_cntr_wrap & out_cntr_en; // @[Compute.scala 141:25]
  assign _GEN_14 = _T_301 ? 16'h0 : _GEN_10; // @[Compute.scala 141:41]
  assign _GEN_15 = done ? 2'h0 : _GEN_11; // @[Compute.scala 137:15]
  assign _T_303 = io_gemm_queue_valid & idle; // @[Compute.scala 145:29]
  assign _GEN_286 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 163:33]
  assign _T_316 = dram_base + _GEN_286; // @[Compute.scala 163:33]
  assign _T_317 = dram_base + _GEN_286; // @[Compute.scala 163:33]
  assign _GEN_287 = {{3'd0}, _T_317}; // @[Compute.scala 163:49]
  assign uop_dram_addr = _GEN_287 << 2'h2; // @[Compute.scala 163:49]
  assign _T_319 = sram_base + uop_cntr_val; // @[Compute.scala 164:33]
  assign uop_sram_addr = sram_base + uop_cntr_val; // @[Compute.scala 164:33]
  assign _GEN_288 = {{12'd0}, y_offset}; // @[Compute.scala 176:35]
  assign _T_328 = dram_base + _GEN_288; // @[Compute.scala 176:35]
  assign _T_329 = dram_base + _GEN_288; // @[Compute.scala 176:35]
  assign _GEN_289 = {{28'd0}, x_pad_0}; // @[Compute.scala 176:46]
  assign _T_330 = _T_329 + _GEN_289; // @[Compute.scala 176:46]
  assign _T_331 = _T_329 + _GEN_289; // @[Compute.scala 176:46]
  assign _T_333 = _T_331 * 32'h1; // @[Compute.scala 176:57]
  assign _GEN_290 = {{17'd0}, acc_cntr_val}; // @[Compute.scala 176:67]
  assign _T_334 = _T_333 + _GEN_290; // @[Compute.scala 176:67]
  assign _T_335 = _T_333 + _GEN_290; // @[Compute.scala 176:67]
  assign _GEN_291 = {{7'd0}, _T_335}; // @[Compute.scala 176:83]
  assign acc_dram_addr = _GEN_291 << 3'h4; // @[Compute.scala 176:83]
  assign _GEN_292 = {{4'd0}, sram_base}; // @[Compute.scala 177:35]
  assign _T_337 = _GEN_292 + y_offset; // @[Compute.scala 177:35]
  assign _T_338 = _GEN_292 + y_offset; // @[Compute.scala 177:35]
  assign _GEN_293 = {{16'd0}, x_pad_0}; // @[Compute.scala 177:46]
  assign _T_339 = _T_338 + _GEN_293; // @[Compute.scala 177:46]
  assign _T_340 = _T_338 + _GEN_293; // @[Compute.scala 177:46]
  assign _T_342 = _T_340 * 20'h1; // @[Compute.scala 177:57]
  assign _GEN_294 = {{5'd0}, acc_cntr_val}; // @[Compute.scala 177:67]
  assign _T_343 = _T_342 + _GEN_294; // @[Compute.scala 177:67]
  assign _T_344 = _T_342 + _GEN_294; // @[Compute.scala 177:67]
  assign _T_346 = _T_344 >> 2'h2; // @[Compute.scala 177:83]
  assign _T_348 = _T_346 - 21'h1; // @[Compute.scala 177:91]
  assign _T_349 = $unsigned(_T_348); // @[Compute.scala 177:91]
  assign acc_sram_addr = _T_349[20:0]; // @[Compute.scala 177:91]
  assign _T_351 = done == 1'h0; // @[Compute.scala 179:33]
  assign _GEN_16 = acc_cntr_val % 16'h4; // @[Compute.scala 185:30]
  assign _T_357 = _GEN_16[2:0]; // @[Compute.scala 185:30]
  assign _GEN_23 = 3'h0 == _T_357 ? io_biases_readdata : biases_data_0; // @[Compute.scala 185:67]
  assign _GEN_24 = 3'h1 == _T_357 ? io_biases_readdata : biases_data_1; // @[Compute.scala 185:67]
  assign _GEN_25 = 3'h2 == _T_357 ? io_biases_readdata : biases_data_2; // @[Compute.scala 185:67]
  assign _GEN_26 = 3'h3 == _T_357 ? io_biases_readdata : biases_data_3; // @[Compute.scala 185:67]
  assign _T_363 = _T_357 == 3'h0; // @[Compute.scala 186:64]
  assign _T_366 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_367 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 196:24]
  assign use_imm = insn[110]; // @[Compute.scala 197:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 198:21]
  assign _T_369 = $signed(imm_raw); // @[Compute.scala 199:25]
  assign _T_371 = $signed(_T_369) < $signed(16'sh0); // @[Compute.scala 199:32]
  assign _T_373 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_375 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_376 = _T_371 ? _T_373 : {{15'd0}, _T_375}; // @[Compute.scala 199:16]
  assign imm = $signed(_T_376); // @[Compute.scala 199:89]
  assign _T_377 = uop_mem_uop_data[10:0]; // @[Compute.scala 207:20]
  assign _GEN_295 = {{5'd0}, _T_377}; // @[Compute.scala 207:47]
  assign _T_378 = _GEN_295 + dst_offset_in; // @[Compute.scala 207:47]
  assign dst_idx = _GEN_295 + dst_offset_in; // @[Compute.scala 207:47]
  assign _T_379 = uop_mem_uop_data[21:11]; // @[Compute.scala 208:20]
  assign _GEN_296 = {{5'd0}, _T_379}; // @[Compute.scala 208:47]
  assign _T_380 = _GEN_296 + dst_offset_in; // @[Compute.scala 208:47]
  assign src_idx = _GEN_296 + dst_offset_in; // @[Compute.scala 208:47]
  assign _GEN_297 = {{7'd0}, dst_idx}; // @[Compute.scala 225:39]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 228:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 229:38]
  assign _T_822 = insn_valid & out_cntr_en; // @[Compute.scala 248:20]
  assign _T_823 = src_vector[31:0]; // @[Compute.scala 251:31]
  assign _T_824 = $signed(_T_823); // @[Compute.scala 251:72]
  assign _T_825 = dst_vector[31:0]; // @[Compute.scala 252:31]
  assign _T_826 = $signed(_T_825); // @[Compute.scala 252:72]
  assign _T_827 = src_vector[63:32]; // @[Compute.scala 251:31]
  assign _T_828 = $signed(_T_827); // @[Compute.scala 251:72]
  assign _T_829 = dst_vector[63:32]; // @[Compute.scala 252:31]
  assign _T_830 = $signed(_T_829); // @[Compute.scala 252:72]
  assign _T_831 = src_vector[95:64]; // @[Compute.scala 251:31]
  assign _T_832 = $signed(_T_831); // @[Compute.scala 251:72]
  assign _T_833 = dst_vector[95:64]; // @[Compute.scala 252:31]
  assign _T_834 = $signed(_T_833); // @[Compute.scala 252:72]
  assign _T_835 = src_vector[127:96]; // @[Compute.scala 251:31]
  assign _T_836 = $signed(_T_835); // @[Compute.scala 251:72]
  assign _T_837 = dst_vector[127:96]; // @[Compute.scala 252:31]
  assign _T_838 = $signed(_T_837); // @[Compute.scala 252:72]
  assign _T_839 = src_vector[159:128]; // @[Compute.scala 251:31]
  assign _T_840 = $signed(_T_839); // @[Compute.scala 251:72]
  assign _T_841 = dst_vector[159:128]; // @[Compute.scala 252:31]
  assign _T_842 = $signed(_T_841); // @[Compute.scala 252:72]
  assign _T_843 = src_vector[191:160]; // @[Compute.scala 251:31]
  assign _T_844 = $signed(_T_843); // @[Compute.scala 251:72]
  assign _T_845 = dst_vector[191:160]; // @[Compute.scala 252:31]
  assign _T_846 = $signed(_T_845); // @[Compute.scala 252:72]
  assign _T_847 = src_vector[223:192]; // @[Compute.scala 251:31]
  assign _T_848 = $signed(_T_847); // @[Compute.scala 251:72]
  assign _T_849 = dst_vector[223:192]; // @[Compute.scala 252:31]
  assign _T_850 = $signed(_T_849); // @[Compute.scala 252:72]
  assign _T_851 = src_vector[255:224]; // @[Compute.scala 251:31]
  assign _T_852 = $signed(_T_851); // @[Compute.scala 251:72]
  assign _T_853 = dst_vector[255:224]; // @[Compute.scala 252:31]
  assign _T_854 = $signed(_T_853); // @[Compute.scala 252:72]
  assign _T_855 = src_vector[287:256]; // @[Compute.scala 251:31]
  assign _T_856 = $signed(_T_855); // @[Compute.scala 251:72]
  assign _T_857 = dst_vector[287:256]; // @[Compute.scala 252:31]
  assign _T_858 = $signed(_T_857); // @[Compute.scala 252:72]
  assign _T_859 = src_vector[319:288]; // @[Compute.scala 251:31]
  assign _T_860 = $signed(_T_859); // @[Compute.scala 251:72]
  assign _T_861 = dst_vector[319:288]; // @[Compute.scala 252:31]
  assign _T_862 = $signed(_T_861); // @[Compute.scala 252:72]
  assign _T_863 = src_vector[351:320]; // @[Compute.scala 251:31]
  assign _T_864 = $signed(_T_863); // @[Compute.scala 251:72]
  assign _T_865 = dst_vector[351:320]; // @[Compute.scala 252:31]
  assign _T_866 = $signed(_T_865); // @[Compute.scala 252:72]
  assign _T_867 = src_vector[383:352]; // @[Compute.scala 251:31]
  assign _T_868 = $signed(_T_867); // @[Compute.scala 251:72]
  assign _T_869 = dst_vector[383:352]; // @[Compute.scala 252:31]
  assign _T_870 = $signed(_T_869); // @[Compute.scala 252:72]
  assign _T_871 = src_vector[415:384]; // @[Compute.scala 251:31]
  assign _T_872 = $signed(_T_871); // @[Compute.scala 251:72]
  assign _T_873 = dst_vector[415:384]; // @[Compute.scala 252:31]
  assign _T_874 = $signed(_T_873); // @[Compute.scala 252:72]
  assign _T_875 = src_vector[447:416]; // @[Compute.scala 251:31]
  assign _T_876 = $signed(_T_875); // @[Compute.scala 251:72]
  assign _T_877 = dst_vector[447:416]; // @[Compute.scala 252:31]
  assign _T_878 = $signed(_T_877); // @[Compute.scala 252:72]
  assign _T_879 = src_vector[479:448]; // @[Compute.scala 251:31]
  assign _T_880 = $signed(_T_879); // @[Compute.scala 251:72]
  assign _T_881 = dst_vector[479:448]; // @[Compute.scala 252:31]
  assign _T_882 = $signed(_T_881); // @[Compute.scala 252:72]
  assign _T_883 = src_vector[511:480]; // @[Compute.scala 251:31]
  assign _T_884 = $signed(_T_883); // @[Compute.scala 251:72]
  assign _T_885 = dst_vector[511:480]; // @[Compute.scala 252:31]
  assign _T_886 = $signed(_T_885); // @[Compute.scala 252:72]
  assign _GEN_43 = alu_opcode_max_en ? $signed(_T_824) : $signed(_T_826); // @[Compute.scala 249:30]
  assign _GEN_44 = alu_opcode_max_en ? $signed(_T_826) : $signed(_T_824); // @[Compute.scala 249:30]
  assign _GEN_45 = alu_opcode_max_en ? $signed(_T_828) : $signed(_T_830); // @[Compute.scala 249:30]
  assign _GEN_46 = alu_opcode_max_en ? $signed(_T_830) : $signed(_T_828); // @[Compute.scala 249:30]
  assign _GEN_47 = alu_opcode_max_en ? $signed(_T_832) : $signed(_T_834); // @[Compute.scala 249:30]
  assign _GEN_48 = alu_opcode_max_en ? $signed(_T_834) : $signed(_T_832); // @[Compute.scala 249:30]
  assign _GEN_49 = alu_opcode_max_en ? $signed(_T_836) : $signed(_T_838); // @[Compute.scala 249:30]
  assign _GEN_50 = alu_opcode_max_en ? $signed(_T_838) : $signed(_T_836); // @[Compute.scala 249:30]
  assign _GEN_51 = alu_opcode_max_en ? $signed(_T_840) : $signed(_T_842); // @[Compute.scala 249:30]
  assign _GEN_52 = alu_opcode_max_en ? $signed(_T_842) : $signed(_T_840); // @[Compute.scala 249:30]
  assign _GEN_53 = alu_opcode_max_en ? $signed(_T_844) : $signed(_T_846); // @[Compute.scala 249:30]
  assign _GEN_54 = alu_opcode_max_en ? $signed(_T_846) : $signed(_T_844); // @[Compute.scala 249:30]
  assign _GEN_55 = alu_opcode_max_en ? $signed(_T_848) : $signed(_T_850); // @[Compute.scala 249:30]
  assign _GEN_56 = alu_opcode_max_en ? $signed(_T_850) : $signed(_T_848); // @[Compute.scala 249:30]
  assign _GEN_57 = alu_opcode_max_en ? $signed(_T_852) : $signed(_T_854); // @[Compute.scala 249:30]
  assign _GEN_58 = alu_opcode_max_en ? $signed(_T_854) : $signed(_T_852); // @[Compute.scala 249:30]
  assign _GEN_59 = alu_opcode_max_en ? $signed(_T_856) : $signed(_T_858); // @[Compute.scala 249:30]
  assign _GEN_60 = alu_opcode_max_en ? $signed(_T_858) : $signed(_T_856); // @[Compute.scala 249:30]
  assign _GEN_61 = alu_opcode_max_en ? $signed(_T_860) : $signed(_T_862); // @[Compute.scala 249:30]
  assign _GEN_62 = alu_opcode_max_en ? $signed(_T_862) : $signed(_T_860); // @[Compute.scala 249:30]
  assign _GEN_63 = alu_opcode_max_en ? $signed(_T_864) : $signed(_T_866); // @[Compute.scala 249:30]
  assign _GEN_64 = alu_opcode_max_en ? $signed(_T_866) : $signed(_T_864); // @[Compute.scala 249:30]
  assign _GEN_65 = alu_opcode_max_en ? $signed(_T_868) : $signed(_T_870); // @[Compute.scala 249:30]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_870) : $signed(_T_868); // @[Compute.scala 249:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_872) : $signed(_T_874); // @[Compute.scala 249:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_874) : $signed(_T_872); // @[Compute.scala 249:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_876) : $signed(_T_878); // @[Compute.scala 249:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_878) : $signed(_T_876); // @[Compute.scala 249:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_880) : $signed(_T_882); // @[Compute.scala 249:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_882) : $signed(_T_880); // @[Compute.scala 249:30]
  assign _GEN_73 = alu_opcode_max_en ? $signed(_T_884) : $signed(_T_886); // @[Compute.scala 249:30]
  assign _GEN_74 = alu_opcode_max_en ? $signed(_T_886) : $signed(_T_884); // @[Compute.scala 249:30]
  assign _GEN_75 = use_imm ? $signed(imm) : $signed(_GEN_44); // @[Compute.scala 260:20]
  assign _GEN_76 = use_imm ? $signed(imm) : $signed(_GEN_46); // @[Compute.scala 260:20]
  assign _GEN_77 = use_imm ? $signed(imm) : $signed(_GEN_48); // @[Compute.scala 260:20]
  assign _GEN_78 = use_imm ? $signed(imm) : $signed(_GEN_50); // @[Compute.scala 260:20]
  assign _GEN_79 = use_imm ? $signed(imm) : $signed(_GEN_52); // @[Compute.scala 260:20]
  assign _GEN_80 = use_imm ? $signed(imm) : $signed(_GEN_54); // @[Compute.scala 260:20]
  assign _GEN_81 = use_imm ? $signed(imm) : $signed(_GEN_56); // @[Compute.scala 260:20]
  assign _GEN_82 = use_imm ? $signed(imm) : $signed(_GEN_58); // @[Compute.scala 260:20]
  assign _GEN_83 = use_imm ? $signed(imm) : $signed(_GEN_60); // @[Compute.scala 260:20]
  assign _GEN_84 = use_imm ? $signed(imm) : $signed(_GEN_62); // @[Compute.scala 260:20]
  assign _GEN_85 = use_imm ? $signed(imm) : $signed(_GEN_64); // @[Compute.scala 260:20]
  assign _GEN_86 = use_imm ? $signed(imm) : $signed(_GEN_66); // @[Compute.scala 260:20]
  assign _GEN_87 = use_imm ? $signed(imm) : $signed(_GEN_68); // @[Compute.scala 260:20]
  assign _GEN_88 = use_imm ? $signed(imm) : $signed(_GEN_70); // @[Compute.scala 260:20]
  assign _GEN_89 = use_imm ? $signed(imm) : $signed(_GEN_72); // @[Compute.scala 260:20]
  assign _GEN_90 = use_imm ? $signed(imm) : $signed(_GEN_74); // @[Compute.scala 260:20]
  assign src_0_0 = _T_822 ? $signed(_GEN_43) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_0 = _T_822 ? $signed(_GEN_75) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_951 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 265:34]
  assign _T_952 = _T_951 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 265:24]
  assign mix_val_0 = _T_822 ? $signed(_T_952) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_953 = mix_val_0[7:0]; // @[Compute.scala 267:37]
  assign _T_954 = $unsigned(src_0_0); // @[Compute.scala 268:30]
  assign _T_955 = $unsigned(src_1_0); // @[Compute.scala 268:59]
  assign _T_956 = _T_954 + _T_955; // @[Compute.scala 268:49]
  assign _T_957 = _T_954 + _T_955; // @[Compute.scala 268:49]
  assign _T_958 = $signed(_T_957); // @[Compute.scala 268:79]
  assign add_val_0 = _T_822 ? $signed(_T_958) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_0 = _T_822 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_959 = add_res_0[7:0]; // @[Compute.scala 270:37]
  assign _T_961 = src_1_0[4:0]; // @[Compute.scala 271:60]
  assign _T_962 = _T_954 >> _T_961; // @[Compute.scala 271:49]
  assign _T_963 = $signed(_T_962); // @[Compute.scala 271:84]
  assign shr_val_0 = _T_822 ? $signed(_T_963) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_0 = _T_822 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_964 = shr_res_0[7:0]; // @[Compute.scala 273:37]
  assign src_0_1 = _T_822 ? $signed(_GEN_45) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_1 = _T_822 ? $signed(_GEN_76) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_965 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 265:34]
  assign _T_966 = _T_965 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 265:24]
  assign mix_val_1 = _T_822 ? $signed(_T_966) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_967 = mix_val_1[7:0]; // @[Compute.scala 267:37]
  assign _T_968 = $unsigned(src_0_1); // @[Compute.scala 268:30]
  assign _T_969 = $unsigned(src_1_1); // @[Compute.scala 268:59]
  assign _T_970 = _T_968 + _T_969; // @[Compute.scala 268:49]
  assign _T_971 = _T_968 + _T_969; // @[Compute.scala 268:49]
  assign _T_972 = $signed(_T_971); // @[Compute.scala 268:79]
  assign add_val_1 = _T_822 ? $signed(_T_972) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_1 = _T_822 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_973 = add_res_1[7:0]; // @[Compute.scala 270:37]
  assign _T_975 = src_1_1[4:0]; // @[Compute.scala 271:60]
  assign _T_976 = _T_968 >> _T_975; // @[Compute.scala 271:49]
  assign _T_977 = $signed(_T_976); // @[Compute.scala 271:84]
  assign shr_val_1 = _T_822 ? $signed(_T_977) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_1 = _T_822 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_978 = shr_res_1[7:0]; // @[Compute.scala 273:37]
  assign src_0_2 = _T_822 ? $signed(_GEN_47) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_2 = _T_822 ? $signed(_GEN_77) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_979 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 265:34]
  assign _T_980 = _T_979 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 265:24]
  assign mix_val_2 = _T_822 ? $signed(_T_980) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_981 = mix_val_2[7:0]; // @[Compute.scala 267:37]
  assign _T_982 = $unsigned(src_0_2); // @[Compute.scala 268:30]
  assign _T_983 = $unsigned(src_1_2); // @[Compute.scala 268:59]
  assign _T_984 = _T_982 + _T_983; // @[Compute.scala 268:49]
  assign _T_985 = _T_982 + _T_983; // @[Compute.scala 268:49]
  assign _T_986 = $signed(_T_985); // @[Compute.scala 268:79]
  assign add_val_2 = _T_822 ? $signed(_T_986) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_2 = _T_822 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_987 = add_res_2[7:0]; // @[Compute.scala 270:37]
  assign _T_989 = src_1_2[4:0]; // @[Compute.scala 271:60]
  assign _T_990 = _T_982 >> _T_989; // @[Compute.scala 271:49]
  assign _T_991 = $signed(_T_990); // @[Compute.scala 271:84]
  assign shr_val_2 = _T_822 ? $signed(_T_991) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_2 = _T_822 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_992 = shr_res_2[7:0]; // @[Compute.scala 273:37]
  assign src_0_3 = _T_822 ? $signed(_GEN_49) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_3 = _T_822 ? $signed(_GEN_78) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_993 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 265:34]
  assign _T_994 = _T_993 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 265:24]
  assign mix_val_3 = _T_822 ? $signed(_T_994) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_995 = mix_val_3[7:0]; // @[Compute.scala 267:37]
  assign _T_996 = $unsigned(src_0_3); // @[Compute.scala 268:30]
  assign _T_997 = $unsigned(src_1_3); // @[Compute.scala 268:59]
  assign _T_998 = _T_996 + _T_997; // @[Compute.scala 268:49]
  assign _T_999 = _T_996 + _T_997; // @[Compute.scala 268:49]
  assign _T_1000 = $signed(_T_999); // @[Compute.scala 268:79]
  assign add_val_3 = _T_822 ? $signed(_T_1000) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_3 = _T_822 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1001 = add_res_3[7:0]; // @[Compute.scala 270:37]
  assign _T_1003 = src_1_3[4:0]; // @[Compute.scala 271:60]
  assign _T_1004 = _T_996 >> _T_1003; // @[Compute.scala 271:49]
  assign _T_1005 = $signed(_T_1004); // @[Compute.scala 271:84]
  assign shr_val_3 = _T_822 ? $signed(_T_1005) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_3 = _T_822 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1006 = shr_res_3[7:0]; // @[Compute.scala 273:37]
  assign src_0_4 = _T_822 ? $signed(_GEN_51) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_4 = _T_822 ? $signed(_GEN_79) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1007 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 265:34]
  assign _T_1008 = _T_1007 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 265:24]
  assign mix_val_4 = _T_822 ? $signed(_T_1008) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1009 = mix_val_4[7:0]; // @[Compute.scala 267:37]
  assign _T_1010 = $unsigned(src_0_4); // @[Compute.scala 268:30]
  assign _T_1011 = $unsigned(src_1_4); // @[Compute.scala 268:59]
  assign _T_1012 = _T_1010 + _T_1011; // @[Compute.scala 268:49]
  assign _T_1013 = _T_1010 + _T_1011; // @[Compute.scala 268:49]
  assign _T_1014 = $signed(_T_1013); // @[Compute.scala 268:79]
  assign add_val_4 = _T_822 ? $signed(_T_1014) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_4 = _T_822 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1015 = add_res_4[7:0]; // @[Compute.scala 270:37]
  assign _T_1017 = src_1_4[4:0]; // @[Compute.scala 271:60]
  assign _T_1018 = _T_1010 >> _T_1017; // @[Compute.scala 271:49]
  assign _T_1019 = $signed(_T_1018); // @[Compute.scala 271:84]
  assign shr_val_4 = _T_822 ? $signed(_T_1019) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_4 = _T_822 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1020 = shr_res_4[7:0]; // @[Compute.scala 273:37]
  assign src_0_5 = _T_822 ? $signed(_GEN_53) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_5 = _T_822 ? $signed(_GEN_80) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1021 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 265:34]
  assign _T_1022 = _T_1021 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 265:24]
  assign mix_val_5 = _T_822 ? $signed(_T_1022) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1023 = mix_val_5[7:0]; // @[Compute.scala 267:37]
  assign _T_1024 = $unsigned(src_0_5); // @[Compute.scala 268:30]
  assign _T_1025 = $unsigned(src_1_5); // @[Compute.scala 268:59]
  assign _T_1026 = _T_1024 + _T_1025; // @[Compute.scala 268:49]
  assign _T_1027 = _T_1024 + _T_1025; // @[Compute.scala 268:49]
  assign _T_1028 = $signed(_T_1027); // @[Compute.scala 268:79]
  assign add_val_5 = _T_822 ? $signed(_T_1028) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_5 = _T_822 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1029 = add_res_5[7:0]; // @[Compute.scala 270:37]
  assign _T_1031 = src_1_5[4:0]; // @[Compute.scala 271:60]
  assign _T_1032 = _T_1024 >> _T_1031; // @[Compute.scala 271:49]
  assign _T_1033 = $signed(_T_1032); // @[Compute.scala 271:84]
  assign shr_val_5 = _T_822 ? $signed(_T_1033) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_5 = _T_822 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1034 = shr_res_5[7:0]; // @[Compute.scala 273:37]
  assign src_0_6 = _T_822 ? $signed(_GEN_55) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_6 = _T_822 ? $signed(_GEN_81) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1035 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 265:34]
  assign _T_1036 = _T_1035 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 265:24]
  assign mix_val_6 = _T_822 ? $signed(_T_1036) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1037 = mix_val_6[7:0]; // @[Compute.scala 267:37]
  assign _T_1038 = $unsigned(src_0_6); // @[Compute.scala 268:30]
  assign _T_1039 = $unsigned(src_1_6); // @[Compute.scala 268:59]
  assign _T_1040 = _T_1038 + _T_1039; // @[Compute.scala 268:49]
  assign _T_1041 = _T_1038 + _T_1039; // @[Compute.scala 268:49]
  assign _T_1042 = $signed(_T_1041); // @[Compute.scala 268:79]
  assign add_val_6 = _T_822 ? $signed(_T_1042) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_6 = _T_822 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1043 = add_res_6[7:0]; // @[Compute.scala 270:37]
  assign _T_1045 = src_1_6[4:0]; // @[Compute.scala 271:60]
  assign _T_1046 = _T_1038 >> _T_1045; // @[Compute.scala 271:49]
  assign _T_1047 = $signed(_T_1046); // @[Compute.scala 271:84]
  assign shr_val_6 = _T_822 ? $signed(_T_1047) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_6 = _T_822 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1048 = shr_res_6[7:0]; // @[Compute.scala 273:37]
  assign src_0_7 = _T_822 ? $signed(_GEN_57) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_7 = _T_822 ? $signed(_GEN_82) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1049 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 265:34]
  assign _T_1050 = _T_1049 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 265:24]
  assign mix_val_7 = _T_822 ? $signed(_T_1050) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1051 = mix_val_7[7:0]; // @[Compute.scala 267:37]
  assign _T_1052 = $unsigned(src_0_7); // @[Compute.scala 268:30]
  assign _T_1053 = $unsigned(src_1_7); // @[Compute.scala 268:59]
  assign _T_1054 = _T_1052 + _T_1053; // @[Compute.scala 268:49]
  assign _T_1055 = _T_1052 + _T_1053; // @[Compute.scala 268:49]
  assign _T_1056 = $signed(_T_1055); // @[Compute.scala 268:79]
  assign add_val_7 = _T_822 ? $signed(_T_1056) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_7 = _T_822 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1057 = add_res_7[7:0]; // @[Compute.scala 270:37]
  assign _T_1059 = src_1_7[4:0]; // @[Compute.scala 271:60]
  assign _T_1060 = _T_1052 >> _T_1059; // @[Compute.scala 271:49]
  assign _T_1061 = $signed(_T_1060); // @[Compute.scala 271:84]
  assign shr_val_7 = _T_822 ? $signed(_T_1061) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_7 = _T_822 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1062 = shr_res_7[7:0]; // @[Compute.scala 273:37]
  assign src_0_8 = _T_822 ? $signed(_GEN_59) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_8 = _T_822 ? $signed(_GEN_83) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1063 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 265:34]
  assign _T_1064 = _T_1063 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 265:24]
  assign mix_val_8 = _T_822 ? $signed(_T_1064) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1065 = mix_val_8[7:0]; // @[Compute.scala 267:37]
  assign _T_1066 = $unsigned(src_0_8); // @[Compute.scala 268:30]
  assign _T_1067 = $unsigned(src_1_8); // @[Compute.scala 268:59]
  assign _T_1068 = _T_1066 + _T_1067; // @[Compute.scala 268:49]
  assign _T_1069 = _T_1066 + _T_1067; // @[Compute.scala 268:49]
  assign _T_1070 = $signed(_T_1069); // @[Compute.scala 268:79]
  assign add_val_8 = _T_822 ? $signed(_T_1070) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_8 = _T_822 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1071 = add_res_8[7:0]; // @[Compute.scala 270:37]
  assign _T_1073 = src_1_8[4:0]; // @[Compute.scala 271:60]
  assign _T_1074 = _T_1066 >> _T_1073; // @[Compute.scala 271:49]
  assign _T_1075 = $signed(_T_1074); // @[Compute.scala 271:84]
  assign shr_val_8 = _T_822 ? $signed(_T_1075) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_8 = _T_822 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1076 = shr_res_8[7:0]; // @[Compute.scala 273:37]
  assign src_0_9 = _T_822 ? $signed(_GEN_61) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_9 = _T_822 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1077 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 265:34]
  assign _T_1078 = _T_1077 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 265:24]
  assign mix_val_9 = _T_822 ? $signed(_T_1078) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1079 = mix_val_9[7:0]; // @[Compute.scala 267:37]
  assign _T_1080 = $unsigned(src_0_9); // @[Compute.scala 268:30]
  assign _T_1081 = $unsigned(src_1_9); // @[Compute.scala 268:59]
  assign _T_1082 = _T_1080 + _T_1081; // @[Compute.scala 268:49]
  assign _T_1083 = _T_1080 + _T_1081; // @[Compute.scala 268:49]
  assign _T_1084 = $signed(_T_1083); // @[Compute.scala 268:79]
  assign add_val_9 = _T_822 ? $signed(_T_1084) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_9 = _T_822 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1085 = add_res_9[7:0]; // @[Compute.scala 270:37]
  assign _T_1087 = src_1_9[4:0]; // @[Compute.scala 271:60]
  assign _T_1088 = _T_1080 >> _T_1087; // @[Compute.scala 271:49]
  assign _T_1089 = $signed(_T_1088); // @[Compute.scala 271:84]
  assign shr_val_9 = _T_822 ? $signed(_T_1089) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_9 = _T_822 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1090 = shr_res_9[7:0]; // @[Compute.scala 273:37]
  assign src_0_10 = _T_822 ? $signed(_GEN_63) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_10 = _T_822 ? $signed(_GEN_85) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1091 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 265:34]
  assign _T_1092 = _T_1091 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 265:24]
  assign mix_val_10 = _T_822 ? $signed(_T_1092) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1093 = mix_val_10[7:0]; // @[Compute.scala 267:37]
  assign _T_1094 = $unsigned(src_0_10); // @[Compute.scala 268:30]
  assign _T_1095 = $unsigned(src_1_10); // @[Compute.scala 268:59]
  assign _T_1096 = _T_1094 + _T_1095; // @[Compute.scala 268:49]
  assign _T_1097 = _T_1094 + _T_1095; // @[Compute.scala 268:49]
  assign _T_1098 = $signed(_T_1097); // @[Compute.scala 268:79]
  assign add_val_10 = _T_822 ? $signed(_T_1098) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_10 = _T_822 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1099 = add_res_10[7:0]; // @[Compute.scala 270:37]
  assign _T_1101 = src_1_10[4:0]; // @[Compute.scala 271:60]
  assign _T_1102 = _T_1094 >> _T_1101; // @[Compute.scala 271:49]
  assign _T_1103 = $signed(_T_1102); // @[Compute.scala 271:84]
  assign shr_val_10 = _T_822 ? $signed(_T_1103) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_10 = _T_822 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1104 = shr_res_10[7:0]; // @[Compute.scala 273:37]
  assign src_0_11 = _T_822 ? $signed(_GEN_65) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_11 = _T_822 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1105 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 265:34]
  assign _T_1106 = _T_1105 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 265:24]
  assign mix_val_11 = _T_822 ? $signed(_T_1106) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1107 = mix_val_11[7:0]; // @[Compute.scala 267:37]
  assign _T_1108 = $unsigned(src_0_11); // @[Compute.scala 268:30]
  assign _T_1109 = $unsigned(src_1_11); // @[Compute.scala 268:59]
  assign _T_1110 = _T_1108 + _T_1109; // @[Compute.scala 268:49]
  assign _T_1111 = _T_1108 + _T_1109; // @[Compute.scala 268:49]
  assign _T_1112 = $signed(_T_1111); // @[Compute.scala 268:79]
  assign add_val_11 = _T_822 ? $signed(_T_1112) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_11 = _T_822 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1113 = add_res_11[7:0]; // @[Compute.scala 270:37]
  assign _T_1115 = src_1_11[4:0]; // @[Compute.scala 271:60]
  assign _T_1116 = _T_1108 >> _T_1115; // @[Compute.scala 271:49]
  assign _T_1117 = $signed(_T_1116); // @[Compute.scala 271:84]
  assign shr_val_11 = _T_822 ? $signed(_T_1117) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_11 = _T_822 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1118 = shr_res_11[7:0]; // @[Compute.scala 273:37]
  assign src_0_12 = _T_822 ? $signed(_GEN_67) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_12 = _T_822 ? $signed(_GEN_87) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1119 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 265:34]
  assign _T_1120 = _T_1119 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 265:24]
  assign mix_val_12 = _T_822 ? $signed(_T_1120) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1121 = mix_val_12[7:0]; // @[Compute.scala 267:37]
  assign _T_1122 = $unsigned(src_0_12); // @[Compute.scala 268:30]
  assign _T_1123 = $unsigned(src_1_12); // @[Compute.scala 268:59]
  assign _T_1124 = _T_1122 + _T_1123; // @[Compute.scala 268:49]
  assign _T_1125 = _T_1122 + _T_1123; // @[Compute.scala 268:49]
  assign _T_1126 = $signed(_T_1125); // @[Compute.scala 268:79]
  assign add_val_12 = _T_822 ? $signed(_T_1126) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_12 = _T_822 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1127 = add_res_12[7:0]; // @[Compute.scala 270:37]
  assign _T_1129 = src_1_12[4:0]; // @[Compute.scala 271:60]
  assign _T_1130 = _T_1122 >> _T_1129; // @[Compute.scala 271:49]
  assign _T_1131 = $signed(_T_1130); // @[Compute.scala 271:84]
  assign shr_val_12 = _T_822 ? $signed(_T_1131) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_12 = _T_822 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1132 = shr_res_12[7:0]; // @[Compute.scala 273:37]
  assign src_0_13 = _T_822 ? $signed(_GEN_69) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_13 = _T_822 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1133 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 265:34]
  assign _T_1134 = _T_1133 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 265:24]
  assign mix_val_13 = _T_822 ? $signed(_T_1134) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1135 = mix_val_13[7:0]; // @[Compute.scala 267:37]
  assign _T_1136 = $unsigned(src_0_13); // @[Compute.scala 268:30]
  assign _T_1137 = $unsigned(src_1_13); // @[Compute.scala 268:59]
  assign _T_1138 = _T_1136 + _T_1137; // @[Compute.scala 268:49]
  assign _T_1139 = _T_1136 + _T_1137; // @[Compute.scala 268:49]
  assign _T_1140 = $signed(_T_1139); // @[Compute.scala 268:79]
  assign add_val_13 = _T_822 ? $signed(_T_1140) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_13 = _T_822 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1141 = add_res_13[7:0]; // @[Compute.scala 270:37]
  assign _T_1143 = src_1_13[4:0]; // @[Compute.scala 271:60]
  assign _T_1144 = _T_1136 >> _T_1143; // @[Compute.scala 271:49]
  assign _T_1145 = $signed(_T_1144); // @[Compute.scala 271:84]
  assign shr_val_13 = _T_822 ? $signed(_T_1145) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_13 = _T_822 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1146 = shr_res_13[7:0]; // @[Compute.scala 273:37]
  assign src_0_14 = _T_822 ? $signed(_GEN_71) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_14 = _T_822 ? $signed(_GEN_89) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1147 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 265:34]
  assign _T_1148 = _T_1147 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 265:24]
  assign mix_val_14 = _T_822 ? $signed(_T_1148) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1149 = mix_val_14[7:0]; // @[Compute.scala 267:37]
  assign _T_1150 = $unsigned(src_0_14); // @[Compute.scala 268:30]
  assign _T_1151 = $unsigned(src_1_14); // @[Compute.scala 268:59]
  assign _T_1152 = _T_1150 + _T_1151; // @[Compute.scala 268:49]
  assign _T_1153 = _T_1150 + _T_1151; // @[Compute.scala 268:49]
  assign _T_1154 = $signed(_T_1153); // @[Compute.scala 268:79]
  assign add_val_14 = _T_822 ? $signed(_T_1154) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_14 = _T_822 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1155 = add_res_14[7:0]; // @[Compute.scala 270:37]
  assign _T_1157 = src_1_14[4:0]; // @[Compute.scala 271:60]
  assign _T_1158 = _T_1150 >> _T_1157; // @[Compute.scala 271:49]
  assign _T_1159 = $signed(_T_1158); // @[Compute.scala 271:84]
  assign shr_val_14 = _T_822 ? $signed(_T_1159) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_14 = _T_822 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1160 = shr_res_14[7:0]; // @[Compute.scala 273:37]
  assign src_0_15 = _T_822 ? $signed(_GEN_73) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign src_1_15 = _T_822 ? $signed(_GEN_90) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1161 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 265:34]
  assign _T_1162 = _T_1161 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 265:24]
  assign mix_val_15 = _T_822 ? $signed(_T_1162) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1163 = mix_val_15[7:0]; // @[Compute.scala 267:37]
  assign _T_1164 = $unsigned(src_0_15); // @[Compute.scala 268:30]
  assign _T_1165 = $unsigned(src_1_15); // @[Compute.scala 268:59]
  assign _T_1166 = _T_1164 + _T_1165; // @[Compute.scala 268:49]
  assign _T_1167 = _T_1164 + _T_1165; // @[Compute.scala 268:49]
  assign _T_1168 = $signed(_T_1167); // @[Compute.scala 268:79]
  assign add_val_15 = _T_822 ? $signed(_T_1168) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign add_res_15 = _T_822 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1169 = add_res_15[7:0]; // @[Compute.scala 270:37]
  assign _T_1171 = src_1_15[4:0]; // @[Compute.scala 271:60]
  assign _T_1172 = _T_1164 >> _T_1171; // @[Compute.scala 271:49]
  assign _T_1173 = $signed(_T_1172); // @[Compute.scala 271:84]
  assign shr_val_15 = _T_822 ? $signed(_T_1173) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign shr_res_15 = _T_822 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 248:36]
  assign _T_1174 = shr_res_15[7:0]; // @[Compute.scala 273:37]
  assign short_cmp_res_0 = _T_822 ? _T_953 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_0 = _T_822 ? _T_959 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_0 = _T_822 ? _T_964 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_1 = _T_822 ? _T_967 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_1 = _T_822 ? _T_973 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_1 = _T_822 ? _T_978 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_2 = _T_822 ? _T_981 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_2 = _T_822 ? _T_987 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_2 = _T_822 ? _T_992 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_3 = _T_822 ? _T_995 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_3 = _T_822 ? _T_1001 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_3 = _T_822 ? _T_1006 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_4 = _T_822 ? _T_1009 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_4 = _T_822 ? _T_1015 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_4 = _T_822 ? _T_1020 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_5 = _T_822 ? _T_1023 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_5 = _T_822 ? _T_1029 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_5 = _T_822 ? _T_1034 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_6 = _T_822 ? _T_1037 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_6 = _T_822 ? _T_1043 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_6 = _T_822 ? _T_1048 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_7 = _T_822 ? _T_1051 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_7 = _T_822 ? _T_1057 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_7 = _T_822 ? _T_1062 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_8 = _T_822 ? _T_1065 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_8 = _T_822 ? _T_1071 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_8 = _T_822 ? _T_1076 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_9 = _T_822 ? _T_1079 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_9 = _T_822 ? _T_1085 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_9 = _T_822 ? _T_1090 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_10 = _T_822 ? _T_1093 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_10 = _T_822 ? _T_1099 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_10 = _T_822 ? _T_1104 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_11 = _T_822 ? _T_1107 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_11 = _T_822 ? _T_1113 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_11 = _T_822 ? _T_1118 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_12 = _T_822 ? _T_1121 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_12 = _T_822 ? _T_1127 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_12 = _T_822 ? _T_1132 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_13 = _T_822 ? _T_1135 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_13 = _T_822 ? _T_1141 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_13 = _T_822 ? _T_1146 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_14 = _T_822 ? _T_1149 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_14 = _T_822 ? _T_1155 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_14 = _T_822 ? _T_1160 : 8'h0; // @[Compute.scala 248:36]
  assign short_cmp_res_15 = _T_822 ? _T_1163 : 8'h0; // @[Compute.scala 248:36]
  assign short_add_res_15 = _T_822 ? _T_1169 : 8'h0; // @[Compute.scala 248:36]
  assign short_shr_res_15 = _T_822 ? _T_1174 : 8'h0; // @[Compute.scala 248:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 280:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 281:39]
  assign _T_1182 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1190 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1182}; // @[Cat.scala 30:58]
  assign _T_1197 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1205 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1197}; // @[Cat.scala 30:58]
  assign _T_1212 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1220 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1212}; // @[Cat.scala 30:58]
  assign _T_1221 = alu_opcode_add_en ? _T_1205 : _T_1220; // @[Compute.scala 283:30]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 158:23]
  assign io_done_readdata = _T_313; // @[Compute.scala 159:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 167:19]
  assign io_uops_read = uops_read; // @[Compute.scala 166:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 32'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 180:21]
  assign io_biases_read = biases_read; // @[Compute.scala 181:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = _T_303 ? insn_valid : 1'h0; // @[Compute.scala 148:27 Compute.scala 150:27 Compute.scala 154:25]
  assign io_l2g_dep_queue_ready = 1'h0;
  assign io_s2g_dep_queue_ready = 1'h0;
  assign io_g2l_dep_queue_valid = g2l_queue_io_deq_valid; // @[Compute.scala 287:26]
  assign io_g2l_dep_queue_data = g2l_queue_io_deq_bits; // @[Compute.scala 289:26]
  assign io_g2s_dep_queue_valid = g2s_queue_io_deq_valid; // @[Compute.scala 290:26]
  assign io_g2s_dep_queue_data = g2s_queue_io_deq_bits; // @[Compute.scala 292:26]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = out_mem_addr[16:0]; // @[Compute.scala 277:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_write_en; // @[Compute.scala 279:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1190 : _T_1221; // @[Compute.scala 282:24]
  assign g2l_queue_clock = clock;
  assign g2l_queue_reset = reset;
  assign g2l_queue_io_enq_valid = push_prev_dep & out_cntr_wrap; // @[Compute.scala 296:26]
  assign g2l_queue_io_deq_ready = io_g2l_dep_queue_ready; // @[Compute.scala 288:26]
  assign g2s_queue_clock = clock;
  assign g2s_queue_reset = reset;
  assign g2s_queue_io_enq_valid = push_next_dep & out_cntr_wrap; // @[Compute.scala 297:26]
  assign g2s_queue_io_deq_ready = io_g2s_dep_queue_ready; // @[Compute.scala 291:26]
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
  uop_cntr_val = _RAND_3[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  acc_cntr_val = _RAND_4[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{`RANDOM}};
  dst_offset_in = _RAND_5[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  uops_read = _RAND_6[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {1{`RANDOM}};
  uops_data = _RAND_7[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {1{`RANDOM}};
  biases_read = _RAND_8[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {4{`RANDOM}};
  biases_data_0 = _RAND_9[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {4{`RANDOM}};
  biases_data_1 = _RAND_10[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_11 = {4{`RANDOM}};
  biases_data_2 = _RAND_11[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_12 = {4{`RANDOM}};
  biases_data_3 = _RAND_12[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_13 = {1{`RANDOM}};
  state = _RAND_13[3:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_14 = {1{`RANDOM}};
  _T_313 = _RAND_14[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_15 = {16{`RANDOM}};
  dst_vector = _RAND_15[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_16 = {16{`RANDOM}};
  src_vector = _RAND_16[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_17 = {1{`RANDOM}};
  out_mem_addr = _RAND_17[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_18 = {1{`RANDOM}};
  out_mem_write_en = _RAND_18[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_365_en & acc_mem__T_365_mask) begin
      acc_mem[acc_mem__T_365_addr] <= acc_mem__T_365_data; // @[Compute.scala 33:20]
    end
    if(uop_mem__T_327_en & uop_mem__T_327_mask) begin
      uop_mem[uop_mem__T_327_addr] <= uop_mem__T_327_data; // @[Compute.scala 34:20]
    end
    if (_T_303) begin
      insn <= io_gemm_queue_data;
    end
    if (done) begin
      if (_T_297) begin
        uop_cntr_val <= 16'h0;
      end else begin
        if (_T_269) begin
          if (_T_271) begin
            uop_cntr_val <= _T_274;
          end
        end
      end
    end else begin
      if (_T_269) begin
        if (_T_271) begin
          uop_cntr_val <= _T_274;
        end
      end
    end
    if (done) begin
      if (_T_299) begin
        acc_cntr_val <= 16'h0;
      end else begin
        if (_T_279) begin
          if (_T_281) begin
            acc_cntr_val <= _T_284;
          end
        end
      end
    end else begin
      if (_T_279) begin
        if (_T_281) begin
          acc_cntr_val <= _T_284;
        end
      end
    end
    if (done) begin
      if (_T_301) begin
        dst_offset_in <= 16'h0;
      end else begin
        if (_T_289) begin
          if (_T_291) begin
            dst_offset_in <= _T_294;
          end
        end
      end
    end else begin
      if (_T_289) begin
        if (_T_291) begin
          dst_offset_in <= _T_294;
        end
      end
    end
    uops_read <= uop_cntr_en & _T_251;
    if (_T_268) begin
      uops_data <= io_uops_readdata;
    end
    biases_read <= acc_cntr_en & _T_351;
    if (_T_278) begin
      if (3'h0 == _T_357) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_278) begin
      if (3'h1 == _T_357) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_278) begin
      if (3'h2 == _T_357) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_278) begin
      if (3'h3 == _T_357) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    state <= {{2'd0}, _GEN_15};
    _T_313 <= opcode_finish_en & io_done_read;
    dst_vector <= acc_mem__T_382_data;
    src_vector <= acc_mem__T_385_data;
    out_mem_addr <= _GEN_297 << 3'h4;
    out_mem_write_en <= opcode == 3'h4;
  end
endmodule
