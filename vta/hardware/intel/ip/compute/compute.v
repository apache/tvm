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
  wire [511:0] acc_mem__T_387_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_387_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_390_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_390_addr; // @[Compute.scala 33:20]
  wire [511:0] acc_mem__T_370_data; // @[Compute.scala 33:20]
  wire [7:0] acc_mem__T_370_addr; // @[Compute.scala 33:20]
  wire  acc_mem__T_370_mask; // @[Compute.scala 33:20]
  wire  acc_mem__T_370_en; // @[Compute.scala 33:20]
  reg [31:0] uop_mem [0:1023] /* synthesis ramstyle = "M20K" */; // @[Compute.scala 34:20]
  reg [31:0] _RAND_1;
  wire [31:0] uop_mem_uop_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem_uop_addr; // @[Compute.scala 34:20]
  wire [31:0] uop_mem__T_335_data; // @[Compute.scala 34:20]
  wire [9:0] uop_mem__T_335_addr; // @[Compute.scala 34:20]
  wire  uop_mem__T_335_mask; // @[Compute.scala 34:20]
  wire  uop_mem__T_335_en; // @[Compute.scala 34:20]
  wire  g2l_queue_clock; // @[Compute.scala 313:25]
  wire  g2l_queue_reset; // @[Compute.scala 313:25]
  wire  g2l_queue_io_enq_ready; // @[Compute.scala 313:25]
  wire  g2l_queue_io_enq_valid; // @[Compute.scala 313:25]
  wire  g2l_queue_io_deq_ready; // @[Compute.scala 313:25]
  wire  g2l_queue_io_deq_valid; // @[Compute.scala 313:25]
  wire  g2l_queue_io_deq_bits; // @[Compute.scala 313:25]
  wire  g2s_queue_clock; // @[Compute.scala 314:25]
  wire  g2s_queue_reset; // @[Compute.scala 314:25]
  wire  g2s_queue_io_enq_ready; // @[Compute.scala 314:25]
  wire  g2s_queue_io_enq_valid; // @[Compute.scala 314:25]
  wire  g2s_queue_io_deq_ready; // @[Compute.scala 314:25]
  wire  g2s_queue_io_deq_valid; // @[Compute.scala 314:25]
  wire  g2s_queue_io_deq_bits; // @[Compute.scala 314:25]
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
  wire [15:0] _GEN_279; // @[Compute.scala 58:30]
  wire [15:0] _GEN_281; // @[Compute.scala 59:30]
  wire [16:0] _T_204; // @[Compute.scala 59:30]
  wire [15:0] _T_205; // @[Compute.scala 59:30]
  wire [15:0] _GEN_282; // @[Compute.scala 59:39]
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
  wire  uop_cntr_en; // @[Compute.scala 73:59]
  reg [15:0] uop_cntr_val; // @[Compute.scala 75:25]
  reg [31:0] _RAND_3;
  wire  _T_219; // @[Compute.scala 76:45]
  reg  uop_cntr_wrap; // @[Compute.scala 76:30]
  reg [31:0] _RAND_4;
  wire  acc_cntr_en; // @[Compute.scala 79:37]
  reg [15:0] acc_cntr_val; // @[Compute.scala 81:25]
  reg [31:0] _RAND_5;
  wire  _T_224; // @[Compute.scala 82:44]
  reg  acc_cntr_wrap; // @[Compute.scala 82:30]
  reg [31:0] _RAND_6;
  wire  out_cntr_en; // @[Compute.scala 85:36]
  reg [15:0] dst_offset_in; // @[Compute.scala 87:25]
  reg [31:0] _RAND_7;
  wire  _T_229; // @[Compute.scala 88:44]
  reg  out_cntr_wrap; // @[Compute.scala 88:30]
  reg [31:0] _RAND_8;
  reg  uops_read; // @[Compute.scala 91:24]
  reg [31:0] _RAND_9;
  reg [31:0] uops_data; // @[Compute.scala 92:24]
  reg [31:0] _RAND_10;
  reg  biases_read; // @[Compute.scala 95:24]
  reg [31:0] _RAND_11;
  reg [127:0] biases_data_0; // @[Compute.scala 96:24]
  reg [127:0] _RAND_12;
  reg [127:0] biases_data_1; // @[Compute.scala 96:24]
  reg [127:0] _RAND_13;
  reg [127:0] biases_data_2; // @[Compute.scala 96:24]
  reg [127:0] _RAND_14;
  reg [127:0] biases_data_3; // @[Compute.scala 96:24]
  reg [127:0] _RAND_15;
  reg [3:0] state; // @[Compute.scala 100:18]
  reg [31:0] _RAND_16;
  wire  busy; // @[Compute.scala 101:20]
  wire  _T_250; // @[Compute.scala 104:31]
  wire  _T_251; // @[Compute.scala 104:28]
  wire  _T_254; // @[Compute.scala 105:31]
  wire  _T_255; // @[Compute.scala 105:28]
  wire  _T_258; // @[Compute.scala 106:31]
  wire  _T_259; // @[Compute.scala 106:28]
  wire  _T_263; // @[Compute.scala 105:15]
  wire  _T_264; // @[Compute.scala 104:15]
  wire  _T_266; // @[Compute.scala 109:22]
  wire  _T_267; // @[Compute.scala 109:19]
  wire  _T_268; // @[Compute.scala 109:37]
  wire [16:0] _T_272; // @[Compute.scala 111:36]
  wire [15:0] _T_273; // @[Compute.scala 111:36]
  wire [15:0] _GEN_0; // @[Compute.scala 112:57]
  wire  _GEN_1; // @[Compute.scala 112:57]
  wire  _GEN_5; // @[Compute.scala 109:46]
  wire  _T_282; // @[Compute.scala 120:24]
  wire  _T_283; // @[Compute.scala 120:21]
  wire  _T_284; // @[Compute.scala 120:39]
  wire  _T_286; // @[Compute.scala 121:24]
  wire [16:0] _T_288; // @[Compute.scala 122:36]
  wire [15:0] _T_289; // @[Compute.scala 122:36]
  wire [15:0] _GEN_6; // @[Compute.scala 123:57]
  wire  _GEN_7; // @[Compute.scala 123:57]
  wire [15:0] _GEN_8; // @[Compute.scala 121:48]
  wire  _GEN_9; // @[Compute.scala 121:48]
  wire  _GEN_11; // @[Compute.scala 120:48]
  wire  _T_298; // @[Compute.scala 131:24]
  wire  _T_299; // @[Compute.scala 131:21]
  wire  _T_300; // @[Compute.scala 131:39]
  wire  _T_302; // @[Compute.scala 132:24]
  wire [16:0] _T_304; // @[Compute.scala 133:36]
  wire [15:0] _T_305; // @[Compute.scala 133:36]
  wire [15:0] _GEN_12; // @[Compute.scala 134:57]
  wire  _GEN_13; // @[Compute.scala 134:57]
  wire [15:0] _GEN_14; // @[Compute.scala 132:48]
  wire  _GEN_15; // @[Compute.scala 132:48]
  wire  _GEN_17; // @[Compute.scala 131:48]
  wire  _T_314; // @[Compute.scala 143:32]
  wire  _T_315; // @[Compute.scala 143:29]
  reg  _T_324; // @[Compute.scala 153:30]
  reg [31:0] _RAND_17;
  wire [31:0] _GEN_284; // @[Compute.scala 157:33]
  wire [32:0] _T_327; // @[Compute.scala 157:33]
  wire [31:0] _T_328; // @[Compute.scala 157:33]
  wire [34:0] _GEN_285; // @[Compute.scala 157:49]
  wire [34:0] uop_dram_addr; // @[Compute.scala 157:49]
  wire [16:0] _T_330; // @[Compute.scala 158:33]
  wire [15:0] uop_sram_addr; // @[Compute.scala 158:33]
  wire [31:0] _GEN_286; // @[Compute.scala 170:35]
  wire [32:0] _T_336; // @[Compute.scala 170:35]
  wire [31:0] _T_337; // @[Compute.scala 170:35]
  wire [31:0] _GEN_287; // @[Compute.scala 170:46]
  wire [32:0] _T_338; // @[Compute.scala 170:46]
  wire [31:0] _T_339; // @[Compute.scala 170:46]
  wire [32:0] _T_341; // @[Compute.scala 170:57]
  wire [32:0] _GEN_288; // @[Compute.scala 170:67]
  wire [33:0] _T_342; // @[Compute.scala 170:67]
  wire [32:0] _T_343; // @[Compute.scala 170:67]
  wire [39:0] _GEN_289; // @[Compute.scala 170:83]
  wire [39:0] acc_dram_addr; // @[Compute.scala 170:83]
  wire [19:0] _GEN_290; // @[Compute.scala 171:35]
  wire [20:0] _T_345; // @[Compute.scala 171:35]
  wire [19:0] _T_346; // @[Compute.scala 171:35]
  wire [19:0] _GEN_291; // @[Compute.scala 171:46]
  wire [20:0] _T_347; // @[Compute.scala 171:46]
  wire [19:0] _T_348; // @[Compute.scala 171:46]
  wire [20:0] _T_350; // @[Compute.scala 171:57]
  wire [20:0] _GEN_292; // @[Compute.scala 171:67]
  wire [21:0] _T_351; // @[Compute.scala 171:67]
  wire [20:0] _T_352; // @[Compute.scala 171:67]
  wire [20:0] _T_354; // @[Compute.scala 171:83]
  wire [21:0] _T_356; // @[Compute.scala 171:91]
  wire [21:0] _T_357; // @[Compute.scala 171:91]
  wire [20:0] acc_sram_addr; // @[Compute.scala 171:91]
  wire [15:0] _GEN_2; // @[Compute.scala 179:30]
  wire [2:0] _T_362; // @[Compute.scala 179:30]
  wire [127:0] _GEN_21; // @[Compute.scala 179:67]
  wire [127:0] _GEN_22; // @[Compute.scala 179:67]
  wire [127:0] _GEN_23; // @[Compute.scala 179:67]
  wire [127:0] _GEN_24; // @[Compute.scala 179:67]
  wire  _T_368; // @[Compute.scala 180:64]
  wire [255:0] _T_371; // @[Cat.scala 30:58]
  wire [255:0] _T_372; // @[Cat.scala 30:58]
  wire [1:0] alu_opcode; // @[Compute.scala 190:24]
  wire  use_imm; // @[Compute.scala 191:21]
  wire [15:0] imm_raw; // @[Compute.scala 192:21]
  wire [15:0] _T_374; // @[Compute.scala 193:25]
  wire  _T_376; // @[Compute.scala 193:32]
  wire [31:0] _T_378; // @[Cat.scala 30:58]
  wire [16:0] _T_380; // @[Cat.scala 30:58]
  wire [31:0] _T_381; // @[Compute.scala 193:16]
  wire [31:0] imm; // @[Compute.scala 193:89]
  wire [10:0] _T_382; // @[Compute.scala 201:20]
  wire [15:0] _GEN_293; // @[Compute.scala 201:47]
  wire [16:0] _T_383; // @[Compute.scala 201:47]
  wire [15:0] dst_idx; // @[Compute.scala 201:47]
  wire [10:0] _T_384; // @[Compute.scala 202:20]
  wire [15:0] _GEN_294; // @[Compute.scala 202:47]
  wire [16:0] _T_385; // @[Compute.scala 202:47]
  wire [15:0] src_idx; // @[Compute.scala 202:47]
  reg [511:0] dst_vector; // @[Compute.scala 205:27]
  reg [511:0] _RAND_18;
  reg [511:0] src_vector; // @[Compute.scala 206:27]
  reg [511:0] _RAND_19;
  wire [22:0] _GEN_295; // @[Compute.scala 219:39]
  reg [22:0] out_mem_addr; // @[Compute.scala 219:30]
  reg [31:0] _RAND_20;
  reg  out_mem_write_en; // @[Compute.scala 220:34]
  reg [31:0] _RAND_21;
  wire  alu_opcode_min_en; // @[Compute.scala 222:38]
  wire  alu_opcode_max_en; // @[Compute.scala 223:38]
  wire  _T_827; // @[Compute.scala 242:20]
  wire [31:0] _T_828; // @[Compute.scala 245:31]
  wire [31:0] _T_829; // @[Compute.scala 245:72]
  wire [31:0] _T_830; // @[Compute.scala 246:31]
  wire [31:0] _T_831; // @[Compute.scala 246:72]
  wire [31:0] _T_832; // @[Compute.scala 245:31]
  wire [31:0] _T_833; // @[Compute.scala 245:72]
  wire [31:0] _T_834; // @[Compute.scala 246:31]
  wire [31:0] _T_835; // @[Compute.scala 246:72]
  wire [31:0] _T_836; // @[Compute.scala 245:31]
  wire [31:0] _T_837; // @[Compute.scala 245:72]
  wire [31:0] _T_838; // @[Compute.scala 246:31]
  wire [31:0] _T_839; // @[Compute.scala 246:72]
  wire [31:0] _T_840; // @[Compute.scala 245:31]
  wire [31:0] _T_841; // @[Compute.scala 245:72]
  wire [31:0] _T_842; // @[Compute.scala 246:31]
  wire [31:0] _T_843; // @[Compute.scala 246:72]
  wire [31:0] _T_844; // @[Compute.scala 245:31]
  wire [31:0] _T_845; // @[Compute.scala 245:72]
  wire [31:0] _T_846; // @[Compute.scala 246:31]
  wire [31:0] _T_847; // @[Compute.scala 246:72]
  wire [31:0] _T_848; // @[Compute.scala 245:31]
  wire [31:0] _T_849; // @[Compute.scala 245:72]
  wire [31:0] _T_850; // @[Compute.scala 246:31]
  wire [31:0] _T_851; // @[Compute.scala 246:72]
  wire [31:0] _T_852; // @[Compute.scala 245:31]
  wire [31:0] _T_853; // @[Compute.scala 245:72]
  wire [31:0] _T_854; // @[Compute.scala 246:31]
  wire [31:0] _T_855; // @[Compute.scala 246:72]
  wire [31:0] _T_856; // @[Compute.scala 245:31]
  wire [31:0] _T_857; // @[Compute.scala 245:72]
  wire [31:0] _T_858; // @[Compute.scala 246:31]
  wire [31:0] _T_859; // @[Compute.scala 246:72]
  wire [31:0] _T_860; // @[Compute.scala 245:31]
  wire [31:0] _T_861; // @[Compute.scala 245:72]
  wire [31:0] _T_862; // @[Compute.scala 246:31]
  wire [31:0] _T_863; // @[Compute.scala 246:72]
  wire [31:0] _T_864; // @[Compute.scala 245:31]
  wire [31:0] _T_865; // @[Compute.scala 245:72]
  wire [31:0] _T_866; // @[Compute.scala 246:31]
  wire [31:0] _T_867; // @[Compute.scala 246:72]
  wire [31:0] _T_868; // @[Compute.scala 245:31]
  wire [31:0] _T_869; // @[Compute.scala 245:72]
  wire [31:0] _T_870; // @[Compute.scala 246:31]
  wire [31:0] _T_871; // @[Compute.scala 246:72]
  wire [31:0] _T_872; // @[Compute.scala 245:31]
  wire [31:0] _T_873; // @[Compute.scala 245:72]
  wire [31:0] _T_874; // @[Compute.scala 246:31]
  wire [31:0] _T_875; // @[Compute.scala 246:72]
  wire [31:0] _T_876; // @[Compute.scala 245:31]
  wire [31:0] _T_877; // @[Compute.scala 245:72]
  wire [31:0] _T_878; // @[Compute.scala 246:31]
  wire [31:0] _T_879; // @[Compute.scala 246:72]
  wire [31:0] _T_880; // @[Compute.scala 245:31]
  wire [31:0] _T_881; // @[Compute.scala 245:72]
  wire [31:0] _T_882; // @[Compute.scala 246:31]
  wire [31:0] _T_883; // @[Compute.scala 246:72]
  wire [31:0] _T_884; // @[Compute.scala 245:31]
  wire [31:0] _T_885; // @[Compute.scala 245:72]
  wire [31:0] _T_886; // @[Compute.scala 246:31]
  wire [31:0] _T_887; // @[Compute.scala 246:72]
  wire [31:0] _T_888; // @[Compute.scala 245:31]
  wire [31:0] _T_889; // @[Compute.scala 245:72]
  wire [31:0] _T_890; // @[Compute.scala 246:31]
  wire [31:0] _T_891; // @[Compute.scala 246:72]
  wire [31:0] _GEN_41; // @[Compute.scala 243:30]
  wire [31:0] _GEN_42; // @[Compute.scala 243:30]
  wire [31:0] _GEN_43; // @[Compute.scala 243:30]
  wire [31:0] _GEN_44; // @[Compute.scala 243:30]
  wire [31:0] _GEN_45; // @[Compute.scala 243:30]
  wire [31:0] _GEN_46; // @[Compute.scala 243:30]
  wire [31:0] _GEN_47; // @[Compute.scala 243:30]
  wire [31:0] _GEN_48; // @[Compute.scala 243:30]
  wire [31:0] _GEN_49; // @[Compute.scala 243:30]
  wire [31:0] _GEN_50; // @[Compute.scala 243:30]
  wire [31:0] _GEN_51; // @[Compute.scala 243:30]
  wire [31:0] _GEN_52; // @[Compute.scala 243:30]
  wire [31:0] _GEN_53; // @[Compute.scala 243:30]
  wire [31:0] _GEN_54; // @[Compute.scala 243:30]
  wire [31:0] _GEN_55; // @[Compute.scala 243:30]
  wire [31:0] _GEN_56; // @[Compute.scala 243:30]
  wire [31:0] _GEN_57; // @[Compute.scala 243:30]
  wire [31:0] _GEN_58; // @[Compute.scala 243:30]
  wire [31:0] _GEN_59; // @[Compute.scala 243:30]
  wire [31:0] _GEN_60; // @[Compute.scala 243:30]
  wire [31:0] _GEN_61; // @[Compute.scala 243:30]
  wire [31:0] _GEN_62; // @[Compute.scala 243:30]
  wire [31:0] _GEN_63; // @[Compute.scala 243:30]
  wire [31:0] _GEN_64; // @[Compute.scala 243:30]
  wire [31:0] _GEN_65; // @[Compute.scala 243:30]
  wire [31:0] _GEN_66; // @[Compute.scala 243:30]
  wire [31:0] _GEN_67; // @[Compute.scala 243:30]
  wire [31:0] _GEN_68; // @[Compute.scala 243:30]
  wire [31:0] _GEN_69; // @[Compute.scala 243:30]
  wire [31:0] _GEN_70; // @[Compute.scala 243:30]
  wire [31:0] _GEN_71; // @[Compute.scala 243:30]
  wire [31:0] _GEN_72; // @[Compute.scala 243:30]
  wire [31:0] _GEN_73; // @[Compute.scala 254:20]
  wire [31:0] _GEN_74; // @[Compute.scala 254:20]
  wire [31:0] _GEN_75; // @[Compute.scala 254:20]
  wire [31:0] _GEN_76; // @[Compute.scala 254:20]
  wire [31:0] _GEN_77; // @[Compute.scala 254:20]
  wire [31:0] _GEN_78; // @[Compute.scala 254:20]
  wire [31:0] _GEN_79; // @[Compute.scala 254:20]
  wire [31:0] _GEN_80; // @[Compute.scala 254:20]
  wire [31:0] _GEN_81; // @[Compute.scala 254:20]
  wire [31:0] _GEN_82; // @[Compute.scala 254:20]
  wire [31:0] _GEN_83; // @[Compute.scala 254:20]
  wire [31:0] _GEN_84; // @[Compute.scala 254:20]
  wire [31:0] _GEN_85; // @[Compute.scala 254:20]
  wire [31:0] _GEN_86; // @[Compute.scala 254:20]
  wire [31:0] _GEN_87; // @[Compute.scala 254:20]
  wire [31:0] _GEN_88; // @[Compute.scala 254:20]
  wire [31:0] src_0_0; // @[Compute.scala 242:36]
  wire [31:0] src_1_0; // @[Compute.scala 242:36]
  wire  _T_956; // @[Compute.scala 259:34]
  wire [31:0] _T_957; // @[Compute.scala 259:24]
  wire [31:0] mix_val_0; // @[Compute.scala 242:36]
  wire [7:0] _T_958; // @[Compute.scala 261:37]
  wire [31:0] _T_959; // @[Compute.scala 262:30]
  wire [31:0] _T_960; // @[Compute.scala 262:59]
  wire [32:0] _T_961; // @[Compute.scala 262:49]
  wire [31:0] _T_962; // @[Compute.scala 262:49]
  wire [31:0] _T_963; // @[Compute.scala 262:79]
  wire [31:0] add_val_0; // @[Compute.scala 242:36]
  wire [31:0] add_res_0; // @[Compute.scala 242:36]
  wire [7:0] _T_964; // @[Compute.scala 264:37]
  wire [4:0] _T_966; // @[Compute.scala 265:60]
  wire [31:0] _T_967; // @[Compute.scala 265:49]
  wire [31:0] _T_968; // @[Compute.scala 265:84]
  wire [31:0] shr_val_0; // @[Compute.scala 242:36]
  wire [31:0] shr_res_0; // @[Compute.scala 242:36]
  wire [7:0] _T_969; // @[Compute.scala 267:37]
  wire [31:0] src_0_1; // @[Compute.scala 242:36]
  wire [31:0] src_1_1; // @[Compute.scala 242:36]
  wire  _T_970; // @[Compute.scala 259:34]
  wire [31:0] _T_971; // @[Compute.scala 259:24]
  wire [31:0] mix_val_1; // @[Compute.scala 242:36]
  wire [7:0] _T_972; // @[Compute.scala 261:37]
  wire [31:0] _T_973; // @[Compute.scala 262:30]
  wire [31:0] _T_974; // @[Compute.scala 262:59]
  wire [32:0] _T_975; // @[Compute.scala 262:49]
  wire [31:0] _T_976; // @[Compute.scala 262:49]
  wire [31:0] _T_977; // @[Compute.scala 262:79]
  wire [31:0] add_val_1; // @[Compute.scala 242:36]
  wire [31:0] add_res_1; // @[Compute.scala 242:36]
  wire [7:0] _T_978; // @[Compute.scala 264:37]
  wire [4:0] _T_980; // @[Compute.scala 265:60]
  wire [31:0] _T_981; // @[Compute.scala 265:49]
  wire [31:0] _T_982; // @[Compute.scala 265:84]
  wire [31:0] shr_val_1; // @[Compute.scala 242:36]
  wire [31:0] shr_res_1; // @[Compute.scala 242:36]
  wire [7:0] _T_983; // @[Compute.scala 267:37]
  wire [31:0] src_0_2; // @[Compute.scala 242:36]
  wire [31:0] src_1_2; // @[Compute.scala 242:36]
  wire  _T_984; // @[Compute.scala 259:34]
  wire [31:0] _T_985; // @[Compute.scala 259:24]
  wire [31:0] mix_val_2; // @[Compute.scala 242:36]
  wire [7:0] _T_986; // @[Compute.scala 261:37]
  wire [31:0] _T_987; // @[Compute.scala 262:30]
  wire [31:0] _T_988; // @[Compute.scala 262:59]
  wire [32:0] _T_989; // @[Compute.scala 262:49]
  wire [31:0] _T_990; // @[Compute.scala 262:49]
  wire [31:0] _T_991; // @[Compute.scala 262:79]
  wire [31:0] add_val_2; // @[Compute.scala 242:36]
  wire [31:0] add_res_2; // @[Compute.scala 242:36]
  wire [7:0] _T_992; // @[Compute.scala 264:37]
  wire [4:0] _T_994; // @[Compute.scala 265:60]
  wire [31:0] _T_995; // @[Compute.scala 265:49]
  wire [31:0] _T_996; // @[Compute.scala 265:84]
  wire [31:0] shr_val_2; // @[Compute.scala 242:36]
  wire [31:0] shr_res_2; // @[Compute.scala 242:36]
  wire [7:0] _T_997; // @[Compute.scala 267:37]
  wire [31:0] src_0_3; // @[Compute.scala 242:36]
  wire [31:0] src_1_3; // @[Compute.scala 242:36]
  wire  _T_998; // @[Compute.scala 259:34]
  wire [31:0] _T_999; // @[Compute.scala 259:24]
  wire [31:0] mix_val_3; // @[Compute.scala 242:36]
  wire [7:0] _T_1000; // @[Compute.scala 261:37]
  wire [31:0] _T_1001; // @[Compute.scala 262:30]
  wire [31:0] _T_1002; // @[Compute.scala 262:59]
  wire [32:0] _T_1003; // @[Compute.scala 262:49]
  wire [31:0] _T_1004; // @[Compute.scala 262:49]
  wire [31:0] _T_1005; // @[Compute.scala 262:79]
  wire [31:0] add_val_3; // @[Compute.scala 242:36]
  wire [31:0] add_res_3; // @[Compute.scala 242:36]
  wire [7:0] _T_1006; // @[Compute.scala 264:37]
  wire [4:0] _T_1008; // @[Compute.scala 265:60]
  wire [31:0] _T_1009; // @[Compute.scala 265:49]
  wire [31:0] _T_1010; // @[Compute.scala 265:84]
  wire [31:0] shr_val_3; // @[Compute.scala 242:36]
  wire [31:0] shr_res_3; // @[Compute.scala 242:36]
  wire [7:0] _T_1011; // @[Compute.scala 267:37]
  wire [31:0] src_0_4; // @[Compute.scala 242:36]
  wire [31:0] src_1_4; // @[Compute.scala 242:36]
  wire  _T_1012; // @[Compute.scala 259:34]
  wire [31:0] _T_1013; // @[Compute.scala 259:24]
  wire [31:0] mix_val_4; // @[Compute.scala 242:36]
  wire [7:0] _T_1014; // @[Compute.scala 261:37]
  wire [31:0] _T_1015; // @[Compute.scala 262:30]
  wire [31:0] _T_1016; // @[Compute.scala 262:59]
  wire [32:0] _T_1017; // @[Compute.scala 262:49]
  wire [31:0] _T_1018; // @[Compute.scala 262:49]
  wire [31:0] _T_1019; // @[Compute.scala 262:79]
  wire [31:0] add_val_4; // @[Compute.scala 242:36]
  wire [31:0] add_res_4; // @[Compute.scala 242:36]
  wire [7:0] _T_1020; // @[Compute.scala 264:37]
  wire [4:0] _T_1022; // @[Compute.scala 265:60]
  wire [31:0] _T_1023; // @[Compute.scala 265:49]
  wire [31:0] _T_1024; // @[Compute.scala 265:84]
  wire [31:0] shr_val_4; // @[Compute.scala 242:36]
  wire [31:0] shr_res_4; // @[Compute.scala 242:36]
  wire [7:0] _T_1025; // @[Compute.scala 267:37]
  wire [31:0] src_0_5; // @[Compute.scala 242:36]
  wire [31:0] src_1_5; // @[Compute.scala 242:36]
  wire  _T_1026; // @[Compute.scala 259:34]
  wire [31:0] _T_1027; // @[Compute.scala 259:24]
  wire [31:0] mix_val_5; // @[Compute.scala 242:36]
  wire [7:0] _T_1028; // @[Compute.scala 261:37]
  wire [31:0] _T_1029; // @[Compute.scala 262:30]
  wire [31:0] _T_1030; // @[Compute.scala 262:59]
  wire [32:0] _T_1031; // @[Compute.scala 262:49]
  wire [31:0] _T_1032; // @[Compute.scala 262:49]
  wire [31:0] _T_1033; // @[Compute.scala 262:79]
  wire [31:0] add_val_5; // @[Compute.scala 242:36]
  wire [31:0] add_res_5; // @[Compute.scala 242:36]
  wire [7:0] _T_1034; // @[Compute.scala 264:37]
  wire [4:0] _T_1036; // @[Compute.scala 265:60]
  wire [31:0] _T_1037; // @[Compute.scala 265:49]
  wire [31:0] _T_1038; // @[Compute.scala 265:84]
  wire [31:0] shr_val_5; // @[Compute.scala 242:36]
  wire [31:0] shr_res_5; // @[Compute.scala 242:36]
  wire [7:0] _T_1039; // @[Compute.scala 267:37]
  wire [31:0] src_0_6; // @[Compute.scala 242:36]
  wire [31:0] src_1_6; // @[Compute.scala 242:36]
  wire  _T_1040; // @[Compute.scala 259:34]
  wire [31:0] _T_1041; // @[Compute.scala 259:24]
  wire [31:0] mix_val_6; // @[Compute.scala 242:36]
  wire [7:0] _T_1042; // @[Compute.scala 261:37]
  wire [31:0] _T_1043; // @[Compute.scala 262:30]
  wire [31:0] _T_1044; // @[Compute.scala 262:59]
  wire [32:0] _T_1045; // @[Compute.scala 262:49]
  wire [31:0] _T_1046; // @[Compute.scala 262:49]
  wire [31:0] _T_1047; // @[Compute.scala 262:79]
  wire [31:0] add_val_6; // @[Compute.scala 242:36]
  wire [31:0] add_res_6; // @[Compute.scala 242:36]
  wire [7:0] _T_1048; // @[Compute.scala 264:37]
  wire [4:0] _T_1050; // @[Compute.scala 265:60]
  wire [31:0] _T_1051; // @[Compute.scala 265:49]
  wire [31:0] _T_1052; // @[Compute.scala 265:84]
  wire [31:0] shr_val_6; // @[Compute.scala 242:36]
  wire [31:0] shr_res_6; // @[Compute.scala 242:36]
  wire [7:0] _T_1053; // @[Compute.scala 267:37]
  wire [31:0] src_0_7; // @[Compute.scala 242:36]
  wire [31:0] src_1_7; // @[Compute.scala 242:36]
  wire  _T_1054; // @[Compute.scala 259:34]
  wire [31:0] _T_1055; // @[Compute.scala 259:24]
  wire [31:0] mix_val_7; // @[Compute.scala 242:36]
  wire [7:0] _T_1056; // @[Compute.scala 261:37]
  wire [31:0] _T_1057; // @[Compute.scala 262:30]
  wire [31:0] _T_1058; // @[Compute.scala 262:59]
  wire [32:0] _T_1059; // @[Compute.scala 262:49]
  wire [31:0] _T_1060; // @[Compute.scala 262:49]
  wire [31:0] _T_1061; // @[Compute.scala 262:79]
  wire [31:0] add_val_7; // @[Compute.scala 242:36]
  wire [31:0] add_res_7; // @[Compute.scala 242:36]
  wire [7:0] _T_1062; // @[Compute.scala 264:37]
  wire [4:0] _T_1064; // @[Compute.scala 265:60]
  wire [31:0] _T_1065; // @[Compute.scala 265:49]
  wire [31:0] _T_1066; // @[Compute.scala 265:84]
  wire [31:0] shr_val_7; // @[Compute.scala 242:36]
  wire [31:0] shr_res_7; // @[Compute.scala 242:36]
  wire [7:0] _T_1067; // @[Compute.scala 267:37]
  wire [31:0] src_0_8; // @[Compute.scala 242:36]
  wire [31:0] src_1_8; // @[Compute.scala 242:36]
  wire  _T_1068; // @[Compute.scala 259:34]
  wire [31:0] _T_1069; // @[Compute.scala 259:24]
  wire [31:0] mix_val_8; // @[Compute.scala 242:36]
  wire [7:0] _T_1070; // @[Compute.scala 261:37]
  wire [31:0] _T_1071; // @[Compute.scala 262:30]
  wire [31:0] _T_1072; // @[Compute.scala 262:59]
  wire [32:0] _T_1073; // @[Compute.scala 262:49]
  wire [31:0] _T_1074; // @[Compute.scala 262:49]
  wire [31:0] _T_1075; // @[Compute.scala 262:79]
  wire [31:0] add_val_8; // @[Compute.scala 242:36]
  wire [31:0] add_res_8; // @[Compute.scala 242:36]
  wire [7:0] _T_1076; // @[Compute.scala 264:37]
  wire [4:0] _T_1078; // @[Compute.scala 265:60]
  wire [31:0] _T_1079; // @[Compute.scala 265:49]
  wire [31:0] _T_1080; // @[Compute.scala 265:84]
  wire [31:0] shr_val_8; // @[Compute.scala 242:36]
  wire [31:0] shr_res_8; // @[Compute.scala 242:36]
  wire [7:0] _T_1081; // @[Compute.scala 267:37]
  wire [31:0] src_0_9; // @[Compute.scala 242:36]
  wire [31:0] src_1_9; // @[Compute.scala 242:36]
  wire  _T_1082; // @[Compute.scala 259:34]
  wire [31:0] _T_1083; // @[Compute.scala 259:24]
  wire [31:0] mix_val_9; // @[Compute.scala 242:36]
  wire [7:0] _T_1084; // @[Compute.scala 261:37]
  wire [31:0] _T_1085; // @[Compute.scala 262:30]
  wire [31:0] _T_1086; // @[Compute.scala 262:59]
  wire [32:0] _T_1087; // @[Compute.scala 262:49]
  wire [31:0] _T_1088; // @[Compute.scala 262:49]
  wire [31:0] _T_1089; // @[Compute.scala 262:79]
  wire [31:0] add_val_9; // @[Compute.scala 242:36]
  wire [31:0] add_res_9; // @[Compute.scala 242:36]
  wire [7:0] _T_1090; // @[Compute.scala 264:37]
  wire [4:0] _T_1092; // @[Compute.scala 265:60]
  wire [31:0] _T_1093; // @[Compute.scala 265:49]
  wire [31:0] _T_1094; // @[Compute.scala 265:84]
  wire [31:0] shr_val_9; // @[Compute.scala 242:36]
  wire [31:0] shr_res_9; // @[Compute.scala 242:36]
  wire [7:0] _T_1095; // @[Compute.scala 267:37]
  wire [31:0] src_0_10; // @[Compute.scala 242:36]
  wire [31:0] src_1_10; // @[Compute.scala 242:36]
  wire  _T_1096; // @[Compute.scala 259:34]
  wire [31:0] _T_1097; // @[Compute.scala 259:24]
  wire [31:0] mix_val_10; // @[Compute.scala 242:36]
  wire [7:0] _T_1098; // @[Compute.scala 261:37]
  wire [31:0] _T_1099; // @[Compute.scala 262:30]
  wire [31:0] _T_1100; // @[Compute.scala 262:59]
  wire [32:0] _T_1101; // @[Compute.scala 262:49]
  wire [31:0] _T_1102; // @[Compute.scala 262:49]
  wire [31:0] _T_1103; // @[Compute.scala 262:79]
  wire [31:0] add_val_10; // @[Compute.scala 242:36]
  wire [31:0] add_res_10; // @[Compute.scala 242:36]
  wire [7:0] _T_1104; // @[Compute.scala 264:37]
  wire [4:0] _T_1106; // @[Compute.scala 265:60]
  wire [31:0] _T_1107; // @[Compute.scala 265:49]
  wire [31:0] _T_1108; // @[Compute.scala 265:84]
  wire [31:0] shr_val_10; // @[Compute.scala 242:36]
  wire [31:0] shr_res_10; // @[Compute.scala 242:36]
  wire [7:0] _T_1109; // @[Compute.scala 267:37]
  wire [31:0] src_0_11; // @[Compute.scala 242:36]
  wire [31:0] src_1_11; // @[Compute.scala 242:36]
  wire  _T_1110; // @[Compute.scala 259:34]
  wire [31:0] _T_1111; // @[Compute.scala 259:24]
  wire [31:0] mix_val_11; // @[Compute.scala 242:36]
  wire [7:0] _T_1112; // @[Compute.scala 261:37]
  wire [31:0] _T_1113; // @[Compute.scala 262:30]
  wire [31:0] _T_1114; // @[Compute.scala 262:59]
  wire [32:0] _T_1115; // @[Compute.scala 262:49]
  wire [31:0] _T_1116; // @[Compute.scala 262:49]
  wire [31:0] _T_1117; // @[Compute.scala 262:79]
  wire [31:0] add_val_11; // @[Compute.scala 242:36]
  wire [31:0] add_res_11; // @[Compute.scala 242:36]
  wire [7:0] _T_1118; // @[Compute.scala 264:37]
  wire [4:0] _T_1120; // @[Compute.scala 265:60]
  wire [31:0] _T_1121; // @[Compute.scala 265:49]
  wire [31:0] _T_1122; // @[Compute.scala 265:84]
  wire [31:0] shr_val_11; // @[Compute.scala 242:36]
  wire [31:0] shr_res_11; // @[Compute.scala 242:36]
  wire [7:0] _T_1123; // @[Compute.scala 267:37]
  wire [31:0] src_0_12; // @[Compute.scala 242:36]
  wire [31:0] src_1_12; // @[Compute.scala 242:36]
  wire  _T_1124; // @[Compute.scala 259:34]
  wire [31:0] _T_1125; // @[Compute.scala 259:24]
  wire [31:0] mix_val_12; // @[Compute.scala 242:36]
  wire [7:0] _T_1126; // @[Compute.scala 261:37]
  wire [31:0] _T_1127; // @[Compute.scala 262:30]
  wire [31:0] _T_1128; // @[Compute.scala 262:59]
  wire [32:0] _T_1129; // @[Compute.scala 262:49]
  wire [31:0] _T_1130; // @[Compute.scala 262:49]
  wire [31:0] _T_1131; // @[Compute.scala 262:79]
  wire [31:0] add_val_12; // @[Compute.scala 242:36]
  wire [31:0] add_res_12; // @[Compute.scala 242:36]
  wire [7:0] _T_1132; // @[Compute.scala 264:37]
  wire [4:0] _T_1134; // @[Compute.scala 265:60]
  wire [31:0] _T_1135; // @[Compute.scala 265:49]
  wire [31:0] _T_1136; // @[Compute.scala 265:84]
  wire [31:0] shr_val_12; // @[Compute.scala 242:36]
  wire [31:0] shr_res_12; // @[Compute.scala 242:36]
  wire [7:0] _T_1137; // @[Compute.scala 267:37]
  wire [31:0] src_0_13; // @[Compute.scala 242:36]
  wire [31:0] src_1_13; // @[Compute.scala 242:36]
  wire  _T_1138; // @[Compute.scala 259:34]
  wire [31:0] _T_1139; // @[Compute.scala 259:24]
  wire [31:0] mix_val_13; // @[Compute.scala 242:36]
  wire [7:0] _T_1140; // @[Compute.scala 261:37]
  wire [31:0] _T_1141; // @[Compute.scala 262:30]
  wire [31:0] _T_1142; // @[Compute.scala 262:59]
  wire [32:0] _T_1143; // @[Compute.scala 262:49]
  wire [31:0] _T_1144; // @[Compute.scala 262:49]
  wire [31:0] _T_1145; // @[Compute.scala 262:79]
  wire [31:0] add_val_13; // @[Compute.scala 242:36]
  wire [31:0] add_res_13; // @[Compute.scala 242:36]
  wire [7:0] _T_1146; // @[Compute.scala 264:37]
  wire [4:0] _T_1148; // @[Compute.scala 265:60]
  wire [31:0] _T_1149; // @[Compute.scala 265:49]
  wire [31:0] _T_1150; // @[Compute.scala 265:84]
  wire [31:0] shr_val_13; // @[Compute.scala 242:36]
  wire [31:0] shr_res_13; // @[Compute.scala 242:36]
  wire [7:0] _T_1151; // @[Compute.scala 267:37]
  wire [31:0] src_0_14; // @[Compute.scala 242:36]
  wire [31:0] src_1_14; // @[Compute.scala 242:36]
  wire  _T_1152; // @[Compute.scala 259:34]
  wire [31:0] _T_1153; // @[Compute.scala 259:24]
  wire [31:0] mix_val_14; // @[Compute.scala 242:36]
  wire [7:0] _T_1154; // @[Compute.scala 261:37]
  wire [31:0] _T_1155; // @[Compute.scala 262:30]
  wire [31:0] _T_1156; // @[Compute.scala 262:59]
  wire [32:0] _T_1157; // @[Compute.scala 262:49]
  wire [31:0] _T_1158; // @[Compute.scala 262:49]
  wire [31:0] _T_1159; // @[Compute.scala 262:79]
  wire [31:0] add_val_14; // @[Compute.scala 242:36]
  wire [31:0] add_res_14; // @[Compute.scala 242:36]
  wire [7:0] _T_1160; // @[Compute.scala 264:37]
  wire [4:0] _T_1162; // @[Compute.scala 265:60]
  wire [31:0] _T_1163; // @[Compute.scala 265:49]
  wire [31:0] _T_1164; // @[Compute.scala 265:84]
  wire [31:0] shr_val_14; // @[Compute.scala 242:36]
  wire [31:0] shr_res_14; // @[Compute.scala 242:36]
  wire [7:0] _T_1165; // @[Compute.scala 267:37]
  wire [31:0] src_0_15; // @[Compute.scala 242:36]
  wire [31:0] src_1_15; // @[Compute.scala 242:36]
  wire  _T_1166; // @[Compute.scala 259:34]
  wire [31:0] _T_1167; // @[Compute.scala 259:24]
  wire [31:0] mix_val_15; // @[Compute.scala 242:36]
  wire [7:0] _T_1168; // @[Compute.scala 261:37]
  wire [31:0] _T_1169; // @[Compute.scala 262:30]
  wire [31:0] _T_1170; // @[Compute.scala 262:59]
  wire [32:0] _T_1171; // @[Compute.scala 262:49]
  wire [31:0] _T_1172; // @[Compute.scala 262:49]
  wire [31:0] _T_1173; // @[Compute.scala 262:79]
  wire [31:0] add_val_15; // @[Compute.scala 242:36]
  wire [31:0] add_res_15; // @[Compute.scala 242:36]
  wire [7:0] _T_1174; // @[Compute.scala 264:37]
  wire [4:0] _T_1176; // @[Compute.scala 265:60]
  wire [31:0] _T_1177; // @[Compute.scala 265:49]
  wire [31:0] _T_1178; // @[Compute.scala 265:84]
  wire [31:0] shr_val_15; // @[Compute.scala 242:36]
  wire [31:0] shr_res_15; // @[Compute.scala 242:36]
  wire [7:0] _T_1179; // @[Compute.scala 267:37]
  wire [7:0] short_cmp_res_0; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_0; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_0; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_1; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_1; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_1; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_2; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_2; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_2; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_3; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_3; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_3; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_4; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_4; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_4; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_5; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_5; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_5; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_6; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_6; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_6; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_7; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_7; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_7; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_8; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_8; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_8; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_9; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_9; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_9; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_10; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_10; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_10; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_11; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_11; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_11; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_12; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_12; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_12; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_13; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_13; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_13; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_14; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_14; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_14; // @[Compute.scala 242:36]
  wire [7:0] short_cmp_res_15; // @[Compute.scala 242:36]
  wire [7:0] short_add_res_15; // @[Compute.scala 242:36]
  wire [7:0] short_shr_res_15; // @[Compute.scala 242:36]
  wire  alu_opcode_minmax_en; // @[Compute.scala 274:48]
  wire  alu_opcode_add_en; // @[Compute.scala 275:39]
  wire [63:0] _T_1187; // @[Cat.scala 30:58]
  wire [127:0] _T_1195; // @[Cat.scala 30:58]
  wire [63:0] _T_1202; // @[Cat.scala 30:58]
  wire [127:0] _T_1210; // @[Cat.scala 30:58]
  wire [63:0] _T_1217; // @[Cat.scala 30:58]
  wire [127:0] _T_1225; // @[Cat.scala 30:58]
  wire [127:0] _T_1226; // @[Compute.scala 277:30]
  BinaryQueue g2l_queue ( // @[Compute.scala 313:25]
    .clock(g2l_queue_clock),
    .reset(g2l_queue_reset),
    .io_enq_ready(g2l_queue_io_enq_ready),
    .io_enq_valid(g2l_queue_io_enq_valid),
    .io_deq_ready(g2l_queue_io_deq_ready),
    .io_deq_valid(g2l_queue_io_deq_valid),
    .io_deq_bits(g2l_queue_io_deq_bits)
  );
  BinaryQueue g2s_queue ( // @[Compute.scala 314:25]
    .clock(g2s_queue_clock),
    .reset(g2s_queue_reset),
    .io_enq_ready(g2s_queue_io_enq_ready),
    .io_enq_valid(g2s_queue_io_enq_valid),
    .io_deq_ready(g2s_queue_io_deq_ready),
    .io_deq_valid(g2s_queue_io_deq_valid),
    .io_deq_bits(g2s_queue_io_deq_bits)
  );
  assign acc_mem__T_387_addr = dst_idx[7:0];
  assign acc_mem__T_387_data = acc_mem[acc_mem__T_387_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_390_addr = src_idx[7:0];
  assign acc_mem__T_390_data = acc_mem[acc_mem__T_390_addr]; // @[Compute.scala 33:20]
  assign acc_mem__T_370_data = {_T_372,_T_371};
  assign acc_mem__T_370_addr = acc_sram_addr[7:0];
  assign acc_mem__T_370_mask = 1'h1;
  assign acc_mem__T_370_en = _T_283 ? _T_368 : 1'h0;
  assign uop_mem_uop_addr = 10'h0;
  assign uop_mem_uop_data = uop_mem[uop_mem_uop_addr]; // @[Compute.scala 34:20]
  assign uop_mem__T_335_data = uops_data;
  assign uop_mem__T_335_addr = uop_sram_addr[9:0];
  assign uop_mem__T_335_mask = 1'h1;
  assign uop_mem__T_335_en = 1'h1;
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
  assign _GEN_279 = {{12'd0}, y_pad_0}; // @[Compute.scala 58:30]
  assign _GEN_281 = {{12'd0}, x_pad_0}; // @[Compute.scala 59:30]
  assign _T_204 = _GEN_281 + x_size; // @[Compute.scala 59:30]
  assign _T_205 = _GEN_281 + x_size; // @[Compute.scala 59:30]
  assign _GEN_282 = {{12'd0}, x_pad_1}; // @[Compute.scala 59:39]
  assign _T_206 = _T_205 + _GEN_282; // @[Compute.scala 59:39]
  assign x_size_total = _T_205 + _GEN_282; // @[Compute.scala 59:39]
  assign y_offset = x_size_total * _GEN_279; // @[Compute.scala 60:31]
  assign opcode_finish_en = opcode == 3'h3; // @[Compute.scala 63:34]
  assign _T_209 = opcode == 3'h0; // @[Compute.scala 64:32]
  assign _T_211 = opcode == 3'h1; // @[Compute.scala 64:60]
  assign opcode_load_en = _T_209 | _T_211; // @[Compute.scala 64:50]
  assign opcode_gemm_en = opcode == 3'h2; // @[Compute.scala 65:32]
  assign opcode_alu_en = opcode == 3'h4; // @[Compute.scala 66:31]
  assign memory_type_uop_en = memory_type == 2'h0; // @[Compute.scala 68:40]
  assign memory_type_acc_en = memory_type == 2'h3; // @[Compute.scala 69:40]
  assign _T_216 = opcode_load_en & memory_type_uop_en; // @[Compute.scala 73:37]
  assign uop_cntr_en = _T_216 & started; // @[Compute.scala 73:59]
  assign _T_219 = uop_cntr_val == 16'h0; // @[Compute.scala 76:45]
  assign acc_cntr_en = opcode_load_en & memory_type_acc_en; // @[Compute.scala 79:37]
  assign _T_224 = acc_cntr_val == 16'h1f; // @[Compute.scala 82:44]
  assign out_cntr_en = opcode_alu_en | opcode_gemm_en; // @[Compute.scala 85:36]
  assign _T_229 = dst_offset_in == 16'h7; // @[Compute.scala 88:44]
  assign busy = state == 4'h1; // @[Compute.scala 101:20]
  assign _T_250 = uop_cntr_wrap == 1'h0; // @[Compute.scala 104:31]
  assign _T_251 = uop_cntr_en & _T_250; // @[Compute.scala 104:28]
  assign _T_254 = acc_cntr_wrap == 1'h0; // @[Compute.scala 105:31]
  assign _T_255 = acc_cntr_en & _T_254; // @[Compute.scala 105:28]
  assign _T_258 = out_cntr_wrap == 1'h0; // @[Compute.scala 106:31]
  assign _T_259 = out_cntr_en & _T_258; // @[Compute.scala 106:28]
  assign _T_263 = _T_255 ? 1'h1 : _T_259; // @[Compute.scala 105:15]
  assign _T_264 = _T_251 ? 1'h1 : _T_263; // @[Compute.scala 104:15]
  assign _T_266 = io_uops_waitrequest == 1'h0; // @[Compute.scala 109:22]
  assign _T_267 = uops_read & _T_266; // @[Compute.scala 109:19]
  assign _T_268 = _T_267 & busy; // @[Compute.scala 109:37]
  assign _T_272 = uop_cntr_val + 16'h1; // @[Compute.scala 111:36]
  assign _T_273 = uop_cntr_val + 16'h1; // @[Compute.scala 111:36]
  assign _GEN_0 = _T_219 ? _T_273 : 16'h0; // @[Compute.scala 112:57]
  assign _GEN_1 = _T_219 ? _T_264 : 1'h0; // @[Compute.scala 112:57]
  assign _GEN_5 = _T_268 ? _GEN_1 : _T_264; // @[Compute.scala 109:46]
  assign _T_282 = io_biases_waitrequest == 1'h0; // @[Compute.scala 120:24]
  assign _T_283 = biases_read & _T_282; // @[Compute.scala 120:21]
  assign _T_284 = _T_283 & busy; // @[Compute.scala 120:39]
  assign _T_286 = acc_cntr_val < 16'h1f; // @[Compute.scala 121:24]
  assign _T_288 = acc_cntr_val + 16'h1; // @[Compute.scala 122:36]
  assign _T_289 = acc_cntr_val + 16'h1; // @[Compute.scala 122:36]
  assign _GEN_6 = _T_224 ? _T_289 : 16'h0; // @[Compute.scala 123:57]
  assign _GEN_7 = _T_224 ? _GEN_5 : 1'h0; // @[Compute.scala 123:57]
  assign _GEN_8 = _T_286 ? _T_289 : _GEN_6; // @[Compute.scala 121:48]
  assign _GEN_9 = _T_286 ? _GEN_5 : _GEN_7; // @[Compute.scala 121:48]
  assign _GEN_11 = _T_284 ? _GEN_9 : _GEN_5; // @[Compute.scala 120:48]
  assign _T_298 = io_out_mem_waitrequest == 1'h0; // @[Compute.scala 131:24]
  assign _T_299 = out_cntr_en & _T_298; // @[Compute.scala 131:21]
  assign _T_300 = _T_299 & busy; // @[Compute.scala 131:39]
  assign _T_302 = dst_offset_in < 16'h7; // @[Compute.scala 132:24]
  assign _T_304 = dst_offset_in + 16'h1; // @[Compute.scala 133:36]
  assign _T_305 = dst_offset_in + 16'h1; // @[Compute.scala 133:36]
  assign _GEN_12 = _T_229 ? _T_305 : 16'h0; // @[Compute.scala 134:57]
  assign _GEN_13 = _T_229 ? _GEN_11 : 1'h0; // @[Compute.scala 134:57]
  assign _GEN_14 = _T_302 ? _T_305 : _GEN_12; // @[Compute.scala 132:48]
  assign _GEN_15 = _T_302 ? _GEN_11 : _GEN_13; // @[Compute.scala 132:48]
  assign _GEN_17 = _T_300 ? _GEN_15 : _GEN_11; // @[Compute.scala 131:48]
  assign _T_314 = busy == 1'h0; // @[Compute.scala 143:32]
  assign _T_315 = io_gemm_queue_valid & _T_314; // @[Compute.scala 143:29]
  assign _GEN_284 = {{16'd0}, uop_cntr_val}; // @[Compute.scala 157:33]
  assign _T_327 = dram_base + _GEN_284; // @[Compute.scala 157:33]
  assign _T_328 = dram_base + _GEN_284; // @[Compute.scala 157:33]
  assign _GEN_285 = {{3'd0}, _T_328}; // @[Compute.scala 157:49]
  assign uop_dram_addr = _GEN_285 << 2'h2; // @[Compute.scala 157:49]
  assign _T_330 = sram_base + uop_cntr_val; // @[Compute.scala 158:33]
  assign uop_sram_addr = sram_base + uop_cntr_val; // @[Compute.scala 158:33]
  assign _GEN_286 = {{12'd0}, y_offset}; // @[Compute.scala 170:35]
  assign _T_336 = dram_base + _GEN_286; // @[Compute.scala 170:35]
  assign _T_337 = dram_base + _GEN_286; // @[Compute.scala 170:35]
  assign _GEN_287 = {{28'd0}, x_pad_0}; // @[Compute.scala 170:46]
  assign _T_338 = _T_337 + _GEN_287; // @[Compute.scala 170:46]
  assign _T_339 = _T_337 + _GEN_287; // @[Compute.scala 170:46]
  assign _T_341 = _T_339 * 32'h1; // @[Compute.scala 170:57]
  assign _GEN_288 = {{17'd0}, acc_cntr_val}; // @[Compute.scala 170:67]
  assign _T_342 = _T_341 + _GEN_288; // @[Compute.scala 170:67]
  assign _T_343 = _T_341 + _GEN_288; // @[Compute.scala 170:67]
  assign _GEN_289 = {{7'd0}, _T_343}; // @[Compute.scala 170:83]
  assign acc_dram_addr = _GEN_289 << 3'h4; // @[Compute.scala 170:83]
  assign _GEN_290 = {{4'd0}, sram_base}; // @[Compute.scala 171:35]
  assign _T_345 = _GEN_290 + y_offset; // @[Compute.scala 171:35]
  assign _T_346 = _GEN_290 + y_offset; // @[Compute.scala 171:35]
  assign _GEN_291 = {{16'd0}, x_pad_0}; // @[Compute.scala 171:46]
  assign _T_347 = _T_346 + _GEN_291; // @[Compute.scala 171:46]
  assign _T_348 = _T_346 + _GEN_291; // @[Compute.scala 171:46]
  assign _T_350 = _T_348 * 20'h1; // @[Compute.scala 171:57]
  assign _GEN_292 = {{5'd0}, acc_cntr_val}; // @[Compute.scala 171:67]
  assign _T_351 = _T_350 + _GEN_292; // @[Compute.scala 171:67]
  assign _T_352 = _T_350 + _GEN_292; // @[Compute.scala 171:67]
  assign _T_354 = _T_352 >> 2'h2; // @[Compute.scala 171:83]
  assign _T_356 = _T_354 - 21'h1; // @[Compute.scala 171:91]
  assign _T_357 = $unsigned(_T_356); // @[Compute.scala 171:91]
  assign acc_sram_addr = _T_357[20:0]; // @[Compute.scala 171:91]
  assign _GEN_2 = acc_cntr_val % 16'h4; // @[Compute.scala 179:30]
  assign _T_362 = _GEN_2[2:0]; // @[Compute.scala 179:30]
  assign _GEN_21 = 3'h0 == _T_362 ? io_biases_readdata : biases_data_0; // @[Compute.scala 179:67]
  assign _GEN_22 = 3'h1 == _T_362 ? io_biases_readdata : biases_data_1; // @[Compute.scala 179:67]
  assign _GEN_23 = 3'h2 == _T_362 ? io_biases_readdata : biases_data_2; // @[Compute.scala 179:67]
  assign _GEN_24 = 3'h3 == _T_362 ? io_biases_readdata : biases_data_3; // @[Compute.scala 179:67]
  assign _T_368 = _T_362 == 3'h0; // @[Compute.scala 180:64]
  assign _T_371 = {biases_data_1,biases_data_0}; // @[Cat.scala 30:58]
  assign _T_372 = {biases_data_3,biases_data_2}; // @[Cat.scala 30:58]
  assign alu_opcode = insn[109:108]; // @[Compute.scala 190:24]
  assign use_imm = insn[110]; // @[Compute.scala 191:21]
  assign imm_raw = insn[126:111]; // @[Compute.scala 192:21]
  assign _T_374 = $signed(imm_raw); // @[Compute.scala 193:25]
  assign _T_376 = $signed(_T_374) < $signed(16'sh0); // @[Compute.scala 193:32]
  assign _T_378 = {16'hffff,imm_raw}; // @[Cat.scala 30:58]
  assign _T_380 = {1'h0,imm_raw}; // @[Cat.scala 30:58]
  assign _T_381 = _T_376 ? _T_378 : {{15'd0}, _T_380}; // @[Compute.scala 193:16]
  assign imm = $signed(_T_381); // @[Compute.scala 193:89]
  assign _T_382 = uop_mem_uop_data[10:0]; // @[Compute.scala 201:20]
  assign _GEN_293 = {{5'd0}, _T_382}; // @[Compute.scala 201:47]
  assign _T_383 = _GEN_293 + dst_offset_in; // @[Compute.scala 201:47]
  assign dst_idx = _GEN_293 + dst_offset_in; // @[Compute.scala 201:47]
  assign _T_384 = uop_mem_uop_data[21:11]; // @[Compute.scala 202:20]
  assign _GEN_294 = {{5'd0}, _T_384}; // @[Compute.scala 202:47]
  assign _T_385 = _GEN_294 + dst_offset_in; // @[Compute.scala 202:47]
  assign src_idx = _GEN_294 + dst_offset_in; // @[Compute.scala 202:47]
  assign _GEN_295 = {{7'd0}, dst_idx}; // @[Compute.scala 219:39]
  assign alu_opcode_min_en = alu_opcode == 2'h0; // @[Compute.scala 222:38]
  assign alu_opcode_max_en = alu_opcode == 2'h1; // @[Compute.scala 223:38]
  assign _T_827 = insn_valid & out_cntr_en; // @[Compute.scala 242:20]
  assign _T_828 = src_vector[31:0]; // @[Compute.scala 245:31]
  assign _T_829 = $signed(_T_828); // @[Compute.scala 245:72]
  assign _T_830 = dst_vector[31:0]; // @[Compute.scala 246:31]
  assign _T_831 = $signed(_T_830); // @[Compute.scala 246:72]
  assign _T_832 = src_vector[63:32]; // @[Compute.scala 245:31]
  assign _T_833 = $signed(_T_832); // @[Compute.scala 245:72]
  assign _T_834 = dst_vector[63:32]; // @[Compute.scala 246:31]
  assign _T_835 = $signed(_T_834); // @[Compute.scala 246:72]
  assign _T_836 = src_vector[95:64]; // @[Compute.scala 245:31]
  assign _T_837 = $signed(_T_836); // @[Compute.scala 245:72]
  assign _T_838 = dst_vector[95:64]; // @[Compute.scala 246:31]
  assign _T_839 = $signed(_T_838); // @[Compute.scala 246:72]
  assign _T_840 = src_vector[127:96]; // @[Compute.scala 245:31]
  assign _T_841 = $signed(_T_840); // @[Compute.scala 245:72]
  assign _T_842 = dst_vector[127:96]; // @[Compute.scala 246:31]
  assign _T_843 = $signed(_T_842); // @[Compute.scala 246:72]
  assign _T_844 = src_vector[159:128]; // @[Compute.scala 245:31]
  assign _T_845 = $signed(_T_844); // @[Compute.scala 245:72]
  assign _T_846 = dst_vector[159:128]; // @[Compute.scala 246:31]
  assign _T_847 = $signed(_T_846); // @[Compute.scala 246:72]
  assign _T_848 = src_vector[191:160]; // @[Compute.scala 245:31]
  assign _T_849 = $signed(_T_848); // @[Compute.scala 245:72]
  assign _T_850 = dst_vector[191:160]; // @[Compute.scala 246:31]
  assign _T_851 = $signed(_T_850); // @[Compute.scala 246:72]
  assign _T_852 = src_vector[223:192]; // @[Compute.scala 245:31]
  assign _T_853 = $signed(_T_852); // @[Compute.scala 245:72]
  assign _T_854 = dst_vector[223:192]; // @[Compute.scala 246:31]
  assign _T_855 = $signed(_T_854); // @[Compute.scala 246:72]
  assign _T_856 = src_vector[255:224]; // @[Compute.scala 245:31]
  assign _T_857 = $signed(_T_856); // @[Compute.scala 245:72]
  assign _T_858 = dst_vector[255:224]; // @[Compute.scala 246:31]
  assign _T_859 = $signed(_T_858); // @[Compute.scala 246:72]
  assign _T_860 = src_vector[287:256]; // @[Compute.scala 245:31]
  assign _T_861 = $signed(_T_860); // @[Compute.scala 245:72]
  assign _T_862 = dst_vector[287:256]; // @[Compute.scala 246:31]
  assign _T_863 = $signed(_T_862); // @[Compute.scala 246:72]
  assign _T_864 = src_vector[319:288]; // @[Compute.scala 245:31]
  assign _T_865 = $signed(_T_864); // @[Compute.scala 245:72]
  assign _T_866 = dst_vector[319:288]; // @[Compute.scala 246:31]
  assign _T_867 = $signed(_T_866); // @[Compute.scala 246:72]
  assign _T_868 = src_vector[351:320]; // @[Compute.scala 245:31]
  assign _T_869 = $signed(_T_868); // @[Compute.scala 245:72]
  assign _T_870 = dst_vector[351:320]; // @[Compute.scala 246:31]
  assign _T_871 = $signed(_T_870); // @[Compute.scala 246:72]
  assign _T_872 = src_vector[383:352]; // @[Compute.scala 245:31]
  assign _T_873 = $signed(_T_872); // @[Compute.scala 245:72]
  assign _T_874 = dst_vector[383:352]; // @[Compute.scala 246:31]
  assign _T_875 = $signed(_T_874); // @[Compute.scala 246:72]
  assign _T_876 = src_vector[415:384]; // @[Compute.scala 245:31]
  assign _T_877 = $signed(_T_876); // @[Compute.scala 245:72]
  assign _T_878 = dst_vector[415:384]; // @[Compute.scala 246:31]
  assign _T_879 = $signed(_T_878); // @[Compute.scala 246:72]
  assign _T_880 = src_vector[447:416]; // @[Compute.scala 245:31]
  assign _T_881 = $signed(_T_880); // @[Compute.scala 245:72]
  assign _T_882 = dst_vector[447:416]; // @[Compute.scala 246:31]
  assign _T_883 = $signed(_T_882); // @[Compute.scala 246:72]
  assign _T_884 = src_vector[479:448]; // @[Compute.scala 245:31]
  assign _T_885 = $signed(_T_884); // @[Compute.scala 245:72]
  assign _T_886 = dst_vector[479:448]; // @[Compute.scala 246:31]
  assign _T_887 = $signed(_T_886); // @[Compute.scala 246:72]
  assign _T_888 = src_vector[511:480]; // @[Compute.scala 245:31]
  assign _T_889 = $signed(_T_888); // @[Compute.scala 245:72]
  assign _T_890 = dst_vector[511:480]; // @[Compute.scala 246:31]
  assign _T_891 = $signed(_T_890); // @[Compute.scala 246:72]
  assign _GEN_41 = alu_opcode_max_en ? $signed(_T_829) : $signed(_T_831); // @[Compute.scala 243:30]
  assign _GEN_42 = alu_opcode_max_en ? $signed(_T_831) : $signed(_T_829); // @[Compute.scala 243:30]
  assign _GEN_43 = alu_opcode_max_en ? $signed(_T_833) : $signed(_T_835); // @[Compute.scala 243:30]
  assign _GEN_44 = alu_opcode_max_en ? $signed(_T_835) : $signed(_T_833); // @[Compute.scala 243:30]
  assign _GEN_45 = alu_opcode_max_en ? $signed(_T_837) : $signed(_T_839); // @[Compute.scala 243:30]
  assign _GEN_46 = alu_opcode_max_en ? $signed(_T_839) : $signed(_T_837); // @[Compute.scala 243:30]
  assign _GEN_47 = alu_opcode_max_en ? $signed(_T_841) : $signed(_T_843); // @[Compute.scala 243:30]
  assign _GEN_48 = alu_opcode_max_en ? $signed(_T_843) : $signed(_T_841); // @[Compute.scala 243:30]
  assign _GEN_49 = alu_opcode_max_en ? $signed(_T_845) : $signed(_T_847); // @[Compute.scala 243:30]
  assign _GEN_50 = alu_opcode_max_en ? $signed(_T_847) : $signed(_T_845); // @[Compute.scala 243:30]
  assign _GEN_51 = alu_opcode_max_en ? $signed(_T_849) : $signed(_T_851); // @[Compute.scala 243:30]
  assign _GEN_52 = alu_opcode_max_en ? $signed(_T_851) : $signed(_T_849); // @[Compute.scala 243:30]
  assign _GEN_53 = alu_opcode_max_en ? $signed(_T_853) : $signed(_T_855); // @[Compute.scala 243:30]
  assign _GEN_54 = alu_opcode_max_en ? $signed(_T_855) : $signed(_T_853); // @[Compute.scala 243:30]
  assign _GEN_55 = alu_opcode_max_en ? $signed(_T_857) : $signed(_T_859); // @[Compute.scala 243:30]
  assign _GEN_56 = alu_opcode_max_en ? $signed(_T_859) : $signed(_T_857); // @[Compute.scala 243:30]
  assign _GEN_57 = alu_opcode_max_en ? $signed(_T_861) : $signed(_T_863); // @[Compute.scala 243:30]
  assign _GEN_58 = alu_opcode_max_en ? $signed(_T_863) : $signed(_T_861); // @[Compute.scala 243:30]
  assign _GEN_59 = alu_opcode_max_en ? $signed(_T_865) : $signed(_T_867); // @[Compute.scala 243:30]
  assign _GEN_60 = alu_opcode_max_en ? $signed(_T_867) : $signed(_T_865); // @[Compute.scala 243:30]
  assign _GEN_61 = alu_opcode_max_en ? $signed(_T_869) : $signed(_T_871); // @[Compute.scala 243:30]
  assign _GEN_62 = alu_opcode_max_en ? $signed(_T_871) : $signed(_T_869); // @[Compute.scala 243:30]
  assign _GEN_63 = alu_opcode_max_en ? $signed(_T_873) : $signed(_T_875); // @[Compute.scala 243:30]
  assign _GEN_64 = alu_opcode_max_en ? $signed(_T_875) : $signed(_T_873); // @[Compute.scala 243:30]
  assign _GEN_65 = alu_opcode_max_en ? $signed(_T_877) : $signed(_T_879); // @[Compute.scala 243:30]
  assign _GEN_66 = alu_opcode_max_en ? $signed(_T_879) : $signed(_T_877); // @[Compute.scala 243:30]
  assign _GEN_67 = alu_opcode_max_en ? $signed(_T_881) : $signed(_T_883); // @[Compute.scala 243:30]
  assign _GEN_68 = alu_opcode_max_en ? $signed(_T_883) : $signed(_T_881); // @[Compute.scala 243:30]
  assign _GEN_69 = alu_opcode_max_en ? $signed(_T_885) : $signed(_T_887); // @[Compute.scala 243:30]
  assign _GEN_70 = alu_opcode_max_en ? $signed(_T_887) : $signed(_T_885); // @[Compute.scala 243:30]
  assign _GEN_71 = alu_opcode_max_en ? $signed(_T_889) : $signed(_T_891); // @[Compute.scala 243:30]
  assign _GEN_72 = alu_opcode_max_en ? $signed(_T_891) : $signed(_T_889); // @[Compute.scala 243:30]
  assign _GEN_73 = use_imm ? $signed(imm) : $signed(_GEN_42); // @[Compute.scala 254:20]
  assign _GEN_74 = use_imm ? $signed(imm) : $signed(_GEN_44); // @[Compute.scala 254:20]
  assign _GEN_75 = use_imm ? $signed(imm) : $signed(_GEN_46); // @[Compute.scala 254:20]
  assign _GEN_76 = use_imm ? $signed(imm) : $signed(_GEN_48); // @[Compute.scala 254:20]
  assign _GEN_77 = use_imm ? $signed(imm) : $signed(_GEN_50); // @[Compute.scala 254:20]
  assign _GEN_78 = use_imm ? $signed(imm) : $signed(_GEN_52); // @[Compute.scala 254:20]
  assign _GEN_79 = use_imm ? $signed(imm) : $signed(_GEN_54); // @[Compute.scala 254:20]
  assign _GEN_80 = use_imm ? $signed(imm) : $signed(_GEN_56); // @[Compute.scala 254:20]
  assign _GEN_81 = use_imm ? $signed(imm) : $signed(_GEN_58); // @[Compute.scala 254:20]
  assign _GEN_82 = use_imm ? $signed(imm) : $signed(_GEN_60); // @[Compute.scala 254:20]
  assign _GEN_83 = use_imm ? $signed(imm) : $signed(_GEN_62); // @[Compute.scala 254:20]
  assign _GEN_84 = use_imm ? $signed(imm) : $signed(_GEN_64); // @[Compute.scala 254:20]
  assign _GEN_85 = use_imm ? $signed(imm) : $signed(_GEN_66); // @[Compute.scala 254:20]
  assign _GEN_86 = use_imm ? $signed(imm) : $signed(_GEN_68); // @[Compute.scala 254:20]
  assign _GEN_87 = use_imm ? $signed(imm) : $signed(_GEN_70); // @[Compute.scala 254:20]
  assign _GEN_88 = use_imm ? $signed(imm) : $signed(_GEN_72); // @[Compute.scala 254:20]
  assign src_0_0 = _T_827 ? $signed(_GEN_41) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_0 = _T_827 ? $signed(_GEN_73) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_956 = $signed(src_0_0) < $signed(src_1_0); // @[Compute.scala 259:34]
  assign _T_957 = _T_956 ? $signed(src_0_0) : $signed(src_1_0); // @[Compute.scala 259:24]
  assign mix_val_0 = _T_827 ? $signed(_T_957) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_958 = mix_val_0[7:0]; // @[Compute.scala 261:37]
  assign _T_959 = $unsigned(src_0_0); // @[Compute.scala 262:30]
  assign _T_960 = $unsigned(src_1_0); // @[Compute.scala 262:59]
  assign _T_961 = _T_959 + _T_960; // @[Compute.scala 262:49]
  assign _T_962 = _T_959 + _T_960; // @[Compute.scala 262:49]
  assign _T_963 = $signed(_T_962); // @[Compute.scala 262:79]
  assign add_val_0 = _T_827 ? $signed(_T_963) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_0 = _T_827 ? $signed(add_val_0) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_964 = add_res_0[7:0]; // @[Compute.scala 264:37]
  assign _T_966 = src_1_0[4:0]; // @[Compute.scala 265:60]
  assign _T_967 = _T_959 >> _T_966; // @[Compute.scala 265:49]
  assign _T_968 = $signed(_T_967); // @[Compute.scala 265:84]
  assign shr_val_0 = _T_827 ? $signed(_T_968) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_0 = _T_827 ? $signed(shr_val_0) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_969 = shr_res_0[7:0]; // @[Compute.scala 267:37]
  assign src_0_1 = _T_827 ? $signed(_GEN_43) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_1 = _T_827 ? $signed(_GEN_74) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_970 = $signed(src_0_1) < $signed(src_1_1); // @[Compute.scala 259:34]
  assign _T_971 = _T_970 ? $signed(src_0_1) : $signed(src_1_1); // @[Compute.scala 259:24]
  assign mix_val_1 = _T_827 ? $signed(_T_971) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_972 = mix_val_1[7:0]; // @[Compute.scala 261:37]
  assign _T_973 = $unsigned(src_0_1); // @[Compute.scala 262:30]
  assign _T_974 = $unsigned(src_1_1); // @[Compute.scala 262:59]
  assign _T_975 = _T_973 + _T_974; // @[Compute.scala 262:49]
  assign _T_976 = _T_973 + _T_974; // @[Compute.scala 262:49]
  assign _T_977 = $signed(_T_976); // @[Compute.scala 262:79]
  assign add_val_1 = _T_827 ? $signed(_T_977) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_1 = _T_827 ? $signed(add_val_1) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_978 = add_res_1[7:0]; // @[Compute.scala 264:37]
  assign _T_980 = src_1_1[4:0]; // @[Compute.scala 265:60]
  assign _T_981 = _T_973 >> _T_980; // @[Compute.scala 265:49]
  assign _T_982 = $signed(_T_981); // @[Compute.scala 265:84]
  assign shr_val_1 = _T_827 ? $signed(_T_982) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_1 = _T_827 ? $signed(shr_val_1) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_983 = shr_res_1[7:0]; // @[Compute.scala 267:37]
  assign src_0_2 = _T_827 ? $signed(_GEN_45) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_2 = _T_827 ? $signed(_GEN_75) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_984 = $signed(src_0_2) < $signed(src_1_2); // @[Compute.scala 259:34]
  assign _T_985 = _T_984 ? $signed(src_0_2) : $signed(src_1_2); // @[Compute.scala 259:24]
  assign mix_val_2 = _T_827 ? $signed(_T_985) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_986 = mix_val_2[7:0]; // @[Compute.scala 261:37]
  assign _T_987 = $unsigned(src_0_2); // @[Compute.scala 262:30]
  assign _T_988 = $unsigned(src_1_2); // @[Compute.scala 262:59]
  assign _T_989 = _T_987 + _T_988; // @[Compute.scala 262:49]
  assign _T_990 = _T_987 + _T_988; // @[Compute.scala 262:49]
  assign _T_991 = $signed(_T_990); // @[Compute.scala 262:79]
  assign add_val_2 = _T_827 ? $signed(_T_991) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_2 = _T_827 ? $signed(add_val_2) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_992 = add_res_2[7:0]; // @[Compute.scala 264:37]
  assign _T_994 = src_1_2[4:0]; // @[Compute.scala 265:60]
  assign _T_995 = _T_987 >> _T_994; // @[Compute.scala 265:49]
  assign _T_996 = $signed(_T_995); // @[Compute.scala 265:84]
  assign shr_val_2 = _T_827 ? $signed(_T_996) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_2 = _T_827 ? $signed(shr_val_2) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_997 = shr_res_2[7:0]; // @[Compute.scala 267:37]
  assign src_0_3 = _T_827 ? $signed(_GEN_47) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_3 = _T_827 ? $signed(_GEN_76) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_998 = $signed(src_0_3) < $signed(src_1_3); // @[Compute.scala 259:34]
  assign _T_999 = _T_998 ? $signed(src_0_3) : $signed(src_1_3); // @[Compute.scala 259:24]
  assign mix_val_3 = _T_827 ? $signed(_T_999) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1000 = mix_val_3[7:0]; // @[Compute.scala 261:37]
  assign _T_1001 = $unsigned(src_0_3); // @[Compute.scala 262:30]
  assign _T_1002 = $unsigned(src_1_3); // @[Compute.scala 262:59]
  assign _T_1003 = _T_1001 + _T_1002; // @[Compute.scala 262:49]
  assign _T_1004 = _T_1001 + _T_1002; // @[Compute.scala 262:49]
  assign _T_1005 = $signed(_T_1004); // @[Compute.scala 262:79]
  assign add_val_3 = _T_827 ? $signed(_T_1005) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_3 = _T_827 ? $signed(add_val_3) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1006 = add_res_3[7:0]; // @[Compute.scala 264:37]
  assign _T_1008 = src_1_3[4:0]; // @[Compute.scala 265:60]
  assign _T_1009 = _T_1001 >> _T_1008; // @[Compute.scala 265:49]
  assign _T_1010 = $signed(_T_1009); // @[Compute.scala 265:84]
  assign shr_val_3 = _T_827 ? $signed(_T_1010) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_3 = _T_827 ? $signed(shr_val_3) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1011 = shr_res_3[7:0]; // @[Compute.scala 267:37]
  assign src_0_4 = _T_827 ? $signed(_GEN_49) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_4 = _T_827 ? $signed(_GEN_77) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1012 = $signed(src_0_4) < $signed(src_1_4); // @[Compute.scala 259:34]
  assign _T_1013 = _T_1012 ? $signed(src_0_4) : $signed(src_1_4); // @[Compute.scala 259:24]
  assign mix_val_4 = _T_827 ? $signed(_T_1013) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1014 = mix_val_4[7:0]; // @[Compute.scala 261:37]
  assign _T_1015 = $unsigned(src_0_4); // @[Compute.scala 262:30]
  assign _T_1016 = $unsigned(src_1_4); // @[Compute.scala 262:59]
  assign _T_1017 = _T_1015 + _T_1016; // @[Compute.scala 262:49]
  assign _T_1018 = _T_1015 + _T_1016; // @[Compute.scala 262:49]
  assign _T_1019 = $signed(_T_1018); // @[Compute.scala 262:79]
  assign add_val_4 = _T_827 ? $signed(_T_1019) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_4 = _T_827 ? $signed(add_val_4) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1020 = add_res_4[7:0]; // @[Compute.scala 264:37]
  assign _T_1022 = src_1_4[4:0]; // @[Compute.scala 265:60]
  assign _T_1023 = _T_1015 >> _T_1022; // @[Compute.scala 265:49]
  assign _T_1024 = $signed(_T_1023); // @[Compute.scala 265:84]
  assign shr_val_4 = _T_827 ? $signed(_T_1024) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_4 = _T_827 ? $signed(shr_val_4) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1025 = shr_res_4[7:0]; // @[Compute.scala 267:37]
  assign src_0_5 = _T_827 ? $signed(_GEN_51) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_5 = _T_827 ? $signed(_GEN_78) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1026 = $signed(src_0_5) < $signed(src_1_5); // @[Compute.scala 259:34]
  assign _T_1027 = _T_1026 ? $signed(src_0_5) : $signed(src_1_5); // @[Compute.scala 259:24]
  assign mix_val_5 = _T_827 ? $signed(_T_1027) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1028 = mix_val_5[7:0]; // @[Compute.scala 261:37]
  assign _T_1029 = $unsigned(src_0_5); // @[Compute.scala 262:30]
  assign _T_1030 = $unsigned(src_1_5); // @[Compute.scala 262:59]
  assign _T_1031 = _T_1029 + _T_1030; // @[Compute.scala 262:49]
  assign _T_1032 = _T_1029 + _T_1030; // @[Compute.scala 262:49]
  assign _T_1033 = $signed(_T_1032); // @[Compute.scala 262:79]
  assign add_val_5 = _T_827 ? $signed(_T_1033) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_5 = _T_827 ? $signed(add_val_5) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1034 = add_res_5[7:0]; // @[Compute.scala 264:37]
  assign _T_1036 = src_1_5[4:0]; // @[Compute.scala 265:60]
  assign _T_1037 = _T_1029 >> _T_1036; // @[Compute.scala 265:49]
  assign _T_1038 = $signed(_T_1037); // @[Compute.scala 265:84]
  assign shr_val_5 = _T_827 ? $signed(_T_1038) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_5 = _T_827 ? $signed(shr_val_5) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1039 = shr_res_5[7:0]; // @[Compute.scala 267:37]
  assign src_0_6 = _T_827 ? $signed(_GEN_53) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_6 = _T_827 ? $signed(_GEN_79) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1040 = $signed(src_0_6) < $signed(src_1_6); // @[Compute.scala 259:34]
  assign _T_1041 = _T_1040 ? $signed(src_0_6) : $signed(src_1_6); // @[Compute.scala 259:24]
  assign mix_val_6 = _T_827 ? $signed(_T_1041) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1042 = mix_val_6[7:0]; // @[Compute.scala 261:37]
  assign _T_1043 = $unsigned(src_0_6); // @[Compute.scala 262:30]
  assign _T_1044 = $unsigned(src_1_6); // @[Compute.scala 262:59]
  assign _T_1045 = _T_1043 + _T_1044; // @[Compute.scala 262:49]
  assign _T_1046 = _T_1043 + _T_1044; // @[Compute.scala 262:49]
  assign _T_1047 = $signed(_T_1046); // @[Compute.scala 262:79]
  assign add_val_6 = _T_827 ? $signed(_T_1047) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_6 = _T_827 ? $signed(add_val_6) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1048 = add_res_6[7:0]; // @[Compute.scala 264:37]
  assign _T_1050 = src_1_6[4:0]; // @[Compute.scala 265:60]
  assign _T_1051 = _T_1043 >> _T_1050; // @[Compute.scala 265:49]
  assign _T_1052 = $signed(_T_1051); // @[Compute.scala 265:84]
  assign shr_val_6 = _T_827 ? $signed(_T_1052) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_6 = _T_827 ? $signed(shr_val_6) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1053 = shr_res_6[7:0]; // @[Compute.scala 267:37]
  assign src_0_7 = _T_827 ? $signed(_GEN_55) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_7 = _T_827 ? $signed(_GEN_80) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1054 = $signed(src_0_7) < $signed(src_1_7); // @[Compute.scala 259:34]
  assign _T_1055 = _T_1054 ? $signed(src_0_7) : $signed(src_1_7); // @[Compute.scala 259:24]
  assign mix_val_7 = _T_827 ? $signed(_T_1055) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1056 = mix_val_7[7:0]; // @[Compute.scala 261:37]
  assign _T_1057 = $unsigned(src_0_7); // @[Compute.scala 262:30]
  assign _T_1058 = $unsigned(src_1_7); // @[Compute.scala 262:59]
  assign _T_1059 = _T_1057 + _T_1058; // @[Compute.scala 262:49]
  assign _T_1060 = _T_1057 + _T_1058; // @[Compute.scala 262:49]
  assign _T_1061 = $signed(_T_1060); // @[Compute.scala 262:79]
  assign add_val_7 = _T_827 ? $signed(_T_1061) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_7 = _T_827 ? $signed(add_val_7) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1062 = add_res_7[7:0]; // @[Compute.scala 264:37]
  assign _T_1064 = src_1_7[4:0]; // @[Compute.scala 265:60]
  assign _T_1065 = _T_1057 >> _T_1064; // @[Compute.scala 265:49]
  assign _T_1066 = $signed(_T_1065); // @[Compute.scala 265:84]
  assign shr_val_7 = _T_827 ? $signed(_T_1066) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_7 = _T_827 ? $signed(shr_val_7) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1067 = shr_res_7[7:0]; // @[Compute.scala 267:37]
  assign src_0_8 = _T_827 ? $signed(_GEN_57) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_8 = _T_827 ? $signed(_GEN_81) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1068 = $signed(src_0_8) < $signed(src_1_8); // @[Compute.scala 259:34]
  assign _T_1069 = _T_1068 ? $signed(src_0_8) : $signed(src_1_8); // @[Compute.scala 259:24]
  assign mix_val_8 = _T_827 ? $signed(_T_1069) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1070 = mix_val_8[7:0]; // @[Compute.scala 261:37]
  assign _T_1071 = $unsigned(src_0_8); // @[Compute.scala 262:30]
  assign _T_1072 = $unsigned(src_1_8); // @[Compute.scala 262:59]
  assign _T_1073 = _T_1071 + _T_1072; // @[Compute.scala 262:49]
  assign _T_1074 = _T_1071 + _T_1072; // @[Compute.scala 262:49]
  assign _T_1075 = $signed(_T_1074); // @[Compute.scala 262:79]
  assign add_val_8 = _T_827 ? $signed(_T_1075) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_8 = _T_827 ? $signed(add_val_8) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1076 = add_res_8[7:0]; // @[Compute.scala 264:37]
  assign _T_1078 = src_1_8[4:0]; // @[Compute.scala 265:60]
  assign _T_1079 = _T_1071 >> _T_1078; // @[Compute.scala 265:49]
  assign _T_1080 = $signed(_T_1079); // @[Compute.scala 265:84]
  assign shr_val_8 = _T_827 ? $signed(_T_1080) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_8 = _T_827 ? $signed(shr_val_8) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1081 = shr_res_8[7:0]; // @[Compute.scala 267:37]
  assign src_0_9 = _T_827 ? $signed(_GEN_59) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_9 = _T_827 ? $signed(_GEN_82) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1082 = $signed(src_0_9) < $signed(src_1_9); // @[Compute.scala 259:34]
  assign _T_1083 = _T_1082 ? $signed(src_0_9) : $signed(src_1_9); // @[Compute.scala 259:24]
  assign mix_val_9 = _T_827 ? $signed(_T_1083) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1084 = mix_val_9[7:0]; // @[Compute.scala 261:37]
  assign _T_1085 = $unsigned(src_0_9); // @[Compute.scala 262:30]
  assign _T_1086 = $unsigned(src_1_9); // @[Compute.scala 262:59]
  assign _T_1087 = _T_1085 + _T_1086; // @[Compute.scala 262:49]
  assign _T_1088 = _T_1085 + _T_1086; // @[Compute.scala 262:49]
  assign _T_1089 = $signed(_T_1088); // @[Compute.scala 262:79]
  assign add_val_9 = _T_827 ? $signed(_T_1089) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_9 = _T_827 ? $signed(add_val_9) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1090 = add_res_9[7:0]; // @[Compute.scala 264:37]
  assign _T_1092 = src_1_9[4:0]; // @[Compute.scala 265:60]
  assign _T_1093 = _T_1085 >> _T_1092; // @[Compute.scala 265:49]
  assign _T_1094 = $signed(_T_1093); // @[Compute.scala 265:84]
  assign shr_val_9 = _T_827 ? $signed(_T_1094) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_9 = _T_827 ? $signed(shr_val_9) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1095 = shr_res_9[7:0]; // @[Compute.scala 267:37]
  assign src_0_10 = _T_827 ? $signed(_GEN_61) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_10 = _T_827 ? $signed(_GEN_83) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1096 = $signed(src_0_10) < $signed(src_1_10); // @[Compute.scala 259:34]
  assign _T_1097 = _T_1096 ? $signed(src_0_10) : $signed(src_1_10); // @[Compute.scala 259:24]
  assign mix_val_10 = _T_827 ? $signed(_T_1097) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1098 = mix_val_10[7:0]; // @[Compute.scala 261:37]
  assign _T_1099 = $unsigned(src_0_10); // @[Compute.scala 262:30]
  assign _T_1100 = $unsigned(src_1_10); // @[Compute.scala 262:59]
  assign _T_1101 = _T_1099 + _T_1100; // @[Compute.scala 262:49]
  assign _T_1102 = _T_1099 + _T_1100; // @[Compute.scala 262:49]
  assign _T_1103 = $signed(_T_1102); // @[Compute.scala 262:79]
  assign add_val_10 = _T_827 ? $signed(_T_1103) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_10 = _T_827 ? $signed(add_val_10) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1104 = add_res_10[7:0]; // @[Compute.scala 264:37]
  assign _T_1106 = src_1_10[4:0]; // @[Compute.scala 265:60]
  assign _T_1107 = _T_1099 >> _T_1106; // @[Compute.scala 265:49]
  assign _T_1108 = $signed(_T_1107); // @[Compute.scala 265:84]
  assign shr_val_10 = _T_827 ? $signed(_T_1108) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_10 = _T_827 ? $signed(shr_val_10) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1109 = shr_res_10[7:0]; // @[Compute.scala 267:37]
  assign src_0_11 = _T_827 ? $signed(_GEN_63) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_11 = _T_827 ? $signed(_GEN_84) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1110 = $signed(src_0_11) < $signed(src_1_11); // @[Compute.scala 259:34]
  assign _T_1111 = _T_1110 ? $signed(src_0_11) : $signed(src_1_11); // @[Compute.scala 259:24]
  assign mix_val_11 = _T_827 ? $signed(_T_1111) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1112 = mix_val_11[7:0]; // @[Compute.scala 261:37]
  assign _T_1113 = $unsigned(src_0_11); // @[Compute.scala 262:30]
  assign _T_1114 = $unsigned(src_1_11); // @[Compute.scala 262:59]
  assign _T_1115 = _T_1113 + _T_1114; // @[Compute.scala 262:49]
  assign _T_1116 = _T_1113 + _T_1114; // @[Compute.scala 262:49]
  assign _T_1117 = $signed(_T_1116); // @[Compute.scala 262:79]
  assign add_val_11 = _T_827 ? $signed(_T_1117) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_11 = _T_827 ? $signed(add_val_11) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1118 = add_res_11[7:0]; // @[Compute.scala 264:37]
  assign _T_1120 = src_1_11[4:0]; // @[Compute.scala 265:60]
  assign _T_1121 = _T_1113 >> _T_1120; // @[Compute.scala 265:49]
  assign _T_1122 = $signed(_T_1121); // @[Compute.scala 265:84]
  assign shr_val_11 = _T_827 ? $signed(_T_1122) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_11 = _T_827 ? $signed(shr_val_11) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1123 = shr_res_11[7:0]; // @[Compute.scala 267:37]
  assign src_0_12 = _T_827 ? $signed(_GEN_65) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_12 = _T_827 ? $signed(_GEN_85) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1124 = $signed(src_0_12) < $signed(src_1_12); // @[Compute.scala 259:34]
  assign _T_1125 = _T_1124 ? $signed(src_0_12) : $signed(src_1_12); // @[Compute.scala 259:24]
  assign mix_val_12 = _T_827 ? $signed(_T_1125) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1126 = mix_val_12[7:0]; // @[Compute.scala 261:37]
  assign _T_1127 = $unsigned(src_0_12); // @[Compute.scala 262:30]
  assign _T_1128 = $unsigned(src_1_12); // @[Compute.scala 262:59]
  assign _T_1129 = _T_1127 + _T_1128; // @[Compute.scala 262:49]
  assign _T_1130 = _T_1127 + _T_1128; // @[Compute.scala 262:49]
  assign _T_1131 = $signed(_T_1130); // @[Compute.scala 262:79]
  assign add_val_12 = _T_827 ? $signed(_T_1131) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_12 = _T_827 ? $signed(add_val_12) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1132 = add_res_12[7:0]; // @[Compute.scala 264:37]
  assign _T_1134 = src_1_12[4:0]; // @[Compute.scala 265:60]
  assign _T_1135 = _T_1127 >> _T_1134; // @[Compute.scala 265:49]
  assign _T_1136 = $signed(_T_1135); // @[Compute.scala 265:84]
  assign shr_val_12 = _T_827 ? $signed(_T_1136) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_12 = _T_827 ? $signed(shr_val_12) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1137 = shr_res_12[7:0]; // @[Compute.scala 267:37]
  assign src_0_13 = _T_827 ? $signed(_GEN_67) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_13 = _T_827 ? $signed(_GEN_86) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1138 = $signed(src_0_13) < $signed(src_1_13); // @[Compute.scala 259:34]
  assign _T_1139 = _T_1138 ? $signed(src_0_13) : $signed(src_1_13); // @[Compute.scala 259:24]
  assign mix_val_13 = _T_827 ? $signed(_T_1139) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1140 = mix_val_13[7:0]; // @[Compute.scala 261:37]
  assign _T_1141 = $unsigned(src_0_13); // @[Compute.scala 262:30]
  assign _T_1142 = $unsigned(src_1_13); // @[Compute.scala 262:59]
  assign _T_1143 = _T_1141 + _T_1142; // @[Compute.scala 262:49]
  assign _T_1144 = _T_1141 + _T_1142; // @[Compute.scala 262:49]
  assign _T_1145 = $signed(_T_1144); // @[Compute.scala 262:79]
  assign add_val_13 = _T_827 ? $signed(_T_1145) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_13 = _T_827 ? $signed(add_val_13) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1146 = add_res_13[7:0]; // @[Compute.scala 264:37]
  assign _T_1148 = src_1_13[4:0]; // @[Compute.scala 265:60]
  assign _T_1149 = _T_1141 >> _T_1148; // @[Compute.scala 265:49]
  assign _T_1150 = $signed(_T_1149); // @[Compute.scala 265:84]
  assign shr_val_13 = _T_827 ? $signed(_T_1150) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_13 = _T_827 ? $signed(shr_val_13) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1151 = shr_res_13[7:0]; // @[Compute.scala 267:37]
  assign src_0_14 = _T_827 ? $signed(_GEN_69) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_14 = _T_827 ? $signed(_GEN_87) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1152 = $signed(src_0_14) < $signed(src_1_14); // @[Compute.scala 259:34]
  assign _T_1153 = _T_1152 ? $signed(src_0_14) : $signed(src_1_14); // @[Compute.scala 259:24]
  assign mix_val_14 = _T_827 ? $signed(_T_1153) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1154 = mix_val_14[7:0]; // @[Compute.scala 261:37]
  assign _T_1155 = $unsigned(src_0_14); // @[Compute.scala 262:30]
  assign _T_1156 = $unsigned(src_1_14); // @[Compute.scala 262:59]
  assign _T_1157 = _T_1155 + _T_1156; // @[Compute.scala 262:49]
  assign _T_1158 = _T_1155 + _T_1156; // @[Compute.scala 262:49]
  assign _T_1159 = $signed(_T_1158); // @[Compute.scala 262:79]
  assign add_val_14 = _T_827 ? $signed(_T_1159) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_14 = _T_827 ? $signed(add_val_14) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1160 = add_res_14[7:0]; // @[Compute.scala 264:37]
  assign _T_1162 = src_1_14[4:0]; // @[Compute.scala 265:60]
  assign _T_1163 = _T_1155 >> _T_1162; // @[Compute.scala 265:49]
  assign _T_1164 = $signed(_T_1163); // @[Compute.scala 265:84]
  assign shr_val_14 = _T_827 ? $signed(_T_1164) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_14 = _T_827 ? $signed(shr_val_14) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1165 = shr_res_14[7:0]; // @[Compute.scala 267:37]
  assign src_0_15 = _T_827 ? $signed(_GEN_71) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign src_1_15 = _T_827 ? $signed(_GEN_88) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1166 = $signed(src_0_15) < $signed(src_1_15); // @[Compute.scala 259:34]
  assign _T_1167 = _T_1166 ? $signed(src_0_15) : $signed(src_1_15); // @[Compute.scala 259:24]
  assign mix_val_15 = _T_827 ? $signed(_T_1167) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1168 = mix_val_15[7:0]; // @[Compute.scala 261:37]
  assign _T_1169 = $unsigned(src_0_15); // @[Compute.scala 262:30]
  assign _T_1170 = $unsigned(src_1_15); // @[Compute.scala 262:59]
  assign _T_1171 = _T_1169 + _T_1170; // @[Compute.scala 262:49]
  assign _T_1172 = _T_1169 + _T_1170; // @[Compute.scala 262:49]
  assign _T_1173 = $signed(_T_1172); // @[Compute.scala 262:79]
  assign add_val_15 = _T_827 ? $signed(_T_1173) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign add_res_15 = _T_827 ? $signed(add_val_15) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1174 = add_res_15[7:0]; // @[Compute.scala 264:37]
  assign _T_1176 = src_1_15[4:0]; // @[Compute.scala 265:60]
  assign _T_1177 = _T_1169 >> _T_1176; // @[Compute.scala 265:49]
  assign _T_1178 = $signed(_T_1177); // @[Compute.scala 265:84]
  assign shr_val_15 = _T_827 ? $signed(_T_1178) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign shr_res_15 = _T_827 ? $signed(shr_val_15) : $signed(32'sh0); // @[Compute.scala 242:36]
  assign _T_1179 = shr_res_15[7:0]; // @[Compute.scala 267:37]
  assign short_cmp_res_0 = _T_827 ? _T_958 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_0 = _T_827 ? _T_964 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_0 = _T_827 ? _T_969 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_1 = _T_827 ? _T_972 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_1 = _T_827 ? _T_978 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_1 = _T_827 ? _T_983 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_2 = _T_827 ? _T_986 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_2 = _T_827 ? _T_992 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_2 = _T_827 ? _T_997 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_3 = _T_827 ? _T_1000 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_3 = _T_827 ? _T_1006 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_3 = _T_827 ? _T_1011 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_4 = _T_827 ? _T_1014 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_4 = _T_827 ? _T_1020 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_4 = _T_827 ? _T_1025 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_5 = _T_827 ? _T_1028 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_5 = _T_827 ? _T_1034 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_5 = _T_827 ? _T_1039 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_6 = _T_827 ? _T_1042 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_6 = _T_827 ? _T_1048 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_6 = _T_827 ? _T_1053 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_7 = _T_827 ? _T_1056 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_7 = _T_827 ? _T_1062 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_7 = _T_827 ? _T_1067 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_8 = _T_827 ? _T_1070 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_8 = _T_827 ? _T_1076 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_8 = _T_827 ? _T_1081 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_9 = _T_827 ? _T_1084 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_9 = _T_827 ? _T_1090 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_9 = _T_827 ? _T_1095 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_10 = _T_827 ? _T_1098 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_10 = _T_827 ? _T_1104 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_10 = _T_827 ? _T_1109 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_11 = _T_827 ? _T_1112 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_11 = _T_827 ? _T_1118 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_11 = _T_827 ? _T_1123 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_12 = _T_827 ? _T_1126 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_12 = _T_827 ? _T_1132 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_12 = _T_827 ? _T_1137 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_13 = _T_827 ? _T_1140 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_13 = _T_827 ? _T_1146 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_13 = _T_827 ? _T_1151 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_14 = _T_827 ? _T_1154 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_14 = _T_827 ? _T_1160 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_14 = _T_827 ? _T_1165 : 8'h0; // @[Compute.scala 242:36]
  assign short_cmp_res_15 = _T_827 ? _T_1168 : 8'h0; // @[Compute.scala 242:36]
  assign short_add_res_15 = _T_827 ? _T_1174 : 8'h0; // @[Compute.scala 242:36]
  assign short_shr_res_15 = _T_827 ? _T_1179 : 8'h0; // @[Compute.scala 242:36]
  assign alu_opcode_minmax_en = alu_opcode_min_en | alu_opcode_max_en; // @[Compute.scala 274:48]
  assign alu_opcode_add_en = alu_opcode == 2'h2; // @[Compute.scala 275:39]
  assign _T_1187 = {short_cmp_res_7,short_cmp_res_6,short_cmp_res_5,short_cmp_res_4,short_cmp_res_3,short_cmp_res_2,short_cmp_res_1,short_cmp_res_0}; // @[Cat.scala 30:58]
  assign _T_1195 = {short_cmp_res_15,short_cmp_res_14,short_cmp_res_13,short_cmp_res_12,short_cmp_res_11,short_cmp_res_10,short_cmp_res_9,short_cmp_res_8,_T_1187}; // @[Cat.scala 30:58]
  assign _T_1202 = {short_add_res_7,short_add_res_6,short_add_res_5,short_add_res_4,short_add_res_3,short_add_res_2,short_add_res_1,short_add_res_0}; // @[Cat.scala 30:58]
  assign _T_1210 = {short_add_res_15,short_add_res_14,short_add_res_13,short_add_res_12,short_add_res_11,short_add_res_10,short_add_res_9,short_add_res_8,_T_1202}; // @[Cat.scala 30:58]
  assign _T_1217 = {short_shr_res_7,short_shr_res_6,short_shr_res_5,short_shr_res_4,short_shr_res_3,short_shr_res_2,short_shr_res_1,short_shr_res_0}; // @[Cat.scala 30:58]
  assign _T_1225 = {short_shr_res_15,short_shr_res_14,short_shr_res_13,short_shr_res_12,short_shr_res_11,short_shr_res_10,short_shr_res_9,short_shr_res_8,_T_1217}; // @[Cat.scala 30:58]
  assign _T_1226 = alu_opcode_add_en ? _T_1210 : _T_1225; // @[Compute.scala 277:30]
  assign io_done_waitrequest = 1'h0; // @[Compute.scala 152:23]
  assign io_done_readdata = _T_324; // @[Compute.scala 153:20]
  assign io_uops_address = uop_dram_addr[31:0]; // @[Compute.scala 161:19]
  assign io_uops_read = uops_read; // @[Compute.scala 160:16]
  assign io_uops_write = 1'h0;
  assign io_uops_writedata = 32'h0;
  assign io_biases_address = acc_dram_addr[31:0]; // @[Compute.scala 174:21]
  assign io_biases_read = biases_read; // @[Compute.scala 175:18]
  assign io_biases_write = 1'h0;
  assign io_biases_writedata = 128'h0;
  assign io_gemm_queue_ready = io_gemm_queue_valid & _T_314; // @[Compute.scala 145:25 Compute.scala 148:25]
  assign io_l2g_dep_queue_ready = 1'h0;
  assign io_s2g_dep_queue_ready = 1'h0;
  assign io_g2l_dep_queue_valid = g2l_queue_io_deq_valid; // @[Compute.scala 315:26]
  assign io_g2l_dep_queue_data = g2l_queue_io_deq_bits; // @[Compute.scala 317:26]
  assign io_g2s_dep_queue_valid = g2s_queue_io_deq_valid; // @[Compute.scala 318:26]
  assign io_g2s_dep_queue_data = g2s_queue_io_deq_bits; // @[Compute.scala 320:26]
  assign io_inp_mem_address = 15'h0;
  assign io_inp_mem_read = 1'h0;
  assign io_inp_mem_write = 1'h0;
  assign io_inp_mem_writedata = 64'h0;
  assign io_wgt_mem_address = 18'h0;
  assign io_wgt_mem_read = 1'h0;
  assign io_wgt_mem_write = 1'h0;
  assign io_wgt_mem_writedata = 64'h0;
  assign io_out_mem_address = out_mem_addr[16:0]; // @[Compute.scala 271:22]
  assign io_out_mem_read = 1'h0;
  assign io_out_mem_write = out_mem_write_en; // @[Compute.scala 273:20]
  assign io_out_mem_writedata = alu_opcode_minmax_en ? _T_1195 : _T_1226; // @[Compute.scala 276:24]
  assign g2l_queue_clock = clock;
  assign g2l_queue_reset = reset;
  assign g2l_queue_io_enq_valid = push_prev_dep & out_cntr_wrap; // @[Compute.scala 324:26]
  assign g2l_queue_io_deq_ready = io_g2l_dep_queue_ready; // @[Compute.scala 316:26]
  assign g2s_queue_clock = clock;
  assign g2s_queue_reset = reset;
  assign g2s_queue_io_enq_valid = push_next_dep & out_cntr_wrap; // @[Compute.scala 325:26]
  assign g2s_queue_io_deq_ready = io_g2s_dep_queue_ready; // @[Compute.scala 319:26]
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
  uop_cntr_wrap = _RAND_4[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{`RANDOM}};
  acc_cntr_val = _RAND_5[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  acc_cntr_wrap = _RAND_6[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {1{`RANDOM}};
  dst_offset_in = _RAND_7[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {1{`RANDOM}};
  out_cntr_wrap = _RAND_8[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {1{`RANDOM}};
  uops_read = _RAND_9[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {1{`RANDOM}};
  uops_data = _RAND_10[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_11 = {1{`RANDOM}};
  biases_read = _RAND_11[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_12 = {4{`RANDOM}};
  biases_data_0 = _RAND_12[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_13 = {4{`RANDOM}};
  biases_data_1 = _RAND_13[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_14 = {4{`RANDOM}};
  biases_data_2 = _RAND_14[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_15 = {4{`RANDOM}};
  biases_data_3 = _RAND_15[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_16 = {1{`RANDOM}};
  state = _RAND_16[3:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_17 = {1{`RANDOM}};
  _T_324 = _RAND_17[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_18 = {16{`RANDOM}};
  dst_vector = _RAND_18[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_19 = {16{`RANDOM}};
  src_vector = _RAND_19[511:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_20 = {1{`RANDOM}};
  out_mem_addr = _RAND_20[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_21 = {1{`RANDOM}};
  out_mem_write_en = _RAND_21[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if(acc_mem__T_370_en & acc_mem__T_370_mask) begin
      acc_mem[acc_mem__T_370_addr] <= acc_mem__T_370_data; // @[Compute.scala 33:20]
    end
    if(uop_mem__T_335_en & uop_mem__T_335_mask) begin
      uop_mem[uop_mem__T_335_addr] <= uop_mem__T_335_data; // @[Compute.scala 34:20]
    end
    if (_T_315) begin
      insn <= io_gemm_queue_data;
    end
    if (_T_268) begin
      if (_T_219) begin
        uop_cntr_val <= _T_273;
      end else begin
        uop_cntr_val <= 16'h0;
      end
    end
    uop_cntr_wrap <= _T_219 & uop_cntr_en;
    if (_T_284) begin
      if (_T_286) begin
        acc_cntr_val <= _T_289;
      end else begin
        if (_T_224) begin
          acc_cntr_val <= _T_289;
        end else begin
          acc_cntr_val <= 16'h0;
        end
      end
    end
    acc_cntr_wrap <= _T_224 & acc_cntr_en;
    if (_T_300) begin
      if (_T_302) begin
        dst_offset_in <= _T_305;
      end else begin
        if (_T_229) begin
          dst_offset_in <= _T_305;
        end else begin
          dst_offset_in <= 16'h0;
        end
      end
    end
    out_cntr_wrap <= _T_229 & out_cntr_en;
    uops_read <= _T_216 & started;
    if (_T_267) begin
      uops_data <= io_uops_readdata;
    end
    biases_read <= opcode_load_en & memory_type_acc_en;
    if (_T_283) begin
      if (3'h0 == _T_362) begin
        biases_data_0 <= io_biases_readdata;
      end
    end
    if (_T_283) begin
      if (3'h1 == _T_362) begin
        biases_data_1 <= io_biases_readdata;
      end
    end
    if (_T_283) begin
      if (3'h2 == _T_362) begin
        biases_data_2 <= io_biases_readdata;
      end
    end
    if (_T_283) begin
      if (3'h3 == _T_362) begin
        biases_data_3 <= io_biases_readdata;
      end
    end
    state <= {{3'd0}, _GEN_17};
    _T_324 <= opcode_finish_en & io_done_read;
    dst_vector <= acc_mem__T_387_data;
    src_vector <= acc_mem__T_390_data;
    out_mem_addr <= _GEN_295 << 3'h4;
    out_mem_write_en <= opcode == 3'h4;
  end
endmodule
