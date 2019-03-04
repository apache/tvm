module Queue(
  input          clock,
  input          reset,
  output         io_enq_ready,
  input          io_enq_valid,
  input  [127:0] io_enq_bits,
  input          io_deq_ready,
  output         io_deq_valid,
  output [127:0] io_deq_bits
);
  reg [127:0] _T_35 [0:7] /* synthesis ramstyle = "M20K" */; // @[Decoupled.scala 215:24]
  reg [127:0] _RAND_0;
  wire [127:0] _T_35__T_68_data; // @[Decoupled.scala 215:24]
  wire [2:0] _T_35__T_68_addr; // @[Decoupled.scala 215:24]
  wire [127:0] _T_35__T_54_data; // @[Decoupled.scala 215:24]
  wire [2:0] _T_35__T_54_addr; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_mask; // @[Decoupled.scala 215:24]
  wire  _T_35__T_54_en; // @[Decoupled.scala 215:24]
  reg [2:0] value; // @[Counter.scala 26:33]
  reg [31:0] _RAND_1;
  reg [2:0] value_1; // @[Counter.scala 26:33]
  reg [31:0] _RAND_2;
  reg  _T_42; // @[Decoupled.scala 218:35]
  reg [31:0] _RAND_3;
  wire  _T_43; // @[Decoupled.scala 220:41]
  wire  _T_45; // @[Decoupled.scala 221:36]
  wire  _T_46; // @[Decoupled.scala 221:33]
  wire  _T_47; // @[Decoupled.scala 222:32]
  wire  _T_48; // @[Decoupled.scala 37:37]
  wire  _T_51; // @[Decoupled.scala 37:37]
  wire [3:0] _T_57; // @[Counter.scala 35:22]
  wire [2:0] _T_58; // @[Counter.scala 35:22]
  wire [2:0] _GEN_5; // @[Decoupled.scala 226:17]
  wire [3:0] _T_61; // @[Counter.scala 35:22]
  wire [2:0] _T_62; // @[Counter.scala 35:22]
  wire [2:0] _GEN_6; // @[Decoupled.scala 230:17]
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
  assign _T_57 = value + 3'h1; // @[Counter.scala 35:22]
  assign _T_58 = value + 3'h1; // @[Counter.scala 35:22]
  assign _GEN_5 = _T_48 ? _T_58 : value; // @[Decoupled.scala 226:17]
  assign _T_61 = value_1 + 3'h1; // @[Counter.scala 35:22]
  assign _T_62 = value_1 + 3'h1; // @[Counter.scala 35:22]
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
  _RAND_0 = {4{`RANDOM}};
  `ifdef RANDOMIZE_MEM_INIT
  for (initvar = 0; initvar < 8; initvar = initvar+1)
    _T_35[initvar] = _RAND_0[127:0];
  `endif // RANDOMIZE_MEM_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  value = _RAND_1[2:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  value_1 = _RAND_2[2:0];
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
      value <= 3'h0;
    end else begin
      if (_T_48) begin
        value <= _T_58;
      end
    end
    if (reset) begin
      value_1 <= 3'h0;
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
module Store(
  input          clock,
  input          reset,
  input          io_outputs_waitrequest,
  output [31:0]  io_outputs_address,
  output         io_outputs_read,
  input  [127:0] io_outputs_readdata,
  output         io_outputs_write,
  output [127:0] io_outputs_writedata,
  output         io_store_queue_ready,
  input          io_store_queue_valid,
  input  [127:0] io_store_queue_data,
  input          io_s2g_dep_queue_ready,
  output         io_s2g_dep_queue_valid,
  output         io_s2g_dep_queue_data,
  output         io_g2s_dep_queue_ready,
  input          io_g2s_dep_queue_valid,
  input          io_g2s_dep_queue_data,
  input          io_out_mem_waitrequest,
  output [16:0]  io_out_mem_address,
  output         io_out_mem_read,
  input  [127:0] io_out_mem_readdata,
  output         io_out_mem_write,
  output [127:0] io_out_mem_writedata
);
  wire  out_queue_clock; // @[Store.scala 52:25]
  wire  out_queue_reset; // @[Store.scala 52:25]
  wire  out_queue_io_enq_ready; // @[Store.scala 52:25]
  wire  out_queue_io_enq_valid; // @[Store.scala 52:25]
  wire [127:0] out_queue_io_enq_bits; // @[Store.scala 52:25]
  wire  out_queue_io_deq_ready; // @[Store.scala 52:25]
  wire  out_queue_io_deq_valid; // @[Store.scala 52:25]
  wire [127:0] out_queue_io_deq_bits; // @[Store.scala 52:25]
  wire  started; // @[Store.scala 19:17]
  reg [127:0] insn; // @[Store.scala 21:28]
  reg [127:0] _RAND_0;
  wire  _T_87; // @[Store.scala 22:31]
  wire  deq_cntr_en; // @[Store.scala 22:40]
  wire  pop_prev_dep; // @[Store.scala 29:26]
  wire  push_prev_dep; // @[Store.scala 31:27]
  wire [15:0] sram_base; // @[Store.scala 34:25]
  wire [31:0] dram_base; // @[Store.scala 35:25]
  wire [15:0] y_size; // @[Store.scala 36:25]
  wire [15:0] x_size; // @[Store.scala 37:25]
  wire [3:0] y_pad_0; // @[Store.scala 39:25]
  wire [3:0] x_pad_0; // @[Store.scala 41:25]
  wire [3:0] x_pad_1; // @[Store.scala 42:25]
  wire [15:0] _GEN_22; // @[Store.scala 44:30]
  wire [15:0] _GEN_24; // @[Store.scala 45:30]
  wire [16:0] _T_94; // @[Store.scala 45:30]
  wire [15:0] _T_95; // @[Store.scala 45:30]
  wire [15:0] _GEN_25; // @[Store.scala 45:39]
  wire [16:0] _T_96; // @[Store.scala 45:39]
  wire [15:0] x_size_total; // @[Store.scala 45:39]
  wire [19:0] y_offset; // @[Store.scala 46:31]
  wire [19:0] _GEN_27; // @[Store.scala 48:29]
  wire [20:0] _T_97; // @[Store.scala 48:29]
  wire [19:0] _T_98; // @[Store.scala 48:29]
  wire [19:0] _GEN_28; // @[Store.scala 48:41]
  wire [20:0] _T_99; // @[Store.scala 48:41]
  wire [19:0] sram_idx; // @[Store.scala 48:41]
  reg [2:0] state; // @[Store.scala 55:18]
  reg [31:0] _RAND_1;
  wire  idle; // @[Store.scala 57:20]
  wire  dump; // @[Store.scala 58:20]
  wire  busy; // @[Store.scala 59:20]
  wire  push; // @[Store.scala 60:20]
  wire  done; // @[Store.scala 61:20]
  wire [31:0] enq_cntr_max; // @[Store.scala 64:29]
  wire  _T_102; // @[Store.scala 66:23]
  wire  enq_cntr_wait; // @[Store.scala 66:47]
  reg [15:0] enq_cntr_val; // @[Store.scala 67:25]
  reg [31:0] _RAND_2;
  wire [31:0] _GEN_29; // @[Store.scala 68:37]
  wire  enq_cntr_wrap; // @[Store.scala 68:37]
  reg [15:0] deq_cntr_val; // @[Store.scala 73:25]
  reg [31:0] _RAND_3;
  wire [31:0] _GEN_30; // @[Store.scala 74:37]
  wire  deq_cntr_wrap; // @[Store.scala 74:37]
  reg  out_mem_read; // @[Store.scala 76:29]
  reg [31:0] _RAND_4;
  reg  pop_prev_dep_ready; // @[Store.scala 79:35]
  reg [31:0] _RAND_5;
  wire  push_prev_dep_valid; // @[Store.scala 80:43]
  reg  push_prev_dep_ready; // @[Store.scala 81:36]
  reg [31:0] _RAND_6;
  wire  _T_115; // @[Store.scala 84:25]
  wire  _T_116; // @[Store.scala 84:22]
  wire  _T_118; // @[Store.scala 84:60]
  wire  _T_119; // @[Store.scala 84:57]
  wire  _T_120; // @[Store.scala 84:41]
  wire  _T_122; // @[Store.scala 85:27]
  wire  _T_123; // @[Store.scala 85:24]
  wire [2:0] _GEN_0; // @[Store.scala 85:48]
  wire [2:0] _GEN_1; // @[Store.scala 84:77]
  wire  _T_124; // @[Store.scala 91:22]
  wire  _T_125; // @[Store.scala 91:56]
  wire  _T_126; // @[Store.scala 91:40]
  wire  _T_128; // @[Store.scala 92:28]
  wire  _T_129; // @[Store.scala 92:25]
  wire [2:0] _GEN_2; // @[Store.scala 92:50]
  wire [2:0] _GEN_3; // @[Store.scala 91:75]
  wire  _T_130; // @[Store.scala 100:14]
  wire [2:0] _GEN_4; // @[Store.scala 100:38]
  wire  _T_131; // @[Store.scala 101:14]
  wire  _T_133; // @[Store.scala 106:22]
  wire  _T_134; // @[Store.scala 106:48]
  wire  _GEN_6; // @[Store.scala 106:57]
  wire  _T_137; // @[Store.scala 111:29]
  wire  _T_138; // @[Store.scala 111:55]
  wire  _GEN_7; // @[Store.scala 111:64]
  wire  _T_141; // @[Store.scala 116:25]
  wire  _T_142; // @[Store.scala 116:22]
  wire  _T_143; // @[Store.scala 116:40]
  wire  _T_144; // @[Store.scala 116:64]
  wire  _T_145; // @[Store.scala 116:48]
  wire [16:0] _T_147; // @[Store.scala 117:34]
  wire [15:0] _T_148; // @[Store.scala 117:34]
  wire [15:0] _GEN_8; // @[Store.scala 116:80]
  wire  _T_149; // @[Store.scala 121:39]
  wire  _T_150; // @[Store.scala 121:30]
  wire [15:0] _GEN_11; // @[Store.scala 121:49]
  wire  _GEN_12; // @[Store.scala 121:49]
  wire  _GEN_13; // @[Store.scala 121:49]
  wire [20:0] _T_159; // @[Store.scala 138:33]
  wire [20:0] _GEN_32; // @[Store.scala 138:43]
  wire [21:0] _T_160; // @[Store.scala 138:43]
  wire [20:0] _T_161; // @[Store.scala 138:43]
  wire [27:0] _GEN_33; // @[Store.scala 138:59]
  wire [27:0] out_sram_addr; // @[Store.scala 138:59]
  wire  _T_166; // @[Store.scala 139:49]
  wire [32:0] _T_174; // @[Store.scala 153:35]
  wire [32:0] _GEN_34; // @[Store.scala 153:45]
  wire [33:0] _T_175; // @[Store.scala 153:45]
  wire [32:0] _T_176; // @[Store.scala 153:45]
  wire [39:0] _GEN_35; // @[Store.scala 153:61]
  wire [39:0] _T_178; // @[Store.scala 153:61]
  wire  _T_179; // @[Store.scala 156:21]
  wire  _T_180; // @[Store.scala 156:47]
  wire  _T_183; // @[Store.scala 158:11]
  wire [16:0] _T_185; // @[Store.scala 159:36]
  wire [15:0] _T_186; // @[Store.scala 159:36]
  wire [15:0] _GEN_17; // @[Store.scala 158:36]
  Queue out_queue ( // @[Store.scala 52:25]
    .clock(out_queue_clock),
    .reset(out_queue_reset),
    .io_enq_ready(out_queue_io_enq_ready),
    .io_enq_valid(out_queue_io_enq_valid),
    .io_enq_bits(out_queue_io_enq_bits),
    .io_deq_ready(out_queue_io_deq_ready),
    .io_deq_valid(out_queue_io_deq_valid),
    .io_deq_bits(out_queue_io_deq_bits)
  );
  assign started = reset == 1'h0; // @[Store.scala 19:17]
  assign _T_87 = insn != 128'h0; // @[Store.scala 22:31]
  assign deq_cntr_en = _T_87 & started; // @[Store.scala 22:40]
  assign pop_prev_dep = insn[3]; // @[Store.scala 29:26]
  assign push_prev_dep = insn[5]; // @[Store.scala 31:27]
  assign sram_base = insn[24:9]; // @[Store.scala 34:25]
  assign dram_base = insn[56:25]; // @[Store.scala 35:25]
  assign y_size = insn[79:64]; // @[Store.scala 36:25]
  assign x_size = insn[95:80]; // @[Store.scala 37:25]
  assign y_pad_0 = insn[115:112]; // @[Store.scala 39:25]
  assign x_pad_0 = insn[123:120]; // @[Store.scala 41:25]
  assign x_pad_1 = insn[127:124]; // @[Store.scala 42:25]
  assign _GEN_22 = {{12'd0}, y_pad_0}; // @[Store.scala 44:30]
  assign _GEN_24 = {{12'd0}, x_pad_0}; // @[Store.scala 45:30]
  assign _T_94 = _GEN_24 + x_size; // @[Store.scala 45:30]
  assign _T_95 = _GEN_24 + x_size; // @[Store.scala 45:30]
  assign _GEN_25 = {{12'd0}, x_pad_1}; // @[Store.scala 45:39]
  assign _T_96 = _T_95 + _GEN_25; // @[Store.scala 45:39]
  assign x_size_total = _T_95 + _GEN_25; // @[Store.scala 45:39]
  assign y_offset = x_size_total * _GEN_22; // @[Store.scala 46:31]
  assign _GEN_27 = {{4'd0}, sram_base}; // @[Store.scala 48:29]
  assign _T_97 = _GEN_27 + y_offset; // @[Store.scala 48:29]
  assign _T_98 = _GEN_27 + y_offset; // @[Store.scala 48:29]
  assign _GEN_28 = {{16'd0}, x_pad_0}; // @[Store.scala 48:41]
  assign _T_99 = _T_98 + _GEN_28; // @[Store.scala 48:41]
  assign sram_idx = _T_98 + _GEN_28; // @[Store.scala 48:41]
  assign idle = state == 3'h0; // @[Store.scala 57:20]
  assign dump = state == 3'h1; // @[Store.scala 58:20]
  assign busy = state == 3'h2; // @[Store.scala 59:20]
  assign push = state == 3'h3; // @[Store.scala 60:20]
  assign done = state == 3'h4; // @[Store.scala 61:20]
  assign enq_cntr_max = x_size * y_size; // @[Store.scala 64:29]
  assign _T_102 = out_queue_io_enq_ready == 1'h0; // @[Store.scala 66:23]
  assign enq_cntr_wait = _T_102 | io_out_mem_waitrequest; // @[Store.scala 66:47]
  assign _GEN_29 = {{16'd0}, enq_cntr_val}; // @[Store.scala 68:37]
  assign enq_cntr_wrap = _GEN_29 == enq_cntr_max; // @[Store.scala 68:37]
  assign _GEN_30 = {{16'd0}, deq_cntr_val}; // @[Store.scala 74:37]
  assign deq_cntr_wrap = _GEN_30 == enq_cntr_max; // @[Store.scala 74:37]
  assign push_prev_dep_valid = push_prev_dep & push; // @[Store.scala 80:43]
  assign _T_115 = enq_cntr_wrap == 1'h0; // @[Store.scala 84:25]
  assign _T_116 = deq_cntr_en & _T_115; // @[Store.scala 84:22]
  assign _T_118 = deq_cntr_wrap == 1'h0; // @[Store.scala 84:60]
  assign _T_119 = deq_cntr_en & _T_118; // @[Store.scala 84:57]
  assign _T_120 = _T_116 & _T_119; // @[Store.scala 84:41]
  assign _T_122 = pop_prev_dep_ready == 1'h0; // @[Store.scala 85:27]
  assign _T_123 = pop_prev_dep & _T_122; // @[Store.scala 85:24]
  assign _GEN_0 = _T_123 ? 3'h1 : 3'h2; // @[Store.scala 85:48]
  assign _GEN_1 = _T_120 ? _GEN_0 : state; // @[Store.scala 84:77]
  assign _T_124 = deq_cntr_en & enq_cntr_wrap; // @[Store.scala 91:22]
  assign _T_125 = deq_cntr_en & deq_cntr_wrap; // @[Store.scala 91:56]
  assign _T_126 = _T_124 & _T_125; // @[Store.scala 91:40]
  assign _T_128 = push_prev_dep_ready == 1'h0; // @[Store.scala 92:28]
  assign _T_129 = push_prev_dep & _T_128; // @[Store.scala 92:25]
  assign _GEN_2 = _T_129 ? 3'h3 : 3'h4; // @[Store.scala 92:50]
  assign _GEN_3 = _T_126 ? _GEN_2 : _GEN_1; // @[Store.scala 91:75]
  assign _T_130 = dump & pop_prev_dep_ready; // @[Store.scala 100:14]
  assign _GEN_4 = _T_130 ? 3'h2 : _GEN_3; // @[Store.scala 100:38]
  assign _T_131 = push & push_prev_dep_ready; // @[Store.scala 101:14]
  assign _T_133 = pop_prev_dep & io_g2s_dep_queue_valid; // @[Store.scala 106:22]
  assign _T_134 = _T_133 & dump; // @[Store.scala 106:48]
  assign _GEN_6 = _T_134 ? 1'h1 : pop_prev_dep_ready; // @[Store.scala 106:57]
  assign _T_137 = push_prev_dep_valid & io_s2g_dep_queue_ready; // @[Store.scala 111:29]
  assign _T_138 = _T_137 & push; // @[Store.scala 111:55]
  assign _GEN_7 = _T_138 ? 1'h1 : push_prev_dep_ready; // @[Store.scala 111:64]
  assign _T_141 = enq_cntr_wait == 1'h0; // @[Store.scala 116:25]
  assign _T_142 = out_mem_read & _T_141; // @[Store.scala 116:22]
  assign _T_143 = _T_142 & busy; // @[Store.scala 116:40]
  assign _T_144 = _GEN_29 < enq_cntr_max; // @[Store.scala 116:64]
  assign _T_145 = _T_143 & _T_144; // @[Store.scala 116:48]
  assign _T_147 = enq_cntr_val + 16'h1; // @[Store.scala 117:34]
  assign _T_148 = enq_cntr_val + 16'h1; // @[Store.scala 117:34]
  assign _GEN_8 = _T_145 ? _T_148 : enq_cntr_val; // @[Store.scala 116:80]
  assign _T_149 = idle | done; // @[Store.scala 121:39]
  assign _T_150 = io_store_queue_valid & _T_149; // @[Store.scala 121:30]
  assign _GEN_11 = _T_150 ? 16'h0 : deq_cntr_val; // @[Store.scala 121:49]
  assign _GEN_12 = _T_150 ? 1'h0 : _GEN_6; // @[Store.scala 121:49]
  assign _GEN_13 = _T_150 ? 1'h0 : _GEN_7; // @[Store.scala 121:49]
  assign _T_159 = sram_idx * 20'h1; // @[Store.scala 138:33]
  assign _GEN_32 = {{5'd0}, enq_cntr_val}; // @[Store.scala 138:43]
  assign _T_160 = _T_159 + _GEN_32; // @[Store.scala 138:43]
  assign _T_161 = _T_159 + _GEN_32; // @[Store.scala 138:43]
  assign _GEN_33 = {{7'd0}, _T_161}; // @[Store.scala 138:59]
  assign out_sram_addr = _GEN_33 << 3'h4; // @[Store.scala 138:59]
  assign _T_166 = _T_116 & busy; // @[Store.scala 139:49]
  assign _T_174 = dram_base * 32'h1; // @[Store.scala 153:35]
  assign _GEN_34 = {{17'd0}, deq_cntr_val}; // @[Store.scala 153:45]
  assign _T_175 = _T_174 + _GEN_34; // @[Store.scala 153:45]
  assign _T_176 = _T_174 + _GEN_34; // @[Store.scala 153:45]
  assign _GEN_35 = {{7'd0}, _T_176}; // @[Store.scala 153:61]
  assign _T_178 = _GEN_35 << 3'h4; // @[Store.scala 153:61]
  assign _T_179 = deq_cntr_en & out_queue_io_deq_valid; // @[Store.scala 156:21]
  assign _T_180 = _T_179 & busy; // @[Store.scala 156:47]
  assign _T_183 = io_outputs_waitrequest == 1'h0; // @[Store.scala 158:11]
  assign _T_185 = deq_cntr_val + 16'h1; // @[Store.scala 159:36]
  assign _T_186 = deq_cntr_val + 16'h1; // @[Store.scala 159:36]
  assign _GEN_17 = _T_183 ? _T_186 : _GEN_11; // @[Store.scala 158:36]
  assign io_outputs_address = _T_178[31:0]; // @[Store.scala 153:22]
  assign io_outputs_read = 1'h0;
  assign io_outputs_write = _T_179 & busy; // @[Store.scala 157:22 Store.scala 165:22]
  assign io_outputs_writedata = out_queue_io_deq_bits; // @[Store.scala 152:24]
  assign io_store_queue_ready = _T_150 ? deq_cntr_en : 1'h0; // @[Store.scala 128:28 Store.scala 130:28 Store.scala 134:26]
  assign io_s2g_dep_queue_valid = push_prev_dep & push; // @[Store.scala 110:26]
  assign io_s2g_dep_queue_data = 1'h1; // @[Store.scala 109:25]
  assign io_g2s_dep_queue_ready = pop_prev_dep_ready & dump; // @[Store.scala 104:26]
  assign io_out_mem_address = out_sram_addr[16:0]; // @[Store.scala 141:22]
  assign io_out_mem_read = out_mem_read; // @[Store.scala 140:19]
  assign io_out_mem_write = 1'h0;
  assign io_out_mem_writedata = 128'h0;
  assign out_queue_clock = clock;
  assign out_queue_reset = reset;
  assign out_queue_io_enq_valid = _T_142 & busy; // @[Store.scala 146:28 Store.scala 148:28]
  assign out_queue_io_enq_bits = io_out_mem_readdata; // @[Store.scala 144:25]
  assign out_queue_io_deq_ready = _T_180 ? _T_183 : 1'h0; // @[Store.scala 160:30 Store.scala 162:30 Store.scala 166:28]
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
  `ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {4{`RANDOM}};
  insn = _RAND_0[127:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  state = _RAND_1[2:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  enq_cntr_val = _RAND_2[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{`RANDOM}};
  deq_cntr_val = _RAND_3[15:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{`RANDOM}};
  out_mem_read = _RAND_4[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{`RANDOM}};
  pop_prev_dep_ready = _RAND_5[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{`RANDOM}};
  push_prev_dep_ready = _RAND_6[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if (_T_150) begin
      insn <= io_store_queue_data;
    end
    if (_T_131) begin
      state <= 3'h4;
    end else begin
      if (_T_130) begin
        state <= 3'h2;
      end else begin
        if (_T_126) begin
          if (_T_129) begin
            state <= 3'h3;
          end else begin
            state <= 3'h4;
          end
        end else begin
          if (_T_120) begin
            if (_T_123) begin
              state <= 3'h1;
            end else begin
              state <= 3'h2;
            end
          end
        end
      end
    end
    if (_T_150) begin
      enq_cntr_val <= 16'h0;
    end else begin
      if (_T_145) begin
        enq_cntr_val <= _T_148;
      end
    end
    if (_T_180) begin
      if (_T_183) begin
        deq_cntr_val <= _T_186;
      end else begin
        if (_T_150) begin
          deq_cntr_val <= 16'h0;
        end
      end
    end else begin
      if (_T_150) begin
        deq_cntr_val <= 16'h0;
      end
    end
    if (reset) begin
      out_mem_read <= 1'h0;
    end else begin
      out_mem_read <= _T_166;
    end
    if (reset) begin
      pop_prev_dep_ready <= 1'h0;
    end else begin
      if (_T_150) begin
        pop_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_134) begin
          pop_prev_dep_ready <= 1'h1;
        end
      end
    end
    if (reset) begin
      push_prev_dep_ready <= 1'h0;
    end else begin
      if (_T_150) begin
        push_prev_dep_ready <= 1'h0;
      end else begin
        if (_T_138) begin
          push_prev_dep_ready <= 1'h1;
        end
      end
    end
  end
endmodule
