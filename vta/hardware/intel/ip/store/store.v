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
  wire  out_queue_clock; // @[Store.scala 50:25]
  wire  out_queue_reset; // @[Store.scala 50:25]
  wire  out_queue_io_enq_ready; // @[Store.scala 50:25]
  wire  out_queue_io_enq_valid; // @[Store.scala 50:25]
  wire [127:0] out_queue_io_enq_bits; // @[Store.scala 50:25]
  wire  out_queue_io_deq_ready; // @[Store.scala 50:25]
  wire  out_queue_io_deq_valid; // @[Store.scala 50:25]
  wire [127:0] out_queue_io_deq_bits; // @[Store.scala 50:25]
  reg [127:0] insn; // @[Store.scala 19:28]
  reg [127:0] _RAND_0;
  wire  pop_prev_dependence; // @[Store.scala 27:33]
  wire [15:0] sram_base; // @[Store.scala 32:25]
  wire [31:0] dram_base; // @[Store.scala 33:25]
  wire [15:0] y_size; // @[Store.scala 34:25]
  wire [15:0] x_size; // @[Store.scala 35:25]
  wire [3:0] y_pad_0; // @[Store.scala 37:25]
  wire [3:0] x_pad_0; // @[Store.scala 39:25]
  wire [3:0] x_pad_1; // @[Store.scala 40:25]
  wire [15:0] _GEN_18; // @[Store.scala 42:30]
  wire [15:0] _GEN_20; // @[Store.scala 43:30]
  wire [16:0] _T_91; // @[Store.scala 43:30]
  wire [15:0] _T_92; // @[Store.scala 43:30]
  wire [15:0] _GEN_21; // @[Store.scala 43:39]
  wire [16:0] _T_93; // @[Store.scala 43:39]
  wire [15:0] x_size_total; // @[Store.scala 43:39]
  wire [19:0] y_offset; // @[Store.scala 44:31]
  wire [19:0] _GEN_23; // @[Store.scala 46:29]
  wire [20:0] _T_94; // @[Store.scala 46:29]
  wire [19:0] _T_95; // @[Store.scala 46:29]
  wire [19:0] _GEN_24; // @[Store.scala 46:41]
  wire [20:0] _T_96; // @[Store.scala 46:41]
  wire [19:0] sram_idx; // @[Store.scala 46:41]
  wire [31:0] enq_cntr_max; // @[Store.scala 53:29]
  wire  _T_98; // @[Store.scala 54:23]
  wire  enq_cntr_wait; // @[Store.scala 54:47]
  reg [31:0] enq_cntr_val; // @[Store.scala 55:25]
  reg [31:0] _RAND_1;
  wire  enq_cntr_en; // @[Store.scala 57:34]
  reg [31:0] deq_cntr_val; // @[Store.scala 62:25]
  reg [31:0] _RAND_2;
  wire  _T_115; // @[Store.scala 69:34]
  wire  _T_116; // @[Store.scala 69:31]
  wire  _T_118; // @[Store.scala 69:50]
  wire  _T_121; // @[Store.scala 70:34]
  wire  _T_122; // @[Store.scala 70:31]
  wire  _T_123; // @[Store.scala 70:67]
  wire  _T_124; // @[Store.scala 70:50]
  wire  busy; // @[Store.scala 69:17]
  wire [32:0] _T_134; // @[Store.scala 75:36]
  wire [31:0] _T_135; // @[Store.scala 75:36]
  wire [31:0] _GEN_0; // @[Store.scala 74:24]
  wire  _T_138; // @[Store.scala 80:16]
  wire [31:0] _GEN_2; // @[Store.scala 80:23]
  wire [32:0] _T_145; // @[Store.scala 89:36]
  wire [31:0] _T_146; // @[Store.scala 89:36]
  wire [31:0] _GEN_6; // @[Store.scala 88:24]
  wire [31:0] _GEN_8; // @[Store.scala 94:23]
  wire  _T_154; // @[Store.scala 102:30]
  wire [20:0] _T_162; // @[Store.scala 119:35]
  wire [31:0] _GEN_25; // @[Store.scala 119:45]
  wire [32:0] _T_163; // @[Store.scala 119:45]
  wire [31:0] _T_164; // @[Store.scala 119:45]
  wire [38:0] _GEN_26; // @[Store.scala 119:61]
  wire [38:0] _T_166; // @[Store.scala 119:61]
  wire [32:0] _T_177; // @[Store.scala 130:35]
  wire [32:0] _GEN_27; // @[Store.scala 130:45]
  wire [33:0] _T_178; // @[Store.scala 130:45]
  wire [32:0] _T_179; // @[Store.scala 130:45]
  wire [39:0] _GEN_28; // @[Store.scala 130:61]
  wire [39:0] _T_181; // @[Store.scala 130:61]
  Queue out_queue ( // @[Store.scala 50:25]
    .clock(out_queue_clock),
    .reset(out_queue_reset),
    .io_enq_ready(out_queue_io_enq_ready),
    .io_enq_valid(out_queue_io_enq_valid),
    .io_enq_bits(out_queue_io_enq_bits),
    .io_deq_ready(out_queue_io_deq_ready),
    .io_deq_valid(out_queue_io_deq_valid),
    .io_deq_bits(out_queue_io_deq_bits)
  );
  assign pop_prev_dependence = insn[3]; // @[Store.scala 27:33]
  assign sram_base = insn[24:9]; // @[Store.scala 32:25]
  assign dram_base = insn[56:25]; // @[Store.scala 33:25]
  assign y_size = insn[79:64]; // @[Store.scala 34:25]
  assign x_size = insn[95:80]; // @[Store.scala 35:25]
  assign y_pad_0 = insn[115:112]; // @[Store.scala 37:25]
  assign x_pad_0 = insn[123:120]; // @[Store.scala 39:25]
  assign x_pad_1 = insn[127:124]; // @[Store.scala 40:25]
  assign _GEN_18 = {{12'd0}, y_pad_0}; // @[Store.scala 42:30]
  assign _GEN_20 = {{12'd0}, x_pad_0}; // @[Store.scala 43:30]
  assign _T_91 = _GEN_20 + x_size; // @[Store.scala 43:30]
  assign _T_92 = _GEN_20 + x_size; // @[Store.scala 43:30]
  assign _GEN_21 = {{12'd0}, x_pad_1}; // @[Store.scala 43:39]
  assign _T_93 = _T_92 + _GEN_21; // @[Store.scala 43:39]
  assign x_size_total = _T_92 + _GEN_21; // @[Store.scala 43:39]
  assign y_offset = x_size_total * _GEN_18; // @[Store.scala 44:31]
  assign _GEN_23 = {{4'd0}, sram_base}; // @[Store.scala 46:29]
  assign _T_94 = _GEN_23 + y_offset; // @[Store.scala 46:29]
  assign _T_95 = _GEN_23 + y_offset; // @[Store.scala 46:29]
  assign _GEN_24 = {{16'd0}, x_pad_0}; // @[Store.scala 46:41]
  assign _T_96 = _T_95 + _GEN_24; // @[Store.scala 46:41]
  assign sram_idx = _T_95 + _GEN_24; // @[Store.scala 46:41]
  assign enq_cntr_max = x_size * y_size; // @[Store.scala 53:29]
  assign _T_98 = out_queue_io_enq_ready == 1'h0; // @[Store.scala 54:23]
  assign enq_cntr_wait = _T_98 | io_out_mem_waitrequest; // @[Store.scala 54:47]
  assign enq_cntr_en = enq_cntr_val < enq_cntr_max; // @[Store.scala 57:34]
  assign _T_115 = enq_cntr_wait == 1'h0; // @[Store.scala 69:34]
  assign _T_116 = enq_cntr_en & _T_115; // @[Store.scala 69:31]
  assign _T_118 = _T_116 & enq_cntr_en; // @[Store.scala 69:50]
  assign _T_121 = io_outputs_waitrequest == 1'h0; // @[Store.scala 70:34]
  assign _T_122 = out_queue_io_deq_valid & _T_121; // @[Store.scala 70:31]
  assign _T_123 = deq_cntr_val < enq_cntr_max; // @[Store.scala 70:67]
  assign _T_124 = _T_122 & _T_123; // @[Store.scala 70:50]
  assign busy = _T_118 ? 1'h1 : _T_124; // @[Store.scala 69:17]
  assign _T_134 = enq_cntr_val + 32'h1; // @[Store.scala 75:36]
  assign _T_135 = enq_cntr_val + 32'h1; // @[Store.scala 75:36]
  assign _GEN_0 = enq_cntr_en ? _T_135 : enq_cntr_val; // @[Store.scala 74:24]
  assign _T_138 = busy == 1'h0; // @[Store.scala 80:16]
  assign _GEN_2 = _T_138 ? 32'h0 : enq_cntr_val; // @[Store.scala 80:23]
  assign _T_145 = deq_cntr_val + 32'h1; // @[Store.scala 89:36]
  assign _T_146 = deq_cntr_val + 32'h1; // @[Store.scala 89:36]
  assign _GEN_6 = out_queue_io_deq_valid ? _T_146 : deq_cntr_val; // @[Store.scala 88:24]
  assign _GEN_8 = _T_138 ? 32'h0 : deq_cntr_val; // @[Store.scala 94:23]
  assign _T_154 = io_store_queue_valid & _T_138; // @[Store.scala 102:30]
  assign _T_162 = sram_idx * 20'h1; // @[Store.scala 119:35]
  assign _GEN_25 = {{11'd0}, _T_162}; // @[Store.scala 119:45]
  assign _T_163 = _GEN_25 + enq_cntr_val; // @[Store.scala 119:45]
  assign _T_164 = _GEN_25 + enq_cntr_val; // @[Store.scala 119:45]
  assign _GEN_26 = {{7'd0}, _T_164}; // @[Store.scala 119:61]
  assign _T_166 = _GEN_26 << 3'h4; // @[Store.scala 119:61]
  assign _T_177 = dram_base * 32'h1; // @[Store.scala 130:35]
  assign _GEN_27 = {{1'd0}, deq_cntr_val}; // @[Store.scala 130:45]
  assign _T_178 = _T_177 + _GEN_27; // @[Store.scala 130:45]
  assign _T_179 = _T_177 + _GEN_27; // @[Store.scala 130:45]
  assign _GEN_28 = {{7'd0}, _T_179}; // @[Store.scala 130:61]
  assign _T_181 = _GEN_28 << 3'h4; // @[Store.scala 130:61]
  assign io_outputs_address = _T_181[31:0]; // @[Store.scala 130:22]
  assign io_outputs_read = 1'h0; // @[Store.scala 131:19]
  assign io_outputs_write = out_queue_io_deq_valid; // @[Store.scala 128:20]
  assign io_outputs_writedata = out_queue_io_deq_bits; // @[Store.scala 129:24]
  assign io_store_queue_ready = io_store_queue_valid & _T_138; // @[Store.scala 104:26 Store.scala 107:26]
  assign io_s2g_dep_queue_valid = 1'h0; // @[Store.scala 136:28 Store.scala 139:28]
  assign io_s2g_dep_queue_data = 1'h1; // @[Store.scala 134:25]
  assign io_g2s_dep_queue_ready = pop_prev_dependence & io_g2s_dep_queue_valid; // @[Store.scala 112:28 Store.scala 115:28]
  assign io_out_mem_address = _T_166[16:0]; // @[Store.scala 119:22]
  assign io_out_mem_read = enq_cntr_val < enq_cntr_max; // @[Store.scala 122:19]
  assign io_out_mem_write = 1'h0; // @[Store.scala 120:20]
  assign io_out_mem_writedata = 128'h0;
  assign out_queue_clock = clock;
  assign out_queue_reset = reset;
  assign out_queue_io_enq_valid = enq_cntr_en & _T_115; // @[Store.scala 123:26]
  assign out_queue_io_enq_bits = io_out_mem_readdata; // @[Store.scala 124:25]
  assign out_queue_io_deq_ready = out_queue_io_deq_valid & _T_121; // @[Store.scala 127:26]
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
  enq_cntr_val = _RAND_1[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{`RANDOM}};
  deq_cntr_val = _RAND_2[31:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if (_T_154) begin
      insn <= io_store_queue_data;
    end
    if (_T_116) begin
      if (enq_cntr_en) begin
        enq_cntr_val <= _T_135;
      end
    end else begin
      if (_T_138) begin
        enq_cntr_val <= 32'h0;
      end
    end
    if (_T_122) begin
      if (out_queue_io_deq_valid) begin
        deq_cntr_val <= _T_146;
      end
    end else begin
      if (_T_138) begin
        deq_cntr_val <= 32'h0;
      end
    end
  end
endmodule
