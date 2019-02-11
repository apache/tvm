module MemArbiter(
  input          clock,
  input          reset,
  output         io_ins_cache_waitrequest,
  input  [31:0]  io_ins_cache_address,
  input          io_ins_cache_read,
  output [127:0] io_ins_cache_readdata,
  input          io_ins_cache_write,
  input  [127:0] io_ins_cache_writedata,
  output         io_inp_cache_waitrequest,
  input  [31:0]  io_inp_cache_address,
  input          io_inp_cache_read,
  output [127:0] io_inp_cache_readdata,
  input          io_inp_cache_write,
  input  [127:0] io_inp_cache_writedata,
  output         io_wgt_cache_waitrequest,
  input  [31:0]  io_wgt_cache_address,
  input          io_wgt_cache_read,
  output [127:0] io_wgt_cache_readdata,
  input          io_wgt_cache_write,
  input  [127:0] io_wgt_cache_writedata,
  output         io_uop_cache_waitrequest,
  input  [31:0]  io_uop_cache_address,
  input          io_uop_cache_read,
  output [127:0] io_uop_cache_readdata,
  input          io_uop_cache_write,
  input  [127:0] io_uop_cache_writedata,
  output         io_acc_cache_waitrequest,
  input  [31:0]  io_acc_cache_address,
  input          io_acc_cache_read,
  output [127:0] io_acc_cache_readdata,
  input          io_acc_cache_write,
  input  [127:0] io_acc_cache_writedata,
  output         io_out_cache_waitrequest,
  input  [31:0]  io_out_cache_address,
  input          io_out_cache_read,
  output [127:0] io_out_cache_readdata,
  input          io_out_cache_write,
  input  [127:0] io_out_cache_writedata,
  input          io_axi_master_waitrequest,
  output [31:0]  io_axi_master_address,
  output         io_axi_master_read,
  input  [127:0] io_axi_master_readdata,
  output         io_axi_master_write,
  output [127:0] io_axi_master_writedata
);
  reg [2:0] state; // @[MemArbiter.scala 23:22]
  reg [31:0] _RAND_0;
  wire  ins_cache_read; // @[MemArbiter.scala 25:31]
  wire  inp_cache_read; // @[MemArbiter.scala 26:31]
  wire  wgt_cache_read; // @[MemArbiter.scala 27:31]
  wire  uop_cache_read; // @[MemArbiter.scala 28:31]
  wire  acc_cache_read; // @[MemArbiter.scala 29:31]
  wire  out_cache_write; // @[MemArbiter.scala 30:31]
  wire  out_cache_ack; // @[MemArbiter.scala 31:31]
  wire  _T_105; // @[Mux.scala 46:19]
  wire  _T_107; // @[Mux.scala 46:19]
  wire [31:0] _T_108; // @[Mux.scala 46:16]
  wire  _T_109; // @[Mux.scala 46:19]
  wire [31:0] _T_110; // @[Mux.scala 46:16]
  wire  _T_111; // @[Mux.scala 46:19]
  wire [31:0] _T_112; // @[Mux.scala 46:16]
  wire  _T_113; // @[Mux.scala 46:19]
  wire [31:0] _T_114; // @[Mux.scala 46:16]
  wire  _T_115; // @[Mux.scala 46:19]
  wire [31:0] _T_116; // @[Mux.scala 46:16]
  wire  _T_117; // @[Mux.scala 46:19]
  wire [31:0] _T_118; // @[Mux.scala 46:16]
  wire  _T_119; // @[Mux.scala 46:19]
  wire  _T_122; // @[MemArbiter.scala 46:77]
  wire [2:0] _T_128; // @[Mux.scala 46:16]
  wire [2:0] _T_130; // @[Mux.scala 46:16]
  wire [2:0] _T_132; // @[Mux.scala 46:16]
  wire [2:0] _T_134; // @[Mux.scala 46:16]
  wire [2:0] axi_master_read; // @[Mux.scala 46:16]
  wire  _T_136; // @[MemArbiter.scala 62:58]
  wire  _T_138; // @[MemArbiter.scala 62:81]
  wire  _T_139; // @[MemArbiter.scala 62:97]
  wire  _T_141; // @[MemArbiter.scala 63:58]
  wire  _T_143; // @[MemArbiter.scala 63:81]
  wire  _T_144; // @[MemArbiter.scala 63:97]
  wire  _T_146; // @[MemArbiter.scala 64:58]
  wire  _T_148; // @[MemArbiter.scala 64:81]
  wire  _T_149; // @[MemArbiter.scala 64:97]
  wire  _T_151; // @[MemArbiter.scala 65:58]
  wire  _T_153; // @[MemArbiter.scala 65:81]
  wire  _T_154; // @[MemArbiter.scala 65:97]
  wire  _T_156; // @[MemArbiter.scala 66:58]
  wire  _T_158; // @[MemArbiter.scala 66:81]
  wire  _T_159; // @[MemArbiter.scala 66:97]
  wire [2:0] _GEN_0; // @[MemArbiter.scala 82:39]
  wire [2:0] _GEN_1; // @[MemArbiter.scala 80:39]
  wire [2:0] _GEN_2; // @[MemArbiter.scala 78:39]
  wire [2:0] _GEN_3; // @[MemArbiter.scala 76:39]
  wire [2:0] _GEN_4; // @[MemArbiter.scala 74:39]
  wire [2:0] _GEN_5; // @[MemArbiter.scala 72:40]
  wire [2:0] _GEN_6; // @[MemArbiter.scala 70:32]
  wire  _T_164; // @[MemArbiter.scala 89:13]
  wire [2:0] _GEN_7; // @[MemArbiter.scala 89:41]
  wire [2:0] _GEN_12; // @[MemArbiter.scala 114:41]
  wire [2:0] _GEN_13; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_14; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_15; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_16; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_17; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_18; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_19; // @[Conditional.scala 39:67]
  wire [2:0] _GEN_20; // @[Conditional.scala 40:58]
  assign ins_cache_read = state == 3'h1; // @[MemArbiter.scala 25:31]
  assign inp_cache_read = state == 3'h2; // @[MemArbiter.scala 26:31]
  assign wgt_cache_read = state == 3'h3; // @[MemArbiter.scala 27:31]
  assign uop_cache_read = state == 3'h4; // @[MemArbiter.scala 28:31]
  assign acc_cache_read = state == 3'h5; // @[MemArbiter.scala 29:31]
  assign out_cache_write = state == 3'h6; // @[MemArbiter.scala 30:31]
  assign out_cache_ack = state == 3'h7; // @[MemArbiter.scala 31:31]
  assign _T_105 = 3'h0 == state; // @[Mux.scala 46:19]
  assign _T_107 = 3'h7 == state; // @[Mux.scala 46:19]
  assign _T_108 = _T_107 ? io_out_cache_address : 32'h0; // @[Mux.scala 46:16]
  assign _T_109 = 3'h6 == state; // @[Mux.scala 46:19]
  assign _T_110 = _T_109 ? io_out_cache_address : _T_108; // @[Mux.scala 46:16]
  assign _T_111 = 3'h5 == state; // @[Mux.scala 46:19]
  assign _T_112 = _T_111 ? io_acc_cache_address : _T_110; // @[Mux.scala 46:16]
  assign _T_113 = 3'h4 == state; // @[Mux.scala 46:19]
  assign _T_114 = _T_113 ? io_uop_cache_address : _T_112; // @[Mux.scala 46:16]
  assign _T_115 = 3'h3 == state; // @[Mux.scala 46:19]
  assign _T_116 = _T_115 ? io_wgt_cache_address : _T_114; // @[Mux.scala 46:16]
  assign _T_117 = 3'h2 == state; // @[Mux.scala 46:19]
  assign _T_118 = _T_117 ? io_inp_cache_address : _T_116; // @[Mux.scala 46:16]
  assign _T_119 = 3'h1 == state; // @[Mux.scala 46:19]
  assign _T_122 = out_cache_write | out_cache_ack; // @[MemArbiter.scala 46:77]
  assign _T_128 = _T_111 ? {{2'd0}, io_acc_cache_read} : 3'h0; // @[Mux.scala 46:16]
  assign _T_130 = _T_113 ? {{2'd0}, io_uop_cache_read} : _T_128; // @[Mux.scala 46:16]
  assign _T_132 = _T_115 ? {{2'd0}, io_wgt_cache_read} : _T_130; // @[Mux.scala 46:16]
  assign _T_134 = _T_117 ? {{2'd0}, io_inp_cache_read} : _T_132; // @[Mux.scala 46:16]
  assign axi_master_read = _T_119 ? {{2'd0}, io_ins_cache_read} : _T_134; // @[Mux.scala 46:16]
  assign _T_136 = io_axi_master_waitrequest & ins_cache_read; // @[MemArbiter.scala 62:58]
  assign _T_138 = ins_cache_read == 1'h0; // @[MemArbiter.scala 62:81]
  assign _T_139 = _T_138 & io_ins_cache_read; // @[MemArbiter.scala 62:97]
  assign _T_141 = io_axi_master_waitrequest & inp_cache_read; // @[MemArbiter.scala 63:58]
  assign _T_143 = inp_cache_read == 1'h0; // @[MemArbiter.scala 63:81]
  assign _T_144 = _T_143 & io_inp_cache_read; // @[MemArbiter.scala 63:97]
  assign _T_146 = io_axi_master_waitrequest & wgt_cache_read; // @[MemArbiter.scala 64:58]
  assign _T_148 = wgt_cache_read == 1'h0; // @[MemArbiter.scala 64:81]
  assign _T_149 = _T_148 & io_wgt_cache_read; // @[MemArbiter.scala 64:97]
  assign _T_151 = io_axi_master_waitrequest & uop_cache_read; // @[MemArbiter.scala 65:58]
  assign _T_153 = uop_cache_read == 1'h0; // @[MemArbiter.scala 65:81]
  assign _T_154 = _T_153 & io_uop_cache_read; // @[MemArbiter.scala 65:97]
  assign _T_156 = io_axi_master_waitrequest & acc_cache_read; // @[MemArbiter.scala 66:58]
  assign _T_158 = acc_cache_read == 1'h0; // @[MemArbiter.scala 66:81]
  assign _T_159 = _T_158 & io_acc_cache_read; // @[MemArbiter.scala 66:97]
  assign _GEN_0 = io_acc_cache_read ? 3'h5 : 3'h0; // @[MemArbiter.scala 82:39]
  assign _GEN_1 = io_uop_cache_read ? 3'h4 : _GEN_0; // @[MemArbiter.scala 80:39]
  assign _GEN_2 = io_wgt_cache_read ? 3'h3 : _GEN_1; // @[MemArbiter.scala 78:39]
  assign _GEN_3 = io_inp_cache_read ? 3'h3 : _GEN_2; // @[MemArbiter.scala 76:39]
  assign _GEN_4 = io_inp_cache_read ? 3'h2 : _GEN_3; // @[MemArbiter.scala 74:39]
  assign _GEN_5 = io_out_cache_write ? 3'h6 : _GEN_4; // @[MemArbiter.scala 72:40]
  assign _GEN_6 = io_ins_cache_read ? 3'h1 : _GEN_5; // @[MemArbiter.scala 70:32]
  assign _T_164 = io_axi_master_waitrequest == 1'h0; // @[MemArbiter.scala 89:13]
  assign _GEN_7 = _T_164 ? 3'h0 : state; // @[MemArbiter.scala 89:41]
  assign _GEN_12 = _T_164 ? 3'h7 : state; // @[MemArbiter.scala 114:41]
  assign _GEN_13 = _T_107 ? 3'h0 : state; // @[Conditional.scala 39:67]
  assign _GEN_14 = _T_109 ? _GEN_12 : _GEN_13; // @[Conditional.scala 39:67]
  assign _GEN_15 = _T_111 ? _GEN_7 : _GEN_14; // @[Conditional.scala 39:67]
  assign _GEN_16 = _T_113 ? _GEN_7 : _GEN_15; // @[Conditional.scala 39:67]
  assign _GEN_17 = _T_115 ? _GEN_7 : _GEN_16; // @[Conditional.scala 39:67]
  assign _GEN_18 = _T_117 ? _GEN_7 : _GEN_17; // @[Conditional.scala 39:67]
  assign _GEN_19 = _T_119 ? _GEN_7 : _GEN_18; // @[Conditional.scala 39:67]
  assign _GEN_20 = _T_105 ? _GEN_6 : _GEN_19; // @[Conditional.scala 40:58]
  assign io_ins_cache_waitrequest = _T_136 | _T_139; // @[MemArbiter.scala 62:28]
  assign io_ins_cache_readdata = io_axi_master_readdata; // @[MemArbiter.scala 57:25]
  assign io_inp_cache_waitrequest = _T_141 | _T_144; // @[MemArbiter.scala 63:28]
  assign io_inp_cache_readdata = io_axi_master_readdata; // @[MemArbiter.scala 58:25]
  assign io_wgt_cache_waitrequest = _T_146 | _T_149; // @[MemArbiter.scala 64:28]
  assign io_wgt_cache_readdata = io_axi_master_readdata; // @[MemArbiter.scala 59:25]
  assign io_uop_cache_waitrequest = _T_151 | _T_154; // @[MemArbiter.scala 65:28]
  assign io_uop_cache_readdata = io_axi_master_readdata; // @[MemArbiter.scala 60:25]
  assign io_acc_cache_waitrequest = _T_156 | _T_159; // @[MemArbiter.scala 66:28]
  assign io_acc_cache_readdata = io_axi_master_readdata; // @[MemArbiter.scala 61:25]
  assign io_out_cache_waitrequest = io_axi_master_waitrequest & _T_122; // @[MemArbiter.scala 46:28]
  assign io_out_cache_readdata = 128'h0;
  assign io_axi_master_address = _T_119 ? io_ins_cache_address : _T_118; // @[MemArbiter.scala 34:25]
  assign io_axi_master_read = axi_master_read[0]; // @[MemArbiter.scala 56:22]
  assign io_axi_master_write = io_out_cache_write & out_cache_write; // @[MemArbiter.scala 45:23]
  assign io_axi_master_writedata = io_out_cache_writedata; // @[MemArbiter.scala 44:28]
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
  _RAND_0 = {1{`RANDOM}};
  state = _RAND_0[2:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if (reset) begin
      state <= 3'h0;
    end else begin
      if (_T_105) begin
        if (io_ins_cache_read) begin
          state <= 3'h1;
        end else begin
          if (io_out_cache_write) begin
            state <= 3'h6;
          end else begin
            if (io_inp_cache_read) begin
              state <= 3'h2;
            end else begin
              if (io_inp_cache_read) begin
                state <= 3'h3;
              end else begin
                if (io_wgt_cache_read) begin
                  state <= 3'h3;
                end else begin
                  if (io_uop_cache_read) begin
                    state <= 3'h4;
                  end else begin
                    if (io_acc_cache_read) begin
                      state <= 3'h5;
                    end else begin
                      state <= 3'h0;
                    end
                  end
                end
              end
            end
          end
        end
      end else begin
        if (_T_119) begin
          if (_T_164) begin
            state <= 3'h0;
          end
        end else begin
          if (_T_117) begin
            if (_T_164) begin
              state <= 3'h0;
            end
          end else begin
            if (_T_115) begin
              if (_T_164) begin
                state <= 3'h0;
              end
            end else begin
              if (_T_113) begin
                if (_T_164) begin
                  state <= 3'h0;
                end
              end else begin
                if (_T_111) begin
                  state <= _GEN_7;
                end else begin
                  if (_T_109) begin
                    if (_T_164) begin
                      state <= 3'h7;
                    end
                  end else begin
                    if (_T_107) begin
                      state <= 3'h0;
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end
endmodule
