module vta_adaptor (
 input                clock_clk
,input                reset_reset
//=================================================
// configure ACP
//=================================================
// acp_cfg_master  
,output  reg   [7:0]  acp_cfg_master_address
,input                acp_cfg_master_waitrequest
,output  reg          acp_cfg_master_write
,output  reg  [31:0]  acp_cfg_master_writedata
//=================================================
// map base address
//=================================================
// uops_cfg slave 
,input                uops_cfg_address
,output  reg          uops_cfg_waitrequest
,input                uops_cfg_write
,input        [31:0]  uops_cfg_writedata
// uops slave
,input        [31:0]  uops_address
,output wire          uops_waitrequest
,input                uops_read
,output      [127:0]  uops_readdata
// insns_cfg slave 
,input                insns_cfg_address
,output  reg          insns_cfg_waitrequest
,input                insns_cfg_write
,input        [31:0]  insns_cfg_writedata
// insns slave
,input        [31:0]  insns_address
,output wire          insns_waitrequest
,input                insns_read
,output      [127:0]  insns_readdata
// biases_cfg slave 
,input                biases_cfg_address
,output  reg          biases_cfg_waitrequest
,input                biases_cfg_write
,input        [31:0]  biases_cfg_writedata
// biases slave
,input        [31:0]  biases_address
,output wire          biases_waitrequest
,input                biases_read
,output      [127:0]  biases_readdata
// outputs_cfg slave 
,input                outputs_cfg_address
,output  reg          outputs_cfg_waitrequest
,input                outputs_cfg_write
,input        [31:0]  outputs_cfg_writedata
// outputs slave
,input        [31:0]  outputs_address
,output wire          outputs_waitrequest
,input                outputs_write
,input       [127:0]  outputs_writedata

// axi_master
,output wire  [31:0]  axi_master_address
,input                axi_master_waitrequest
,output wire          axi_master_read
,input       [127:0]  axi_master_readdata
,output wire          axi_master_write
,output wire [127:0]  axi_master_writedata
);

reg   [3:0]    awcache = 4'b1111;
reg   [4:0]    awuser = 5'b00001;
reg   [3:0]    arcache = 4'b1111;
reg   [4:0]    aruser = 5'b00001;
reg   [2:0]    awprot = 3'b100;
reg   [2:0]    arprot = 3'b100;

reg   [7:0]    cfg_state;

reg   [31:0]   insns_base_address = 0;
reg   [31:0]   uops_base_address = 0;
reg   [31:0]   biases_base_address = 0;
reg   [31:0]   outputs_base_address = 0;

// ins_master
wire  [31:0]  ins_master_address    ;
wire          ins_master_waitrequest;
wire          ins_master_read       ;
wire [127:0]  ins_master_readdata   ;
// uop_master
wire  [31:0]  uop_master_address    ;
wire          uop_master_waitrequest;
wire          uop_master_read       ;
wire [127:0]  uop_master_readdata   ;
// acc_master                              
wire  [31:0]  acc_master_address    ;
wire          acc_master_waitrequest;
wire          acc_master_read       ;
wire [127:0]  acc_master_readdata   ;
// out_master                              
wire  [31:0]  out_master_address    ;
wire          out_master_waitrequest;
wire          out_master_write      ;
wire [127:0]  out_master_writedata  ;

MemArbiter arbiter (
  .clock(clock_clk),
  .reset(reset_reset),
  .io_ins_cache_waitrequest (ins_master_waitrequest),
  .io_ins_cache_address     (ins_master_address),
  .io_ins_cache_read        (ins_master_read),
  .io_ins_cache_readdata    (ins_master_readdata),
  .io_inp_cache_waitrequest (),
  .io_inp_cache_address     (),
  .io_inp_cache_read        (),
  .io_inp_cache_readdata    (),
  .io_inp_cache_write       (),
  .io_inp_cache_writedata   (),
  .io_wgt_cache_waitrequest (),
  .io_wgt_cache_address     (),
  .io_wgt_cache_read        (),
  .io_wgt_cache_readdata    (),
  .io_wgt_cache_write       (),
  .io_wgt_cache_writedata   (),
  .io_uop_cache_address     (uop_master_address),
  .io_uop_cache_waitrequest (uop_master_waitrequest),
  .io_uop_cache_read        (uop_master_read),
  .io_uop_cache_readdata    (uop_master_readdata),
  .io_acc_cache_address     (acc_master_address),
  .io_acc_cache_waitrequest (acc_master_waitrequest),
  .io_acc_cache_read        (acc_master_read),
  .io_acc_cache_readdata    (acc_master_readdata),
  .io_out_cache_address     (out_master_address),
  .io_out_cache_waitrequest (out_master_waitrequest),
  .io_out_cache_write       (out_master_write),
  .io_out_cache_writedata   (out_master_writedata),
  .io_axi_master_address    (axi_master_address),
  .io_axi_master_waitrequest(axi_master_waitrequest),
  .io_axi_master_read       (axi_master_read),
  .io_axi_master_readdata   (axi_master_readdata),
  .io_axi_master_write      (axi_master_write),
  .io_axi_master_writedata  (axi_master_writedata)
);
                     
assign ins_master_address[31:0] = insns_address[31:0] + insns_base_address[31:0];
assign insns_waitrequest = ins_master_waitrequest;
assign ins_master_read = insns_read;
assign insns_readdata[127:0] = ins_master_readdata[127:0];

assign uop_master_address[31:0] = uops_address[31:0] + uops_base_address[31:0];
assign uops_waitrequest = uop_master_waitrequest;
assign uop_master_read = uops_read;
assign uops_readdata[127:0] = uop_master_readdata[127:0];

assign acc_master_address[31:0] = biases_address[31:0] + biases_base_address[31:0];
assign biases_waitrequest = acc_master_waitrequest;
assign acc_master_read = biases_read;
assign biases_readdata[127:0] = acc_master_readdata[127:0];

assign out_master_address[31:0] = outputs_address[31:0] + outputs_base_address[31:0];
assign outputs_waitrequest = out_master_waitrequest;
assign out_master_write = outputs_write;
assign out_master_writedata[127:0] = outputs_writedata[127:0];


///////////////////////////////
// state controller
///////////////////////////////

always @(posedge clock_clk or posedge reset_reset) begin
  if (reset_reset) begin
    cfg_state <= 8'h07;
  end
  else begin
    if (outputs_cfg_write) begin
     cfg_state <= 0;
    end
    if (!acp_cfg_master_waitrequest) begin
      if (cfg_state == 8'h07) begin
        cfg_state <= cfg_state;
      end
      else begin
        cfg_state <= cfg_state+1;
      end
    end
   end
end

///////////////////////////////
// axi_slave
///////////////////////////////

always @(posedge clock_clk or posedge reset_reset) begin
  if (reset_reset) begin
    insns_base_address <= 0;
    uops_base_address <= 0;
    biases_base_address <= 0;
    outputs_base_address <= 0;
  end
  else begin
     if (insns_cfg_write) begin
       insns_base_address <= insns_cfg_writedata;
     end
     if (uops_cfg_write) begin
       uops_base_address <= uops_cfg_writedata;
     end
     if (biases_cfg_write) begin
       biases_base_address <= biases_cfg_writedata;
     end
     if (outputs_cfg_write) begin
       outputs_base_address <= outputs_cfg_writedata;
     end
  end
end

///////////////////////////////
// avalon_csr_master
///////////////////////////////

always @(posedge clock_clk or posedge reset_reset) begin
  if (reset_reset) begin
    acp_cfg_master_writedata <= 0;
    acp_cfg_master_address <= 0;
    acp_cfg_master_write <= 0;
  end
  else begin
    case (cfg_state) 
      8'h00: begin acp_cfg_master_address <= 8'h00; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= awcache; end
      8'h01: begin acp_cfg_master_address <= 8'h04; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= awprot ; end
      8'h02: begin acp_cfg_master_address <= 8'h08; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= awuser ; end
      8'h03: begin acp_cfg_master_address <= 8'h10; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= arcache; end
      8'h04: begin acp_cfg_master_address <= 8'h14; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= arprot ; end
      8'h05: begin acp_cfg_master_address <= 8'h18; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= aruser ; end
      8'h06: begin acp_cfg_master_address <= 8'h1C; acp_cfg_master_write <= 1; acp_cfg_master_writedata <= 0      ; end
      8'h07: begin                                  acp_cfg_master_write <= 0; end
    endcase
  end
end

endmodule
