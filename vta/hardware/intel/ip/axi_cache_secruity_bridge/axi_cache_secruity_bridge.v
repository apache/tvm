/*
  Legal Notice: (C)2015 Altera Corporation. All rights reserved.  Your
  use of Altera Corporation's design tools, logic functions and other
  software and tools, and its AMPP partner logic functions, and any
  output files any of the foregoing (including device programming or
  simulation files), and any associated documentation or information are
  expressly subject to the terms and conditions of the Altera Program
  License Subscription Agreement or other applicable license agreement,
  including, without limitation, that your use is for the sole purpose
  of programming logic devices manufactured by Altera and sold by Altera
  or its authorized distributors.  Please refer to the applicable
  agreement for further details.
*/


/*

PRBS generator core

Author:  JCJB
Date:    11/30/2015

Version:  1.0

Description:  AXI to AXI bridge that optionally allows a host to overwrite the master Ax_CACHE, Ax_PROT, and Ax_USER signals with values written to this component.
              The write channel settings are located in addresses 0x0, 0x4, 0x8, and 0x1C bits 0-2.  The read channel settings are located in addresses 0x10, 0x14,
              0x18, and 0x1C bits 4-6.  The typical use case is software will write values to addresses 0x0, 0x4, 0x8, 0x10, 0x14, 0x18, and write all zeros to address
              0x1C so that non of the incoming Ax_CACHE, Ax_PROT, and Ax_USER signals are fowarded to the master and instead the host set values are used instead.

Register map:

|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|  Address   |   Access Type   |                                                            Bits                                                        |
|            |                 |------------------------------------------------------------------------------------------------------------------------|
|            |                 |            31..24            |            23..16            |            15..8            |            7..0            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|     0x00   |       r/w       |                                                 Master write channel AW_CACHE                                          |
|     0x04   |       r/w       |                                                 Master write channel AW_PROT                                           |
|     0x08   |       r/w       |                                                 Master write channel AW_USER                                           |
|     0x0C   |       N/A       |                                                         Reserved                                                       |
|     0x10   |       r/w       |                                                 Master read channel AR_CACHE                                           |
|     0x14   |       r/w       |                                                 Master read channel AR_PROT                                            |
|     0x18   |       r/w       |                                                 Master read channel AR_USER                                            |
|     0x1C   |       r/w       |                                                        MUX select                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|


Address 0x00 --> m_awcache[3:0] master AW_CACHE value.  This value resides in bits 3:0 and is driven out to the master if mux_select[0] is set to 0.
Address 0x04 --> m_awprot[2:0] master AW_PROT value.  This value resides in bits 2:0 and is driven out to the master if mux_select[1] is set to 0.
Address 0x08 --> m_awuser[AXI_USER_WIDTH-1:0] master AW_USER value.  This value resides in the least significant bits and is driven out to the master if mux_select[2] is set to 0.
Address 0x0C --> Reserved.  Writes will have no effect and reads will return mux_select value.
Address 0x10 --> m_arcache[3:0] master AR_CACHE value.  This value resides in bits 3:0 and is driven out to the master if mux_select[4] is set to 0.
Address 0x14 --> m_arprot[2:0] master AR_PROT value.  This value resides in bits 2:0 and is driven out to the master if mux_select[5] is set to 0.
Address 0x18 --> m_aruser[AXI_USER_WIDTH-1:0] master AR_USER value.  This value resides in the least significant bits and is driven out to the master if mux_select[6] is set to 0.
Address 0x1C --> mux_select which is used to determine if the software programmable fields are issued to the master or if the incoming signals from the slave
                 port route to the master.  The bit decoding is shown below:

                 mux_select[0] = Master write mux select for AW_CACHE, when set to 0 the value m_awcache is sent to the master, when sent to 1 the slave s_awcache is sent to master.
                 mux_select[1] = Master write mux select for AW_PROT, when set to 0 the value m_awprot is sent to the master, when sent to 1 the slave s_awuser is sent to master.
                 mux_select[2] = Master write mux select for AW_USER, when set to 0 the value m_awuser is sent to the master, when sent to 1 the slave s_awuser is sent to master.
                 mux_select[3] = Reserved, writing to this bit will have no effect on the hardware.
                 mux_select[4] = Master read mux select for AR_CACHE, when set to 0 the value m_arcache is sent to the master, when sent to 1 the slave s_arcache is sent to master.
                 mux_select[5] = Master read mux select for AR_PROT, when set to 0 the value m_arprot is sent to the master, when sent to 1 the slave s_arprot is sent to master.
                 mux_select[6] = Master read mux select for AR_USER, when set to 0 the value m_aruser is sent to the master, when sent to 1 the slave s_aruser is sent to master.
                 mux_select[7] = Reserved, writing to this bit will have no effect on the hardware.


Revision History:

1.0 (11/30/2010)   Initial version based on the old 13.1 axi_conduit_merger component from the Cyclone V SoC data mover.


*/

module axi_cache_secruity_bridge #(
  parameter ID_WIDTH      = 1,
  parameter DATA_WIDTH    = 32,
  parameter ADDRESS_WIDTH = 32,
  parameter AXUSER_WIDTH  = 5  
) (
// axi master
output       				m_awvalid,  
output [3:0] 				m_awlen  ,  
output [2:0] 				m_awsize ,  
output [1:0] 				m_awburst,  
output [1:0] 				m_awlock ,  
output [3:0] 				m_awcache,  
output [2:0] 				m_awprot ,  
input        				m_awready,  
output [AXUSER_WIDTH-1:0] 	m_awuser ,  
output       				m_arvalid,  
output [3:0] 				m_arlen  ,  
output [2:0] 				m_arsize ,  
output [1:0] 				m_arburst,  
output [1:0] 				m_arlock ,  
output [3:0] 				m_arcache,  
output [2:0] 				m_arprot ,  
input        				m_arready,  
output [AXUSER_WIDTH-1:0] 	m_aruser ,  
input        				m_rvalid ,  
input        				m_rlast  ,  
input  [1:0] 				m_rresp  ,  
output       				m_rready ,  
output       				m_wvalid ,  
output       				m_wlast  ,  
input        				m_wready ,  
input        				m_bvalid ,  
input  [1:0] 				m_bresp  ,  
output       				m_bready ,  
output [ADDRESS_WIDTH-1:0] 	m_awaddr ,   
output [ID_WIDTH-1:0] 		m_awid   ,   
output [ADDRESS_WIDTH-1:0] 	m_araddr ,   
output [ID_WIDTH-1:0] 		m_arid   ,   
input  [DATA_WIDTH-1:0] 	m_rdata  ,   
input  [ID_WIDTH-1:0] 		m_rid    ,   
output [DATA_WIDTH-1:0] 	m_wdata  ,   
output [DATA_WIDTH/8-1:0]  	m_wstrb  ,   
output [ID_WIDTH-1:0] 		m_wid    ,   
input  [ID_WIDTH-1:0] 		m_bid    ,   

// axi slave
input       				s_awvalid,  
input  [3:0] 				s_awlen  ,  
input  [2:0] 				s_awsize ,  
input  [1:0] 				s_awburst,  
input  [1:0] 				s_awlock ,  
input  [3:0] 				s_awcache,  
input  [2:0] 				s_awprot ,  
output         				s_awready,  
input  [AXUSER_WIDTH-1:0] 	s_awuser ,  
input       				s_arvalid,  
input  [3:0] 				s_arlen  ,  
input  [2:0] 				s_arsize ,  
input  [1:0] 				s_arburst,  
input  [1:0] 				s_arlock ,  
input  [3:0] 				s_arcache,  
input  [2:0] 				s_arprot ,  
output         				s_arready,  
input  [AXUSER_WIDTH-1:0] 	s_aruser ,  
output        				s_rvalid ,  
output        				s_rlast  ,  
output [1:0] 				s_rresp  ,  
input       				s_rready ,  
input       				s_wvalid ,  
input       				s_wlast  ,  
output        				s_wready ,  
output        				s_bvalid ,  
output [1:0] 				s_bresp  ,  
input       				s_bready ,  
input  [ADDRESS_WIDTH-1:0] 	s_awaddr,   
input  [ID_WIDTH-1:0] 		s_awid  ,   
input  [ADDRESS_WIDTH-1:0] 	s_araddr,   
input  [ID_WIDTH-1:0] 		s_arid  ,   
output [DATA_WIDTH-1:0] 	s_rdata ,   
output [ID_WIDTH-1:0] 		s_rid   ,   
input  [DATA_WIDTH-1:0] 	s_wdata ,   
input  [DATA_WIDTH/8-1:0]  	s_wstrb ,   
input  [ID_WIDTH-1:0] 		s_wid   ,   
output [ID_WIDTH-1:0] 		s_bid   ,  

// Avalon-MM control slave
input [2:0] csr_address,
input       csr_write,
input       csr_read,
input [31:0] csr_writedata,
input [3:0] csr_byteenable,
output [31:0] csr_readdata,


// clock and reset
input    					clk,
input    					rst_n 
);


// registers that software will access to setup the forced master signals
reg [31:0] csr_awcache;
reg [31:0] csr_awprot;
reg [31:0] csr_awuser;
reg [31:0] csr_arcache;
reg [31:0] csr_arprot;
reg [31:0] csr_aruser;
reg [31:0] csr_mux_selects;   // bit 0 --> use s_awcache, bit 1 --> use s_awprot, bit 2 --> use s_awusers, bit 4 --> use s_arcache, bit 5 --> use s_arprot, bit 6 --> use s_aruser
reg [31:0] csr_readdata_d1;


// shadow registers for the cache, prot, and user signals for the read and write channels, these are optionally used based on the settings in 'csr_mux_selects'
reg  [3:0] 				       	r_awcache;
reg  [2:0] 				      	r_awprot;
reg  [AXUSER_WIDTH-1:0] 	r_awuser;
reg  [3:0] 				      	r_arcache;
reg  [2:0] 				      	r_arprot;
reg  [AXUSER_WIDTH-1:0] 	r_aruser;


always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_awcache <= 32'h00000000;
  end
  else if ((csr_address == 3'h0) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_awcache[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_awcache[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_awcache[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_awcache[31:24] <= csr_writedata[31:24];
  end
end

always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_awprot <= 32'h00000000;
  end
  else if ((csr_address == 3'h1) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_awprot[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_awprot[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_awprot[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_awprot[31:24] <= csr_writedata[31:24];
  end
end

always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_awuser <= 32'h00000000;
  end
  else if ((csr_address == 3'h2) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_awuser[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_awuser[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_awuser[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_awuser[31:24] <= csr_writedata[31:24];
  end
end


always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_arcache <= 32'h00000000;
  end
  else if ((csr_address == 3'h4) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_arcache[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_arcache[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_arcache[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_arcache[31:24] <= csr_writedata[31:24];
  end
end

always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_arprot <= 32'h00000000;
  end
  else if ((csr_address == 3'h5) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_arprot[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_arprot[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_arprot[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_arprot[31:24] <= csr_writedata[31:24];
  end
end

always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_aruser <= 32'h00000000;
  end
  else if ((csr_address == 3'h6) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_aruser[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_aruser[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_aruser[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_aruser[31:24] <= csr_writedata[31:24];
  end
end

always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    csr_mux_selects <= 32'h00000000;
  end
  else if ((csr_address == 3'h7) & (csr_write == 1'b1))
  begin
    if (csr_byteenable[0] == 1'b1)
      csr_mux_selects[7:0] <= csr_writedata[7:0];
    if (csr_byteenable[1] == 1'b1)
      csr_mux_selects[15:8] <= csr_writedata[15:8];
    if (csr_byteenable[2] == 1'b1)
      csr_mux_selects[23:16] <= csr_writedata[23:16];
    if (csr_byteenable[3] == 1'b1)
      csr_mux_selects[31:24] <= csr_writedata[31:24];
  end
end

// csr_ registers will clear so this register will settle one cycle later (no reset needed)
always @ (posedge clk)
begin
  case (csr_address)
    3'h0:  csr_readdata_d1 <= csr_awcache;
    3'h1:  csr_readdata_d1 <= csr_awprot;
    3'h2:  csr_readdata_d1 <= csr_awuser;
    3'h3:  csr_readdata_d1 <= csr_mux_selects; 
    3'h4:  csr_readdata_d1 <= csr_arcache;
    3'h5:  csr_readdata_d1 <= csr_arprot;
    3'h6:  csr_readdata_d1 <= csr_aruser;
    3'h7:  csr_readdata_d1 <= csr_mux_selects;
    default: csr_readdata_d1 <= csr_mux_selects;  
  endcase
end

// shadow registers for the write channel
always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    r_awcache <= 4'h0;
    r_awprot <= 3'h0;
    r_awuser <= {AXUSER_WIDTH{1'b0}};
  end
  else if (s_awvalid == 0)  // make sure write channel is idle before updating  
  begin
    r_awcache <= csr_awcache[3:0];
    r_awprot <= csr_awprot[2:0];
    r_awuser <= csr_awuser[AXUSER_WIDTH-1:0];
  end
end

// shadow registers for the read channel
always @ (posedge clk)
begin
  if (rst_n == 0)
  begin
    r_arcache <= 4'h0;
    r_arprot <= 3'h0;
    r_aruser <= {AXUSER_WIDTH{1'b0}};
  end
  else if (s_arvalid == 0)  // make sure read channel is idle before updating  
  begin
    r_arcache <= csr_arcache[3:0];
    r_arprot <= csr_arprot[2:0];
    r_aruser <= csr_aruser[AXUSER_WIDTH-1:0];
  end
end


assign csr_readdata = csr_readdata_d1;


// axi bus assignment
assign m_awvalid = s_awvalid   ;
assign m_awlen   = s_awlen     ;
assign m_awsize  = s_awsize    ;
assign m_awburst = s_awburst   ;
assign m_awlock  = s_awlock    ;
assign m_awcache = (csr_mux_selects[0] == 1)? s_awcache : r_awcache;
assign m_awprot  = (csr_mux_selects[1] == 1)? s_awprot : r_awprot;
assign m_awuser  = (csr_mux_selects[2] == 1)? s_awuser : r_awuser;
assign m_awaddr  = s_awaddr    ;
assign m_awid    = s_awid      ;
assign s_awready = m_awready   ;
assign m_arvalid = s_arvalid   ;
assign m_arlen   = s_arlen     ;
assign m_arsize  = s_arsize    ;
assign m_arburst = s_arburst   ;
assign m_arlock  = s_arlock    ;
assign m_arcache = (csr_mux_selects[4] == 1)? s_arcache : r_arcache;
assign m_arprot  = (csr_mux_selects[5] == 1)? s_arprot : r_arprot;
assign m_aruser  = (csr_mux_selects[6] == 1)? s_aruser : r_aruser;
assign m_araddr  = s_araddr    ;
assign m_arid    = s_arid      ;
assign s_arready = m_arready   ;
assign s_rvalid  = m_rvalid    ;
assign s_rlast   = m_rlast     ;
assign s_rresp   = m_rresp     ;
assign s_rdata   = m_rdata     ;
assign s_rid     = m_rid       ;
assign m_rready  = s_rready    ;
assign m_wvalid  = s_wvalid    ;
assign m_wlast   = s_wlast     ;
assign m_wdata   = s_wdata     ;
assign m_wstrb   = s_wstrb     ;
assign m_wid     = s_wid       ;
assign s_wready  = m_wready    ;
assign s_bvalid  = m_bvalid    ;
assign s_bresp   = m_bresp     ;
assign s_bid     = m_bid       ;
assign m_bready  = s_bready    ;      

endmodule
