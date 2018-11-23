module stream_acp_adaptor (input            clock_clk,
                           input            reset,
									// conduit
									input   [31:0]   data_length_data,
									// axi_slave
									input                 axi_slave_address,
									output  reg           axi_slave_waitrequest,
									input                 axi_slave_read,
									output  reg  [31:0]   axi_slave_readdata,
									input                 axi_slave_write,
									input        [31:0]   axi_slave_writedata,
									// avmm_csr
									output  reg  [7:0]    avmm_csr_address,
									input                 avmm_csr_waitrequest,
									output  reg           avmm_csr_read,
									input        [31:0]   avmm_csr_readdata,
									output  reg           avmm_csr_write,
									output  reg  [31:0]   avmm_csr_writedata,
									// avmm_desc     
									output  reg  [7:0]    avmm_desc_address,
									input                 avmm_desc_waitrequest,
									output  reg           avmm_desc_read,
									input        [127:0]  avmm_desc_readdata,
									output  reg           avmm_desc_write,
									output  reg  [127:0]  avmm_desc_writedata,
									// cfg_master     
									output  reg  [7:0]    cfg_master_address,
									input                 cfg_master_waitrequest,
									output  reg           cfg_master_read,
									input        [31:0]   cfg_master_readdata,
									output  reg           cfg_master_write,
									output  reg  [31:0]   cfg_master_writedata
                           );

reg   [3:0]    awcache = 4'b1111;
reg   [4:0]    awuser = 5'b00001;
reg   [3:0]    arcache = 4'b1111;
reg   [4:0]    aruser = 5'b00001;
reg   [2:0]    awprot = 3'b100;
reg   [2:0]    arprot = 3'b100;
     
reg   [7:0]    csr_state;
reg   [7:0]    cfg_state;
     
reg   [1:0]    csr_status;
reg   [31:0]   dma_address = 0;

///////////////////////////////
// state controller
///////////////////////////////

always @(posedge clock_clk or posedge reset) begin
	if (reset) begin
	  csr_state <= 3; // idle
	  cfg_state <= 7;
	end
	else begin
	  if (axi_slave_write) begin
	    csr_state <= 0; // trigger
		 cfg_state <= 0;
	  end
	  else if (!avmm_desc_waitrequest) begin
	    if (csr_state == 3) begin
	      csr_state <= csr_state;
	    end
	    else begin
	  	   if ((csr_state==1) && ((avmm_csr_readdata & 8'h02)!=2)) begin
           csr_state <= csr_state; 
	  	   end
	      else begin
	        csr_state <= csr_state+1;
	      end
	    end
	  end
	  if (!cfg_master_waitrequest) begin
	    if (cfg_state == 7) begin
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

always @(posedge clock_clk or posedge reset) begin
	if (reset) begin
	  dma_address <= 0;
	end
	else begin
     if (axi_slave_write) begin
       dma_address <= axi_slave_writedata;
     end
	end
end

///////////////////////////////
// avmm_desc
///////////////////////////////

always @(posedge clock_clk or posedge reset) begin
	if (reset) begin
	  avmm_csr_writedata <= 0;
	  avmm_csr_address <= 0;
	  avmm_desc_writedata <= 0;
	  avmm_desc_address <= 0;
	  csr_status <= 0;
     avmm_csr_read <= 0;
	  avmm_csr_write <= 0; 
	  avmm_desc_write <= 0;
	  avmm_desc_read <= 0; 
	end
	else begin
	  case (csr_state) 
	    8'h00: begin 
	      avmm_csr_address <= 8'h00;
	  	   avmm_csr_read <= 1;
	  	   avmm_csr_write <= 0; 
	  	 end
	    8'h01: begin 
	      csr_status <= avmm_csr_readdata & 8'h02;
	  	   avmm_csr_read <= 1;
	  	   avmm_csr_write <= 0; 
	  	   avmm_desc_write <= 1; 
	  	   avmm_desc_read <= 0; 
	  	 end
	    8'h02: // read address for msgdma
	  	 if (csr_status==2) begin 
	  	   avmm_desc_address <= 8'h00;
	  	   avmm_csr_read <= 0;
	  	   avmm_csr_write <= 0; 
	  	   avmm_desc_write <= 1; 
	  	   avmm_desc_read <= 0; 
	  	   avmm_desc_writedata <= {32'h80000300,data_length_data,32'h00000000,dma_address};
	  	 end 
		 8'h03: begin // idle
	  	   avmm_csr_read <= 0;
	  	   avmm_csr_write <= 0; 
		   avmm_desc_write <= 0;
	  	   avmm_desc_read <= 0; 
		 end
	  endcase
	end
end

///////////////////////////////
// cfg_master
///////////////////////////////

always @(posedge clock_clk or posedge reset) begin
	if (reset) begin
	  cfg_master_writedata <= 0;
	  cfg_master_address <= 0;
	  cfg_master_write <= 0;
	  cfg_master_read <= 0; 
	end
	else begin
	  case (cfg_state) 
	    8'h00: begin cfg_master_address <= 8'h00; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= awcache; end
	    8'h01: begin cfg_master_address <= 8'h04; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= awprot ; end
	    8'h02: begin cfg_master_address <= 8'h08; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= awuser ; end
	    8'h03: begin cfg_master_address <= 8'h10; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= arcache; end
	    8'h04: begin cfg_master_address <= 8'h14; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= arprot ; end
	    8'h05: begin cfg_master_address <= 8'h18; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= aruser ; end
	    8'h06: begin cfg_master_address <= 8'h1C; cfg_master_write <= 1; cfg_master_read <= 0; cfg_master_writedata <= 0      ; end
		 8'h07: begin cfg_master_write <= 0; cfg_master_read <= 0; end
	  endcase
	end
end




endmodule
