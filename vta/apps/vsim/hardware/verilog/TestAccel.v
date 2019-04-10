/** Test accelerator.
  * 
  * Instantiate host/memory DPI modules and connect them to the accelerator.
  *    
  */
module TestAccel
(
  input clock,
  input reset
);

  localparam HOST_ADDR_BITS = 8;
  localparam HOST_DATA_BITS = 32;

  logic                      host_req_valid;
  logic                      host_req_opcode;
  logic [HOST_ADDR_BITS-1:0] host_req_addr;
  logic [HOST_DATA_BITS-1:0] host_req_value;
  logic                      host_req_deq;
  logic                      host_resp_valid;
  logic [HOST_DATA_BITS-1:0] host_resp_bits;

  localparam MEM_LEN_BITS = 8;
  localparam MEM_ADDR_BITS = 64;
  localparam MEM_DATA_BITS = 64;

  logic                     mem_req_valid;
  logic                     mem_req_opcode;
  logic  [MEM_LEN_BITS-1:0] mem_req_len;
  logic [MEM_ADDR_BITS-1:0] mem_req_addr;
  logic                     mem_wr_valid;
  logic [MEM_DATA_BITS-1:0] mem_wr_bits;
  logic                     mem_rd_valid;
  logic [MEM_DATA_BITS-1:0] mem_rd_bits;
  logic                     mem_rd_ready;

  VTAHostDPI host
  (
    .clock          (clock),
    .reset          (reset),

    .dpi_req_valid  (host_req_valid),
    .dpi_req_opcode (host_req_opcode),
    .dpi_req_addr   (host_req_addr),
    .dpi_req_value  (host_req_value),
    .dpi_req_deq    (host_req_deq),
    .dpi_resp_valid (host_resp_valid),
    .dpi_resp_bits  (host_resp_bits)
  );

  VTAMemDPI mem
  (
    .clock          (clock),
    .reset          (reset),

    .dpi_req_valid  (mem_req_valid),
    .dpi_req_opcode (mem_req_opcode),
    .dpi_req_len    (mem_req_len),
    .dpi_req_addr   (mem_req_addr),
    .dpi_wr_valid   (mem_wr_valid),
    .dpi_wr_bits    (mem_wr_bits),
    .dpi_rd_valid   (mem_rd_valid),
    .dpi_rd_bits    (mem_rd_bits),
    .dpi_rd_ready   (mem_rd_ready)
  );

  Accel #
  (
    .HOST_ADDR_BITS(HOST_ADDR_BITS),
    .HOST_DATA_BITS(HOST_DATA_BITS),
    .MEM_LEN_BITS(MEM_LEN_BITS),
    .MEM_ADDR_BITS(MEM_ADDR_BITS),
    .MEM_DATA_BITS(MEM_DATA_BITS)
  )
  accel
  (
    .clock           (clock),
    .reset           (reset),

    .host_req_valid  (host_req_valid),
    .host_req_opcode (host_req_opcode),
    .host_req_addr   (host_req_addr),
    .host_req_value  (host_req_value),
    .host_req_deq    (host_req_deq),
    .host_resp_valid (host_resp_valid),
    .host_resp_bits  (host_resp_bits),

    .mem_req_valid   (mem_req_valid),
    .mem_req_opcode  (mem_req_opcode),
    .mem_req_len     (mem_req_len),
    .mem_req_addr    (mem_req_addr),
    .mem_wr_valid    (mem_wr_valid),
    .mem_wr_bits     (mem_wr_bits),
    .mem_rd_valid    (mem_rd_valid),
    .mem_rd_bits     (mem_rd_bits),
    .mem_rd_ready    (mem_rd_ready)
  );
endmodule
