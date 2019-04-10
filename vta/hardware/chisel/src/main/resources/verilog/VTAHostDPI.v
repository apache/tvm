module VTAHostDPI #
( parameter ADDR_BITS = 8,
  parameter DATA_BITS = 32
)
(
  input                        clock,
  input                        reset,
  output logic                 dpi_req_valid,
  output logic                 dpi_req_opcode,
  output logic [ADDR_BITS-1:0] dpi_req_addr,
  output logic [DATA_BITS-1:0] dpi_req_value,
  input                        dpi_req_deq,
  input                        dpi_resp_valid,
  input        [DATA_BITS-1:0] dpi_resp_bits
);

  import "DPI-C" function void VTAHostDPI
  (
    output byte unsigned exit,
    output byte unsigned req_valid,
    output byte unsigned req_opcode,
    output byte unsigned req_addr,
    output int  unsigned req_value,
    input  byte unsigned req_deq,
    input  byte unsigned resp_valid,
    input  int  unsigned resp_value
  );

  typedef logic        dpi1_t;
  typedef logic  [7:0] dpi8_t;
  typedef logic [31:0] dpi32_t;

  dpi1_t  __reset;
  dpi8_t  __exit;
  dpi8_t  __req_valid;
  dpi8_t  __req_opcode;
  dpi8_t  __req_addr;
  dpi32_t __req_value;
  dpi8_t  __req_deq;
  dpi8_t  __resp_valid;
  dpi32_t __resp_bits;

  // reset
  always_ff @(posedge clock) begin
    __reset <= reset;
  end

  // delaying outputs by one-cycle
  // since verilator does not support delays
  always_ff @(posedge clock) begin
    dpi_req_valid  <= dpi1_t ' (__req_valid);
    dpi_req_opcode <= dpi1_t ' (__req_opcode);
    dpi_req_addr   <= __req_addr;
    dpi_req_value  <= __req_value;
  end

  assign __req_deq    = dpi8_t ' (dpi_req_deq);
  assign __resp_valid = dpi8_t ' (dpi_resp_valid);
  assign __resp_bits  = dpi_resp_bits;

  // evaluate DPI function
  always_ff @(posedge clock) begin
    if (reset | __reset) begin
      __exit = 0;
      __req_valid = 0;
      __req_opcode = 0;
      __req_addr = 0;
      __req_value = 0;
    end
    else begin
      VTAHostDPI(
        __exit,
        __req_valid,
        __req_opcode,
        __req_addr,
        __req_value,
        __req_deq,
        __resp_valid,
        __resp_bits);
    end
  end

  logic [63:0] cycles;

  always_ff @(posedge clock) begin
    if (reset | __reset) begin
      cycles <= 'd0;
    end
    else begin
      cycles <= cycles + 1'b1;
    end
  end

  always_ff @(posedge clock) begin
    if (__exit == 'd1) begin
      $display("[DONE] at cycle:%016d", cycles);
      $finish;
    end
  end

endmodule
