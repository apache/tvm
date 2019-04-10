/** Register File.
  * 
  * Six 32-bit register file.
  *
  * -------------------------------
  *  Register description    | addr
  * -------------------------|-----
  *  Control status register | 0x00
  *  Length value register   | 0x04
  *  Input pointer lsb       | 0x08
  *  Input pointer msb       | 0x0c
  *  Output pointer lsb      | 0x10
  *  Output pointer msb      | 0x14
  * -------------------------------

  * ------------------------------
  *  Control status register | bit
  * ------------------------------
  *  Launch                  | 0
  *  Finish                  | 1
  * ------------------------------
  */
module RegFile #
( parameter HOST_ADDR_BITS = 8,
  parameter HOST_DATA_BITS = 32
)
(
  input                         clock,
  input                         reset,

  input                         host_req_valid,
  input                         host_req_opcode,
  input    [HOST_ADDR_BITS-1:0] host_req_addr,
  input    [HOST_DATA_BITS-1:0] host_req_value,
  output                        host_req_deq,
  output                        host_resp_valid,
  output   [HOST_DATA_BITS-1:0] host_resp_bits,

  output                        launch,
  input                         finish,
  output                 [31:0] length,
  output                 [63:0] inp_baddr,
  output                 [63:0] out_baddr
);

  typedef enum logic {IDLE, READ} state_t;
  state_t state_n, state_r;

  always_ff @(posedge clock) begin
    if (reset) begin
      state_r <= IDLE;
    end else begin
      state_r <= state_n;
    end
  end

  always_comb begin
    state_n = IDLE;
    case (state_r)
      IDLE: begin
        if (host_req_valid & ~host_req_opcode) begin
          state_n = READ;
	end
      end

      READ: begin
        state_n = IDLE;
      end
    endcase
  end

  assign host_req_deq = (state_r == IDLE) ? host_req_valid : 1'b0;

  logic [HOST_DATA_BITS-1:0] rf [5:0];

  genvar i;
  for (i = 0; i < 6; i++) begin
    logic wen = (state_r == IDLE)? host_req_valid & host_req_opcode & i*4 == host_req_addr : 1'b0;
    if (i == 0) begin
      always_ff @(posedge clock) begin
        if (reset) begin
	end else if (finish) begin
	  rf[i] <= 'd2;
	end else if (wen) begin
	  rf[i] <= host_req_value;
	end
      end
    end else begin
      always_ff @(posedge clock) begin
        if (reset) begin
	end else if (wen) begin
	  rf[i] <= host_req_value;
	end
      end
    end
  end

  logic [HOST_DATA_BITS-1:0] rdata;
  always_ff @(posedge clock) begin
    if (reset) begin
      rdata <= 'd0;
    end else if ((state_r == IDLE) & host_req_valid & ~host_req_opcode) begin
      if (host_req_addr == 'h00) begin
        rdata <= rf[0];
      end else if (host_req_addr == 'h04) begin
        rdata <= rf[1];
      end else if (host_req_addr == 'h08) begin
        rdata <= rf[2];
      end else if (host_req_addr == 'h0c) begin
        rdata <= rf[3];
      end else if (host_req_addr == 'h10) begin
        rdata <= rf[4];
      end else if (host_req_addr == 'h14) begin
        rdata <= rf[5];
      end else begin
        rdata <= 'd0;
      end
    end
  end

  assign host_resp_valid = (state_r == READ);
  assign host_resp_bits = rdata;

  assign launch = rf[0][0];
  assign length = rf[1];
  assign inp_baddr = {rf[3], rf[2]};
  assign out_baddr = {rf[5], rf[4]};

endmodule
