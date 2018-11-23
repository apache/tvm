module altera_edge_detector #(
parameter PULSE_EXT = 0, // 0, 1 = edge detection generate single cycle pulse, >1 = pulse extended for specified clock cycle
parameter EDGE_TYPE = 0, // 0 = falling edge, 1 or else = rising edge
parameter IGNORE_RST_WHILE_BUSY = 0  // 0 = module internal reset will be default whenever rst_n asserted, 1 = rst_n request will be ignored while generating pulse out
) (
input      clk,
input      rst_n,
input      signal_in,
output     pulse_out
);

localparam IDLE = 0, ARM = 1, CAPT = 2;
localparam SIGNAL_ASSERT   = EDGE_TYPE ? 1'b1 : 1'b0;
localparam SIGNAL_DEASSERT = EDGE_TYPE ? 1'b0 : 1'b1;

reg [1:0] state, next_state;
reg       pulse_detect;
wire      busy_pulsing;

assign busy_pulsing = (IGNORE_RST_WHILE_BUSY)? pulse_out : 1'b0;
assign reset_qual_n = rst_n | busy_pulsing;

generate
if (PULSE_EXT > 1) begin: pulse_extend
  integer i;
  reg [PULSE_EXT-1:0] extend_pulse;
  always @(posedge clk or negedge reset_qual_n) begin
    if (!reset_qual_n)
      extend_pulse <= {{PULSE_EXT}{1'b0}};
    else begin
      for (i = 1; i < PULSE_EXT; i = i+1) begin
        extend_pulse[i] <= extend_pulse[i-1];
        end
      extend_pulse[0] <= pulse_detect;
      end
    end
  assign pulse_out = |extend_pulse;
  end
else begin: single_pulse
  reg pulse_reg;
  always @(posedge clk or negedge reset_qual_n) begin
    if (!reset_qual_n)
      pulse_reg <= 1'b0;
    else
      pulse_reg <= pulse_detect;
    end
  assign pulse_out = pulse_reg;
  end
endgenerate

always @(posedge clk) begin
  if (!rst_n)
    state <= IDLE;
  else
    state <= next_state;
end

// edge detect
always @(*) begin
  next_state = state;
  pulse_detect = 1'b0;
  case (state)
    IDLE : begin
           pulse_detect = 1'b0;
           if (signal_in == SIGNAL_DEASSERT) next_state = ARM;
           else next_state = IDLE;
           end
    ARM  : begin
           pulse_detect = 1'b0;
           if (signal_in == SIGNAL_ASSERT)   next_state = CAPT;
           else next_state = ARM;
           end
    CAPT : begin
           pulse_detect = 1'b1;
           if (signal_in == SIGNAL_DEASSERT) next_state = ARM;
           else next_state = IDLE;
           end
    default : begin
           pulse_detect = 1'b0;
           next_state = IDLE;
           end
  endcase
end

endmodule
