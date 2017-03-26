module main();

    // Parameters
    parameter PER=10;

    // FIFO parameters
    parameter DATA_WIDTH = 8;
    parameter DEPTH = 32;
    parameter CNTR_WIDTH = 6; // floor(log(32)) + 1
    parameter RD_WINDOW = 1;
    parameter RD_ADVANCE = 1;
    parameter RD_ADDR_WIDTH = 1;
    parameter WR_WINDOW = 1;
    parameter WR_ADVANCE = 1;
    parameter WR_ADDR_WIDTH = 1;

    // Clock & reset
    reg clk;
    reg rst;

    // Module inputs
    reg [DATA_WIDTH-1:0] write_data;
    // FIFO interface abstraction:
    // Connect deq to read_advance and read_ready
    // Connect enq to write_advance and write_valid
    // Set read_addr and write_addr to 0
    reg deq;
    reg enq;

    // Module outputs
    wire [DATA_WIDTH-1:0] read_data;
    wire read_valid;
    wire write_ready;
    wire [CNTR_WIDTH-1:0] status_counter;

    // Module instantiation
    tvm_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .CNTR_WIDTH(CNTR_WIDTH),
        .RD_WINDOW(RD_WINDOW),
        .RD_ADVANCE(RD_ADVANCE),
        .RD_ADDR_WIDTH(RD_ADDR_WIDTH),
        .WR_WINDOW(WR_WINDOW),
        .WR_ADVANCE(WR_ADVANCE),
        .WR_ADDR_WIDTH(WR_ADDR_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .read_advance(deq),
        .read_addr({RD_ADDR_WIDTH{1'b0}}),
        .read_ready(deq),
        .read_valid(read_valid),
        .read_data(read_data),
        .write_advance(enq),
        .write_addr({WR_ADDR_WIDTH{1'b0}}),
        .write_ready(write_ready),
        .write_valid(enq),
        .write_data(write_data),
        .status_counter(status_counter)
    );

    // clock generation
    always begin
      #(PER/2) clk =~ clk;
    end

    // fifo read logic
    always @(posedge clk) begin
        if (rst)
            deq <= 0;
        else
            deq <= read_valid;
    end

    // read_data_valid logic
    reg read_data_valid;
    always @(posedge clk) begin
        if (rst)
            read_data_valid <= 0;
        else
            read_data_valid <= deq;
    end

    initial begin
        // This will allow tvm session to be called every cycle.
        $tvm_session(clk);
    end
endmodule
