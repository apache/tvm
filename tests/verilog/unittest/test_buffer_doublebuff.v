module main();

    // Parameters
    parameter PER=10;

    // Double buffer parameters
    parameter DATA_WIDTH = 8;
    parameter DEPTH = 32;
    parameter CNTR_WIDTH = 6; // floor(log(32)) + 1
    parameter RD_WINDOW = 16;
    parameter RD_ADVANCE = 16;
    parameter RD_ADDR_WIDTH = 5; // floor(log(16)) + 1
    parameter WR_WINDOW = 16;
    parameter WR_ADVANCE = 16;
    parameter WR_ADDR_WIDTH = 5; // floor(log(16)) + 1

    // Clock & reset
    reg clk;
    reg rst;

    // Read port inputs
    reg read_advance;
    reg [RD_ADDR_WIDTH-1:0] read_addr;
    reg read_ready;
    // Write port outputs
    reg write_advance;
    reg [DATA_WIDTH-1:0] write_data;
    reg [WR_ADDR_WIDTH-1:0] write_addr;
    reg write_valid;

    // Outputs
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
        .read_advance(read_advance),
        .read_data(read_data),
        .read_addr(read_addr),
        .read_ready(read_ready),
        .read_valid(read_valid),
        .write_advance(write_advance),
        .write_data(write_data),
        .write_addr(write_addr),
        .write_ready(write_ready),
        .write_valid(write_valid),
        .status_counter(status_counter)
    );

    // clock generation
    always begin
      #(PER/2) clk =~ clk;
    end

    // read logic
    always @(posedge clk) begin
        if (rst) begin
            read_advance <= 0;
            read_addr <= 0;
            read_ready <= 0;
        end else begin
            if (read_valid) begin
                read_ready <= 1;
            end else begin
                read_ready <= 0;
            end
            if (read_addr%RD_WINDOW==RD_WINDOW-2) begin
                read_advance <= 1;
            end else begin
                read_advance <= 0;
            end
            if (read_ready) begin
                read_addr <= (read_addr+1) % WR_WINDOW;
            end else begin
                read_addr <= read_addr % WR_WINDOW;
            end
        end
    end

    // read_data_valid logic
    reg read_data_valid;
    always @(posedge clk) begin
        if (rst)
            read_data_valid <= 0;
        else
            read_data_valid <= read_ready;
    end

    initial begin
        // This will allow tvm session to be called every cycle.
        $tvm_session(clk);
    end
endmodule
