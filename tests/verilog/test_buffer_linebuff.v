module main();

    // Parameters
    parameter PER=10;

    // In this example we perform a 3x3 convolution of an 8x8 input image
    // Therefore the window size here is (3-1)*8+3 = 19
    parameter IMAGE_WIDTH = 8;
    parameter KERNEL_WIDTH = 3;
    // Line buffer parameters
    parameter DATA_WIDTH = 8;
    parameter DEPTH = 20; // (3-1)*8+3+1
    parameter CNTR_WIDTH = 5; // floor(log(20)) + 1
    parameter RD_WINDOW = 19; // (3-1)*8+3
    parameter RD_ADVANCE = 1;
    parameter RD_ADDR_WIDTH = 5; // floor(log(19)) + 1
    parameter WR_WINDOW = 1;
    parameter WR_ADVANCE = 1;
    parameter WR_ADDR_WIDTH = 1;

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
        .write_addr({WR_ADDR_WIDTH{1'b0}}),
        .write_ready(write_ready),
        .write_valid(write_valid),
        .status_counter(status_counter)
    );

    // clock generation
    always begin
      #(PER/2) clk =~ clk;
    end

    // read logic
    localparam KERNEL_SIZE = KERNEL_WIDTH*KERNEL_WIDTH;
    reg [3:0] read_counter;
    always @(posedge clk) begin
        if (rst) begin
            read_counter <= KERNEL_SIZE-1;
            read_advance <= 0;
            read_addr <= -1;
            read_ready <= 0;
        end else begin
            if (read_valid) begin
                read_counter <= (read_counter+1)%KERNEL_SIZE;
                read_ready <= 1;
                // Only advance at the last inner loop iteration
                if (read_counter==KERNEL_SIZE-2) begin
                    read_advance <= 1;
                end else begin
                    read_advance <= 0;
                end
                // Read address should describe a loop
                if (read_counter==KERNEL_SIZE-1) begin
                    read_addr <= 0;
                end else if (read_counter%KERNEL_WIDTH==KERNEL_WIDTH-1) begin
                    read_addr <= read_addr+IMAGE_WIDTH-KERNEL_WIDTH+1;
                end else begin
                    read_addr <= read_addr+1;
                end
            end else begin
                read_counter <= read_counter;
                read_advance <= 0;
                read_addr <= read_addr;
                read_ready <= 0;
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
