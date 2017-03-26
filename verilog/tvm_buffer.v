// Buffer used to add intermediate data buffering in channels
//
// Data within the read/write window is directly accessible via rd_addr/wr_addr.
// The read_advance/write_advance signals update the read/write data pointers by adding RD_WINDOW/WR_WINDOW.
// The status_counter indicate how many items are currently in the buffer (only registered after an advance signal is asserted).
// The ready/valid signals are used to implement a handshake protocol.
//
// Usage: create and pass instance to additional arguments of $tvm_session.


module tvm_buffer #(
    parameter DATA_WIDTH = 256,
    parameter DEPTH = 1024,
    parameter CNTR_WIDTH = 10, // log base 2 of BUFF_DEPTH
    parameter RD_WINDOW = 8, // set to 1 for FIFO behavior, or DEPTH for SRAM behavior
    parameter RD_ADVANCE = 2, // window advance (set to 1 for FIFO behavior)
    parameter RD_ADDR_WIDTH = 3, // log base 2 of RD_WINDOW
    parameter WR_WINDOW = 8, // set to 1 for FIFO behavior, or DEPTH for SRAM behavior
    parameter WR_ADVANCE = 2, // window advance (set to 1 for FIFO behavior)
    parameter WR_ADDR_WIDTH = 3 // log base 2 of WR_WINDOW
) (
    input clk,
    input rst,
    // Read ports
    input                       read_advance,   // Window advance (read pointer)
    input [RD_ADDR_WIDTH-1:0]   read_addr,      // Read address offset
    input                       read_ready,     // Read ready (dequeue)
    output                      read_valid,     // Read valid (not empty)
    output [DATA_WIDTH-1:0]     read_data,      // Read data port
    // Write ports
    input                       write_advance,  // Window advance (write pointer)
    input [WR_ADDR_WIDTH-1:0]   write_addr,     // Write address offset
    output                      write_ready,    // Write ready (not full)
    input                       write_valid,    // Write valid (enqueue)
    input [DATA_WIDTH-1:0]      write_data,     // Write data port
    // Other outputs
    output [CNTR_WIDTH-1:0]     status_counter  // Number of elements currently in FIFO
);

    // Outputs that need to be latched
    reg read_data;
    reg status_counter;

    // Internal registers (read pointer, write pointer)
    reg[CNTR_WIDTH-1:0] read_ptr;
    reg[CNTR_WIDTH-1:0] write_ptr;

    // RAM instance
    reg [DATA_WIDTH-1:0] ram[DEPTH-1:0];

    // Empty and full logic
    assign read_valid = (status_counter>=RD_WINDOW) ? 1'b1 : 1'b0;
    assign write_ready = (status_counter<(DEPTH-WR_WINDOW)) ? 1'b1 : 1'b0;

    // Counter logic (only affected by enq and deq)
    always @(posedge clk) begin
        // Case 1: system reset
        if (rst==1'b1) begin
            status_counter <= 0;
        // Case 2: simultaneous write advance and read advance and deq
        end else if ((write_advance && write_ready) && (read_advance && read_valid)) begin
            status_counter <= status_counter + (WR_ADVANCE - RD_ADVANCE);
        // Case 3: write advance
        end else if (write_advance && write_ready) begin
            status_counter <= status_counter + WR_ADVANCE;
        // Case 4: deq
        end else if (read_advance && read_valid) begin
            status_counter <= status_counter - RD_ADVANCE;
        // Default
        end else begin
            status_counter <= status_counter;
        end
    end

    // Output logic
    always @(posedge clk) begin
        if (rst==1'b1) begin 
            read_data <= 0;
        end else begin
            if(read_ready) begin
                read_data <= ram[(read_ptr+read_addr)%DEPTH];
            end else begin
                read_data <= read_data;
            end
        end
    end

    // RAM writing logic
    always @(posedge clk) begin
        if(write_valid) begin
            ram[((write_ptr+write_addr)%DEPTH)] <= write_data;
        end
    end

    // Read and write pointer logic
    always@(posedge clk) begin
        if (rst==1'b1) begin
            write_ptr <= 0;
            read_ptr <= 0;
        end else begin
            // Increment write pointer by WR_ADVANCE when asserting write_advance
            // When performing a write, no need to update the write pointer
            if (write_advance && write_ready) begin
                write_ptr <= (write_ptr + WR_ADVANCE) % DEPTH;
            end else begin
                write_ptr <= write_ptr;
            end
            // Increment read pointer by RD_ADVANCE when asserting read_advance
            // When performing a read, no need to update the read pointer
            if(read_advance && read_valid) begin
                read_ptr <= (read_ptr + RD_ADVANCE) % DEPTH;
            end else begin
                read_ptr <= read_ptr;
            end
        end
    end

endmodule // tvm_buffer
