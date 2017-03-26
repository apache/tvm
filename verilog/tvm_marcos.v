// Nonstop version of loop
// Always keeps looping when increase == true
// At end is a signal to indicate the next cycle is end
// Use that to signal parent loop to advance.
`define NONSTOP_LOOP(iter, width, init, ready, finish, min, extent)\
    reg [width-1:0] iter;\
    wire finish;\
    always@(posedge clk) begin\
        if (rst || init) begin\
            iter <= (min);\
        end else if(ready) begin\
            if (iter != ((extent)-1)) begin\
                iter <= iter + 1;\
            end else begin\
              iter <= (min);\
            end\
        end else begin\
            iter <= iter;\
        end\
    end\
    assign finish = (ready && (iter == (extent) - 1));


// Wrap a nonstop loop to normal loop that loop only once.
// Use done signal to control the non-stop body to stop.
// The init and done behaves like normal loop
`define WRAP_LOOP_ONCE(init, valid, ready, body_finish, body_ready)\
    reg valid;\
    wire body_ready;\
    always@(posedge clk) begin\
        if (rst || init) begin\
            valid <= 1;\
        end else if(body_finish) begin\
            valid <= 0;\
        end else begin\
            valid <= valid;\
        end\
    end\
    assign body_ready = (valid && ready);

// Assign dst as src delayed by specific cycles.
`define DELAY(dst, src, width, delay, not_stall)\
    reg [(width)*(delay)-1:0] src``_dly_chain;\
    always@(posedge clk) begin\
        if(rst) begin\
            src``_dly_chain <= 0;\
        end else if (not_stall) begin\
            src``_dly_chain[(width)-1:0] <= src;\
            if((delay) != 1) begin\
                src``_dly_chain[(delay)*(width)-1:(width)] <= src``_dly_chain[((delay)-1)*(width)-1:0];\
            end\
        end else begin\
            src``_dly_chain <= src``_dly_chain;\
        end\
    end\
    assign dst = src``_dly_chain[(delay)*(width)-1:((delay)-1)*(width)];

// TVM generate clock signal
`define TVM_DEFINE_TEST_SIGNAL(clk, rst)\
   parameter PER = 10;\
   reg clk;\
   reg rst;\
   always begin\
      #(PER/2) clk =~ clk;\
   end

// Control logic on buffer/RAM read valid.
// This delays the valid signal by one cycle and retain it when write_ready == 0
`define BUFFER_READ_VALID_DELAY(dst, data_valid, write_ready)\
    reg dst;\
    always@(posedge clk) begin\
        if(rst) begin\
            dst <= 0;\
        end else if (write_ready) begin\
            dst <= (data_valid);\
        end else begin\
            dst <= dst;\
        end\
    end\

// A cache register that add one cycle lag to the ready signal
// This allows the signal to flow more smoothly
`define CACHE_REG(width, in_data, in_valid, in_ready, out_data, out_valid, out_ready)\
    reg [width-1:0] out_data``_state_;\
    reg [width-1:0] out_data``_overflow_;\
    reg out_valid``_state_;\
    reg out_valid``_overflow_;\
    always@(posedge clk) begin\
        if(rst) begin\
            out_valid``_overflow_ <= 0;\
            out_valid``_state_ <= 0;\
        end else if (out_valid``_overflow_) begin\
            if (out_ready) begin\
               out_valid``_state_ <= 1;\
               out_data``_state_ <= out_data``_overflow_;\
               out_valid``_overflow_ <= 0;\
               out_data``_overflow_ <= 0;\
            end else begin\
               out_valid``_state_ <= 1;\
               out_data``_state_ <= out_data``_state_;\
               out_valid``_overflow_ <= out_valid``_overflow_;\
               out_data``_overflow_ <= out_data``_overflow_;\
            end\
        end else begin\
            if (!out_ready && out_valid``_state_) begin\
               out_valid``_state_ <= 1;\
               out_data``_state_ <= out_data``_state_;\
               out_valid``_overflow_ <= in_valid;\
               out_data``_overflow_ <= in_data;\
            end else begin\
               out_valid``_state_ <= in_valid;\
               out_data``_state_ <= in_data;\
               out_valid``_overflow_ <= out_valid``_overflow_;\
               out_data``_overflow_ <= out_data``_overflow_;\
            end\
        end\
    end\ // always@ (posedge clk)
    assign in_ready = !out_valid``_overflow_;\
    assign out_data = out_data``_state_;\
    assign out_valid = out_valid``_state_;
