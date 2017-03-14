// Leaf of a normal loop nest
// Starts at done = 1
// Need init to reset to done = 0
// increases when enabled = 1
`define NORMAL_LOOP_LEAF(iter, width, init, enable, done, min, max, incr)\
    reg [width-1:0] iter;\
    reg valid;\
    reg done;\
    always@(posedge clk) begin\
        if(rst) begin\
            iter <= 0;\
            done <= 1;\
        end else if(init) begin\
            iter <= (min);\
            done <= 0;\
        end else if(done) begin\
            iter <= 0;\
            done <= 1;\
        end else if(enable) begin\
            if (iter < ((max)-(incr))) begin\
                iter <= iter + (incr);\
                done <= 0;\
            end else begin\
                iter <= 0;\
                done <= 1;\
            end\
        end else begin\
            iter <= iter;\
            done <= done;\
        end\
    end

// Normal loop nest that can connect to a child which is a normal loop
`define NORMAL_LOOP_NEST(iter, width, init, body_done, done, min, max, incr, body_init)\
    reg [width-1:0] iter;\
    reg done;\
    reg body_init;\
    always@(posedge clk) begin\
        if(rst) begin\
            iter <= 0;\
            done <= 1;\
            body_init <= 0;\
        end else if(init) begin\
            iter <= (min);\
            done <= 0;\
            body_init <= 1;\
        end else if(done) begin\
            iter <= 0;\
            done <= 1;\
            body_init <= 0;\
        end else if (body_init) begin\
            iter <= iter;\
            done <= done;\
            body_init <= 0;\
        end else if (body_done) begin\
            if (iter < ((max)-(incr))) begin\
                iter <= iter + (incr);\
                done <= 0;\
                body_init <= 1;\
            end else begin\
                iter <= 0;\
                done <= 1;\
                body_init <= 0;\
            end\
        end else begin\
            iter <= iter;\
            done <= done;\
            body_init <= 0;\
        end\
    end
