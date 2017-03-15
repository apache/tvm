module main();
   parameter PER = 10;
   parameter DATA_WIDTH = 8;
   parameter ADDR_WIDTH = 8;
   reg clk;
   reg rst;
   // read channels
   reg [ADDR_WIDTH-1:0] read_addr;
   wire [DATA_WIDTH-1:0] read_data;
   // write channels
   reg [ADDR_WIDTH-1:0]  write_addr;
   reg [DATA_WIDTH-1:0]  write_data;
   reg                   write_en;
   // mmap base
   reg [31:0]            mmap_addr;

   always begin
      #(PER/2) clk =~ clk;
   end

   tvm_vpi_read_mmap #
     (
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH)
      )
   rmmap
     (
      .clk(clk),
      .rst(rst),
      .addr(read_addr),
      .data_out(read_data),
      .mmap_addr(mmap_addr)
      );

   tvm_vpi_write_mmap #
     (
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH)
    )
   wmmap
     (
      .clk(clk),
      .rst(rst),
      .addr(write_addr),
      .data_in(write_data),
      .en(write_en),
      .mmap_addr(mmap_addr)
      );

   initial begin
      $tvm_session(clk, rmmap, wmmap);
   end
endmodule
