module main();
   parameter PER = 10;
   parameter WIDTH = 8;

   reg clk;
   reg rst;
   // read channels
   reg read_en;
   wire [WIDTH-1:0] read_data;
   wire             read_data_valid;
   // write channels
   reg              write_en;
   reg [WIDTH-1:0]  write_data;
   wire             write_data_ready;
   // controls
   reg              read_req;
   reg [31:0]       read_addr;
   reg [31:0]       read_size;
   reg              write_req;
   reg [31:0]       write_addr;
   reg [31:0]       write_size;


   always begin
      #(PER/2) clk =~ clk;
   end

   tvm_vpi_mem_interface #
     (
      .READ_WIDTH(WIDTH),
      .WRITE_WIDTH(WIDTH),
      .ADDR_WIDTH(32),
      .SIZE_WIDTH(32)
    )
   mem
     (
      .clk(clk),
      .rst(rst),
      .read_en(read_en),
      .read_data_out(read_data),
      .read_data_valid(read_data_valid),
      .write_en(write_en),
      .write_data_in(write_data),
      .write_data_ready(write_data_ready),
      .host_read_req(read_req),
      .host_read_addr(read_addr),
      .host_read_size(read_size),
      .host_write_req(write_req),
      .host_write_addr(write_addr),
      .host_write_size(write_size)
      );

   initial begin
      // pass myram to session to hook it up with simulation
      $tvm_session(clk, mem);
   end
endmodule
