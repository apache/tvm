/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

module VTAMemDPI #
( parameter LEN_BITS = 8,
  parameter ADDR_BITS = 64,
  parameter DATA_BITS = 64
)
(
  input                        clock,
  input                        reset,
  input                        dpi_req_valid,
  input                        dpi_req_opcode,
  input         [LEN_BITS-1:0] dpi_req_len,
  input        [ADDR_BITS-1:0] dpi_req_addr,
  input                        dpi_wr_valid,
  input        [DATA_BITS-1:0] dpi_wr_bits,
  output logic                 dpi_rd_valid,
  output logic [DATA_BITS-1:0] dpi_rd_bits,
  input                        dpi_rd_ready
);

  import "DPI-C" function void VTAMemDPI
  (
    input  byte     unsigned req_valid,
    input  byte     unsigned req_opcode,
    input  byte     unsigned req_len,
    input  longint  unsigned req_addr,
    input  byte     unsigned wr_valid,
    input  longint  unsigned wr_value,
    output byte     unsigned rd_valid,
    output longint  unsigned rd_value,
    input  byte     unsigned rd_ready
  );

  typedef logic        dpi1_t;
  typedef logic  [7:0] dpi8_t;
  typedef logic [31:0] dpi32_t;
  typedef logic [63:0] dpi64_t;

  dpi1_t  __reset;
  dpi8_t  __req_valid;
  dpi8_t  __req_opcode;
  dpi8_t  __req_len;
  dpi64_t __req_addr;
  dpi8_t  __wr_valid;
  dpi64_t __wr_value;
  dpi8_t  __rd_valid;
  dpi64_t __rd_value;
  dpi8_t  __rd_ready;

  always_ff @(posedge clock) begin
    __reset <= reset;
  end

  // delaying outputs by one-cycle
  // since verilator does not support delays
  always_ff @(posedge clock) begin
    dpi_rd_valid <= dpi1_t ' (__rd_valid);
    dpi_rd_bits  <= __rd_value;
  end

  assign __req_valid  = dpi8_t ' (dpi_req_valid);
  assign __req_opcode = dpi8_t ' (dpi_req_opcode);
  assign __req_len    = dpi_req_len;
  assign __req_addr   = dpi_req_addr;
  assign __wr_valid   = dpi8_t ' (dpi_wr_valid);
  assign __wr_value   = dpi_wr_bits;
  assign __rd_ready   = dpi8_t ' (dpi_rd_ready);

  // evaluate DPI function
  always_ff @(posedge clock) begin
    if (reset | __reset) begin
      __rd_valid = 0;
      __rd_value = 0;
    end
    else begin
      VTAMemDPI(
        __req_valid,
        __req_opcode,
        __req_len,
        __req_addr,
        __wr_valid,
        __wr_value,
        __rd_valid,
        __rd_value,
        __rd_ready);
    end
  end
endmodule
