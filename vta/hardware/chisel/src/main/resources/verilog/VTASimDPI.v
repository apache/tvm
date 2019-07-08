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

module VTASimDPI
(
  input                        clock,
  input                        reset,
  output logic                 dpi_wait
);

  import "DPI-C" function void VTASimDPI
  (
    output byte unsigned sim_wait,
    output byte unsigned sim_exit
  );

  typedef logic        dpi1_t;
  typedef logic  [7:0] dpi8_t;

  dpi1_t __reset;
  dpi8_t __wait;
  dpi8_t __exit;

  // reset
  always_ff @(posedge clock) begin
    __reset <= reset;
  end

  // evaluate DPI function
  always_ff @(posedge clock) begin
    if (reset | __reset) begin
      __wait = 0;
      __exit = 0;
    end
    else begin
      VTASimDPI(
        __wait,
	__exit);
    end
  end

  logic wait_reg;

  always_ff @(posedge clock) begin
    if (reset | __reset) begin
      wait_reg <= 1'b0;
    end else if (__wait == 1) begin
      wait_reg <= 1'b1;
    end else begin
      wait_reg <= 1'b0;
    end
  end

  assign dpi_wait = wait_reg;

  always_ff @(posedge clock) begin
    if (__exit == 1) begin
      $finish;
    end
  end

endmodule
