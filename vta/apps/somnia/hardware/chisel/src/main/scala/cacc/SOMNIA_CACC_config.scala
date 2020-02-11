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
 
 //soDLA lisence see https://github.com/soDLA-publishment/soDLA/blob/soDLA_beta/LICENSE.soDLA 

 package somnia

 import chisel3._
 import chisel3.experimental._
 import chisel3.util._

 class caccConfiguration extends cmacConfiguration
 {
     val CACC_PARSUM_WIDTH = 34  //sum result width for one layer operation.
     val CACC_FINAL_WIDTH = 32  //sum result width for one layer operation with saturaton.
     val CACC_IN_WIDTH = PE_MAC_RESULT_WIDTH  //16+log2(atomC),sum result width for one atomic operation.
     val CACC_ATOMK = PE_MAC_ATOMIC_K_SIZE 
     val CACC_ATOMK_LOG2 = PE_MAC_ATOMIC_K_SIZE_LOG2

    //assembly buffer
     val CACC_ABUF_WIDTH = CACC_PARSUM_WIDTH*CACC_ATOMK
     val CACC_ABUF_DEPTH = PE_MAC_ATOMIC_K_SIZE*2 //2*atomK, to store two slots of result.
     val CACC_ABUF_AWIDTH  = PE_MAC_ATOMIC_K_SIZE_LOG2+1   //log2(abuf_depth)

    //delivery buffer
     val CACC_DBUF_WIDTH = CACC_FINAL_WIDTH*CACC_ATOMK
     val CACC_DBUF_DEPTH = CACC_ABUF_DEPTH
     val CACC_DBUF_AWIDTH = CACC_ABUF_AWIDTH    //address width

     val CACC_PPU_DATA_WIDTH = CACC_FINAL_WIDTH * PPU_THROUGHPUT
     val CACC_PPU_WIDTH = CACC_PPU_DATA_WIDTH + 2    //cacc to sdp pd width
     val CACC_DWIDTH_DIV_PWIDTH = CACC_DBUF_WIDTH/CACC_PPU_DATA_WIDTH

     val CACC_CELL_PARTIAL_LATENCY = 2
     val CACC_CELL_FINAL_LATENCY = 2
     val CACC_D_RAM_WRITE_LATENCY = 1

    
 }

 class cacc_dual_reg_flop_outputs extends Bundle{
     val batches = Output(UInt(5.W))
     val clip_truncate = Output(UInt(5.W))
     val cya = Output(UInt(32.W))
     val dataout_addr = Output(UInt(32.W))
     val line_packed = Output(Bool())
     val surf_packed = Output(Bool())
     val dataout_height = Output(UInt(13.W))
     val dataout_width = Output(UInt(13.W))
     val dataout_channel = Output(UInt(13.W))
     val line_stride = Output(UInt(24.W))
     val conv_mode = Output(Bool())
     val proc_precision = Output(UInt(2.W))
     val surf_stride = Output(UInt(24.W))

     val bn_relu_bypass = Output(Bool())
 }