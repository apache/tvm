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
import chisel3.util._
import chisel3.experimental._


//  flow valid
class cmac2cacc_if(implicit val conf: somniaConfig) extends Bundle{
    val mask = Output(Vec(conf.CMAC_ATOMK, Bool()))
    val data = Output(Vec(conf.CMAC_ATOMK, UInt(conf.CMAC_RESULT_WIDTH.W)))
    //val mode = Output(Bool())
//pd
//   field batch_index 5
//   field stripe_st 1
//   field stripe_end 1
//   field channel_end 1
//   field layer_end 1
    val pd = Output(UInt(9.W))
}

class csb2dp_if extends Bundle{
    val req = Flipped(ValidIO(UInt(63.W)))
    val resp = ValidIO(UInt(34.W))
}

class somnia_clock_if extends Bundle{
    val somnia_core_clk = Output(Clock())
    val dla_clk_ovr_on_sync = Output(Clock())
    val global_clk_ovr_on_sync = Output(Clock())
    val tmc2slcg_disable_clock_gating = Output(Bool())
}

// Register control interface
class reg_control_if extends Bundle{
    val rd_data = Output(UInt(32.W))
    val offset = Input(UInt(12.W))
    val wr_data = Input(UInt(32.W))
    val wr_en = Input(Bool())
}


class nvdla_wr_if(addr_width:Int, width:Int) extends Bundle{
    val addr = ValidIO(UInt(addr_width.W))
    val data = Output(UInt(width.W))

    override def cloneType: this.type =
    new nvdla_wr_if(addr_width:Int, width:Int).asInstanceOf[this.type]
}

class nvdla_rd_if(addr_width:Int, width:Int) extends Bundle{
    val addr = ValidIO(UInt(addr_width.W))
    val data = Input(UInt(width.W))

    override def cloneType: this.type =
    new nvdla_rd_if(addr_width:Int, width:Int).asInstanceOf[this.type]
}