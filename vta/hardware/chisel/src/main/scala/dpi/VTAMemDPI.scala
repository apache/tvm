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

package vta.dpi

import chisel3._
import chisel3.util._

/** Memory DPI parameters */
trait VTAMemDPIParams {
  val dpiLenBits = 8
  val dpiAddrBits = 64
  val dpiDataBits = 64
}

/** Memory master interface.
  *
  * This interface is tipically used by the Accelerator
  */
class VTAMemDPIMaster extends Bundle with VTAMemDPIParams {
  val req = new Bundle {
    val valid = Output(Bool())
    val opcode = Output(Bool())
    val len = Output(UInt(dpiLenBits.W))
    val addr = Output(UInt(dpiAddrBits.W))
  }
  val wr = ValidIO(UInt(dpiDataBits.W))
  val rd = Flipped(Decoupled(UInt(dpiDataBits.W)))
}

/** Memory client interface.
  *
  * This interface is tipically used by the Host
  */
class VTAMemDPIClient extends Bundle with VTAMemDPIParams {
  val req = new Bundle {
    val valid = Input(Bool())
    val opcode = Input(Bool())
    val len = Input(UInt(dpiLenBits.W))
    val addr = Input(UInt(dpiAddrBits.W))
  }
  val wr = Flipped(ValidIO(UInt(dpiDataBits.W)))
  val rd = Decoupled(UInt(dpiDataBits.W))
}

/** Memory DPI module.
  *
  * Wrapper for Memory Verilog DPI module.
  */
class VTAMemDPI extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val dpi = new VTAMemDPIClient
  })
  setResource("/verilog/VTAMemDPI.v")
}
