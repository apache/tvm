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

/** Host DPI parameters */
trait VTAHostDPIParams {
  val dpiAddrBits = 8
  val dpiDataBits = 32
}

/** Host master interface.
  *
  * This interface is tipically used by the Host
  */
class VTAHostDPIMaster extends Bundle with VTAHostDPIParams {
  val req = new Bundle {
    val valid = Output(Bool())
    val opcode = Output(Bool())
    val addr = Output(UInt(dpiAddrBits.W))
    val value = Output(UInt(dpiDataBits.W))
    val deq = Input(Bool())
  }
  val resp = Flipped(ValidIO(UInt(dpiDataBits.W)))
}

/** Host client interface.
  *
  * This interface is tipically used by the Accelerator
  */
class VTAHostDPIClient extends Bundle with VTAHostDPIParams {
  val req = new Bundle {
    val valid = Input(Bool())
    val opcode = Input(Bool())
    val addr = Input(UInt(dpiAddrBits.W))
    val value = Input(UInt(dpiDataBits.W))
    val deq = Output(Bool())
  }
  val resp = ValidIO(UInt(dpiDataBits.W))
}

/** Host DPI module.
  *
  * Wrapper for Host Verilog DPI module.
  */
class VTAHostDPI extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val dpi = new VTAHostDPIMaster
  })
  setResource("/verilog/VTAHostDPI.v")
}
