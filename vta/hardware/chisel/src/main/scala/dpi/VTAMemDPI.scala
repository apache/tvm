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
import vta.util.config._
import vta.interface.axi._
import vta.shell._

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

class VTAMemDPIToAXI(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val dpi = new VTAMemDPIMaster
    val axi = new AXIClient(p(ShellKey).memParams)
  })
  val opcode = RegInit(false.B)
  val len = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.len)))
  val addr = RegInit(0.U.asTypeOf(chiselTypeOf(io.dpi.req.addr)))
  val sIdle :: sReadAddress :: sReadData :: sWriteAddress :: sWriteData :: sWriteResponse :: Nil = Enum(6)
  val state = RegInit(sIdle)

  switch (state) {
    is (sIdle) {
      when (io.axi.ar.valid) {
        state := sReadAddress
      } .elsewhen (io.axi.aw.valid) {
        state := sWriteAddress
      }
    }
    is (sReadAddress) {
      when (io.axi.ar.valid) {
        state := sReadData
      }
    }
    is (sReadData) {
      when (io.axi.r.ready && io.dpi.rd.valid && len === 0.U) {
        state := sIdle
      }
    }
    is (sWriteAddress) {
      when (io.axi.aw.valid) {
        state := sWriteData
      }
    }
    is (sWriteData) {
      when (io.axi.w.valid && io.axi.w.bits.last) {
        state := sWriteResponse
      }
    }
    is (sWriteResponse) {
      when (io.axi.b.ready) {
        state := sIdle
      }
    }
  }

  when (state === sIdle) {
    when (io.axi.ar.valid) {
      opcode := false.B
      len := io.axi.ar.bits.len
      addr := io.axi.ar.bits.addr
    } .elsewhen (io.axi.aw.valid) {
      opcode := true.B
      len := io.axi.aw.bits.len
      addr := io.axi.aw.bits.addr
    }
  } .elsewhen (state === sReadData) {
    when (io.axi.r.ready && io.dpi.rd.valid && len =/= 0.U) {
      len := len - 1.U
    }
  }

  io.dpi.req.valid := (state === sReadAddress & io.axi.ar.valid) | (state === sWriteAddress & io.axi.aw.valid)
  io.dpi.req.opcode := opcode
  io.dpi.req.len := len
  io.dpi.req.addr := addr

  io.axi.ar.ready := state === sReadAddress
  io.axi.aw.ready := state === sWriteAddress

  io.axi.r.valid := state === sReadData & io.dpi.rd.valid
  io.axi.r.bits.data := io.dpi.rd.bits
  io.axi.r.bits.last := len === 0.U
  io.axi.r.bits.resp := 0.U
  io.axi.r.bits.user := 0.U
  io.axi.r.bits.id := 0.U
  io.dpi.rd.ready := state === sReadData & io.axi.r.ready

  io.dpi.wr.valid := state === sWriteData & io.axi.w.valid
  io.dpi.wr.bits := io.axi.w.bits.data
  io.axi.w.ready := state === sWriteData

  io.axi.b.valid := state === sWriteResponse
  io.axi.b.bits.resp := 0.U
  io.axi.b.bits.user := 0.U
  io.axi.b.bits.id := 0.U

  if (debug) {
    when (state === sReadAddress && io.axi.ar.valid) { printf("[VTAMemDPIToAXI] [AR] addr:%x len:%x\n", addr, len) }
    when (state === sWriteAddress && io.axi.aw.valid) { printf("[VTAMemDPIToAXI] [AW] addr:%x len:%x\n", addr, len) }
    when (io.axi.r.fire()) { printf("[VTAMemDPIToAXI] [R] last:%x data:%x\n", io.axi.r.bits.last, io.axi.r.bits.data) }
    when (io.axi.w.fire()) { printf("[VTAMemDPIToAXI] [W] last:%x data:%x\n", io.axi.w.bits.last, io.axi.w.bits.data) }
  }
}
