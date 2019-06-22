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

package accel

import chisel3._
import chisel3.util._
import vta.dpi._

/** Register File.
  *
  * Six 32-bit register file.
  *
  * -------------------------------
  *  Register description    | addr
  * -------------------------|-----
  *  Control status register | 0x00
  *  Cycle counter           | 0x04
  *  Constant value          | 0x08
  *  Vector length           | 0x0c
  *  Input pointer lsb       | 0x10
  *  Input pointer msb       | 0x14
  *  Output pointer lsb      | 0x18
  *  Output pointer msb      | 0x1c
  * -------------------------------

  * ------------------------------
  *  Control status register | bit
  * ------------------------------
  *  Launch                  | 0
  *  Finish                  | 1
  * ------------------------------
  */
class RegFile(implicit config: AccelConfig) extends Module {
  val io = IO(new Bundle {
    val launch = Output(Bool())
    val finish = Input(Bool())
    val ecnt = Vec(config.nECnt, Flipped(ValidIO(UInt(config.regBits.W))))
    val vals = Output(Vec(config.nVals, UInt(config.regBits.W)))
    val ptrs = Output(Vec(config.nPtrs, UInt(config.ptrBits.W)))
    val host = new VTAHostDPIClient
  })
  val sIdle :: sRead :: Nil = Enum(2)
  val state = RegInit(sIdle)

  switch (state) {
    is (sIdle) {
      when (io.host.req.valid && !io.host.req.opcode) {
        state := sRead
      }
    }
    is (sRead) {
      state := sIdle
    }
  }

  io.host.req.deq := state === sIdle & io.host.req.valid

  val nTotal = config.nCtrl + config.nECnt + config.nVals + (2*config.nPtrs)
  val reg = Seq.fill(nTotal)(RegInit(0.U.asTypeOf(chiselTypeOf(io.host.req.value))))
  val addr = Seq.tabulate(nTotal)(_ * 4)
  val reg_map = (addr zip reg)  map { case (a, r) => a.U -> r }
  val eo = config.nCtrl
  val vo = eo + config.nECnt
  val po = vo + config.nVals

  when (io.finish) {
    reg(0) := "b_10".U
  } .elsewhen (state === sIdle && io.host.req.valid &&
        io.host.req.opcode && addr(0).U === io.host.req.addr) {
    reg(0) := io.host.req.value
  }

  for (i <- 0 until config.nECnt) {
    when (io.ecnt(i).valid) {
      reg(eo + i) := io.ecnt(i).bits
    } .elsewhen (state === sIdle && io.host.req.valid &&
          io.host.req.opcode && addr(eo + i).U === io.host.req.addr) {
      reg(eo + i) := io.host.req.value
    }
  }

  for (i <- 0 until (config.nVals + (2*config.nPtrs))) {
    when (state === sIdle && io.host.req.valid &&
          io.host.req.opcode && addr(vo + i).U === io.host.req.addr) {
      reg(vo + i) := io.host.req.value
    }
  }

  val rdata = RegInit(0.U.asTypeOf(chiselTypeOf(io.host.req.value)))
  when (state === sIdle && io.host.req.valid && !io.host.req.opcode) {
    rdata := MuxLookup(io.host.req.addr, 0.U, reg_map)
  }

  io.host.resp.valid := state === sRead
  io.host.resp.bits := rdata

  io.launch := reg(0)(0)

  for (i <- 0 until config.nVals) {
    io.vals(i) := reg(vo + i)
  }

  for (i <- 0 until config.nPtrs) {
    io.ptrs(i) := Cat(reg(po + 2*i + 1), reg(po + 2*i))
  }
}
