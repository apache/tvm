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

/** Compute
  *
  * Add-by-one procedure:
  *
  * 1. Wait for launch to be asserted
  * 2. Issue a read request for 8-byte value at inp_baddr address
  * 3. Wait for the value
  * 4. Issue a write request for 8-byte value at out_baddr address
  * 5. Increment read-address and write-address for next value
  * 6. Check if counter (cnt) is equal to length to assert finish,
  *    otherwise go to step 2.
  */
class Compute(implicit config: AccelConfig) extends Module {
  val io = IO(new Bundle {
    val launch = Input(Bool())
    val finish = Output(Bool())
    val ecnt = Vec(config.nECnt, ValidIO(UInt(config.regBits.W)))
    val vals = Input(Vec(config.nVals, UInt(config.regBits.W)))
    val ptrs = Input(Vec(config.nPtrs, UInt(config.ptrBits.W)))
    val mem = new VTAMemDPIMaster
  })
  val sIdle :: sReadReq :: sReadData :: sWriteReq :: sWriteData :: Nil = Enum(5)
  val state = RegInit(sIdle)
  val const = io.vals(0)
  val length = io.vals(1)
  val cycles = RegInit(0.U(config.regBits.W))
  val reg = Reg(chiselTypeOf(io.mem.rd.bits))
  val cnt = Reg(UInt(config.regBits.W))
  val raddr = Reg(UInt(config.ptrBits.W))
  val waddr = Reg(UInt(config.ptrBits.W))

  switch (state) {
    is (sIdle) {
      when (io.launch) {
        state := sReadReq
      }
    }
    is (sReadReq) {
      state := sReadData
    }
    is (sReadData) {
      when (io.mem.rd.valid) {
        state := sWriteReq
      }
    }
    is (sWriteReq) {
      state := sWriteData
    }
    is (sWriteData) {
      when (cnt === (length - 1.U)) {
        state := sIdle
      } .otherwise {
        state := sReadReq
      }
    }
  }

  val last = state === sWriteData && cnt === (length - 1.U)

  // cycle counter
  when (state === sIdle) {
    cycles := 0.U
  } .otherwise {
    cycles := cycles + 1.U
  }

  io.ecnt(0).valid := last
  io.ecnt(0).bits := cycles

  // calculate next address
  when (state === sIdle) {
    raddr := io.ptrs(0)
    waddr := io.ptrs(1)
  } .elsewhen (state === sWriteData) { // increment by 8-bytes
    raddr := raddr + 8.U
    waddr := waddr + 8.U
  }

  // create request
  io.mem.req.valid := state === sReadReq | state === sWriteReq
  io.mem.req.opcode := state === sWriteReq
  io.mem.req.len := 0.U // one-word-per-request
  io.mem.req.addr := Mux(state === sReadReq, raddr, waddr)

  // read
  when (state === sReadData && io.mem.rd.valid) {
    reg := io.mem.rd.bits + const
  }
  io.mem.rd.ready := state === sReadData

  // write
  io.mem.wr.valid := state === sWriteData
  io.mem.wr.bits := reg

  // count read/write
  when (state === sIdle) {
    cnt := 0.U
  } .elsewhen (state === sWriteData) {
    cnt := cnt + 1.U
  }

  // done when read/write are equal to length
  io.finish := last
}
