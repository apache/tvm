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
  * Bit Slice GEMM:
  *
  * 1. Wait for launch to be asserted
  * 2. Issue 2 read request for 8-byte value at inp1_baddr address and inp2_baddr address
  * 3. Wait for the value
  * 4. Increment read-address for next value
  * 5. Wait for sliced accumulator
  * 6. Check if counter (cnt) is equal to length process,
       otherwise goto step 2
  * 7. Check if reset slice accumulator
  * 8. Wait for overall accumulator
  * 8. Issue a write request for 8-byte value at out_baddr address
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
  val sIdle :: sReadAReq :: sReadAData :: sReadBReq :: sReadBData :: sWriteReq :: sWriteData :: Nil = Enum(7)
  val state = RegInit(sIdle)
  val shift = io.vals(0)
  val length = io.vals(1)
  val rstAccum = io.vals(2)
  val startDot = io.vals(3)
  val cycles = RegInit(0.U(config.regBits.W))
  val reg1 = Reg(chiselTypeOf(io.mem.rd.bits))
  val reg2 = Reg(chiselTypeOf(io.mem.rd.bits))
  val cnt = Reg(UInt(config.regBits.W))
  val raddr1 = Reg(UInt(config.ptrBits.W))
  val raddr2 = Reg(UInt(config.ptrBits.W))
  val waddr = Reg(UInt(config.ptrBits.W))

  switch (state) {
    is (sIdle) {
      when (io.launch) {
        state := sReadAReq
      }
    }
    // Read
    is (sReadAReq) {
      state := sReadAData
    }
    is (sReadAData) {
      when (io.mem.rd.valid) {
        state := sReadBReq
      }
    }
    is (sReadBReq) {
      state := sReadBData
    }
    is (sReadBData) {
      when (io.mem.rd.valid) {
        state := sWriteReq
      }
    }
    // Write
    is (sWriteReq) {
      state := sWriteData
    }
    is (sWriteData) {
      when (cnt === (length - 1.U)) {
        state := sIdle
      } .otherwise {
        state := sReadAReq
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
    raddr1 := io.ptrs(0)
    raddr2 := io.ptrs(1)
    waddr := io.ptrs(2)
  } .elsewhen (state === sWriteData) { // increment input array by 1-byte
    raddr1 := raddr1 + 1.U
    raddr2 := raddr2 + 1.U
    waddr := waddr
  }

  // create request
  io.mem.req.valid := state === sReadAReq | state === sReadBReq | state === sWriteReq
  io.mem.req.opcode := state === sWriteReq
  io.mem.req.len := 0.U // one-word-per-request
  io.mem.req.addr := Mux(state === sReadAReq | state === sReadBReq, Mux(state === sReadAReq, raddr1, raddr2), waddr)

  // read
  when (state === sReadAData && io.mem.rd.valid) {
    reg1 := io.mem.rd.bits(7, 0)
  }

  when (state === sReadBData && io.mem.rd.valid) {
    reg2 := io.mem.rd.bits(7, 0)
  }

  io.mem.rd.ready := state === sReadAData | state === sReadBData

  
  val sliceAccum = Module(new Accumulator(63))
  val overallAccum = Module(new Accumulator(64))

  sliceAccum.io.valid := state === sWriteReq // 2 inputs have been processed 
  sliceAccum.io.in := reg1 * reg2
  sliceAccum.io.clear := startDot
  overallAccum.io.clear := rstAccum
  overallAccum.io.valid := last // last element has been processed
  overallAccum.io.in := sliceAccum.io.sum << shift(7,0) // limit to 8 bits 

  // write
  io.mem.wr.valid := overallAccum.io.ready 
  io.mem.wr.bits := overallAccum.io.sum
  

  // count read/write
  when (state === sIdle) {
    cnt := 0.U
  } .elsewhen (state === sWriteData) {
    cnt := cnt + 1.U
  }

  io.finish := overallAccum.io.ready // data has been added
}


class Accumulator(dataBits: Int = 8) extends Module {
  val io = IO(new Bundle {
    val clear = Input(Bool())
    val valid = Input(Bool())
    val ready = Output(Bool())
    val in = Input(UInt(dataBits.W))
    val sum = Output(UInt((dataBits).W))
  })

  val reg = RegInit(0.U((dataBits).W))
  val ready = RegNext(io.valid)
  when (io.clear) {
    reg := 0.U
  } .elsewhen (io.valid) {
    reg := reg + io.in
  } 
  io.ready := ready
  io.sum := reg
}

