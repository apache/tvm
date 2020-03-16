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
import vta.core._
import vta.util.config._
import vta.shell._

class TestConfig extends Config(new CoreConfig ++ new PynqConfig)
/** Compute
  *
  * Bit Slice GEMM:
  *
  * 1. Wait for launch to be asserted
  * 2. Issue 1 read request for 8-bit value at inp1_baddr address (read matrix)
  * 3. Wait for the value
  * 4. Increment read-address for next value
  * 5. Repeat until all inp1 data have been read

  * 6. Issue 1 read request for 8-bit value at inp2_baddr address (read vector)
  * 7. Wait for the value
  * 8. Increment read-address for next value
  * 9. Repeat until all inp2 data have been read

  * 10. Wait for output to be calculated
  * 11. Issue a write request for 8-byte value at out_baddr address
  * 12. Increment write-address for next value to write
  * 13. Check if counter (cntout) is equal to length to asser finish,
       otherwise go to step 11
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
  implicit val p: Parameters = new TestConfig
  val sIdle :: sReadAReq :: sReadAData :: sReadADone ::sReadBReq :: sReadBData :: sReadBDone :: sInpDone ::sWait:: sWriteReq :: sWriteData :: sWriteDone :: Nil = Enum(12)
  val state = RegInit(sIdle)
  val shift = io.vals(0)
  val length = io.vals(1)
  val rstAccum = io.vals(2)
  val startDot = io.vals(3)
  val cycles = RegInit(0.U(config.regBits.W))
  val mvc = Module(new MatrixVectorMultiplication)
  val reg1 = Reg(chiselTypeOf(mvc.io.wgt.data.bits))
  val reg2 = Reg(chiselTypeOf(mvc.io.inp.data.bits))
  val cntwgt = Reg(UInt(config.regBits.W))
  val cntinp = Reg(UInt(config.regBits.W))
  val cntout = Reg(UInt(config.regBits.W))
  val raddr1 = Reg(UInt(config.ptrBits.W))
  val raddr2 = Reg(UInt(config.ptrBits.W))
  val waddr = Reg(UInt(config.ptrBits.W))
  val accum = Module(new Accmulator(size = p(CoreKey).blockOut, accBits = p(CoreKey).accBits))

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
        state := sReadADone
      }
    }
    is (sReadADone) {
      when (cntwgt === (length * length) - 1.U) {
        state := sReadBReq
      } .otherwise {
        state := sReadAReq
      }
    }
    is (sReadBReq) {
      state := sReadBData
    }
    is (sReadBData) {
      when (io.mem.rd.valid) {
        state := sReadBDone
      }
    }
    is (sReadBDone) {
      when (cntinp === length-1.U) {
        state := sInpDone
      } .otherwise {
        state := sReadBReq
      }
    }
    // Both input is processed
    is (sInpDone) {
      state := sWait
    }
    // Wait for computation
    is (sWait) {
      when (accum.io.ready) {
        state := sWriteReq
      }
    }
    // Write
    is (sWriteReq) {
      state := sWriteData
    }
    is (sWriteData) {
        state := sWriteDone
    }
    is (sWriteDone) {
      when (cntout === (length - 1.U)) {
        state := sIdle
      } .otherwise {
        state := sWriteReq
      }
    }
  }

  val last = state === sWriteDone && cntout === (length - 1.U)

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
  } .elsewhen (state === sReadADone) { // increment input array by 1-byte
    raddr1 := raddr1 + 1.U
  } .elsewhen (state === sReadBDone) { // increment input array by 1-byte
    raddr2 := raddr2 + 1.U
  } .elsewhen (state === sWriteDone) {
    waddr := waddr + 4.U // writing 4 bytes
  }

  // create request
  io.mem.req.valid := state === sReadAReq | state === sReadBReq | state === sWriteReq
  io.mem.req.opcode := state === sWriteReq
  io.mem.req.len := 0.U // one-word-per-request
  io.mem.req.addr := Mux(state === sReadAReq | state === sReadBReq, Mux(state === sReadAReq, raddr1, raddr2), waddr)

  // read
  when (state === sReadAData && io.mem.rd.valid) {
    reg1(cntwgt/length)(cntwgt%length) := io.mem.rd.bits(7, 0)
  }

  when (state === sReadBData && io.mem.rd.valid) {
    reg2(0)(cntinp) := io.mem.rd.bits(7, 0)
  }

  io.mem.rd.ready := state === sReadAData | state === sReadBData
  mvc.io.inp.data.valid := state === sInpDone // 2 inputs have been processed
  mvc.io.wgt.data.valid := state === sInpDone // 2 inputs have been processed

  mvc.io.wgt.data.bits <> reg1
  mvc.io.inp.data.bits <> reg2
  // Modify when shift operation is supported
  mvc.io.reset := false.B
  mvc.io.acc_i.data.valid := true.B
  for (i <- 0 until p(CoreKey).blockOut) {
    mvc.io.acc_i.data.bits(0)(i) := 0.U
  }

  accum.io.in := mvc.io.acc_o.data.bits
  accum.io.shift := shift
  accum.io.clear := rstAccum
  accum.io.valid := mvc.io.acc_o.data.valid

  // write
  io.mem.wr.valid := state === sWriteData
  io.mem.wr.bits := accum.io.sum(cntout)

  // count read/write
  when (state === sIdle) {
    cntwgt := 0.U
    cntinp := 0.U
    cntout := 0.U
  } .elsewhen (state === sReadADone) {
    cntwgt := cntwgt + 1.U
  } .elsewhen (state === sReadBDone) {
    cntinp := cntinp + 1.U
  } .elsewhen (state === sWriteDone) {
    cntout := cntout + 1.U
  }

  io.finish := last // data has been added
}
// Shift operation until supported in MVM
class Accmulator(size: Int = 16, accBits: Int = 32) extends Module {
  val io = IO(new Bundle {
    val clear = Input(Bool())
    val valid = Input(Bool())
    val ready = Output(Bool())
    val in = Input(Vec(1, Vec(size, (UInt(accBits.W)))))
    val shift = Input(UInt(8.W))
    val sum = Output(Vec(size, (UInt(accBits.W))))
  })
    val reg = RegInit(VecInit(Seq.fill(size)(0.U(accBits.W))))

    for (i <- 0 until size) {
      when (io.clear) {
        reg(i) := 0.U
      } .elsewhen(io.valid) {
        reg(i) := reg(i) + (io.in(0)(i) << io.shift)
      }
    }
    io.ready := RegNext(io.valid)
    io.sum := reg
}

