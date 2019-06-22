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

package vta.core

import chisel3._
import chisel3.util._
import chisel3.experimental._
import vta.util.config._
import scala.math.pow

/** Pipelined multiply and accumulate */
class MAC(dataBits: Int = 8, cBits: Int = 16, outBits: Int = 17) extends Module {
  require (cBits >= dataBits * 2)
  require (outBits >= dataBits * 2)
  val io = IO(new Bundle {
    val a = Input(SInt(dataBits.W))
    val b = Input(SInt(dataBits.W))
    val c = Input(SInt(cBits.W))
    val y = Output(SInt(outBits.W))
  })
  val mult = Wire(SInt(cBits.W))
  val add = Wire(SInt(outBits.W))
  val rA = RegNext(io.a)
  val rB = RegNext(io.b)
  val rC = RegNext(io.c)
  mult := rA * rB
  add := rC + mult
  io.y := add
}

/** Pipelined adder */
class Adder(dataBits: Int = 8, outBits: Int = 17) extends Module {
  require (outBits >= dataBits)
  val io = IO(new Bundle {
    val a = Input(SInt(dataBits.W))
    val b = Input(SInt(dataBits.W))
    val y = Output(SInt(outBits.W))
  })
  val add = Wire(SInt(outBits.W))
  val rA = RegNext(io.a)
  val rB = RegNext(io.b)
  add := rA + rB
  io.y := add
}

/** Pipelined DotProduct based on MAC and Adder */
class DotProduct(dataBits: Int = 8, size: Int = 16) extends Module {
  val errMsg = s"\n\n[VTA] [DotProduct] size must be greater than 4 and a power of 2\n\n"
  require(size >= 4 && isPow2(size), errMsg)
  val b = dataBits * 2
  val outBits = b + log2Ceil(size) + 1
  val io = IO(new Bundle {
    val a = Input(Vec(size, SInt(dataBits.W)))
    val b = Input(Vec(size, SInt(dataBits.W)))
    val y = Output(SInt(outBits.W))
  })
  val p = log2Ceil(size/2)
  val s = Seq.tabulate(log2Ceil(size))(i => pow(2, p - i).toInt)
  val da = Seq.tabulate(s(0))(i => RegNext(io.a(s(0) + i)))
  val db = Seq.tabulate(s(0))(i => RegNext(io.b(s(0) + i)))
  val m = Seq.tabulate(2)(i =>
    Seq.fill(s(0))(Module(new MAC(dataBits = dataBits, cBits = b + i, outBits = b + i + 1)))
  )
  val a = Seq.tabulate(p)(i =>
    Seq.fill(s(i + 1))(Module(new Adder(dataBits = b + i + 2, outBits = b + i + 3)))
  )

  for (i <- 0 until log2Ceil(size)) {
    for (j <- 0 until s(i)) {
      if (i == 0) {
        m(i)(j).io.a := io.a(j)
        m(i)(j).io.b := io.b(j)
        m(i)(j).io.c := 0.S
        m(i + 1)(j).io.a := da(j)
        m(i + 1)(j).io.b := db(j)
        m(i + 1)(j).io.c := m(i)(j).io.y
      } else if (i == 1) {
        a(i - 1)(j).io.a := m(i)(2*j).io.y
        a(i - 1)(j).io.b := m(i)(2*j + 1).io.y
      } else {
        a(i - 1)(j).io.a := a(i - 2)(2*j).io.y
        a(i - 1)(j).io.b := a(i - 2)(2*j + 1).io.y
      }
    }
  }
  io.y := a(p-1)(0).io.y
}

/** Perform matric-vector-multiplication based on DotProduct */
class MatrixVectorCore(implicit p: Parameters) extends Module {
  val accBits = p(CoreKey).accBits
  val size = p(CoreKey).blockOut
  val dataBits = p(CoreKey).inpBits
  val io = IO(new Bundle{
    val reset = Input(Bool()) // FIXME: reset should be replaced by a load-acc instr
    val inp = new TensorMasterData(tensorType = "inp")
    val wgt = new TensorMasterData(tensorType = "wgt")
    val acc_i = new TensorMasterData(tensorType = "acc")
    val acc_o = new TensorClientData(tensorType = "acc")
    val out = new TensorClientData(tensorType = "out")
  })
  val dot = Seq.fill(size)(Module(new DotProduct(dataBits, size)))
  val acc = Seq.fill(size)(Module(new Pipe(UInt(accBits.W), latency = log2Ceil(size) + 1)))
  val add = Seq.fill(size)(Wire(SInt(accBits.W)))
  val vld = Wire(Vec(size, Bool()))

  for (i <- 0 until size) {
    acc(i).io.enq.valid := io.inp.data.valid & io.wgt.data.valid & io.acc_i.data.valid & ~io.reset
    acc(i).io.enq.bits := io.acc_i.data.bits(0)(i)
    for (j <- 0 until size) {
      dot(i).io.a(j) := io.inp.data.bits(0)(j).asSInt
      dot(i).io.b(j) := io.wgt.data.bits(i)(j).asSInt
    }
    add(i) := acc(i).io.deq.bits.asSInt + dot(i).io.y
    io.acc_o.data.bits(0)(i) := Mux(io.reset, 0.U, add(i).asUInt)
    io.out.data.bits(0)(i) := add(i).asUInt
    vld(i) := acc(i).io.deq.valid
  }
  io.acc_o.data.valid := vld.asUInt.andR | io.reset
  io.out.data.valid := vld.asUInt.andR
}

/** TensorGemm.
  *
  * This unit instantiate the MatrixVectorCore and go over the
  * micro-ops (uops) which are used to read inputs, weights and biases,
  * and writes results back to the acc and out scratchpads.
  *
  * Also, the TensorGemm uses the reset field in the Gemm instruction to
  * clear or zero-out the acc-scratchpad locations based on the micro-ops.
  */
class TensorGemm(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
    val uop = new UopMaster
    val inp = new TensorMaster(tensorType = "inp")
    val wgt = new TensorMaster(tensorType = "wgt")
    val acc = new TensorMaster(tensorType = "acc")
    val out = new TensorMaster(tensorType = "out")
  })
  val sIdle :: sReadUop :: sComputeIdx :: sReadTensor :: sExe :: sWait :: Nil = Enum(6)
  val state = RegInit(sIdle)
  val mvc = Module(new MatrixVectorCore)
  val dec = io.inst.asTypeOf(new GemmDecode)
  val uop_idx = Reg(chiselTypeOf(dec.uop_end))
  val uop_end = dec.uop_end
  val uop_acc = Reg(chiselTypeOf(dec.uop_end))
  val uop_inp = Reg(chiselTypeOf(dec.uop_end))
  val uop_wgt = Reg(chiselTypeOf(dec.uop_end))
  val cnt_o = Reg(chiselTypeOf(dec.lp_0))
  val acc_o = Reg(chiselTypeOf(dec.uop_end))
  val inp_o = Reg(chiselTypeOf(dec.uop_end))
  val wgt_o = Reg(chiselTypeOf(dec.uop_end))
  val cnt_i = Reg(chiselTypeOf(dec.lp_1))
  val acc_i = Reg(chiselTypeOf(dec.uop_end))
  val inp_i = Reg(chiselTypeOf(dec.uop_end))
  val wgt_i = Reg(chiselTypeOf(dec.uop_end))
  val pBits = log2Ceil(p(CoreKey).blockOut) + 1
  val inflight = Reg(UInt(pBits.W))
  val wrpipe = Module(new Pipe(chiselTypeOf(dec.uop_end), latency = pBits))
  val done = inflight === 0.U &
             ((state === sExe &
              cnt_o === dec.lp_0 - 1.U &
	      cnt_i === dec.lp_1 - 1.U &
	      uop_idx === uop_end - 1.U &
	      inflight === 0.U) |
	     state === sWait)

  switch (state) {
    is (sIdle) {
      when (io.start) {
        state := sReadUop
      }
    }
    is (sReadUop) {
      state := sComputeIdx
    }
    is (sComputeIdx) {
      state := sReadTensor
    }
    is (sReadTensor) {
      state := sExe
    }
    is (sExe) {
      when ((cnt_o === dec.lp_0 - 1.U) &&
            (cnt_i === dec.lp_1 - 1.U) &&
            (uop_idx === uop_end - 1.U)) {
	when (inflight =/= 0.U) {
          state := sWait
	} .otherwise {
          state := sIdle
	}
      } .otherwise {
        state := sReadUop
      }
    }
    is (sWait) {
      when (inflight === 0.U) {
        state := sIdle
      }
    }
  }

  when (state === sIdle) {
    inflight := 0.U
  } .elsewhen (!dec.reset) {
    when (state === sExe && inflight =/= ((1 << pBits) - 1).asUInt) { // overflow check
      inflight := inflight + 1.U
    } .elsewhen (mvc.io.acc_o.data.valid && inflight =/= 0.U) { // underflow check
      inflight := inflight - 1.U
    }
  }

  when (state === sIdle ||
         (state === sExe &&
	  uop_idx === uop_end - 1.U)) {
    uop_idx := dec.uop_begin
  } .elsewhen (state === sExe) {
    uop_idx := uop_idx + 1.U
  }

  when (state === sIdle) {
    cnt_o := 0.U
    acc_o := 0.U
    inp_o := 0.U
    wgt_o := 0.U
  } .elsewhen (state === sExe &&
	       uop_idx === uop_end - 1.U &&
	       cnt_i === dec.lp_1 - 1.U) {
    cnt_o := cnt_o + 1.U
    acc_o := acc_o + dec.acc_0
    inp_o := inp_o + dec.inp_0
    wgt_o := wgt_o + dec.wgt_0
  }

  when (state === sIdle) {
    cnt_i := 0.U
    acc_i := 0.U
    inp_i := 0.U
    wgt_i := 0.U
  } .elsewhen (state === sReadUop && cnt_i === dec.lp_1) {
    cnt_i := 0.U
    acc_i := acc_o
    inp_i := inp_o
    wgt_i := wgt_o
  } .elsewhen (state === sExe &&
	       uop_idx === uop_end - 1.U) {
    cnt_i := cnt_i + 1.U
    acc_i := acc_i + dec.acc_1
    inp_i := inp_i + dec.inp_1
    wgt_i := wgt_i + dec.wgt_1
  }

  when (state === sComputeIdx && io.uop.data.valid) {
    uop_acc := io.uop.data.bits.u0 + acc_i
    uop_inp := io.uop.data.bits.u1 + inp_i
    uop_wgt := io.uop.data.bits.u2 + wgt_i
  }

  wrpipe.io.enq.valid := state === sExe & ~dec.reset
  wrpipe.io.enq.bits := uop_acc

  // uop
  io.uop.idx.valid := state === sReadUop
  io.uop.idx.bits := uop_idx

  // inp
  io.inp.rd.idx.valid := state === sReadTensor
  io.inp.rd.idx.bits := uop_inp
  io.inp.tieoffWrite() // read-only

  // wgt
  io.wgt.rd.idx.valid := state === sReadTensor
  io.wgt.rd.idx.bits := uop_wgt
  io.wgt.tieoffWrite() // read-only

  // acc_i
  io.acc.rd.idx.valid := state === sReadTensor
  io.acc.rd.idx.bits := uop_acc

  // mvc
  mvc.io.reset := dec.reset & state === sExe
  mvc.io.inp.data <> io.inp.rd.data
  mvc.io.wgt.data <> io.wgt.rd.data
  mvc.io.acc_i.data <> io.acc.rd.data

  // acc_o
  io.acc.wr.valid := mvc.io.acc_o.data.valid & Mux(dec.reset, true.B, wrpipe.io.deq.valid)
  io.acc.wr.bits.idx := Mux(dec.reset, uop_acc, wrpipe.io.deq.bits)
  io.acc.wr.bits.data <> mvc.io.acc_o.data.bits

  // out
  io.out.wr.valid := mvc.io.out.data.valid & wrpipe.io.deq.valid
  io.out.wr.bits.idx := wrpipe.io.deq.bits
  io.out.wr.bits.data <> mvc.io.out.data.bits
  io.out.tieoffRead() // write-only

  io.done := done

  if (debug) {
    when (state === sReadUop && ~dec.reset) {
      printf("[TensorGemm] [uop] idx:%x\n", uop_idx)
    }

    when (state === sReadTensor && ~dec.reset) {
      printf("[TensorGemm] [uop] acc:%x inp:%x wgt:%x\n", uop_acc, uop_inp, uop_wgt)
    }

    io.inp.rd.data.bits.zipWithIndex.foreach { case(r, i) =>
      when (io.inp.rd.data.valid && ~dec.reset) {
        printf("[TensorGemm] [inp] i:%x val:%x\n", i.U, r.asUInt)
      }
    }

    io.wgt.rd.data.bits.zipWithIndex.foreach { case(r, i) =>
      when (io.wgt.rd.data.valid && ~dec.reset) {
        printf("[TensorGemm] [wgt] i:%x val:%x\n", i.U, r.asUInt)
      }
    }

    io.acc.rd.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (io.acc.rd.data.valid && ~dec.reset) {
          printf("[TensorGemm] [acc_i] i:%x val:%x\n", i.U, elem)
        }
      }
    }

    mvc.io.acc_o.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (mvc.io.acc_o.data.valid && ~dec.reset) {
          printf("[TensorGemm] [acc_o] i:%x val:%x\n", i.U, elem)
        }
      }
    }

    mvc.io.out.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (mvc.io.out.data.valid && ~dec.reset) {
          printf("[TensorGemm] [out] i:%x val:%x\n", i.U, elem)
        }
      }
    }
  }
}
