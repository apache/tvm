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
import vta.util.config._

/** ALU datapath */
class Alu(implicit p: Parameters) extends Module {
  val aluBits = p(CoreKey).accBits
  val io = IO(new Bundle {
    val opcode = Input(UInt(C_ALU_OP_BITS.W))
    val a = Input(SInt(aluBits.W))
    val b = Input(SInt(aluBits.W))
    val y = Output(SInt(aluBits.W))
  })

  // FIXME: the following three will change once we support properly SHR and SHL
  val ub = io.b.asUInt
  val width = log2Ceil(aluBits)
  val m = ~ub(width - 1, 0) + 1.U

  val n = ub(width - 1, 0)
  val fop = Seq(Mux(io.a < io.b, io.a, io.b),
                Mux(io.a < io.b, io.b, io.a),
                io.a + io.b,
                io.a >> n,
	        io.a << m)

  val opmux = Seq.tabulate(ALU_OP_NUM)(i => ALU_OP(i) -> fop(i))
  io.y := MuxLookup(io.opcode, io.a, opmux)
}

/** Pipelined ALU */
class AluReg(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val opcode = Input(UInt(C_ALU_OP_BITS.W))
    val a = Flipped(ValidIO(UInt(p(CoreKey).accBits.W)))
    val b = Flipped(ValidIO(UInt(p(CoreKey).accBits.W)))
    val y = ValidIO(UInt(p(CoreKey).accBits.W))
  })
  val alu = Module(new Alu)
  val rA = RegEnable(io.a.bits, io.a.valid)
  val rB = RegEnable(io.b.bits, io.b.valid)
  val valid = RegNext(io.b.valid)

  alu.io.opcode := io.opcode

  // register input
  alu.io.a := rA.asSInt
  alu.io.b := rB.asSInt

  // output
  io.y.valid := valid
  io.y.bits := alu.io.y.asUInt
}

/** Vector of pipeline ALUs */
class AluVector(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val opcode = Input(UInt(C_ALU_OP_BITS.W))
    val acc_a = new TensorMasterData(tensorType = "acc")
    val acc_b = new TensorMasterData(tensorType = "acc")
    val acc_y = new TensorClientData(tensorType = "acc")
    val out = new TensorClientData(tensorType = "out")
  })
  val blockOut = p(CoreKey).blockOut
  val f = Seq.fill(blockOut)(Module(new AluReg))
  val valid = Wire(Vec(blockOut, Bool()))
  for (i <- 0 until blockOut) {
    f(i).io.opcode := io.opcode
    f(i).io.a.valid := io.acc_a.data.valid
    f(i).io.a.bits := io.acc_a.data.bits(0)(i)
    f(i).io.b.valid := io.acc_b.data.valid
    f(i).io.b.bits := io.acc_b.data.bits(0)(i)
    valid(i) := f(i).io.y.valid
    io.acc_y.data.bits(0)(i) := f(i).io.y.bits
    io.out.data.bits(0)(i) := f(i).io.y.bits
  }
  io.acc_y.data.valid := valid.asUInt.andR
  io.out.data.valid := valid.asUInt.andR
}

/** TensorAlu.
  *
  * This unit instantiate the ALU vector unit (AluVector) and go over the
  * micro-ops (uops) which are used to read the source operands (vectors)
  * from the acc-scratchpad and then they are written back the same
  * acc-scratchpad.
  */
class TensorAlu(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
    val uop = new UopMaster
    val acc = new TensorMaster(tensorType = "acc")
    val out = new TensorMaster(tensorType = "out")
  })
  val sIdle :: sReadUop :: sComputeIdx :: sReadTensorA :: sReadTensorB :: sExe :: Nil = Enum(6)
  val state = RegInit(sIdle)
  val alu = Module(new AluVector)
  val dec = io.inst.asTypeOf(new AluDecode)
  val uop_idx = Reg(chiselTypeOf(dec.uop_end))
  val uop_end = dec.uop_end
  val uop_dst = Reg(chiselTypeOf(dec.uop_end))
  val uop_src = Reg(chiselTypeOf(dec.uop_end))
  val cnt_o = Reg(chiselTypeOf(dec.lp_0))
  val dst_o = Reg(chiselTypeOf(dec.uop_end))
  val src_o = Reg(chiselTypeOf(dec.uop_end))
  val cnt_i = Reg(chiselTypeOf(dec.lp_1))
  val dst_i = Reg(chiselTypeOf(dec.uop_end))
  val src_i = Reg(chiselTypeOf(dec.uop_end))
  val done =
    state === sExe &
    alu.io.out.data.valid &
    (cnt_o === dec.lp_0 - 1.U) &
    (cnt_i === dec.lp_1 - 1.U) &
    (uop_idx === uop_end - 1.U)

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
      state := sReadTensorA
    }
    is (sReadTensorA) {
      state := sReadTensorB
    }
    is (sReadTensorB) {
      state := sExe
    }
    is (sExe) {
      when (alu.io.out.data.valid) {
        when ((cnt_o === dec.lp_0 - 1.U) &&
	      (cnt_i === dec.lp_1 - 1.U) &&
	      (uop_idx === uop_end - 1.U)) {
          state := sIdle
        } .otherwise {
          state := sReadUop
        }
      }
    }
  }

  when (state === sIdle ||
         (state === sExe &&
	  alu.io.out.data.valid &&
	  uop_idx === uop_end - 1.U)) {
    uop_idx := dec.uop_begin
  } .elsewhen (state === sExe && alu.io.out.data.valid) {
    uop_idx := uop_idx + 1.U
  }

  when (state === sIdle) {
    cnt_o := 0.U
    dst_o := 0.U
    src_o := 0.U
  } .elsewhen (state === sExe &&
               alu.io.out.data.valid &&
               uop_idx === uop_end - 1.U &&
	       cnt_i === dec.lp_1 - 1.U) {
    cnt_o := cnt_o + 1.U
    dst_o := dst_o + dec.dst_0
    src_o := src_o + dec.src_0
  }

  when (state === sIdle) {
    cnt_i := 0.U
    dst_i := 0.U
    src_i := 0.U
  } .elsewhen (state === sReadUop && cnt_i === dec.lp_1) {
    cnt_i := 0.U
    dst_i := dst_o
    src_i := src_o
  } .elsewhen (state === sExe &&
               alu.io.out.data.valid &&
	       uop_idx === uop_end - 1.U) {
    cnt_i := cnt_i + 1.U
    dst_i := dst_i + dec.dst_1
    src_i := src_i + dec.src_1
  }

  when (state === sComputeIdx && io.uop.data.valid) {
    uop_dst := io.uop.data.bits.u0 + dst_i
    uop_src := io.uop.data.bits.u1 + src_i
  }

  // uop
  io.uop.idx.valid := state === sReadUop
  io.uop.idx.bits := uop_idx

  // acc_i
  io.acc.rd.idx.valid := state === sReadTensorA | (state === sReadTensorB & ~dec.alu_use_imm)
  io.acc.rd.idx.bits := Mux(state === sReadTensorA, uop_dst, uop_src)

  // imm
  val tensorImm = Wire(new TensorClientData(tensorType = "acc"))
  tensorImm.data.valid := state === sReadTensorB
  tensorImm.data.bits.foreach { b => b.foreach { c => c := dec.alu_imm } }

  // alu
  val isSHR = dec.alu_op === ALU_OP(3)
  val neg_shift = isSHR & dec.alu_imm(C_ALU_IMM_BITS-1)
  val fixme_alu_op = Cat(neg_shift, Mux(neg_shift, 0.U, dec.alu_op))
  alu.io.opcode := fixme_alu_op
  alu.io.acc_a.data.valid := io.acc.rd.data.valid & state === sReadTensorB
  alu.io.acc_a.data.bits <> io.acc.rd.data.bits
  alu.io.acc_b.data.valid := Mux(dec.alu_use_imm, tensorImm.data.valid, io.acc.rd.data.valid & state === sExe)
  alu.io.acc_b.data.bits <> Mux(dec.alu_use_imm, tensorImm.data.bits, io.acc.rd.data.bits)

  // acc_o
  io.acc.wr.valid := alu.io.acc_y.data.valid
  io.acc.wr.bits.idx := uop_dst
  io.acc.wr.bits.data <> alu.io.acc_y.data.bits

  // out
  io.out.wr.valid := alu.io.out.data.valid
  io.out.wr.bits.idx := uop_dst
  io.out.wr.bits.data <> alu.io.out.data.bits
  io.out.tieoffRead() // write-only

  io.done := done

  if (debug) {

    when (state === sReadUop) {
      printf("[TensorAlu] [uop] idx:%x\n", uop_idx)
    }

    when (state === sReadTensorA) {
      printf("[TensorAlu] [uop] dst:%x src:%x\n", uop_dst, uop_src)
    }

    when (state === sIdle && io.start) {
      printf(p"[TensorAlu] decode:$dec\n")
    }

    alu.io.acc_a.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (alu.io.acc_a.data.valid) {
          printf("[TensorAlu] [a] i:%x val:%x\n", i.U, elem)
        }
      }
    }

    alu.io.acc_b.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (alu.io.acc_b.data.valid) {
          printf("[TensorAlu] [b] i:%x val:%x\n", i.U, elem)
        }
      }
    }

    alu.io.acc_y.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (alu.io.acc_y.data.valid) {
          printf("[TensorAlu] [y] i:%x val:%x\n", i.U, elem)
        }
      }
    }

    alu.io.out.data.bits.foreach { tensor =>
      tensor.zipWithIndex.foreach { case(elem, i) =>
        when (alu.io.out.data.valid) {
          printf("[TensorAlu] [out] i:%x val:%x\n", i.U, elem)
        }
      }
    }
  }
}
