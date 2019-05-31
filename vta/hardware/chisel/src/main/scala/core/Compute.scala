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
import vta.shell._

/** Compute.
  *
  * The compute unit is in charge of the following:
  * - Loading micro-ops from memory (loadUop module)
  * - Loading biases (acc) from memory (tensorAcc module)
  * - Compute ALU instructions (tensorAlu module)
  * - Compute GEMM instructions (tensorGemm module)
  */
class Compute(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val i_post = Vec(2, Input(Bool()))
    val o_post = Vec(2, Output(Bool()))
    val inst = Flipped(Decoupled(UInt(INST_BITS.W)))
    val uop_baddr = Input(UInt(mp.addrBits.W))
    val acc_baddr = Input(UInt(mp.addrBits.W))
    val vme_rd = Vec(2, new VMEReadMaster)
    val inp = new TensorMaster(tensorType = "inp")
    val wgt = new TensorMaster(tensorType = "wgt")
    val out = new TensorMaster(tensorType = "out")
    val finish = Output(Bool())
  })
  val sIdle :: sSync :: sExe :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val s = Seq.tabulate(2)(_ => Module(new Semaphore(counterBits = 8, counterInitValue = 0)))

  val loadUop = Module(new LoadUop)
  val tensorAcc = Module(new TensorLoad(tensorType = "acc"))
  val tensorGemm = Module(new TensorGemm)
  val tensorAlu = Module(new TensorAlu)

  val inst_q = Module(new Queue(UInt(INST_BITS.W), p(CoreKey).instQueueEntries))

  // decode
  val dec = Module(new ComputeDecode)
  dec.io.inst := inst_q.io.deq.bits

  val inst_type = Cat(dec.io.isFinish,
                      dec.io.isAlu,
		      dec.io.isGemm,
		      dec.io.isLoadAcc,
		      dec.io.isLoadUop).asUInt

  val sprev = inst_q.io.deq.valid & Mux(dec.io.pop_prev, s(0).io.sready, true.B)
  val snext = inst_q.io.deq.valid & Mux(dec.io.pop_next, s(1).io.sready, true.B)
  val start = snext & sprev
  val done =
    MuxLookup(inst_type,
               false.B, // default
      Array(
        "h_01".U -> loadUop.io.done,
        "h_02".U -> tensorAcc.io.done,
        "h_04".U -> tensorGemm.io.done,
        "h_08".U -> tensorAlu.io.done,
        "h_10".U -> true.B // Finish
      )
    )

  // control
  switch (state) {
    is (sIdle) {
      when (start) {
	when (dec.io.isSync) {
          state := sSync
	} .elsewhen (inst_type.orR) {
          state := sExe
	}
      }
    }
    is (sSync) {
      state := sIdle
    }
    is (sExe) {
      when (done) {
        state := sIdle
      }
    }
  }

  // instructions
  inst_q.io.enq <> io.inst
  inst_q.io.deq.ready := (state === sExe & done) | (state === sSync)

  // uop
  loadUop.io.start :=  state === sIdle & start & dec.io.isLoadUop
  loadUop.io.inst := inst_q.io.deq.bits
  loadUop.io.baddr := io.uop_baddr
  io.vme_rd(0) <> loadUop.io.vme_rd
  loadUop.io.uop.idx <> Mux(dec.io.isGemm, tensorGemm.io.uop.idx, tensorAlu.io.uop.idx)

  // acc
  tensorAcc.io.start := state === sIdle & start & dec.io.isLoadAcc
  tensorAcc.io.inst := inst_q.io.deq.bits
  tensorAcc.io.baddr := io.acc_baddr
  tensorAcc.io.tensor.rd.idx <> Mux(dec.io.isGemm, tensorGemm.io.acc.rd.idx, tensorAlu.io.acc.rd.idx)
  tensorAcc.io.tensor.wr <> Mux(dec.io.isGemm, tensorGemm.io.acc.wr, tensorAlu.io.acc.wr)
  io.vme_rd(1) <> tensorAcc.io.vme_rd

  // gemm
  tensorGemm.io.start :=  state === sIdle & start & dec.io.isGemm
  tensorGemm.io.inst := inst_q.io.deq.bits
  tensorGemm.io.uop.data.valid := loadUop.io.uop.data.valid & dec.io.isGemm
  tensorGemm.io.uop.data.bits <> loadUop.io.uop.data.bits
  tensorGemm.io.inp <> io.inp
  tensorGemm.io.wgt <> io.wgt
  tensorGemm.io.acc.rd.data.valid := tensorAcc.io.tensor.rd.data.valid & dec.io.isGemm
  tensorGemm.io.acc.rd.data.bits <> tensorAcc.io.tensor.rd.data.bits
  tensorGemm.io.out.rd.data.valid := io.out.rd.data.valid & dec.io.isGemm
  tensorGemm.io.out.rd.data.bits <> io.out.rd.data.bits

  // alu
  tensorAlu.io.start :=  state === sIdle & start & dec.io.isAlu
  tensorAlu.io.inst := inst_q.io.deq.bits
  tensorAlu.io.uop.data.valid := loadUop.io.uop.data.valid & dec.io.isAlu
  tensorAlu.io.uop.data.bits <> loadUop.io.uop.data.bits
  tensorAlu.io.acc.rd.data.valid := tensorAcc.io.tensor.rd.data.valid & dec.io.isAlu
  tensorAlu.io.acc.rd.data.bits <> tensorAcc.io.tensor.rd.data.bits
  tensorAlu.io.out.rd.data.valid := io.out.rd.data.valid & dec.io.isAlu
  tensorAlu.io.out.rd.data.bits <> io.out.rd.data.bits

  // out
  io.out.rd.idx <> Mux(dec.io.isGemm, tensorGemm.io.out.rd.idx, tensorAlu.io.out.rd.idx)
  io.out.wr <> Mux(dec.io.isGemm, tensorGemm.io.out.wr, tensorAlu.io.out.wr)

  // semaphore
  s(0).io.spost := io.i_post(0)
  s(1).io.spost := io.i_post(1)
  s(0).io.swait := dec.io.pop_prev & (state === sIdle & start)
  s(1).io.swait := dec.io.pop_next & (state === sIdle & start)
  io.o_post(0) := dec.io.push_prev & ((state === sExe & done) | (state === sSync))
  io.o_post(1) := dec.io.push_next & ((state === sExe & done) | (state === sSync))

  // finish
  io.finish := state === sExe & done & dec.io.isFinish

  // debug
  if (debug) {
    // start
    when (state === sIdle && start) {
      when (dec.io.isSync) {
        printf("[Compute] start sync\n")
      } .elsewhen (dec.io.isLoadUop) {
        printf("[Compute] start load uop\n")
      } .elsewhen (dec.io.isLoadAcc) {
        printf("[Compute] start load acc\n")
      } .elsewhen (dec.io.isGemm) {
        printf("[Compute] start gemm\n")
      } .elsewhen (dec.io.isAlu) {
        printf("[Compute] start alu\n")
      } .elsewhen (dec.io.isFinish) {
        printf("[Compute] start finish\n")
      }
    }
    // done
    when (state === sSync) {
      printf("[Compute] done sync\n")
    }
    when (state === sExe) {
      when (done) {
        when (dec.io.isLoadUop) {
          printf("[Compute] done load uop\n")
        } .elsewhen (dec.io.isLoadAcc) {
          printf("[Compute] done load acc\n")
        } .elsewhen (dec.io.isGemm) {
          printf("[Compute] done gemm\n")
        } .elsewhen (dec.io.isAlu) {
          printf("[Compute] done alu\n")
        } .elsewhen (dec.io.isFinish) {
          printf("[Compute] done finish\n")
        }
      }
    }
  }
}
