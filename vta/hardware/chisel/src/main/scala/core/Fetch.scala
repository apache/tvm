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

/** Fetch.
  *
  * The fetch unit reads instructions (tasks) from memory (i.e. DRAM), using the
  * VTA Memory Engine (VME), and push them into an instruction queue called
  * inst_q. Once the instruction queue is full, instructions are dispatched to
  * the Load, Compute and Store module queues based on the instruction opcode.
  * After draining the queue, the fetch unit checks if there are more instructions
  * via the ins_count register which is written by the host.
  *
  * Additionally, instructions are read into two chunks (see sReadLSB and sReadMSB)
  * because we are using a DRAM payload of 8-bytes or half of a VTA instruction.
  * This should be configurable for larger payloads, i.e. 64-bytes, which can load
  * more than one instruction at the time. Finally, the instruction queue is
  * sized (entries_q), depending on the maximum burst allowed in the memory.
  */
class Fetch(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val launch = Input(Bool())
    val ins_baddr = Input(UInt(mp.addrBits.W))
    val ins_count = Input(UInt(vp.regBits.W))
    val vme_rd = new VMEReadMaster
    val inst = new Bundle {
      val ld = Decoupled(UInt(INST_BITS.W))
      val co = Decoupled(UInt(INST_BITS.W))
      val st = Decoupled(UInt(INST_BITS.W))
    }
  })
  val entries_q = 1 << (mp.lenBits - 1) // one-instr-every-two-vme-word
  val inst_q = Module(new Queue(UInt(INST_BITS.W), entries_q))
  val dec = Module(new FetchDecode)

  val s1_launch = RegNext(io.launch)
  val pulse = io.launch & ~s1_launch

  val raddr = Reg(chiselTypeOf(io.vme_rd.cmd.bits.addr))
  val rlen = Reg(chiselTypeOf(io.vme_rd.cmd.bits.len))
  val ilen = Reg(chiselTypeOf(io.vme_rd.cmd.bits.len))

  val xrem = Reg(chiselTypeOf(io.ins_count))
  val xsize = (io.ins_count << 1.U) - 1.U
  val xmax = (1 << mp.lenBits).U
  val xmax_bytes = ((1 << mp.lenBits)*mp.dataBits/8).U

  val sIdle :: sReadCmd :: sReadLSB :: sReadMSB :: sDrain :: Nil = Enum(5)
  val state = RegInit(sIdle)

  // control
  switch (state) {
    is (sIdle) {
      when (pulse) {
        state := sReadCmd
	when (xsize < xmax) {
          rlen := xsize
	  ilen := xsize >> 1.U
          xrem := 0.U
	} .otherwise {
          rlen := xmax - 1.U
	  ilen := (xmax >> 1.U) - 1.U
          xrem := xsize - xmax
	}
      }
    }
    is (sReadCmd) {
      when (io.vme_rd.cmd.ready) {
        state := sReadLSB
      }
    }
    is (sReadLSB) {
      when (io.vme_rd.data.valid) {
          state := sReadMSB
      }
    }
    is (sReadMSB) {
      when (io.vme_rd.data.valid) {
        when (inst_q.io.count === ilen) {
          state := sDrain
        } .otherwise {
          state := sReadLSB
	}
      }
    }
    is (sDrain) {
      when (inst_q.io.count === 0.U) {
        when (xrem === 0.U) {
          state := sIdle
        } .elsewhen (xrem < xmax) {
          state := sReadCmd
          rlen := xrem
	  ilen := xrem >> 1.U
          xrem := 0.U
        } .otherwise {
          state := sReadCmd
          rlen := xmax - 1.U
	  ilen := (xmax >> 1.U) - 1.U
          xrem := xrem - xmax
        }
      }
    }
  }

  // read instructions from dram
  when (state === sIdle) {
    raddr := io.ins_baddr
  } .elsewhen (state === sDrain && inst_q.io.count === 0.U && xrem =/= 0.U) {
    raddr := raddr + xmax_bytes
  }

  io.vme_rd.cmd.valid := state === sReadCmd
  io.vme_rd.cmd.bits.addr := raddr
  io.vme_rd.cmd.bits.len := rlen

  io.vme_rd.data.ready := inst_q.io.enq.ready

  val lsb = Reg(chiselTypeOf(io.vme_rd.data.bits))
  val msb = io.vme_rd.data.bits
  val inst = Cat(msb, lsb)

  when (state === sReadLSB) { lsb := io.vme_rd.data.bits }

  inst_q.io.enq.valid := io.vme_rd.data.valid & state === sReadMSB
  inst_q.io.enq.bits := inst

  // decode
  dec.io.inst := inst_q.io.deq.bits

  // instruction queues
  io.inst.ld.valid := dec.io.isLoad & inst_q.io.deq.valid & state === sDrain
  io.inst.co.valid := dec.io.isCompute & inst_q.io.deq.valid & state === sDrain
  io.inst.st.valid := dec.io.isStore & inst_q.io.deq.valid & state === sDrain

  io.inst.ld.bits := inst_q.io.deq.bits
  io.inst.co.bits := inst_q.io.deq.bits
  io.inst.st.bits := inst_q.io.deq.bits

  // check if selected queue is ready
  val deq_sel = Cat(dec.io.isCompute, dec.io.isStore, dec.io.isLoad).asUInt
  val deq_ready =
    MuxLookup(deq_sel,
               false.B, // default
      Array(
        "h_01".U -> io.inst.ld.ready,
        "h_02".U -> io.inst.st.ready,
        "h_04".U -> io.inst.co.ready
      )
    )

  // dequeue instruction
  inst_q.io.deq.ready := deq_ready & inst_q.io.deq.valid & state === sDrain


  // debug
  if (debug) {
    when (state === sIdle && pulse) {
      printf("[Fetch] Launch\n")
    }
    // instruction
    when (inst_q.io.deq.fire()) {
      when (dec.io.isLoad) {
        printf("[Fetch] [instruction decode] [L] %x\n", inst_q.io.deq.bits)
      }
      when (dec.io.isCompute) {
        printf("[Fetch] [instruction decode] [C] %x\n", inst_q.io.deq.bits)
      }
      when (dec.io.isStore) {
        printf("[Fetch] [instruction decode] [S] %x\n", inst_q.io.deq.bits)
      }
    }
  }
}
