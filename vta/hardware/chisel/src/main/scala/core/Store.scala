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

/** Store.
  *
  * Store results back to memory (DRAM) from scratchpads (SRAMs).
  * This module instantiate the TensorStore unit which is in charge
  * of storing 1D and 2D tensors to main memory.
  */
class Store(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val i_post = Input(Bool())
    val o_post = Output(Bool())
    val inst = Flipped(Decoupled(UInt(INST_BITS.W)))
    val out_baddr = Input(UInt(mp.addrBits.W))
    val vme_wr = new VMEWriteMaster
    val out = new TensorClient(tensorType = "out")
  })
  val sIdle :: sSync :: sExe :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val s = Module(new Semaphore(counterBits = 8, counterInitValue = 0))
  val inst_q = Module(new Queue(UInt(INST_BITS.W), p(CoreKey).instQueueEntries))

  val dec = Module(new StoreDecode)
  dec.io.inst := inst_q.io.deq.bits

  val tensorStore = Module(new TensorStore(tensorType = "out"))

  val start = inst_q.io.deq.valid & Mux(dec.io.pop_prev, s.io.sready, true.B)
  val done = tensorStore.io.done

  // control
  switch (state) {
    is (sIdle) {
      when (start) {
        when (dec.io.isSync) {
          state := sSync
	} .elsewhen (dec.io.isStore) {
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

  // store
  tensorStore.io.start := state === sIdle & start & dec.io.isStore
  tensorStore.io.inst := inst_q.io.deq.bits
  tensorStore.io.baddr := io.out_baddr
  io.vme_wr <> tensorStore.io.vme_wr
  tensorStore.io.tensor <> io.out

  // semaphore
  s.io.spost := io.i_post
  s.io.swait := dec.io.pop_prev & (state === sIdle & start)
  io.o_post := dec.io.push_prev & ((state === sExe & done) | (state === sSync))

  // debug
  if (debug) {
    // start
    when (state === sIdle && start) {
      when (dec.io.isSync) {
        printf("[Store] start sync\n")
      } .elsewhen (dec.io.isStore) {
        printf("[Store] start\n")
      }
    }
    // done
    when (state === sSync) {
      printf("[Store] done sync\n")
    }
    when (state === sExe) {
      when (done) {
	printf("[Store] done\n")
      }
    }
  }
}
