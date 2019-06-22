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
import vta.util.config._
import vta.shell._

/** Core parameters */
case class CoreParams (
  batch: Int = 1,
  blockOut: Int = 16,
  blockIn: Int = 16,
  inpBits: Int = 8,
  wgtBits: Int = 8,
  uopBits: Int = 32,
  accBits: Int = 32,
  outBits: Int = 8,
  uopMemDepth: Int = 512,
  inpMemDepth: Int = 512,
  wgtMemDepth: Int = 512,
  accMemDepth: Int = 512,
  outMemDepth: Int = 512,
  instQueueEntries: Int = 32
)

case object CoreKey extends Field[CoreParams]

/** Core.
  *
  * The core defines the current VTA architecture by connecting memory and
  * compute modules together such as load/store and compute. Most of the
  * connections in the core are bulk (<>), and we should try to keep it this
  * way, because it is easier to understand what is going on.
  *
  * Also, the core must be instantiated by a shell using the
  * VTA Control Register (VCR) and the VTA Memory Engine (VME) interfaces.
  * More info about these interfaces and modules can be found in the shell
  * directory.
  */
class Core(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val vcr = new VCRClient
    val vme = new VMEMaster
  })
  val fetch = Module(new Fetch)
  val load = Module(new Load)
  val compute = Module(new Compute)
  val store = Module(new Store)
  val ecounters = Module(new EventCounters)

  // Read(rd) and write(wr) from/to memory (i.e. DRAM)
  io.vme.rd(0) <> fetch.io.vme_rd
  io.vme.rd(1) <> compute.io.vme_rd(0)
  io.vme.rd(2) <> load.io.vme_rd(0)
  io.vme.rd(3) <> load.io.vme_rd(1)
  io.vme.rd(4) <> compute.io.vme_rd(1)
  io.vme.wr(0) <> store.io.vme_wr

  // Fetch instructions (tasks) from memory (DRAM) into queues (SRAMs)
  fetch.io.launch := io.vcr.launch
  fetch.io.ins_baddr := io.vcr.ptrs(0)
  fetch.io.ins_count := io.vcr.vals(0)

  // Load inputs and weights from memory (DRAM) into scratchpads (SRAMs)
  load.io.i_post := compute.io.o_post(0)
  load.io.inst <> fetch.io.inst.ld
  load.io.inp_baddr := io.vcr.ptrs(2)
  load.io.wgt_baddr := io.vcr.ptrs(3)

  // The compute module performs the following:
  // - Load micro-ops (uops) and accumulations (acc)
  // - Compute dense and ALU instructions (tasks)
  compute.io.i_post(0) := load.io.o_post
  compute.io.i_post(1) := store.io.o_post
  compute.io.inst <> fetch.io.inst.co
  compute.io.uop_baddr := io.vcr.ptrs(1)
  compute.io.acc_baddr := io.vcr.ptrs(4)
  compute.io.inp <> load.io.inp
  compute.io.wgt <> load.io.wgt

  // The store module performs the following:
  // - Writes results from compute into scratchpads (SRAMs)
  // - Store results from scratchpads (SRAMs) to memory (DRAM)
  store.io.i_post := compute.io.o_post(1)
  store.io.inst <> fetch.io.inst.st
  store.io.out_baddr := io.vcr.ptrs(5)
  store.io.out <> compute.io.out

  // Event counters
  ecounters.io.launch := io.vcr.launch
  ecounters.io.finish := compute.io.finish
  io.vcr.ecnt <> ecounters.io.ecnt

  // Finish instruction is executed and asserts the VCR finish flag
  val finish = RegNext(compute.io.finish)
  io.vcr.finish := finish
}
