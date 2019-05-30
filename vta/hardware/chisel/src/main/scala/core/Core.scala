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

class Core(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val vcr = new VCRClient
    val vme = new VMEMaster
  })
  val fetch = Module(new Fetch)
  val load = Module(new Load)
  val compute = Module(new Compute)
  val store = Module(new Store)

  // vme
  io.vme.rd(0) <> fetch.io.vme_rd
  io.vme.rd(1) <> compute.io.vme_rd(0)
  io.vme.rd(2) <> load.io.vme_rd(0)
  io.vme.rd(3) <> load.io.vme_rd(1)
  io.vme.rd(4) <> compute.io.vme_rd(1)
  io.vme.wr(0) <> store.io.vme_wr

  // fetch
  fetch.io.launch := io.vcr.launch
  fetch.io.ins_baddr := io.vcr.ptrs(0)
  fetch.io.ins_count := io.vcr.vals(0)

  // load
  load.io.i_post := compute.io.o_post(0)
  load.io.inst <> fetch.io.inst.ld
  load.io.inp_baddr := io.vcr.ptrs(2)
  load.io.wgt_baddr := io.vcr.ptrs(3)

  // compute
  compute.io.i_post(0) := load.io.o_post
  compute.io.i_post(1) := store.io.o_post
  compute.io.inst <> fetch.io.inst.co
  compute.io.uop_baddr := io.vcr.ptrs(1)
  compute.io.acc_baddr := io.vcr.ptrs(4)
  compute.io.inp <> load.io.inp
  compute.io.wgt <> load.io.wgt

  // store
  store.io.i_post := compute.io.o_post(1)
  store.io.inst <> fetch.io.inst.st
  store.io.out_baddr := io.vcr.ptrs(5)
  store.io.out <> compute.io.out

  // finish
  val finish = RegNext(compute.io.finish)
  io.vcr.finish := finish
}
