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

package test

import chisel3._
import chisel3.experimental.MultiIOModule
import vta.dpi._
import accel._

/** VTA simulation shell.
  *
  * Instantiate Host and Memory DPI modules.
  *
  */
class VTASimShell extends MultiIOModule {
  val host = IO(new VTAHostDPIMaster)
  val mem = IO(new VTAMemDPIClient)
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))
  val mod_sim = Module(new VTASimDPI)
  val mod_host = Module(new VTAHostDPI)
  val mod_mem = Module(new VTAMemDPI)
  mod_mem.io.clock := clock
  mod_mem.io.reset := reset
  mod_mem.io.dpi <> mem
  mod_host.io.clock := clock
  mod_host.io.reset := reset
  host <> mod_host.io.dpi
  mod_sim.io.clock := sim_clock
  mod_sim.io.reset := reset
  sim_wait := mod_sim.io.dpi_wait
}

/** Test accelerator.
  *
  * Instantiate and connect the simulation-shell and the accelerator.
  *
  */
class TestAccel extends MultiIOModule {
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))
  val sim_shell = Module(new VTASimShell)
  val vta_accel = Module(new Accel)
  sim_shell.sim_clock := sim_clock
  sim_wait := sim_shell.sim_wait
  sim_shell.mem <> vta_accel.io.mem
  vta_accel.io.host <> sim_shell.host
}

/** Generate TestAccel as top module */
object Elaborate extends App {
  chisel3.Driver.execute(args, () => new TestAccel)
}
