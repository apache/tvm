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

package vta.shell

import chisel3._
import chisel3.experimental.MultiIOModule
import vta.util.config._
import vta.interface.axi._
import vta.shell._
import vta.dpi._

/** VTAHost.
  *
  * This module translate the DPI protocol into AXI. This is a simulation only
  * module and used to test host-to-VTA communication. This module should be updated
  * for testing hosts using a different bus protocol, other than AXI.
  */
class VTAHost(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val axi = new AXILiteMaster(p(ShellKey).hostParams)
  })
  val host_dpi = Module(new VTAHostDPI)
  val host_axi = Module(new VTAHostDPIToAXI)
  host_dpi.io.reset := reset
  host_dpi.io.clock := clock
  host_axi.io.dpi <> host_dpi.io.dpi
  io.axi <> host_axi.io.axi
}

/** VTAMem.
  *
  * This module translate the DPI protocol into AXI. This is a simulation only
  * module and used to test VTA-to-memory communication. This module should be updated
  * for testing memories using a different bus protocol, other than AXI.
  */
class VTAMem(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val axi = new AXIClient(p(ShellKey).memParams)
  })
  val mem_dpi = Module(new VTAMemDPI)
  val mem_axi = Module(new VTAMemDPIToAXI)
  mem_dpi.io.reset := reset
  mem_dpi.io.clock := clock
  mem_dpi.io.dpi <> mem_axi.io.dpi
  mem_axi.io.axi <> io.axi
}

/** VTASim.
  *
  * This module is used to handle hardware simulation thread, such as halting
  * or terminating the simulation thread. The sim_wait port is used to halt
  * the simulation thread when it is asserted and resume it when it is
  * de-asserted.
  */
class VTASim(implicit p: Parameters) extends MultiIOModule {
  val sim_wait = IO(Output(Bool()))
  val sim = Module(new VTASimDPI)
  sim.io.reset := reset
  sim.io.clock := clock
  sim_wait := sim.io.dpi_wait
}
/** SimShell.
  *
  * The simulation shell instantiate the sim, host and memory DPI modules that
  * are connected to the VTAShell. An extra clock, sim_clock, is used to eval
  * the VTASim DPI function when the main simulation clock is on halt state.
  */
class SimShell(implicit p: Parameters) extends MultiIOModule {
  val mem = IO(new AXIClient(p(ShellKey).memParams))
  val host = IO(new AXILiteMaster(p(ShellKey).hostParams))
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))
  val mod_sim = Module(new VTASim)
  val mod_host = Module(new VTAHost)
  val mod_mem = Module(new VTAMem)
  mem <> mod_mem.io.axi
  host <> mod_host.io.axi
  mod_sim.reset := reset
  mod_sim.clock := sim_clock
  sim_wait := mod_sim.sim_wait
}
