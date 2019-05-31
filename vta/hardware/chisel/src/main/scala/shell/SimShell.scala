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

/** SimShell.
  *
  * The simulation shell instantiate a host and memory simulation modules and it is
  * intended to be connected to the VTAShell.
  */
class SimShell(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val mem = new AXIClient(p(ShellKey).memParams)
    val host = new AXILiteMaster(p(ShellKey).hostParams)
  })
  val host = Module(new VTAHost)
  val mem = Module(new VTAMem)
  io.mem <> mem.io.axi
  io.host <> host.io.axi
}
