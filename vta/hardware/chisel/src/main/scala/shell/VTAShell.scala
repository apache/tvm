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
import vta.core._

/** Shell parameters. */
case class ShellParams(
  hostParams: AXIParams,
  memParams: AXIParams,
  vcrParams: VCRParams,
  vmeParams: VMEParams
)

case object ShellKey extends Field[ShellParams]

/** VTAShell.
  *
  * The VTAShell is based on a VME, VCR and core. This creates a complete VTA
  * system that can be used for simulation or real hardware.
  */
class VTAShell(implicit p: Parameters) extends Module {
  val io = IO(new Bundle{
    val host = new AXILiteClient(p(ShellKey).hostParams)
    val mem = new AXIMaster(p(ShellKey).memParams)
  })

  val vcr = Module(new VCR)
  val vme = Module(new VME)
  val core = Module(new Core)

  core.io.vcr <> vcr.io.vcr
  vme.io.vme <> core.io.vme

  vcr.io.host <> io.host
  io.mem <> vme.io.mem
}
