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
import vta.dpi._
import accel._
import chisel3.experimental.MultiIOModule

/** Test accelerator.
  *
  * Instantiate and connect the simulation-shell and the accelerator.
  *
  */
class TestAccel extends MultiIOModule {
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))
  val host = Module(new VTAHostDPI)
  val mem = Module(new VTAMemDPI)
  val sim = Module(new VTASimDPI)
  val vta = Module(new Accel)
  mem.io.clock := clock
  mem.io.reset := reset
  host.io.clock := clock
  host.io.reset := reset
  sim.io.clock := sim_clock
  sim.io.reset := reset
  vta.io.host <> host.io.dpi
  mem.io.dpi <> vta.io.mem
  sim_wait := sim.io.dpi_wait
}

/** Generate TestAccel as top module */
object Elaborate extends App {
  chisel3.Driver.execute(args, () => new TestAccel)
}
