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

package accel

import chisel3._
import vta.dpi._

/** Add-by-one accelerator.
  *
  * ___________      ___________
  * |         |      |         |
  * | HostDPI | <--> | RegFile | <->|
  * |_________|      |_________|    |
  *                                 |
  * ___________      ___________    |
  * |         |      |         |    |
  * | MemDPI  | <--> | Compute | <->|
  * |_________|      |_________|
  *
  */
case class AccelConfig() {
  val nCtrl = 1
  val nECnt = 1
  val nVals = 4
  val nPtrs = 3
  val regBits = 32
  val ptrBits = 2*regBits
}

class Accel extends Module {
  val io = IO(new Bundle {
    val host = new VTAHostDPIClient
    val mem = new VTAMemDPIMaster
  })
  implicit val config = AccelConfig()
  val rf = Module(new RegFile)
  val ce = Module(new Compute)
  rf.io.host <> io.host
  io.mem <> ce.io.mem
  ce.io.launch := rf.io.launch
  rf.io.finish := ce.io.finish
  rf.io.ecnt <> ce.io.ecnt
  ce.io.vals <> rf.io.vals
  ce.io.ptrs <> rf.io.ptrs
}
