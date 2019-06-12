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

class EventCounters(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val vp = p(ShellKey).vcrParams
  val io = IO(new Bundle{
    val launch = Input(Bool())
    val finish = Input(Bool())
    val ecnt = Vec(vp.nVals, ValidIO(UInt(vp.regBits.W)))
  })
  val cycle_cnt = RegInit(0.U(vp.regBits.W))
  when (io.launch && !io.finish) {
    cycle_cnt := cycle_cnt + 1.U
  } .otherwise {
    cycle_cnt := 0.U
  }
  io.ecnt(0).valid := io.finish
  io.ecnt(0).bits := cycle_cnt
}
