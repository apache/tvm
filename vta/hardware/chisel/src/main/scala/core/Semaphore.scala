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

/** Semaphore.
  *
  * This semaphore is used instead of push/pop fifo, used in the initial
  * version of VTA. This semaphore is incremented (spost) or decremented (swait)
  * depending on the push and pop fields on instructions to prevent RAW and WAR
  * hazards.
  */
class Semaphore(counterBits: Int = 1, counterInitValue: Int = 1) extends Module {
  val io = IO(new Bundle {
    val spost = Input(Bool())
    val swait = Input(Bool())
    val sready = Output(Bool())
  })
  val cnt = RegInit(counterInitValue.U(counterBits.W))
  when (io.spost && !io.swait && cnt =/= ((1 << counterBits) - 1).asUInt) { cnt := cnt + 1.U }
  when (!io.spost && io.swait && cnt =/= 0.U) { cnt := cnt - 1.U }
  io.sready := cnt =/= 0.U
}
