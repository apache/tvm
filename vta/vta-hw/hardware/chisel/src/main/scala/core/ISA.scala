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
import scala.collection.mutable.HashMap

/** ISAConstants.
 *
 * These constants are used for decoding (parsing) fields on instructions.
 */
trait ISAConstants {
  val INST_BITS = 128

  val OP_BITS = 3

  val M_DEP_BITS = 4
  val M_ID_BITS = 2
  val M_SRAM_OFFSET_BITS = 16
  val M_DRAM_OFFSET_BITS = 32
  val M_SIZE_BITS = 16
  val M_STRIDE_BITS = 16
  val M_PAD_BITS = 4

  val C_UOP_BGN_BITS = 13
  val C_UOP_END_BITS = 14
  val C_ITER_BITS = 14
  val C_AIDX_BITS = 11
  val C_IIDX_BITS = 11
  val C_WIDX_BITS = 10
  val C_ALU_DEC_BITS = 2 // FIXME: there should be a SHL and SHR instruction
  val C_ALU_OP_BITS = 3
  val C_ALU_IMM_BITS = 16

  val Y = true.B
  val N = false.B

  val OP_L = 0.asUInt(OP_BITS.W)
  val OP_S = 1.asUInt(OP_BITS.W)
  val OP_G = 2.asUInt(OP_BITS.W)
  val OP_F = 3.asUInt(OP_BITS.W)
  val OP_A = 4.asUInt(OP_BITS.W)
  val OP_X = 5.asUInt(OP_BITS.W)

  val ALU_OP_NUM = 5
  val ALU_OP = Enum(ALU_OP_NUM)

  val M_ID_U = 0.asUInt(M_ID_BITS.W)
  val M_ID_W = 1.asUInt(M_ID_BITS.W)
  val M_ID_I = 2.asUInt(M_ID_BITS.W)
  val M_ID_A = 3.asUInt(M_ID_BITS.W)
}

/** ISA.
 *
 * This is the VTA task ISA
 *
 * TODO: Add VXOR to clear accumulator
 * TODO: Use ISA object for decoding as well
 * TODO: Eventually deprecate ISAConstants
 */
object ISA {
  private val xLen = 128
  private val depBits = 4

  private val idBits: HashMap[String, Int] =
    HashMap(("task", 3), ("mem", 2), ("alu", 2))

  private val taskId: HashMap[String, String] =
    HashMap(("load", "000"),
      ("store", "001"),
      ("gemm", "010"),
      ("finish", "011"),
      ("alu", "100"))

  private val memId: HashMap[String, String] =
    HashMap(("uop", "00"), ("wgt", "01"), ("inp", "10"), ("acc", "11"))

  private val aluId: HashMap[String, String] =
    HashMap(("minpool", "00"),
      ("maxpool", "01"),
      ("add", "10"),
      ("shift", "11"))

  private def dontCare(bits: Int): String = "?" * bits

  private def instPat(bin: String): BitPat = BitPat("b" + bin)

  private def load(id: String): BitPat = {
    val rem = xLen - idBits("mem") - depBits - idBits("task")
    val inst = dontCare(rem) + memId(id) + dontCare(depBits) + taskId("load")
    instPat(inst)
  }

  private def store: BitPat = {
    val rem = xLen - idBits("task")
    val inst = dontCare(rem) + taskId("store")
    instPat(inst)
  }

  private def gemm: BitPat = {
    val rem = xLen - idBits("task")
    val inst = dontCare(rem) + taskId("gemm")
    instPat(inst)
  }

  private def alu(id: String): BitPat = {
    // TODO: move alu id next to task id
    val inst = dontCare(18) + aluId(id) + dontCare(105) + taskId("alu")
    instPat(inst)
  }

  private def finish: BitPat = {
    val rem = xLen - idBits("task")
    val inst = dontCare(rem) + taskId("finish")
    instPat(inst)
  }

  def LUOP = load("uop")
  def LWGT = load("wgt")
  def LINP = load("inp")
  def LACC = load("acc")
  def SOUT = store
  def GEMM = gemm
  def VMIN = alu("minpool")
  def VMAX = alu("maxpool")
  def VADD = alu("add")
  def VSHX = alu("shift")
  def FNSH = finish
}
