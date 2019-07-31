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

import ISA._

/** MemDecode.
  *
  * Decode memory instructions with a Bundle. This is similar to an union,
  * therefore order matters when declaring fields. These are the instructions
  * decoded with this bundle:
  *   - LUOP
  *   - LWGT
  *   - LINP
  *   - LACC
  *   - SOUT
  */
class MemDecode extends Bundle {
  val xpad_1 = UInt(M_PAD_BITS.W)
  val xpad_0 = UInt(M_PAD_BITS.W)
  val ypad_1 = UInt(M_PAD_BITS.W)
  val ypad_0 = UInt(M_PAD_BITS.W)
  val xstride = UInt(M_STRIDE_BITS.W)
  val xsize = UInt(M_SIZE_BITS.W)
  val ysize = UInt(M_SIZE_BITS.W)
  val empty_0 = UInt(7.W) // derive this
  val dram_offset = UInt(M_DRAM_OFFSET_BITS.W)
  val sram_offset = UInt(M_SRAM_OFFSET_BITS.W)
  val id = UInt(M_ID_BITS.W)
  val push_next = Bool()
  val push_prev = Bool()
  val pop_next = Bool()
  val pop_prev = Bool()
  val op = UInt(OP_BITS.W)
}

/** GemmDecode.
  *
  * Decode GEMM instruction with a Bundle. This is similar to an union,
  * therefore order matters when declaring fields.
  */
class GemmDecode extends Bundle {
  val wgt_1 = UInt(C_WIDX_BITS.W)
  val wgt_0 = UInt(C_WIDX_BITS.W)
  val inp_1 = UInt(C_IIDX_BITS.W)
  val inp_0 = UInt(C_IIDX_BITS.W)
  val acc_1 = UInt(C_AIDX_BITS.W)
  val acc_0 = UInt(C_AIDX_BITS.W)
  val empty_0 = Bool()
  val lp_1 = UInt(C_ITER_BITS.W)
  val lp_0 = UInt(C_ITER_BITS.W)
  val uop_end = UInt(C_UOP_END_BITS.W)
  val uop_begin = UInt(C_UOP_BGN_BITS.W)
  val reset = Bool()
  val push_next = Bool()
  val push_prev = Bool()
  val pop_next = Bool()
  val pop_prev = Bool()
  val op = UInt(OP_BITS.W)
}

/** AluDecode.
  *
  * Decode ALU instructions with a Bundle. This is similar to an union,
  * therefore order matters when declaring fields. These are the instructions
  * decoded with this bundle:
  *   - VMIN
  *   - VMAX
  *   - VADD
  *   - VSHX
  */
class AluDecode extends Bundle {
  val empty_1 = Bool()
  val alu_imm = UInt(C_ALU_IMM_BITS.W)
  val alu_use_imm = Bool()
  val alu_op = UInt(C_ALU_DEC_BITS.W)
  val src_1 = UInt(C_IIDX_BITS.W)
  val src_0 = UInt(C_IIDX_BITS.W)
  val dst_1 = UInt(C_AIDX_BITS.W)
  val dst_0 = UInt(C_AIDX_BITS.W)
  val empty_0 = Bool()
  val lp_1 = UInt(C_ITER_BITS.W)
  val lp_0 = UInt(C_ITER_BITS.W)
  val uop_end = UInt(C_UOP_END_BITS.W)
  val uop_begin = UInt(C_UOP_BGN_BITS.W)
  val reset = Bool()
  val push_next = Bool()
  val push_prev = Bool()
  val pop_next = Bool()
  val pop_prev = Bool()
  val op = UInt(OP_BITS.W)
}

/** UopDecode.
  *
  * Decode micro-ops (uops).
  */
class UopDecode extends Bundle {
  val u2 = UInt(10.W)
  val u1 = UInt(11.W)
  val u0 = UInt(11.W)
}

/** FetchDecode.
  *
  * Partial decoding for dispatching instructions to Load, Compute, and Store.
  */
class FetchDecode extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val isLoad = Output(Bool())
    val isCompute = Output(Bool())
    val isStore = Output(Bool())
  })
  val csignals =
    ListLookup(io.inst,
        List(N, OP_X),
      Array(
        LUOP -> List(Y, OP_G),
        LWGT -> List(Y, OP_L),
        LINP -> List(Y, OP_L),
        LACC -> List(Y, OP_G),
        SOUT -> List(Y, OP_S),
        GEMM -> List(Y, OP_G),
        FNSH -> List(Y, OP_G),
        VMIN -> List(Y, OP_G),
        VMAX -> List(Y, OP_G),
        VADD -> List(Y, OP_G),
        VSHX -> List(Y, OP_G)
      )
    )

  val (cs_val_inst: Bool) :: cs_op_type :: Nil = csignals

  io.isLoad := cs_val_inst & cs_op_type === OP_L
  io.isCompute := cs_val_inst & cs_op_type === OP_G
  io.isStore := cs_val_inst & cs_op_type === OP_S
}

/** LoadDecode.
  *
  * Decode dependencies, type and sync for Load module.
  */
class LoadDecode extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val push_next = Output(Bool())
    val pop_next = Output(Bool())
    val isInput = Output(Bool())
    val isWeight = Output(Bool())
    val isSync = Output(Bool())
  })
  val dec = io.inst.asTypeOf(new MemDecode)
  io.push_next := dec.push_next
  io.pop_next := dec.pop_next
  io.isInput := io.inst === LINP & dec.xsize =/= 0.U
  io.isWeight := io.inst === LWGT & dec.xsize =/= 0.U
  io.isSync := (io.inst === LINP | io.inst === LWGT) & dec.xsize === 0.U
}

/** ComputeDecode.
  *
  * Decode dependencies, type and sync for Compute module.
  */
class ComputeDecode extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val push_next = Output(Bool())
    val push_prev = Output(Bool())
    val pop_next = Output(Bool())
    val pop_prev = Output(Bool())
    val isLoadAcc = Output(Bool())
    val isLoadUop = Output(Bool())
    val isSync = Output(Bool())
    val isAlu = Output(Bool())
    val isGemm = Output(Bool())
    val isFinish = Output(Bool())
  })
  val dec = io.inst.asTypeOf(new MemDecode)
  io.push_next := dec.push_next
  io.push_prev := dec.push_prev
  io.pop_next := dec.pop_next
  io.pop_prev := dec.pop_prev
  io.isLoadAcc := io.inst === LACC & dec.xsize =/= 0.U
  io.isLoadUop := io.inst === LUOP & dec.xsize =/= 0.U
  io.isSync := (io.inst === LACC | io.inst === LUOP) & dec.xsize === 0.U
  io.isAlu := io.inst === VMIN | io.inst === VMAX | io.inst === VADD | io.inst === VSHX
  io.isGemm := io.inst === GEMM
  io.isFinish := io.inst === FNSH
}

/** StoreDecode.
  *
  * Decode dependencies, type and sync for Store module.
  */
class StoreDecode extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val push_prev = Output(Bool())
    val pop_prev = Output(Bool())
    val isStore = Output(Bool())
    val isSync = Output(Bool())
  })
  val dec = io.inst.asTypeOf(new MemDecode)
  io.push_prev := dec.push_prev
  io.pop_prev := dec.pop_prev
  io.isStore := io.inst === SOUT & dec.xsize =/= 0.U
  io.isSync := io.inst === SOUT & dec.xsize === 0.U
}
