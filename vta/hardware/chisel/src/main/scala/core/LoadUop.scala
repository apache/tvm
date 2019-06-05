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

/** UopMaster.
  *
  * Uop interface used by a master module, i.e. TensorAlu or TensorGemm,
  * to request a micro-op (uop) from the uop-scratchpad. The index (idx) is
  * used as an address to find the uop in the uop-scratchpad.
  */
class UopMaster(implicit p: Parameters) extends Bundle {
  val addrBits = log2Ceil(p(CoreKey).uopMemDepth)
  val idx = ValidIO(UInt(addrBits.W))
  val data = Flipped(ValidIO(new UopDecode))
  override def cloneType = new UopMaster().asInstanceOf[this.type]
}

/** UopClient.
  *
  * Uop interface used by a client module, i.e. LoadUop, to receive
  * a request from a master module, i.e. TensorAlu or TensorGemm.
  * The index (idx) is used as an address to find the uop in the uop-scratchpad.
  */
class UopClient(implicit p: Parameters) extends Bundle {
  val addrBits = log2Ceil(p(CoreKey).uopMemDepth)
  val idx = Flipped(ValidIO(UInt(addrBits.W)))
  val data = ValidIO(new UopDecode)
  override def cloneType = new UopClient().asInstanceOf[this.type]
}

/** LoadUop.
  *
  * Load micro-ops (uops) from memory, i.e. DRAM, and store them in the
  * uop-scratchpad. Currently, micro-ops are 32-bit wide and loaded in
  * group of 2 given the fact that the DRAM payload is 8-bytes. This module
  * should be modified later on to support different DRAM sizes efficiently.
  */
class LoadUop(debug: Boolean = false)(implicit p: Parameters) extends Module {
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
    val baddr = Input(UInt(mp.addrBits.W))
    val vme_rd = new VMEReadMaster
    val uop = new UopClient
  })
  val numUop = 2 // store two uops per sram word
  val uopBits = p(CoreKey).uopBits
  val uopDepth = p(CoreKey).uopMemDepth / numUop

  val dec = io.inst.asTypeOf(new MemDecode)
  val raddr = Reg(chiselTypeOf(io.vme_rd.cmd.bits.addr))
  val xcnt = Reg(chiselTypeOf(io.vme_rd.cmd.bits.len))
  val xlen = Reg(chiselTypeOf(io.vme_rd.cmd.bits.len))
  val xrem = Reg(chiselTypeOf(dec.xsize))
  val xsize = dec.xsize(0) + (dec.xsize >> log2Ceil(numUop)) - 1.U
  val xmax = (1 << mp.lenBits).U
  val xmax_bytes = ((1 << mp.lenBits)*mp.dataBits/8).U

  val offsetIsEven = (dec.sram_offset % 2.U) === 0.U
  val sizeIsEven = (dec.xsize % 2.U) === 0.U

  val sIdle :: sReadCmd :: sReadData :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // control
  switch (state) {
    is (sIdle) {
      when (io.start) {
        state := sReadCmd
	when (xsize < xmax) {
          xlen := xsize
          xrem := 0.U
	} .otherwise {
          xlen := xmax - 1.U
          xrem := xsize - xmax
	}
      }
    }
    is (sReadCmd) {
      when (io.vme_rd.cmd.ready) {
        state := sReadData
      }
    }
    is (sReadData) {
      when (io.vme_rd.data.valid) {
        when(xcnt === xlen) {
          when (xrem === 0.U) {
            state := sIdle
          } .elsewhen (xrem < xmax) {
            state := sReadCmd
            xlen := xrem
            xrem := 0.U
          } .otherwise {
            state := sReadCmd
            xlen := xmax - 1.U
            xrem := xrem - xmax
          }
        }
      }
    }
  }

  // read-from-dram
  when (state === sIdle) {
    when (offsetIsEven) {
      raddr := io.baddr + dec.dram_offset
    } .otherwise {
      raddr := io.baddr + dec.dram_offset - 4.U
    }
  } .elsewhen (state === sReadData && xcnt === xlen && xrem =/= 0.U) {
    raddr := raddr + xmax_bytes
  }

  io.vme_rd.cmd.valid := state === sReadCmd
  io.vme_rd.cmd.bits.addr := raddr
  io.vme_rd.cmd.bits.len := xlen

  io.vme_rd.data.ready := state === sReadData

  when (state =/= sReadData) {
    xcnt := 0.U
  } .elsewhen (io.vme_rd.data.fire()) {
    xcnt := xcnt + 1.U
  }

  val waddr = Reg(UInt(log2Ceil(uopDepth).W))
  when (state === sIdle) {
    waddr := dec.sram_offset >> log2Ceil(numUop)
  } .elsewhen (io.vme_rd.data.fire()) {
    waddr := waddr + 1.U
  }

  val wdata = Wire(Vec(numUop, UInt(uopBits.W)))
  val mem = SyncReadMem(uopDepth, chiselTypeOf(wdata))
  val wmask = Reg(Vec(numUop, Bool()))

  when (offsetIsEven) {
    when (sizeIsEven) {
      wmask := "b_11".U.asTypeOf(wmask)
    } .elsewhen (io.vme_rd.cmd.fire()) {
      when (dec.xsize === 1.U) {
        wmask := "b_01".U.asTypeOf(wmask)
      } .otherwise {
        wmask := "b_11".U.asTypeOf(wmask)
      }
    } .elsewhen (io.vme_rd.data.fire()) {
      when (xcnt === xlen - 1.U) {
        wmask := "b_01".U.asTypeOf(wmask)
      } .otherwise {
        wmask := "b_11".U.asTypeOf(wmask)
      }
    }
  } .otherwise {
    when (io.vme_rd.cmd.fire()) {
      wmask := "b_10".U.asTypeOf(wmask)
    } .elsewhen (io.vme_rd.data.fire()) {
      when (sizeIsEven && xcnt === xlen - 1.U) {
        wmask := "b_01".U.asTypeOf(wmask)
      } .otherwise {
        wmask := "b_11".U.asTypeOf(wmask)
      }
    }
  }

  wdata := io.vme_rd.data.bits.asTypeOf(wdata)
  when (io.vme_rd.data.fire()) {
    mem.write(waddr, wdata, wmask)
  }

  // read-from-sram
  io.uop.data.valid := RegNext(io.uop.idx.valid)

  val sIdx = io.uop.idx.bits % numUop.U
  val rIdx = io.uop.idx.bits >> log2Ceil(numUop)
  val memRead = mem.read(rIdx, io.uop.idx.valid)
  val sWord = memRead.asUInt.asTypeOf(wdata)
  val sUop = sWord(sIdx).asTypeOf(io.uop.data.bits)

  io.uop.data.bits <> sUop

  // done
  io.done := state === sReadData & io.vme_rd.data.valid & xcnt === xlen & xrem === 0.U

  // debug
  if (debug) {
    when (io.vme_rd.cmd.fire()) {
      printf("[LoadUop] cmd addr:%x len:%x rem:%x\n", raddr, xlen, xrem)
    }
  }
}
