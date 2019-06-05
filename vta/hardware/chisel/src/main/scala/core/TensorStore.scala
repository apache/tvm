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

/** TensorStore.
  *
  * Store 1D and 2D tensors from out-scratchpad (SRAM) to main memory (DRAM).
  */
class TensorStore(tensorType: String = "true", debug: Boolean = false)
  (implicit p: Parameters) extends Module {
  val tp = new TensorParams(tensorType)
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
    val baddr = Input(UInt(mp.addrBits.W))
    val vme_wr = new VMEWriteMaster
    val tensor = new TensorClient(tensorType)
  })
  val tensorLength = tp.tensorLength
  val tensorWidth = tp.tensorWidth
  val tensorElemBits = tp.tensorElemBits
  val memBlockBits = tp.memBlockBits
  val memDepth = tp.memDepth
  val numMemBlock = tp.numMemBlock

  val dec = io.inst.asTypeOf(new MemDecode)
  val waddr_cur = Reg(chiselTypeOf(io.vme_wr.cmd.bits.addr))
  val waddr_nxt = Reg(chiselTypeOf(io.vme_wr.cmd.bits.addr))
  val xcnt = Reg(chiselTypeOf(io.vme_wr.cmd.bits.len))
  val xlen = Reg(chiselTypeOf(io.vme_wr.cmd.bits.len))
  val xrem = Reg(chiselTypeOf(dec.xsize))
  val xsize = (dec.xsize << log2Ceil(tensorLength*numMemBlock)) - 1.U
  val xmax = (1 << mp.lenBits).U
  val xmax_bytes = ((1 << mp.lenBits)*mp.dataBits/8).U
  val ycnt = Reg(chiselTypeOf(dec.ysize))
  val ysize = dec.ysize
  val tag = Reg(UInt(8.W))
  val set = Reg(UInt(8.W))

  val sIdle :: sWriteCmd :: sWriteData :: sReadMem :: sWriteAck :: Nil = Enum(5)
  val state = RegInit(sIdle)

  // control
  switch (state) {
    is (sIdle) {
      when (io.start) {
        state := sWriteCmd
	when (xsize < xmax) {
          xlen := xsize
          xrem := 0.U
	} .otherwise {
          xlen := xmax - 1.U
          xrem := xsize - xmax
	}
      }
    }
    is (sWriteCmd) {
      when (io.vme_wr.cmd.ready) {
        state := sWriteData
      }
    }
    is (sWriteData) {
      when (io.vme_wr.data.ready) {
        when (xcnt === xlen) {
          state := sWriteAck
        } .elsewhen (tag === (numMemBlock - 1).U) {
          state := sReadMem
	}
      }
    }
    is (sReadMem) {
      state := sWriteData
    }
    is (sWriteAck) {
      when (io.vme_wr.ack) {
        when (xrem === 0.U) {
	  when (ycnt === ysize - 1.U) {
            state := sIdle
	  } .otherwise {
            state := sWriteCmd
	    when (xsize < xmax) {
              xlen := xsize
              xrem := 0.U
	    } .otherwise {
              xlen := xmax - 1.U
              xrem := xsize - xmax
	    }
	  }
	} .elsewhen (xrem < xmax) {
          state := sWriteCmd
          xlen := xrem
          xrem := 0.U
	} .otherwise {
          state := sWriteCmd
          xlen := xmax - 1.U
          xrem := xrem - xmax
	}
      }
    }
  }

  // write-to-sram
  val tensorFile = Seq.fill(tensorLength) { SyncReadMem(memDepth, Vec(numMemBlock, UInt(memBlockBits.W))) }
  val wdata_t = Wire(Vec(numMemBlock, UInt(memBlockBits.W)))
  val no_mask = Wire(Vec(numMemBlock, Bool()))

  wdata_t := DontCare
  no_mask.foreach { m => m := true.B }

  for (i <- 0 until tensorLength) {
    val inWrData = io.tensor.wr.bits.data(i).asUInt.asTypeOf(wdata_t)
    when (io.tensor.wr.valid) {
      tensorFile(i).write(io.tensor.wr.bits.idx, inWrData, no_mask)
    }
  }

  // read-from-sram
  val stride = state === sWriteAck &
              io.vme_wr.ack &
              xcnt === xlen + 1.U &
	      xrem === 0.U &
	      ycnt =/= ysize - 1.U

  when (state === sIdle) {
    ycnt := 0.U
  } .elsewhen (stride) {
    ycnt := ycnt + 1.U
  }

  when (state === sWriteCmd || tag === (numMemBlock - 1).U) {
    tag := 0.U
  } .elsewhen (io.vme_wr.data.fire()) {
    tag := tag + 1.U
  }

  when (state === sWriteCmd || (set === (tensorLength - 1).U && tag === (numMemBlock - 1).U)) {
    set := 0.U
  } .elsewhen (io.vme_wr.data.fire() && tag === (numMemBlock - 1).U) {
    set := set + 1.U
  }

  val raddr_cur = Reg(UInt(tp.memAddrBits.W))
  val raddr_nxt = Reg(UInt(tp.memAddrBits.W))
  when (state === sIdle) {
    raddr_cur := dec.sram_offset
    raddr_nxt := dec.sram_offset
  } .elsewhen (io.vme_wr.data.fire() && set === (tensorLength - 1).U && tag === (numMemBlock - 1).U) {
    raddr_cur := raddr_cur + 1.U
  } .elsewhen (stride) {
    raddr_cur := raddr_nxt + dec.xsize
    raddr_nxt := raddr_nxt + dec.xsize
  }

  val tread = Seq.tabulate(tensorLength) { i => i.U ->
    tensorFile(i).read(raddr_cur, state === sWriteCmd | state === sReadMem) }
  val mdata = MuxLookup(set, 0.U.asTypeOf(chiselTypeOf(wdata_t)), tread)

  // write-to-dram
  when (state === sIdle) {
    waddr_cur := io.baddr + dec.dram_offset
    waddr_nxt := io.baddr + dec.dram_offset
  } .elsewhen (state === sWriteAck && io.vme_wr.ack && xrem =/= 0.U) {
    waddr_cur := waddr_cur + xmax_bytes
  } .elsewhen (stride) {
    waddr_cur := waddr_nxt + (dec.xstride << log2Ceil(tensorLength*tensorWidth))
    waddr_nxt := waddr_nxt + (dec.xstride << log2Ceil(tensorLength*tensorWidth))
  }

  io.vme_wr.cmd.valid := state === sWriteCmd
  io.vme_wr.cmd.bits.addr := waddr_cur
  io.vme_wr.cmd.bits.len := xlen

  io.vme_wr.data.valid := state === sWriteData
  io.vme_wr.data.bits := mdata(tag)

  when (state === sWriteCmd) {
    xcnt := 0.U
  } .elsewhen (io.vme_wr.data.fire()) {
    xcnt := xcnt + 1.U
  }

  // disable external read-from-sram requests
  io.tensor.tieoffRead()

  // done
  io.done := state === sWriteAck & io.vme_wr.ack & xrem === 0.U & ycnt === ysize - 1.U

  // debug
  if (debug) {
    when (io.vme_wr.cmd.fire()) {
      printf("[TensorStore] ysize:%x ycnt:%x raddr:%x waddr:%x len:%x rem:%x\n", ysize, ycnt, raddr_cur, waddr_cur, xlen, xrem)
    }
    when (io.vme_wr.data.fire()) {
      printf("[TensorStore] data:%x\n", io.vme_wr.data.bits)
    }
    when (io.vme_wr.ack) {
      printf("[TensorStore] ack\n")
    }
  }
}
