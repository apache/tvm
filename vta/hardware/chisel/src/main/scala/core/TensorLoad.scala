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
  * Load 1D and 2D tensors from main memory (DRAM) to input/weight
  * scratchpads (SRAM). Also, there is support for zero padding, while
  * doing the load. Zero-padding works on the y and x axis, and it is
  * managed by TensorPadCtrl. The TensorDataCtrl is in charge of
  * handling the way tensors are stored on the scratchpads.
  */
class TensorLoad(tensorType: String = "none", debug: Boolean = false)
  (implicit p: Parameters) extends Module {
  val tp = new TensorParams(tensorType)
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
    val baddr = Input(UInt(mp.addrBits.W))
    val vme_rd = new VMEReadMaster
    val tensor = new TensorClient(tensorType)
  })
  val sizeFactor = tp.tensorLength * tp.numMemBlock
  val strideFactor = tp.tensorLength * tp.tensorWidth

  val dec = io.inst.asTypeOf(new MemDecode)
  val dataCtrl = Module(new TensorDataCtrl(sizeFactor, strideFactor))
  val dataCtrlDone = RegInit(false.B)
  val yPadCtrl0 = Module(new TensorPadCtrl(padType = "YPad0", sizeFactor))
  val yPadCtrl1 = Module(new TensorPadCtrl(padType = "YPad1", sizeFactor))
  val xPadCtrl0 = Module(new TensorPadCtrl(padType = "XPad0", sizeFactor))
  val xPadCtrl1 = Module(new TensorPadCtrl(padType = "XPad1", sizeFactor))

  val tag = Reg(UInt(8.W))
  val set = Reg(UInt(8.W))

  val sIdle :: sYPad0 :: sXPad0 :: sReadCmd :: sReadData :: sXPad1 :: sYPad1 :: Nil = Enum(7)
  val state = RegInit(sIdle)

  // control
  switch (state) {
    is (sIdle) {
      when (io.start) {
        when (dec.ypad_0 =/= 0.U) {
          state := sYPad0
	} .elsewhen (dec.xpad_0 =/= 0.U) {
          state := sXPad0
	} .otherwise {
          state := sReadCmd
	}
      }
    }
    is (sYPad0) {
      when (yPadCtrl0.io.done) {
        when (dec.xpad_0 =/= 0.U) {
          state := sXPad0
	} .otherwise {
          state := sReadCmd
	}
      }
    }
    is (sXPad0) {
      when (xPadCtrl0.io.done) {
        state := sReadCmd
      }
    }
    is (sReadCmd) {
      when (io.vme_rd.cmd.ready) {
        state := sReadData
      }
    }
    is (sReadData) {
      when (io.vme_rd.data.valid) {
        when (dataCtrl.io.done) {
	  when (dec.xpad_1 =/= 0.U) {
	    state := sXPad1
	  } .elsewhen (dec.ypad_1 =/= 0.U) {
	    state := sYPad1
	  } .otherwise  {
	    state := sIdle
	  }
	} .elsewhen (dataCtrl.io.stride || dataCtrl.io.split) {
          when (dec.xpad_1 =/= 0.U) {
	    state := sXPad1
	  } .elsewhen (dec.xpad_0 =/= 0.U) {
            state := sXPad0
	  } .otherwise {
              state := sReadCmd
	  }
	}
      }
    }
    is (sXPad1) {
      when (xPadCtrl1.io.done) {
        when (dataCtrlDone) {
          when (dec.ypad_1 =/= 0.U) {
            state := sYPad1
          } .otherwise {
            state := sIdle
          }
        } .otherwise {
          when (dec.xpad_0 =/= 0.U) {
            state := sXPad0
          } .otherwise {
            state := sReadCmd
          }
        }
      }
    }
    is (sYPad1) {
      when (yPadCtrl1.io.done && dataCtrlDone) {
        state := sIdle
      }
    }
  }

  // data controller
  dataCtrl.io.start := state === sIdle & io.start
  dataCtrl.io.inst := io.inst
  dataCtrl.io.baddr := io.baddr
  dataCtrl.io.xinit := io.vme_rd.cmd.fire()
  dataCtrl.io.xupdate := io.vme_rd.data.fire()
  dataCtrl.io.yupdate := io.vme_rd.data.fire()

  when (state === sIdle) {
    dataCtrlDone := false.B
  } .elsewhen (io.vme_rd.data.fire() && dataCtrl.io.done) {
    dataCtrlDone := true.B
  }

  // pad
  yPadCtrl0.io.start := dec.ypad_0 =/= 0.U & state === sIdle & io.start

  yPadCtrl1.io.start := dec.ypad_1 =/= 0.U &
                          ((io.vme_rd.data.fire() & dataCtrl.io.done & dec.xpad_1 === 0.U) |
                           (state === sXPad1 & xPadCtrl1.io.done & dataCtrlDone))

  xPadCtrl0.io.start := dec.xpad_0 =/= 0.U &
                          ((state === sIdle & io.start) |
			  (state === sYPad0 & yPadCtrl0.io.done) |
                          (io.vme_rd.data.fire() & ~dataCtrlDone & (dataCtrl.io.stride | dataCtrl.io.split) & dec.xpad_1 === 0.U) |
			  (state === sXPad1 & xPadCtrl1.io.done & ~dataCtrlDone))

  xPadCtrl1.io.start := dec.xpad_1 =/= 0.U & io.vme_rd.data.fire() &
                          ((dataCtrl.io.done) |
                          (~dataCtrl.io.done & (dataCtrl.io.stride | dataCtrl.io.split) & dec.xpad_1 =/= 0.U))

  yPadCtrl0.io.inst := io.inst
  yPadCtrl1.io.inst := io.inst
  xPadCtrl0.io.inst := io.inst
  xPadCtrl1.io.inst := io.inst

  // read-from-dram
  io.vme_rd.cmd.valid := state === sReadCmd
  io.vme_rd.cmd.bits.addr := dataCtrl.io.addr
  io.vme_rd.cmd.bits.len := dataCtrl.io.len

  io.vme_rd.data.ready := state === sReadData

  // write-to-sram
  val isZeroPad = state === sYPad0 |
                  state === sXPad0 |
		  state === sXPad1 |
		  state === sYPad1

  when (state === sIdle || state === sReadCmd || tag === (tp.numMemBlock - 1).U) {
    tag := 0.U
  } .elsewhen (io.vme_rd.data.fire() || isZeroPad) {
    tag := tag + 1.U
  }

  when (state === sIdle || state === sReadCmd || (set === (tp.tensorLength - 1).U && tag === (tp.numMemBlock - 1).U)) {
    set := 0.U
  } .elsewhen ((io.vme_rd.data.fire() || isZeroPad) && tag === (tp.numMemBlock - 1).U) {
    set := set + 1.U
  }

  val waddr_cur = Reg(UInt(tp.memAddrBits.W))
  val waddr_nxt = Reg(UInt(tp.memAddrBits.W))
  when (state === sIdle) {
    waddr_cur := dec.sram_offset
    waddr_nxt := dec.sram_offset
  } .elsewhen ((io.vme_rd.data.fire() || isZeroPad) && set === (tp.tensorLength - 1).U && tag === (tp.numMemBlock - 1).U) {
    waddr_cur := waddr_cur + 1.U
  } .elsewhen (dataCtrl.io.stride) {
    waddr_cur := waddr_nxt + dec.xsize
    waddr_nxt := waddr_nxt + dec.xsize
  }

  val tensorFile = Seq.fill(tp.tensorLength) { SyncReadMem(tp.memDepth, Vec(tp.numMemBlock, UInt(tp.memBlockBits.W))) }
  val wmask = Seq.fill(tp.tensorLength) { Wire(Vec(tp.numMemBlock, Bool())) }
  val wdata = Seq.fill(tp.tensorLength) { Wire(Vec(tp.numMemBlock, UInt(tp.memBlockBits.W))) }
  val no_mask = Wire(Vec(tp.numMemBlock, Bool()))
  no_mask.foreach { m => m := true.B }

  for (i <- 0 until tp.tensorLength) {
    for (j <- 0 until tp.numMemBlock) {
      wmask(i)(j) := tag === j.U
      wdata(i)(j) := Mux(isZeroPad, 0.U, io.vme_rd.data.bits)
    }
    val tdata = io.tensor.wr.bits.data(i).asUInt.asTypeOf(wdata(i))
    val muxWen = Mux(state === sIdle, io.tensor.wr.valid, (io.vme_rd.data.fire() | isZeroPad) & set === i.U)
    val muxWaddr = Mux(state === sIdle, io.tensor.wr.bits.idx, waddr_cur)
    val muxWdata = Mux(state === sIdle, tdata, wdata(i))
    val muxWmask = Mux(state === sIdle, no_mask, wmask(i))
    when (muxWen) {
      tensorFile(i).write(muxWaddr, muxWdata, muxWmask)
    }
  }

  // read-from-sram
  val rvalid = RegNext(io.tensor.rd.idx.valid)
  io.tensor.rd.data.valid := rvalid

  val rdata = tensorFile.map(_.read(io.tensor.rd.idx.bits, io.tensor.rd.idx.valid))
  rdata.zipWithIndex.foreach { case(r, i) =>
    io.tensor.rd.data.bits(i) := r.asUInt.asTypeOf(io.tensor.rd.data.bits(i))
  }

  // done
  val done_no_pad = io.vme_rd.data.fire() & dataCtrl.io.done & dec.xpad_1 === 0.U & dec.ypad_1 === 0.U
  val done_x_pad = state === sXPad1 & xPadCtrl1.io.done & dataCtrlDone & dec.ypad_1 === 0.U
  val done_y_pad = state === sYPad1 & dataCtrlDone & yPadCtrl1.io.done
  io.done := done_no_pad | done_x_pad | done_y_pad

  // debug
  if (debug) {
    if (tensorType == "inp") {
      when (io.vme_rd.cmd.fire()) {
        printf("[TensorLoad] [inp] cmd addr:%x len:%x\n", dataCtrl.io.addr, dataCtrl.io.len)
      }
      when (state === sYPad0) {
        printf("[TensorLoad] [inp] sYPad0\n")
      }
      when (state === sYPad1) {
        printf("[TensorLoad] [inp] sYPad1\n")
      }
      when (state === sXPad0) {
        printf("[TensorLoad] [inp] sXPad0\n")
      }
      when (state === sXPad1) {
        printf("[TensorLoad] [inp] sXPad1\n")
      }
    } else if (tensorType == "wgt") {
      when (io.vme_rd.cmd.fire()) {
        printf("[TensorLoad] [wgt] cmd addr:%x len:%x\n", dataCtrl.io.addr, dataCtrl.io.len)
      }
    } else if (tensorType == "acc") {
      when (io.vme_rd.cmd.fire()) {
        printf("[TensorLoad] [acc] cmd addr:%x len:%x\n", dataCtrl.io.addr, dataCtrl.io.len)
      }
    }
  }
}
