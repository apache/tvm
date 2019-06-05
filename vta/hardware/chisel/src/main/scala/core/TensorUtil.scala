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

/** TensorParams.
  *
  * This Bundle derives parameters for each tensorType, including inputs (inp),
  * weights (wgt), biases (acc), and outputs (out). This is used to avoid
  * doing the same boring calculations over and over again.
  */
class TensorParams(tensorType: String = "none")(implicit p: Parameters) extends Bundle {
  val errorMsg = s"\n\n[VTA] [TensorParams] only inp, wgt, acc, and out supported\n\n"

  require (tensorType == "inp" || tensorType == "wgt"
    || tensorType == "acc" || tensorType == "out", errorMsg)

  val (tensorLength, tensorWidth, tensorElemBits) =
    if (tensorType == "inp")
      (p(CoreKey).batch, p(CoreKey).blockIn, p(CoreKey).inpBits)
    else if (tensorType == "wgt")
      (p(CoreKey).blockOut, p(CoreKey).blockIn, p(CoreKey).wgtBits)
    else if (tensorType == "acc")
      (p(CoreKey).batch, p(CoreKey).blockOut, p(CoreKey).accBits)
    else
      (p(CoreKey).batch, p(CoreKey).blockOut, p(CoreKey).outBits)

  val memBlockBits = p(ShellKey).memParams.dataBits
  val numMemBlock = (tensorWidth * tensorElemBits) / memBlockBits

  val memDepth =
    if (tensorType == "inp")
      p(CoreKey).inpMemDepth
    else if (tensorType == "wgt")
      p(CoreKey).wgtMemDepth
    else if (tensorType == "acc")
      p(CoreKey).accMemDepth
    else
      p(CoreKey).outMemDepth

  val memAddrBits = log2Ceil(memDepth)
}

/** TensorMaster.
  *
  * This interface issue read and write tensor-requests to scratchpads. For example,
  * The TensorGemm unit uses this interface for managing the inputs (inp), weights (wgt),
  * biases (acc), and outputs (out).
  *
  */
class TensorMaster(tensorType: String = "none")
  (implicit p: Parameters) extends TensorParams(tensorType) {
    val rd = new Bundle {
      val idx = ValidIO(UInt(memAddrBits.W))
      val data = Flipped(ValidIO(Vec(tensorLength, Vec(tensorWidth, UInt(tensorElemBits.W)))))
    }
    val wr = ValidIO(new Bundle {
      val idx = UInt(memAddrBits.W)
      val data = Vec(tensorLength, Vec(tensorWidth, UInt(tensorElemBits.W)))
    })
    def tieoffRead() {
      rd.idx.valid := false.B
      rd.idx.bits := 0.U
    }
    def tieoffWrite() {
      wr.valid := false.B
      wr.bits.idx := 0.U
      wr.bits.data.foreach { b => b.foreach { c => c := 0.U } }
    }
  override def cloneType =
    new TensorMaster(tensorType).asInstanceOf[this.type]
}

/** TensorClient.
  *
  * This interface receives read and write tensor-requests to scratchpads. For example,
  * The TensorLoad unit uses this interface for receiving read and write requests from
  * the TensorGemm unit.
  */
class TensorClient(tensorType: String = "none")
  (implicit p: Parameters) extends TensorParams(tensorType) {
    val rd = new Bundle {
      val idx = Flipped(ValidIO(UInt(memAddrBits.W)))
      val data = ValidIO(Vec(tensorLength, Vec(tensorWidth, UInt(tensorElemBits.W))))
    }
    val wr = Flipped(ValidIO(new Bundle {
      val idx = UInt(memAddrBits.W)
      val data = Vec(tensorLength, Vec(tensorWidth, UInt(tensorElemBits.W)))
    }))
    def tieoffRead() {
      rd.data.valid := false.B
      rd.data.bits.foreach { b => b.foreach { c => c := 0.U } }
    }
  override def cloneType =
    new TensorClient(tensorType).asInstanceOf[this.type]
}

/** TensorMasterData.
  *
  * This interface is only used for datapath only purposes and the direction convention
  * is based on the TensorMaster interface, which means this is an input. This interface
  * is used on datapath only module such MatrixVectorCore or AluVector.
  */
class TensorMasterData(tensorType: String = "none")
  (implicit p: Parameters) extends TensorParams(tensorType) {
  val data = Flipped(ValidIO(Vec(tensorLength, Vec(tensorWidth, UInt(tensorElemBits.W)))))
  override def cloneType =
    new TensorMasterData(tensorType).asInstanceOf[this.type]
}

/** TensorClientData.
  *
  * This interface is only used for datapath only purposes and the direction convention
  * is based on the TensorClient interface, which means this is an output. This interface
  * is used on datapath only module such MatrixVectorCore or AluVector.
  */
class TensorClientData(tensorType: String = "none")
  (implicit p: Parameters) extends TensorParams(tensorType) {
  val data = ValidIO(Vec(tensorLength, Vec(tensorWidth, UInt(tensorElemBits.W))))
  override def cloneType =
    new TensorClientData(tensorType).asInstanceOf[this.type]
}

/** TensorPadCtrl. Zero-padding controller for TensorLoad. */
class TensorPadCtrl(padType: String = "none", sizeFactor: Int = 1) extends Module {
  val errorMsg = s"\n\n\n[VTA-ERROR] only YPad0, YPad1, XPad0, or XPad1 supported\n\n\n"
  require (padType == "YPad0" || padType == "YPad1"
    || padType == "XPad0" || padType == "XPad1", errorMsg)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
  })

  val dec = io.inst.asTypeOf(new MemDecode)

  val xmax = Reg(chiselTypeOf(dec.xsize))
  val ymax = Reg(chiselTypeOf(dec.ypad_0))
  val xcnt = Reg(chiselTypeOf(dec.xsize))
  val ycnt = Reg(chiselTypeOf(dec.ypad_0))

  val xval =
    if (padType == "YPad0" || padType == "YPad1")
      ((dec.xpad_0 + dec.xsize + dec.xpad_1) << log2Ceil(sizeFactor)) - 1.U
    else if (padType == "XPad0")
      (dec.xpad_0 << log2Ceil(sizeFactor)) - 1.U
    else
      (dec.xpad_1 << log2Ceil(sizeFactor)) - 1.U

  val yval =
    if (padType == "YPad0")
      Mux(dec.ypad_0 =/= 0.U, dec.ypad_0 - 1.U, 0.U)
    else if (padType == "YPad1")
      Mux(dec.ypad_1 =/= 0.U, dec.ypad_1 - 1.U, 0.U)
    else
      0.U

  val sIdle :: sActive :: Nil = Enum(2)
  val state = RegInit(sIdle)

  switch (state) {
    is (sIdle) {
      when (io.start) {
        state := sActive
      }
    }
    is (sActive) {
      when (ycnt === ymax && xcnt === xmax) {
        state := sIdle
      }
    }
  }

  when (state === sIdle) {
    xmax := xval
    ymax := yval
  }

  when (state === sIdle || xcnt === xmax) {
    xcnt := 0.U
  } .elsewhen (state === sActive) {
    xcnt := xcnt + 1.U
  }

  when (state === sIdle || ymax === 0.U) {
    ycnt := 0.U
  } .elsewhen (state === sActive && xcnt === xmax) {
    ycnt := ycnt + 1.U
  }

  io.done := state === sActive & ycnt === ymax & xcnt === xmax
}

/** TensorDataCtrl. Data controller for TensorLoad. */
class TensorDataCtrl(sizeFactor: Int = 1, strideFactor: Int = 1)(implicit p: Parameters) extends Module {
  val mp = p(ShellKey).memParams
  val io = IO(new Bundle {
    val start = Input(Bool())
    val done = Output(Bool())
    val inst = Input(UInt(INST_BITS.W))
    val baddr = Input(UInt(mp.addrBits.W))
    val xinit = Input(Bool())
    val xupdate = Input(Bool())
    val yupdate = Input(Bool())
    val stride = Output(Bool())
    val split = Output(Bool())
    val commit = Output(Bool())
    val addr = Output(UInt(mp.addrBits.W))
    val len = Output(UInt(mp.lenBits.W))
  })

  val dec = io.inst.asTypeOf(new MemDecode)

  val caddr = Reg(UInt(mp.addrBits.W))
  val baddr = Reg(UInt(mp.addrBits.W))

  val len = Reg(UInt(mp.lenBits.W))

  val xmax_bytes = ((1 << mp.lenBits)*mp.dataBits/8).U
  val xcnt = Reg(UInt(mp.lenBits.W))
  val xrem = Reg(chiselTypeOf(dec.xsize))
  val xsize = (dec.xsize << log2Ceil(sizeFactor)) - 1.U
  val xmax = (1 << mp.lenBits).U
  val ycnt = Reg(chiselTypeOf(dec.ysize))

  val stride = xcnt === len &
	       xrem === 0.U &
	       ycnt =/= dec.ysize - 1.U

  val split = xcnt === len & xrem =/= 0.U

  when (io.start || (io.xupdate && stride)) {
    when (xsize < xmax) {
      len := xsize
      xrem := 0.U
    } .otherwise {
      len := xmax - 1.U
      xrem := xsize - xmax
    }
  } .elsewhen (io.xupdate && split) {
    when (xrem < xmax) {
      len := xrem
      xrem := 0.U
    } .otherwise {
      len := xmax - 1.U
      xrem := xrem - xmax
    }
  }

  when (io.xinit) {
    xcnt := 0.U
  } .elsewhen (io.xupdate) {
    xcnt := xcnt + 1.U
  }

  when (io.start) {
    ycnt := 0.U
  } .elsewhen (io.yupdate && stride) {
    ycnt := ycnt + 1.U
  }

  when (io.start) {
    caddr := io.baddr + dec.dram_offset
    baddr := io.baddr + dec.dram_offset
  } .elsewhen (io.yupdate) {
    when (split) {
      caddr := caddr + xmax_bytes
    } .elsewhen (stride) {
      caddr := baddr + (dec.xstride << log2Ceil(strideFactor))
      baddr := baddr + (dec.xstride << log2Ceil(strideFactor))
    }
  }

  io.stride := stride
  io.split := split
  io.commit := xcnt === len
  io.addr := caddr
  io.len := len
  io.done := xcnt === len &
	     xrem === 0.U &
	     ycnt === dec.ysize - 1.U
}
