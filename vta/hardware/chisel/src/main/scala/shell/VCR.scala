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

package vta.shell

import chisel3._
import chisel3.util._
import vta.util.config._
import vta.util.genericbundle._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.LinkedHashMap
import vta.interface.axi._

/** VCR parameters.
  *
  * These parameters are used on VCR interfaces and modules.
  */
case class VCRParams()
{
  val nValsReg: Int = 1
  val nPtrsReg: Int = 6
  val regBits: Int = 32
  val nCtrlReg: Int = 4
  val ctrlBaseAddr: Int = 0

  require (nValsReg > 0)
  require (nPtrsReg > 0)
}

/** VCRBase. Parametrize base class. */
abstract class VCRBase(implicit p: Parameters)
  extends GenericParameterizedBundle(p)

/** VCRMaster.
  *
  * This is the master interface used by VCR in the VTAShell to control
  * the Core unit.
  */
class VCRMaster(implicit p: Parameters) extends VCRBase {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val launch = Output(Bool())
  val finish = Input(Bool())
  val irq = Output(Bool())
  val ptrs = Output(Vec(vp.nPtrsReg, UInt(mp.addrBits.W)))
  val vals = Output(Vec(vp.nValsReg, UInt(vp.regBits.W)))
}

/** VCRClient.
  *
  * This is the client interface used by the Core module to communicate
  * to the VCR in the VTAShell.
  */
class VCRClient(implicit p: Parameters) extends VCRBase {
  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val launch = Input(Bool())
  val finish = Output(Bool())
  val irq = Input(Bool())
  val ptrs = Input(Vec(vp.nPtrsReg, UInt(mp.addrBits.W)))
  val vals = Input(Vec(vp.nValsReg, UInt(vp.regBits.W)))
}

/** VTA Control Registers (VCR).
  *
  * This unit provides control registers (32 and 64 bits) to be used by a control'
  * unit, typically a host processor. These registers are read-only by the core
  * at the moment but this will likely change once we add support to general purpose
  * registers that could be used as event counters by the Core unit.
  */
class VCR(implicit p: Parameters) extends Module {
  val io = IO(new Bundle{
    val host = new AXILiteClient(p(ShellKey).hostParams)
    val vcr = new VCRMaster
  })

  val vp = p(ShellKey).vcrParams
  val mp = p(ShellKey).memParams
  val hp = p(ShellKey).hostParams

  // Write control (AW, W, B)
  val waddr = RegInit("h_ffff".U(hp.addrBits.W)) // init with invalid address
  val wdata = io.host.w.bits.data
  val wstrb = io.host.w.bits.strb
  val wmask = Cat(Fill(8, wstrb(3)), Fill(8, wstrb(2)), Fill(8, wstrb(1)), Fill(8, wstrb(0)))
  val sWriteAddress :: sWriteData :: sWriteResponse :: Nil = Enum(3)
  val wstate = RegInit(sWriteAddress)
  switch (wstate) {
    is (sWriteAddress) {
      when (io.host.aw.valid) {
        wstate := sWriteData
      }
    }
    is (sWriteData) {
      when (io.host.w.valid) {
        wstate := sWriteResponse
      }
    }
    is (sWriteResponse) {
      when (io.host.b.ready) {
        wstate := sWriteAddress
      }
    }
  }

  when (io.host.aw.fire()) { waddr := io.host.aw.bits.addr }

  io.host.aw.ready := wstate === sWriteAddress
  io.host.w.ready := wstate === sWriteData
  io.host.b.valid := wstate === sWriteResponse
  io.host.b.bits.resp := "h_0".U

  // read control (AR, R)
  val sReadAddress :: sReadData :: Nil = Enum(2)
  val rstate = RegInit(sReadAddress)

  switch (rstate) {
    is (sReadAddress) {
      when (io.host.ar.valid) {
        rstate := sReadData
      }
    }
    is (sReadData) {
      when (io.host.r.ready) {
        rstate := sReadAddress
      }
    }
  }

  io.host.ar.ready := rstate === sReadAddress
  io.host.r.valid := rstate === sReadData

  val nPtrsReg = vp.nPtrsReg
  val nValsReg = vp.nValsReg
  val regBits = vp.regBits
  val ptrsBits = mp.addrBits
  val nCtrlReg = vp.nCtrlReg
  val rStride = regBits/8
  val pStride = ptrsBits/8
  val ctrlBaseAddr = vp.ctrlBaseAddr
  val valsBaseAddr = ctrlBaseAddr + nCtrlReg*rStride
  val ptrsBaseAddr = valsBaseAddr + nValsReg*rStride

  val ctrlAddr = Seq.tabulate(nCtrlReg)(i => i*rStride + ctrlBaseAddr)
  val valsAddr = Seq.tabulate(nValsReg)(i => i*rStride + valsBaseAddr)

  val ptrsAddr = new ListBuffer[Int]()
  for (i <- 0 until nPtrsReg) {
    ptrsAddr += i*pStride + ptrsBaseAddr
    if (ptrsBits == 64) {
      ptrsAddr += i*pStride + rStride + ptrsBaseAddr
    }
  }

  // AP register
  val c0 = RegInit(VecInit(Seq.fill(regBits)(false.B)))

  // ap start
  when (io.host.w.fire() && waddr === ctrlAddr(0).asUInt && wstrb(0) && wdata(0)) {
    c0(0) := true.B
  } .elsewhen (io.vcr.finish) {
    c0(0) := false.B
  }

  // ap done = finish
  when (io.vcr.finish) {
    c0(1) := true.B
  } .elsewhen (io.host.ar.fire() && io.host.ar.bits.addr === ctrlAddr(0).asUInt) {
    c0(1) := false.B
  }

  val c1 = 0.U
  val c2 = 0.U
  val c3 = 0.U

  val ctrlRegList = List(c0, c1, c2, c3)

  io.vcr.launch := c0(0)

  // interrupts not supported atm
  io.vcr.irq := false.B

  // Write pointer and value registers
  val pvAddr = valsAddr ++ ptrsAddr
  val pvNumReg =  if (ptrsBits == 64) nValsReg + nPtrsReg*2 else nValsReg + nPtrsReg
  val pvReg = RegInit(VecInit(Seq.fill(pvNumReg)(0.U(regBits.W))))
  val pvRegList = new ListBuffer[UInt]()

  for (i <- 0 until pvNumReg) {
    when (io.host.w.fire() && (waddr === pvAddr(i).U)) {
      pvReg(i) := (wdata & wmask) | (pvReg(i) & ~wmask)
    }
    pvRegList += pvReg(i)
  }

  for (i <- 0 until nValsReg) {
    io.vcr.vals(i) := pvReg(i)
  }

  for (i <- 0 until nPtrsReg) {
    if (ptrsBits == 64) {
      io.vcr.ptrs(i) := Cat(pvReg(nValsReg + i*2 + 1), pvReg(nValsReg + i*2))
    } else {
      io.vcr.ptrs(i) := pvReg(nValsReg + i)
    }
  }

  // Read pointer and value registers
  val mapAddr = ctrlAddr ++ valsAddr ++ ptrsAddr
  val mapRegList = ctrlRegList ++ pvRegList

  val rdata = RegInit(0.U(regBits.W))
  val rmap = LinkedHashMap[Int,UInt]()

  val totalReg = mapRegList.length
  for (i <- 0 until totalReg) { rmap += mapAddr(i) -> mapRegList(i).asUInt }

  val decodeAddr = rmap map { case (k, _) => k -> (io.host.ar.bits.addr === k.asUInt) }

  when (io.host.ar.fire()) {
    rdata := Mux1H(for ((k, v) <- rmap) yield decodeAddr(k) -> v)
  }

  io.host.r.bits.resp := 0.U
  io.host.r.bits.data := rdata
}
