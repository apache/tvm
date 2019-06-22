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
import vta.interface.axi._

/** VCR parameters.
  *
  * These parameters are used on VCR interfaces and modules.
  */
case class VCRParams()
{
  val nCtrl = 1
  val nECnt = 1
  val nVals = 1
  val nPtrs = 6
  val regBits = 32
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
  val ecnt = Vec(vp.nECnt, Flipped(ValidIO(UInt(vp.regBits.W))))
  val vals = Output(Vec(vp.nVals, UInt(vp.regBits.W)))
  val ptrs = Output(Vec(vp.nPtrs, UInt(mp.addrBits.W)))
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
  val ecnt = Vec(vp.nECnt, ValidIO(UInt(vp.regBits.W)))
  val vals = Input(Vec(vp.nVals, UInt(vp.regBits.W)))
  val ptrs = Input(Vec(vp.nPtrs, UInt(mp.addrBits.W)))
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
  val sWriteAddress :: sWriteData :: sWriteResponse :: Nil = Enum(3)
  val wstate = RegInit(sWriteAddress)

  // read control (AR, R)
  val sReadAddress :: sReadData :: Nil = Enum(2)
  val rstate = RegInit(sReadAddress)
  val rdata = RegInit(0.U(vp.regBits.W))

  // registers
  val nTotal = vp.nCtrl + vp.nECnt + vp.nVals + (2*vp.nPtrs)
  val reg = Seq.fill(nTotal)(RegInit(0.U(vp.regBits.W)))
  val addr = Seq.tabulate(nTotal)(_ * 4)
  val reg_map = (addr zip reg)  map { case (a, r) => a.U -> r }
  val eo = vp.nCtrl
  val vo = eo + vp.nECnt
  val po = vo + vp.nVals

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
  io.host.b.bits.resp := 0.U


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
  io.host.r.bits.data := rdata
  io.host.r.bits.resp := 0.U

  when (io.vcr.finish) {
    reg(0) := "b_10".U
  } .elsewhen (io.host.w.fire() && addr(0).U === waddr) {
    reg(0) := wdata
  }

  for (i <- 0 until vp.nECnt) {
    when (io.vcr.ecnt(i).valid) {
      reg(eo + i) := io.vcr.ecnt(i).bits
    } .elsewhen (io.host.w.fire() && addr(eo + i).U === waddr) {
      reg(eo + i) := wdata
    }
  }

  for (i <- 0 until (vp.nVals + (2*vp.nPtrs))) {
    when (io.host.w.fire() && addr(vo + i).U === waddr) {
      reg(vo + i) := wdata
    }
  }

  when (io.host.ar.fire()) {
    rdata := MuxLookup(io.host.ar.bits.addr, 0.U, reg_map)
  }

  io.vcr.launch := reg(0)(0)

  for (i <- 0 until vp.nVals) {
    io.vcr.vals(i) := reg(vo + i)
  }

  for (i <- 0 until vp.nPtrs) {
    io.vcr.ptrs(i) := Cat(reg(po + 2*i + 1), reg(po + 2*i))
  }
}
