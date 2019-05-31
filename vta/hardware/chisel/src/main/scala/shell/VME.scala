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

/** VME parameters.
  *
  * These parameters are used on VME interfaces and modules.
  */
case class VMEParams() {
  val nReadClients: Int = 5
  val nWriteClients: Int = 1
  require (nReadClients > 0, s"\n\n[VTA] [VMEParams] nReadClients must be larger than 0\n\n")
  require (nWriteClients == 1, s"\n\n[VTA] [VMEParams] nWriteClients must be 1, only one-write-client support atm\n\n")
}

/** VMEBase. Parametrize base class. */
abstract class VMEBase(implicit p: Parameters)
  extends GenericParameterizedBundle(p)

/** VMECmd.
  *
  * This interface is used for creating write and read requests to memory.
  */
class VMECmd(implicit p: Parameters) extends VMEBase {
  val addrBits = p(ShellKey).memParams.addrBits
  val lenBits = p(ShellKey).memParams.lenBits
  val addr = UInt(addrBits.W)
  val len = UInt(lenBits.W)
}

/** VMEReadMaster.
  *
  * This interface is used by modules inside the core to generate read requests
  * and receive responses from VME.
  */
class VMEReadMaster(implicit p: Parameters) extends Bundle {
  val dataBits = p(ShellKey).memParams.dataBits
  val cmd = Decoupled(new VMECmd)
  val data = Flipped(Decoupled(UInt(dataBits.W)))
  override def cloneType =
    new VMEReadMaster().asInstanceOf[this.type]
}

/** VMEReadClient.
  *
  * This interface is used by the VME to receive read requests and generate
  * responses to modules inside the core.
  */
class VMEReadClient(implicit p: Parameters) extends Bundle {
  val dataBits = p(ShellKey).memParams.dataBits
  val cmd = Flipped(Decoupled(new VMECmd))
  val data = Decoupled(UInt(dataBits.W))
  override def cloneType =
    new VMEReadClient().asInstanceOf[this.type]
}

/** VMEWriteMaster.
  *
  * This interface is used by modules inside the core to generate write requests
  * to the VME.
  */
class VMEWriteMaster(implicit p: Parameters) extends Bundle {
  val dataBits = p(ShellKey).memParams.dataBits
  val cmd = Decoupled(new VMECmd)
  val data = Decoupled(UInt(dataBits.W))
  val ack = Input(Bool())
  override def cloneType =
    new VMEWriteMaster().asInstanceOf[this.type]
}

/** VMEWriteClient.
  *
  * This interface is used by the VME to handle write requests from modules inside
  * the core.
  */
class VMEWriteClient(implicit p: Parameters) extends Bundle {
  val dataBits = p(ShellKey).memParams.dataBits
  val cmd = Flipped(Decoupled(new VMECmd))
  val data = Flipped(Decoupled(UInt(dataBits.W)))
  val ack = Output(Bool())
  override def cloneType =
    new VMEWriteClient().asInstanceOf[this.type]
}

/** VMEMaster.
  *
  * Pack nRd number of VMEReadMaster interfaces and nWr number of VMEWriteMaster
  * interfaces.
  */
class VMEMaster(implicit p: Parameters) extends Bundle {
  val nRd = p(ShellKey).vmeParams.nReadClients
  val nWr = p(ShellKey).vmeParams.nWriteClients
  val rd = Vec(nRd, new VMEReadMaster)
  val wr = Vec(nWr, new VMEWriteMaster)
}

/** VMEClient.
  *
  * Pack nRd number of VMEReadClient interfaces and nWr number of VMEWriteClient
  * interfaces.
  */
class VMEClient(implicit p: Parameters) extends Bundle {
  val nRd = p(ShellKey).vmeParams.nReadClients
  val nWr = p(ShellKey).vmeParams.nWriteClients
  val rd = Vec(nRd, new VMEReadClient)
  val wr = Vec(nWr, new VMEWriteClient)
}

/** VTA Memory Engine (VME).
  *
  * This unit multiplexes the memory controller interface for the Core. Currently,
  * it supports single-writer and multiple-reader mode and it is also based on AXI.
  */
class VME(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val mem = new AXIMaster(p(ShellKey).memParams)
    val vme = new VMEClient
  })

  val nReadClients = p(ShellKey).vmeParams.nReadClients
  val rd_arb = Module(new Arbiter(new VMECmd, nReadClients))
  val rd_arb_chosen = RegEnable(rd_arb.io.chosen, rd_arb.io.out.fire())

  for (i <- 0 until nReadClients) { rd_arb.io.in(i) <> io.vme.rd(i).cmd }

  val sReadIdle :: sReadAddr :: sReadData :: Nil = Enum(3)
  val rstate = RegInit(sReadIdle)

  switch (rstate) {
    is (sReadIdle) {
      when (rd_arb.io.out.valid) {
        rstate := sReadAddr
      }
    }
    is (sReadAddr) {
      when (io.mem.ar.ready) {
        rstate := sReadData
      }
    }
    is (sReadData) {
      when (io.mem.r.fire() && io.mem.r.bits.last) {
        rstate := sReadIdle
      }
    }
  }

  val sWriteIdle :: sWriteAddr :: sWriteData :: sWriteResp :: Nil = Enum(4)
  val wstate = RegInit(sWriteIdle)
  val addrBits = p(ShellKey).memParams.addrBits
  val lenBits = p(ShellKey).memParams.lenBits
  val wr_cnt = RegInit(0.U(lenBits.W))

  when (wstate === sWriteIdle) {
    wr_cnt := 0.U
  } .elsewhen (io.mem.w.fire()) {
    wr_cnt := wr_cnt + 1.U
  }

  switch (wstate) {
    is (sWriteIdle) {
      when (io.vme.wr(0).cmd.valid) {
        wstate := sWriteAddr
      }
    }
    is (sWriteAddr) {
      when (io.mem.aw.ready) {
        wstate := sWriteData
      }
    }
    is (sWriteData) {
      when (io.mem.w.ready && wr_cnt === io.vme.wr(0).cmd.bits.len) {
        wstate := sWriteResp
      }
    }
    is (sWriteResp) {
      when (io.mem.b.valid) {
        wstate := sWriteIdle
      }
    }
  }

  // registers storing read/write cmds

  val rd_len = RegInit(0.U(lenBits.W))
  val wr_len = RegInit(0.U(lenBits.W))
  val rd_addr = RegInit(0.U(addrBits.W))
  val wr_addr = RegInit(0.U(addrBits.W))

  when (rd_arb.io.out.fire()) {
    rd_len := rd_arb.io.out.bits.len
    rd_addr := rd_arb.io.out.bits.addr
  }

  when (io.vme.wr(0).cmd.fire()) {
    wr_len := io.vme.wr(0).cmd.bits.len
    wr_addr := io.vme.wr(0).cmd.bits.addr
  }

  // rd arb
  rd_arb.io.out.ready := rstate === sReadIdle

  // vme
  for (i <- 0 until nReadClients) {
    io.vme.rd(i).data.valid := rd_arb_chosen === i.asUInt & io.mem.r.valid
    io.vme.rd(i).data.bits := io.mem.r.bits.data
  }

  io.vme.wr(0).cmd.ready := wstate === sWriteIdle
  io.vme.wr(0).ack := io.mem.b.fire()
  io.vme.wr(0).data.ready := wstate === sWriteData &  io.mem.w.ready

  // mem
  io.mem.aw.valid := wstate === sWriteAddr
  io.mem.aw.bits.addr := wr_addr
  io.mem.aw.bits.len := wr_len

  io.mem.w.valid := wstate === sWriteData & io.vme.wr(0).data.valid
  io.mem.w.bits.data := io.vme.wr(0).data.bits
  io.mem.w.bits.last := wr_cnt === io.vme.wr(0).cmd.bits.len

  io.mem.b.ready := wstate === sWriteResp

  io.mem.ar.valid := rstate === sReadAddr
  io.mem.ar.bits.addr := rd_addr
  io.mem.ar.bits.len := rd_len

  io.mem.r.ready := rstate === sReadData & io.vme.rd(rd_arb_chosen).data.ready

  // AXI constants - statically defined
  io.mem.setConst()
}
