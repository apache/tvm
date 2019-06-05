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

package vta.interface.axi

import chisel3._
import chisel3.util._
import vta.util.genericbundle._

case class AXIParams(
  addrBits: Int = 32,
  dataBits: Int = 64
)
{
  require (addrBits > 0)
  require (dataBits >= 8 && dataBits % 2 == 0)

  val idBits = 1
  val userBits = 1
  val strbBits = dataBits/8
  val lenBits = 8
  val sizeBits = 3
  val burstBits = 2
  val lockBits = 2
  val cacheBits = 4
  val protBits = 3
  val qosBits = 4
  val regionBits = 4
  val respBits = 2
  val sizeConst = log2Ceil(dataBits/8)
  val idConst = 0
  val userConst = 0
  val burstConst = 1
  val lockConst = 0
  val cacheConst = 3
  val protConst = 0
  val qosConst = 0
  val regionConst = 0
}

abstract class AXIBase(params: AXIParams)
  extends GenericParameterizedBundle(params)

// AXILite

class AXILiteAddress(params: AXIParams) extends AXIBase(params) {
  val addr = UInt(params.addrBits.W)
}

class AXILiteWriteData(params: AXIParams) extends AXIBase(params) {
  val data = UInt(params.dataBits.W)
  val strb = UInt(params.strbBits.W)
}

class AXILiteWriteResponse(params: AXIParams) extends AXIBase(params) {
  val resp = UInt(params.respBits.W)
}

class AXILiteReadData(params: AXIParams) extends AXIBase(params) {
  val data = UInt(params.dataBits.W)
  val resp = UInt(params.respBits.W)
}

class AXILiteMaster(params: AXIParams) extends AXIBase(params) {
  val aw = Decoupled(new AXILiteAddress(params))
  val w = Decoupled(new AXILiteWriteData(params))
  val b = Flipped(Decoupled(new AXILiteWriteResponse(params)))
  val ar = Decoupled(new AXILiteAddress(params))
  val r = Flipped(Decoupled(new AXILiteReadData(params)))

  def tieoff() {
    aw.valid := false.B
    aw.bits.addr := 0.U
    w.valid := false.B
    w.bits.data := 0.U
    w.bits.strb := 0.U
    b.ready := false.B
    ar.valid := false.B
    ar.bits.addr := 0.U
    r.ready := false.B
  }
}

class AXILiteClient(params: AXIParams) extends AXIBase(params) {
  val aw = Flipped(Decoupled(new AXILiteAddress(params)))
  val w = Flipped(Decoupled(new AXILiteWriteData(params)))
  val b = Decoupled(new AXILiteWriteResponse(params))
  val ar = Flipped(Decoupled(new AXILiteAddress(params)))
  val r = Decoupled(new AXILiteReadData(params))

  def tieoff() {
    aw.ready := false.B
    w.ready := false.B
    b.valid := false.B
    b.bits.resp := 0.U
    ar.ready := false.B
    r.valid := false.B
    r.bits.resp := 0.U
    r.bits.data := 0.U
  }
}

// AXI extends AXILite

class AXIAddress(params: AXIParams) extends AXILiteAddress(params) {
  val id = UInt(params.idBits.W)
  val user = UInt(params.userBits.W)
  val len = UInt(params.lenBits.W)
  val size = UInt(params.sizeBits.W)
  val burst = UInt(params.burstBits.W)
  val lock = UInt(params.lockBits.W)
  val cache = UInt(params.cacheBits.W)
  val prot = UInt(params.protBits.W)
  val qos = UInt(params.qosBits.W)
  val region = UInt(params.regionBits.W)
}

class AXIWriteData(params: AXIParams) extends AXILiteWriteData(params) {
  val last = Bool()
  val id = UInt(params.idBits.W)
  val user = UInt(params.userBits.W)
}

class AXIWriteResponse(params: AXIParams) extends AXILiteWriteResponse(params) {
  val id = UInt(params.idBits.W)
  val user = UInt(params.userBits.W)
}

class AXIReadData(params: AXIParams) extends AXILiteReadData(params) {
  val last = Bool()
  val id = UInt(params.idBits.W)
  val user = UInt(params.userBits.W)
}

class AXIMaster(params: AXIParams) extends AXIBase(params) {
  val aw = Decoupled(new AXIAddress(params))
  val w = Decoupled(new AXIWriteData(params))
  val b = Flipped(Decoupled(new AXIWriteResponse(params)))
  val ar = Decoupled(new AXIAddress(params))
  val r = Flipped(Decoupled(new AXIReadData(params)))

  def tieoff() {
    aw.valid := false.B
    aw.bits.addr := 0.U
    aw.bits.id := 0.U
    aw.bits.user := 0.U
    aw.bits.len := 0.U
    aw.bits.size := 0.U
    aw.bits.burst := 0.U
    aw.bits.lock := 0.U
    aw.bits.cache := 0.U
    aw.bits.prot := 0.U
    aw.bits.qos := 0.U
    aw.bits.region := 0.U
    w.valid := false.B
    w.bits.data := 0.U
    w.bits.strb := 0.U
    w.bits.last := false.B
    w.bits.id := 0.U
    w.bits.user := 0.U
    b.ready := false.B
    ar.valid := false.B
    ar.bits.addr := 0.U
    ar.bits.id := 0.U
    ar.bits.user := 0.U
    ar.bits.len := 0.U
    ar.bits.size := 0.U
    ar.bits.burst := 0.U
    ar.bits.lock := 0.U
    ar.bits.cache := 0.U
    ar.bits.prot := 0.U
    ar.bits.qos := 0.U
    ar.bits.region := 0.U
    r.ready := false.B
  }

  def setConst() {
    aw.bits.user := params.userConst.U
    aw.bits.burst := params.burstConst.U
    aw.bits.lock := params.lockConst.U
    aw.bits.cache := params.cacheConst.U
    aw.bits.prot := params.protConst.U
    aw.bits.qos := params.qosConst.U
    aw.bits.region := params.regionConst.U
    aw.bits.size := params.sizeConst.U
    aw.bits.id := params.idConst.U
    w.bits.id := params.idConst.U
    w.bits.user := params.userConst.U
    w.bits.strb := Fill(params.strbBits, true.B)
    ar.bits.user := params.userConst.U
    ar.bits.burst := params.burstConst.U
    ar.bits.lock := params.lockConst.U
    ar.bits.cache := params.cacheConst.U
    ar.bits.prot := params.protConst.U
    ar.bits.qos := params.qosConst.U
    ar.bits.region := params.regionConst.U
    ar.bits.size := params.sizeConst.U
    ar.bits.id := params.idConst.U
  }
}

class AXIClient(params: AXIParams) extends AXIBase(params) {
  val aw = Flipped(Decoupled(new AXIAddress(params)))
  val w = Flipped(Decoupled(new AXIWriteData(params)))
  val b = Decoupled(new AXIWriteResponse(params))
  val ar = Flipped(Decoupled(new AXIAddress(params)))
  val r = Decoupled(new AXIReadData(params))

  def tieoff() {
    aw.ready := false.B
    w.ready := false.B
    b.valid := false.B
    b.bits.resp := 0.U
    b.bits.user := 0.U
    b.bits.id := 0.U
    ar.ready := false.B
    r.valid := false.B
    r.bits.resp := 0.U
    r.bits.data := 0.U
    r.bits.user := 0.U
    r.bits.last := false.B
    r.bits.id := 0.U
  }
}

// XilinxAXILiteClient and XilinxAXIMaster bundles are needed
// for wrapper purposes, because the package RTL tool in Xilinx Vivado
// only allows certain name formats

class XilinxAXILiteClient(params: AXIParams) extends AXIBase(params) {
  val AWVALID = Input(Bool())
  val AWREADY = Output(Bool())
  val AWADDR = Input(UInt(params.addrBits.W))
  val WVALID = Input(Bool())
  val WREADY = Output(Bool())
  val WDATA = Input(UInt(params.dataBits.W))
  val WSTRB = Input(UInt(params.strbBits.W))
  val BVALID = Output(Bool())
  val BREADY = Input(Bool())
  val BRESP = Output(UInt(params.respBits.W))
  val ARVALID = Input(Bool())
  val ARREADY = Output(Bool())
  val ARADDR = Input(UInt(params.addrBits.W))
  val RVALID = Output(Bool())
  val RREADY = Input(Bool())
  val RDATA = Output(UInt(params.dataBits.W))
  val RRESP = Output(UInt(params.respBits.W))
}

class XilinxAXIMaster(params: AXIParams) extends AXIBase(params) {
  val AWVALID = Output(Bool())
  val AWREADY = Input(Bool())
  val AWADDR = Output(UInt(params.addrBits.W))
  val AWID = Output(UInt(params.idBits.W))
  val AWUSER = Output(UInt(params.userBits.W))
  val AWLEN = Output(UInt(params.lenBits.W))
  val AWSIZE = Output(UInt(params.sizeBits.W))
  val AWBURST = Output(UInt(params.burstBits.W))
  val AWLOCK = Output(UInt(params.lockBits.W))
  val AWCACHE = Output(UInt(params.cacheBits.W))
  val AWPROT = Output(UInt(params.protBits.W))
  val AWQOS = Output(UInt(params.qosBits.W))
  val AWREGION = Output(UInt(params.regionBits.W))
  val WVALID = Output(Bool())
  val WREADY = Input(Bool())
  val WDATA = Output(UInt(params.dataBits.W))
  val WSTRB = Output(UInt(params.strbBits.W))
  val WLAST = Output(Bool())
  val WID = Output(UInt(params.idBits.W))
  val WUSER = Output(UInt(params.userBits.W))
  val BVALID = Input(Bool())
  val BREADY = Output(Bool())
  val BRESP = Input(UInt(params.respBits.W))
  val BID = Input(UInt(params.idBits.W))
  val BUSER = Input(UInt(params.userBits.W))
  val ARVALID = Output(Bool())
  val ARREADY = Input(Bool())
  val ARADDR = Output(UInt(params.addrBits.W))
  val ARID = Output(UInt(params.idBits.W))
  val ARUSER = Output(UInt(params.userBits.W))
  val ARLEN = Output(UInt(params.lenBits.W))
  val ARSIZE = Output(UInt(params.sizeBits.W))
  val ARBURST = Output(UInt(params.burstBits.W))
  val ARLOCK = Output(UInt(params.lockBits.W))
  val ARCACHE = Output(UInt(params.cacheBits.W))
  val ARPROT = Output(UInt(params.protBits.W))
  val ARQOS = Output(UInt(params.qosBits.W))
  val ARREGION = Output(UInt(params.regionBits.W))
  val RVALID = Input(Bool())
  val RREADY = Output(Bool())
  val RDATA = Input(UInt(params.dataBits.W))
  val RRESP = Input(UInt(params.respBits.W))
  val RLAST = Input(Bool())
  val RID = Input(UInt(params.idBits.W))
  val RUSER = Input(UInt(params.userBits.W))
}
