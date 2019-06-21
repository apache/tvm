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
import chisel3.experimental.{RawModule, withClockAndReset}
import vta.util.config._
import vta.interface.axi._

/** IntelShell.
  *
  * This is a wrapper shell mostly used to match Intel convention naming,
  * therefore we can pack VTA as an IP for IPI based flows.
  */
class IntelShell(implicit p: Parameters) extends RawModule {

  val hp = p(ShellKey).hostParams
  val mp = p(ShellKey).memParams

  val ap_clk = IO(Input(Clock()))
  val ap_rst_n = IO(Input(Bool()))
  val m_axi_gmem = IO(new IntelAXIMaster(mp))
  val s_axi_control = IO(new IntelAXILiteClient(hp))

  val shell = withClockAndReset (clock = ap_clk, reset = ~ap_rst_n) { Module(new VTAShell) }

  // memory
  m_axi_gmem.awvalid := shell.io.mem.aw.valid
  shell.io.mem.aw.ready := m_axi_gmem.awready
  m_axi_gmem.awaddr := shell.io.mem.aw.bits.addr
  m_axi_gmem.awid := shell.io.mem.aw.bits.id
  m_axi_gmem.awuser := shell.io.mem.aw.bits.user
  m_axi_gmem.awlen := shell.io.mem.aw.bits.len
  m_axi_gmem.awsize := shell.io.mem.aw.bits.size
  m_axi_gmem.awburst := shell.io.mem.aw.bits.burst
  m_axi_gmem.awlock := shell.io.mem.aw.bits.lock
  m_axi_gmem.awcache := shell.io.mem.aw.bits.cache
  m_axi_gmem.awprot := shell.io.mem.aw.bits.prot
  m_axi_gmem.awqos := shell.io.mem.aw.bits.qos
  m_axi_gmem.awregion := shell.io.mem.aw.bits.region

  m_axi_gmem.wvalid := shell.io.mem.w.valid
  shell.io.mem.w.ready := m_axi_gmem.wready
  m_axi_gmem.wdata := shell.io.mem.w.bits.data
  m_axi_gmem.wstrb := shell.io.mem.w.bits.strb
  m_axi_gmem.wlast := shell.io.mem.w.bits.last
  m_axi_gmem.wid := shell.io.mem.w.bits.id
  m_axi_gmem.wuser := shell.io.mem.w.bits.user

  shell.io.mem.b.valid := m_axi_gmem.bvalid
  m_axi_gmem.bready := shell.io.mem.b.valid
  shell.io.mem.b.bits.resp := m_axi_gmem.bresp
  shell.io.mem.b.bits.id := m_axi_gmem.bid
  shell.io.mem.b.bits.user := m_axi_gmem.buser

  m_axi_gmem.arvalid := shell.io.mem.ar.valid
  shell.io.mem.ar.ready := m_axi_gmem.arready
  m_axi_gmem.araddr := shell.io.mem.ar.bits.addr
  m_axi_gmem.arid := shell.io.mem.ar.bits.id
  m_axi_gmem.aruser := shell.io.mem.ar.bits.user
  m_axi_gmem.arlen := shell.io.mem.ar.bits.len
  m_axi_gmem.arsize := shell.io.mem.ar.bits.size
  m_axi_gmem.arburst := shell.io.mem.ar.bits.burst
  m_axi_gmem.arlock := shell.io.mem.ar.bits.lock
  m_axi_gmem.arcache := shell.io.mem.ar.bits.cache
  m_axi_gmem.arprot := shell.io.mem.ar.bits.prot
  m_axi_gmem.arqos := shell.io.mem.ar.bits.qos
  m_axi_gmem.arregion := shell.io.mem.ar.bits.region

  shell.io.mem.r.valid := m_axi_gmem.rvalid
  m_axi_gmem.rready := shell.io.mem.r.ready
  shell.io.mem.r.bits.data := m_axi_gmem.rdata
  shell.io.mem.r.bits.resp := m_axi_gmem.rresp
  shell.io.mem.r.bits.last := m_axi_gmem.rlast
  shell.io.mem.r.bits.id := m_axi_gmem.rid
  shell.io.mem.r.bits.user := m_axi_gmem.ruser

  // host
  shell.io.host.aw.valid := s_axi_control.awvalid
  s_axi_control.awready := shell.io.host.aw.ready
  shell.io.host.aw.bits.addr := s_axi_control.awaddr
  s_axi_control.awprot <> DontCare

  shell.io.host.w.valid := s_axi_control.wvalid
  s_axi_control.wready := shell.io.host.w.ready
  shell.io.host.w.bits.data := s_axi_control.wdata
  shell.io.host.w.bits.strb := s_axi_control.wstrb

  s_axi_control.bvalid := shell.io.host.b.valid
  shell.io.host.b.ready := s_axi_control.bready
  s_axi_control.bresp := shell.io.host.b.bits.resp

  shell.io.host.ar.valid := s_axi_control.arvalid
  s_axi_control.arready := shell.io.host.ar.ready
  shell.io.host.ar.bits.addr := s_axi_control.araddr
  s_axi_control.arprot <> DontCare

  s_axi_control.rvalid := shell.io.host.r.valid
  shell.io.host.r.ready := s_axi_control.rready
  s_axi_control.rdata := shell.io.host.r.bits.data
  s_axi_control.rresp := shell.io.host.r.bits.resp
}
