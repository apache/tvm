// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.{Parameters, Field}

class VTAIO(implicit p: Parameters) extends CoreBundle()(p) {
  val h2f_lite = new AvalonSlaveIO(dataBits = 32, addrBits = 32)
  val f2h = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
}

class VTA(implicit val p: Parameters) extends Module with CoreParams {
  val io = IO(new VTAIO())

  val arb = Module(new MemArbiter())
  val fetch = Module(new Fetch())
  val compute = Module(new Compute())
  val store = Module(new Store())

  io.h2f_lite <> DontCare
  io.f2h <> DontCare

  fetch.io <> DontCare
  compute.io <> DontCare
  store.io <> DontCare

}

