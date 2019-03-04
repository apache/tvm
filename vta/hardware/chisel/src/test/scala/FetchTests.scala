// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class FetchTests(c: Fetch)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  val insn_count = 50.U
  val insn0 = "h00000001000100010000000000000000".U
  val insn1 = "h00000008000800010000000000000180".U
  val insn2 = "h7ffe4002000008000010000800200044".U
  val insn3 = "h00000008000800010000000000000029".U

  step(1)
  poke(c.io.insn_count.writedata, insn_count)
  poke(c.io.insn_count.write, 0.U)
  step(1)
  poke(c.io.insn_count.write, 1.U)
  step(1)
  poke(c.io.insn_count.write, 0.U)
  poke(c.io.insns.readdata, insn0)
  poke(c.io.insns.waitrequest, 1.U)
  step(1)
  poke(c.io.insns.waitrequest, 0.U)
  poke(c.io.insns.readdata, insn1)
  step(1)
  poke(c.io.insns.readdata, insn2)
  step(1)
  poke(c.io.insns.readdata, insn3)
  step(1)
  poke(c.io.insns.waitrequest, 1.U)
  step(1)
  step(1)
  step(1)
  poke(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.gemm_queue.ready, 0.U)
  poke(c.io.store_queue.ready, 1.U)
  step(1)
  poke(c.io.store_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

}

class FetchTester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "Fetch"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new Fetch(), backend)((c) => new FetchTests(c)) should be (true)
    }
  }
}

