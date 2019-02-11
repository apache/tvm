// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class StoreTests(c: Store)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  val insn0 = "h00000008000800010000000000000029".U

  val out_mem = List(
    "hf6c4c7efc1ecf3fcfcfcfcfceec6e9f5".U, "hfcfcfbfcf3defcfafcfccdfcd3c6fcfc".U,
    "hfcfcfcd8fcfcfcfcd1fcfcd0dbf8c7c7".U, "he2fcfcdddbfcfcfcfcd9c5f8fcf8ecfc".U,
    "hf8e1fcf3fcfcdcd3fcfcfcfcfcfcfce3".U, "hfcfcfcfcfcfce9fcfafce9cafcfccefc".U,
    "hd3fcfce7f2d2fcfcfcfcfcfcfcfcdef1".U, "hfafcf5fcfcc8d6fafcf1fcfcf6fcfcfc".U)

  def run() {

  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  poke(c.io.store_queue.data, insn0)
  poke(c.io.store_queue.valid, 1.U)
  poke(c.io.out_mem.readdata, out_mem(0))
  step(1)
  poke(c.io.store_queue.valid, 0.U)
  step(1)
  poke(c.io.g2s_dep_queue.valid, 1.U)
  poke(c.io.g2s_dep_queue.data, 1.U)
  step(1)
  expect(c.io.g2s_dep_queue.ready, 1.U)
  step(1)
  expect(c.io.g2s_dep_queue.ready, 0.U)
  expect(c.io.out_mem.address, "h0000".U)
  step(1)
  step(1)
  expect(c.io.outputs.write, 1.U)
  expect(c.io.outputs.address, "h0000".U)
  expect(c.io.outputs.writedata, out_mem(0))
  poke(c.io.outputs.waitrequest, 1.U)
  expect(c.io.out_mem.address, "h0010".U)
  poke(c.io.out_mem.readdata, out_mem(1))
  step(1)
  expect(c.io.out_mem.address, "h0020".U)
  poke(c.io.out_mem.readdata, out_mem(2))
  step(1)
  poke(c.io.out_mem.readdata, out_mem(3))
  poke(c.io.outputs.waitrequest, 0.U)              // dequeue
  poke(c.io.out_mem.waitrequest, 1.U)              // enqueue
  expect(c.io.outputs.write, 1.U)
  expect(c.io.outputs.address, "h0010".U)
  expect(c.io.outputs.writedata, out_mem(1))
  step(1)
  poke(c.io.out_mem.waitrequest, 0.U)              // enqueue
  expect(c.io.out_mem.read, 1.U)              // enqueue
  expect(c.io.outputs.address, "h0020".U)
  expect(c.io.outputs.writedata, out_mem(2))
  step(1)
  poke(c.io.out_mem.readdata, out_mem(4))
  step(1)
  poke(c.io.out_mem.readdata, out_mem(5))
  step(1)
  poke(c.io.outputs.waitrequest, 1.U)
  poke(c.io.out_mem.readdata, out_mem(6))
  step(1)
  poke(c.io.outputs.waitrequest, 0.U)
  expect(c.io.out_mem.address, "h0070".U)
  poke(c.io.out_mem.readdata, out_mem(7))
  step(1)
  // poke(c.io.out_mem.waitrequest, 1.U)               // stop enqueue
  step(1)
  step(1)
  poke(c.io.outputs.waitrequest, 0.U)
  step(1)

  step(1)
  expect(c.io.s2g_dep_queue.valid, 1.U)
  poke(c.io.s2g_dep_queue.ready, 1.U)
  step(1)
  poke(c.io.s2g_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

  }

  run()
  run()
}

class StoreTester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "Store"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new Store(), backend)((c) => new StoreTests(c)) should be (true)
    }
  }
}

