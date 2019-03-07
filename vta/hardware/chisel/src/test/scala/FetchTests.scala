// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class FetchTests(c: Fetch)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  val insn_count_val = 9
  val insn_count = insn_count_val.U
  val insn0 = "h00000001000100010000000000000000".U // compute - uop fetch (uop_compress=true)
  val insn1 = "h00000008000800010000000000000180".U // compute - acc fetch
  val insn2 = "h7ffe4002000008000010000800200044".U // compute - out_mem
  val insn3 = "h00000008000800010000000000000029".U // store   - outputs
  val insn4 = "h00000008000800010000000000000000".U // compute - uop fetch (uop_compress=false)
  val insn5 = "h00000008000800010000000000000180".U // compute - acc fetch
  val insn6 = "h0002c000000000000002000801000044".U // compute - out_mem
  val insn7 = "h00000008000800010000000000000029".U // store   - outputs
  val insn8 = "h00000000000000000000000000000013".U // compute - finish
  val insns = IndexedSeq(insn0, insn1, insn2, insn3, insn4, insn5, insn6, insn7, insn8)
  val categories = IndexedSeq(1, 1, 1, 2, 1, 1, 1, 2, 1)

  // reset
  poke(c.io.insn_count.write, 0.U)
  poke(c.io.insn_count.writedata, 0.U)
  step(1)

  // write insn_count
  poke(c.io.insn_count.writedata, insn_count)
  step(1)
  poke(c.io.insn_count.write, 1.U)
  step(1)
  poke(c.io.insn_count.write, 0.U)
  poke(c.io.insns.waitrequest, 1.U)
  poke(c.io.gemm_queue.ready, 0.U)
  poke(c.io.store_queue.ready, 0.U)

  val fifo_size = 4

  // try enqueue instructions more than fifo size
  for (i <- 0 to (fifo_size + 1)) {
    expect(c.io.insns.read, 1.U)
    expect(c.io.insns.address, (i.min(fifo_size) << 4).U)
    poke(c.io.insns.readdata, insns(i))
    step(1)
    poke(c.io.insns.waitrequest, 0.U)
    step(1)
    poke(c.io.insns.waitrequest, 1.U)
  }

  // dequeue instructions that have been enqueued
  for (i <- 0 to (fifo_size - 1)){
    expect(c.io.load_queue.valid, (categories(i) == 0).B)
    expect(c.io.gemm_queue.valid, (categories(i) == 1).B)
    expect(c.io.store_queue.valid, (categories(i) == 2).B)
    if (categories(i) == 0) {
      expect(c.io.load_queue.data, insns(i))
      poke(c.io.load_queue.ready, 1.U)
    } else if (categories(i) == 1) {
      expect(c.io.gemm_queue.data, insns(i))
      poke(c.io.gemm_queue.ready, 1.U)
    } else if (categories(i) == 2) {
      expect(c.io.store_queue.data, insns(i))
      poke(c.io.store_queue.ready, 1.U)
    }
    step(1)
    poke(c.io.load_queue.ready, 0.U)
    poke(c.io.gemm_queue.ready, 0.U)
    poke(c.io.store_queue.ready, 0.U)
    step(1)
  }

  // Push rest of the instructions into the fifo,
  // and dequeue instructions that have been enqueued
  for (i <- fifo_size to (insn_count_val - 1)) {
    expect(c.io.insns.read, 1.U)
    expect(c.io.insns.address, (i << 4).U)
    poke(c.io.insns.readdata, insns(i))
    step(1)
    poke(c.io.insns.waitrequest, 0.U)
    step(1)
    poke(c.io.insns.waitrequest, 1.U)
    step(1)
    expect(c.io.load_queue.valid, (categories(i) == 0).B)
    expect(c.io.gemm_queue.valid, (categories(i) == 1).B)
    expect(c.io.store_queue.valid, (categories(i) == 2).B)
    if (categories(i) == 0) {
      expect(c.io.load_queue.data, insns(i))
      poke(c.io.load_queue.ready, 1.U)
    } else if (categories(i) == 1) {
      expect(c.io.gemm_queue.data, insns(i))
      poke(c.io.gemm_queue.ready, 1.U)
    } else if (categories(i) == 2) {
      expect(c.io.store_queue.data, insns(i))
      poke(c.io.store_queue.ready, 1.U)
    }
    step(1)
    poke(c.io.load_queue.ready, 0.U)
    poke(c.io.gemm_queue.ready, 0.U)
    poke(c.io.store_queue.ready, 0.U)
    step(1)
  }

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

