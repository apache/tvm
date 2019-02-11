// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class ComputeTests(c: Compute)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  val insn0 = "h00000001000100010000000000000000".U
  val insn1 = "h00000008000800010000000000000180".U
  val insn2 = "h7ffe4002000008000010000800200044".U

  def run() {

  poke(c.io.gemm_queue.valid, 0.U)
  poke(c.io.uops.readdata, 0.U)

  step(1)
  poke(c.io.uops.waitrequest, 0.U)
  poke(c.io.uops.readdata, "h4000".U)
  step(1)
  poke(c.io.gemm_queue.data, insn0)
  poke(c.io.gemm_queue.valid, 1.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  poke(c.io.uops.waitrequest, 1.U)
  poke(c.io.gemm_queue.data, insn1)
  poke(c.io.gemm_queue.valid, 1.U)
  expect(c.io.gemm_queue.ready, 0.U)
  step(1)
  step(1)
  poke(c.io.uops.waitrequest, 0.U)
  step(1)
  poke(c.io.uops.waitrequest, 1.U)
  step(1)
  poke(c.io.uops.waitrequest, 0.U)
  expect(c.io.gemm_queue.ready, 0.U)
  step(1)
  expect(c.io.gemm_queue.ready, 1.U)
  step(1)
  step(1)

  expect(c.io.biases.read, 1.U)
  expect(c.io.biases.address, "h000".U)
  poke(c.io.biases.readdata, "hffffffeeffffffc6ffffffe9fffffff5".U) // data 0
  step(1)
  expect(c.io.biases.address, "h010".U)
  poke(c.io.biases.readdata, "h000000040000000a0000000b00000039".U)
  step(1)
  expect(c.io.biases.address, "h020".U)
  poke(c.io.biases.readdata, "hffffffc1ffffffecfffffff30000001e".U)
  step(1)
  expect(c.io.biases.address, "h030".U)
  poke(c.io.biases.readdata, "hfffffff6ffffffc4ffffffc7ffffffef".U)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  step(1)

  expect(c.io.biases.address, "h040".U)
  poke(c.io.biases.readdata, "hffffffd3ffffffc60000003400000020".U) // data 1
  step(1)
  poke(c.io.biases.readdata, "h0000002a00000037ffffffcd00000022".U)
  step(1)
  poke(c.io.biases.readdata, "hfffffff3ffffffde00000034fffffffa".U)
  step(1)
  poke(c.io.biases.readdata, "h000000120000003afffffffb00000010".U)
  step(1)

  expect(c.io.biases.address, "h080".U)
  poke(c.io.biases.readdata, "hffffffdbfffffff8ffffffc7ffffffc7".U) // data 2
  step(1)
  poke(c.io.biases.readdata, "hffffffd10000002200000038ffffffd0".U)
  step(1)
  poke(c.io.biases.readdata, "h00000031000000170000000b0000003e".U)
  step(1)
  poke(c.io.biases.waitrequest, 1.U)
  step(1)
  poke(c.io.biases.waitrequest, 0.U)
  poke(c.io.biases.readdata, "h000000310000003efffffffcffffffd8".U)
  step(1)

  expect(c.io.biases.address, "h0c0".U)
  poke(c.io.biases.readdata, "h00000007fffffff8ffffffec00000031".U) // data 3
  step(1)
  poke(c.io.biases.readdata, "h0000002bffffffd9ffffffc5fffffff8".U)
  step(1)
  poke(c.io.biases.readdata, "hffffffdb00000016000000200000000f".U)
  step(1)
  poke(c.io.biases.readdata, "hffffffe20000003c00000023ffffffdd".U)
  step(1)

  expect(c.io.biases.address, "h100".U)
  poke(c.io.biases.readdata, "h00000018000000250000002bffffffe3".U) // data 4
  step(1)
  poke(c.io.biases.readdata, "h00000028000000290000002e0000000f".U)
  step(1)
  poke(c.io.biases.readdata, "h0000001400000017ffffffdcffffffd3".U)
  step(1)
  poke(c.io.biases.readdata, "hfffffff8ffffffe10000000afffffff3".U)
  step(1)

  expect(c.io.biases.address, "h140".U)
  poke(c.io.biases.readdata, "h0000001000000019ffffffce00000021".U) // data 5
  step(1)
  poke(c.io.biases.readdata, "hfffffffa0000001fffffffe9ffffffca".U)
  step(1)
  poke(c.io.biases.readdata, "h0000003b00000016ffffffe90000002d".U)
  step(1)
  poke(c.io.biases.readdata, "h000000020000001b0000000900000030".U)
  step(1)

  expect(c.io.biases.address, "h180".U)
  poke(c.io.biases.readdata, "h0000002f00000034ffffffdefffffff1".U) // data 6
  step(1)
  poke(c.io.biases.readdata, "h0000001100000008000000320000001d".U)
  step(1)
  poke(c.io.biases.readdata, "hfffffff2ffffffd20000002e00000003".U)
  step(1)
  poke(c.io.biases.readdata, "hffffffd3000000160000003affffffe7".U)
  step(1)
  poke(c.io.gemm_queue.data, insn2)
  poke(c.io.gemm_queue.valid, 1.U)

  expect(c.io.biases.address, "h1c0".U)
  poke(c.io.biases.readdata, "hfffffff6000000110000002400000017".U) // data 7
  step(1)
  poke(c.io.biases.readdata, "h0000001ffffffff1000000010000003e".U)
  step(1)
  poke(c.io.biases.readdata, "h00000019ffffffc8ffffffd6fffffffa".U)
  step(1)
  poke(c.io.biases.readdata, "hfffffffa0000003ffffffff50000002f".U)
  step(1)
  step(1)
  step(1)

  step(1)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, 0.U)
  // expect(c.io.out_mem.writedata, "hf6c4c7efc1ecf3fcfcfcfcfceec6e9f5".U)
  poke(c.io.gemm_queue.data, 0.U)
  poke(c.io.gemm_queue.valid, 0.U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h10".U)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.writedata, "hfcfcfbfcf3defcfafcfccdfcd3c6fcfc".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h20".U)
  expect(c.io.out_mem.writedata, "hfcfcfcd8fcfcfcfcd1fcfcd0dbf8c7c7".U)
  step(1)
  expect(c.io.out_mem.write, 1.U)
  expect(c.io.out_mem.address, "h30".U)
  expect(c.io.out_mem.writedata, "he2fcfcdddbfcfcfcfcd9c5f8fcf8ecfc".U)
  step(1)
  expect(c.io.out_mem.address, "h40".U)
  step(1)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h50".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  expect(c.io.out_mem.address, "h60".U)
  step(1)
  poke(c.io.out_mem.waitrequest, 1.U)
  step(1)
  expect(c.io.out_mem.address, "h70".U)
  poke(c.io.out_mem.waitrequest, 0.U)
  step(1)
  step(1)
  expect(c.io.out_mem.write, 0.U)

  poke(c.io.g2s_dep_queue.ready, 1.U)
  expect(c.io.g2s_dep_queue.valid, 1.U)
  step(1)
  poke(c.io.g2s_dep_queue.ready, 0.U)
  step(1)
  step(1)
  step(1)
  step(1)

  }

  run()
  run()
}

class ComputeTester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "Compute"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new Compute(), backend)((c) => new ComputeTests(c)) should be (true)
    }
  }
}

