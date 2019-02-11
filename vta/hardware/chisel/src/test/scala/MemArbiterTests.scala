// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class MemArbiterTests(c: MemArbiter)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  def test_alu() {
  poke(c.io.axi_master.waitrequest, 1.U)
  step(1)
  poke(c.io.uop_cache.read, 0.U)
  poke(c.io.acc_cache.read, 0.U)
  poke(c.io.out_cache.write, 0.U)
  poke(c.io.uop_cache.address, "haaaaaaaa".U)
  poke(c.io.acc_cache.address, "hbbbbbbbb".U)
  poke(c.io.out_cache.address, "hcccccccc".U)
  poke(c.io.out_cache.writedata, "h33333333333333333333333333333333".U)
  step(1)

  // read uop_cache from axi_master
  poke(c.io.uop_cache.read, 1.U)
  poke(c.io.acc_cache.read, 1.U)
  poke(c.io.out_cache.write, 0.U)
  step(1)
  expect(c.io.uop_cache.waitrequest, 1.U)
  expect(c.io.acc_cache.waitrequest, 1.U)
  expect(c.io.axi_master.address, "haaaaaaaa".U)                         // uop_cache address
  poke(c.io.axi_master.readdata, "h11111111111111111111111111111111".U)  // uop_cache databits
  poke(c.io.axi_master.waitrequest, 0.U)
  step(1)
  expect(c.io.uop_cache.readdata, "h11111111111111111111111111111111".U)
  poke(c.io.uop_cache.read, 0.U)
  poke(c.io.acc_cache.read, 0.U)
  poke(c.io.axi_master.readdata, "h22222222222222222222222222222222".U)
  step(1)
  poke(c.io.axi_master.waitrequest, 1.U)
  poke(c.io.acc_cache.read, 0.U)
  expect(c.io.uop_cache.waitrequest, 1.U)
  step(1)
  step(1)

  // write out_cache to axi_master
  poke(c.io.uop_cache.read, 1.U)
  poke(c.io.acc_cache.read, 1.U)
  poke(c.io.out_cache.write, 1.U)
  poke(c.io.out_cache.writedata, "h77777777777777777777777777777777".U)
  poke(c.io.axi_master.readdata, "h44444444444444444444444444444444".U)
  step(1)
  expect(c.io.axi_master.write, 1.U)
  expect(c.io.axi_master.address, "hcccccccc".U)
  expect(c.io.axi_master.writedata, "h77777777777777777777777777777777".U)
  poke(c.io.axi_master.waitrequest, 1.U)
  step(1)
  step(1)
  step(1)
  expect(c.io.axi_master.write, 1.U)
  expect(c.io.axi_master.writedata, "h77777777777777777777777777777777".U)
  poke(c.io.axi_master.waitrequest, 0.U)
  step(1)
  poke(c.io.out_cache.write, 0.U)
  poke(c.io.uop_cache.read, 0.U)
  poke(c.io.acc_cache.read, 0.U)
  poke(c.io.axi_master.waitrequest, 1.U)
  step(1)
  step(1)
    
  // write out_cache to axi_master
  poke(c.io.out_cache.write, 1.U)
  poke(c.io.out_cache.address, "hdddddddd".U)
  poke(c.io.out_cache.writedata, "h88888888888888888888888888888888".U)
  step(1)
  poke(c.io.axi_master.waitrequest, 1.U)
  step(1)
  expect(c.io.axi_master.write, 1.U)
  expect(c.io.axi_master.address, "hdddddddd".U)
  expect(c.io.axi_master.writedata, "h88888888888888888888888888888888".U)
  poke(c.io.axi_master.waitrequest, 0.U)
  step(1)
  poke(c.io.out_cache.write, 0.U)
  poke(c.io.uop_cache.read, 0.U)
  poke(c.io.acc_cache.read, 0.U)
  poke(c.io.axi_master.waitrequest, 1.U)
  step(1)

  // read acc_cache from axi_master
  poke(c.io.out_cache.write, 0.U)
  poke(c.io.acc_cache.read, 0.U)
  poke(c.io.acc_cache.read, 1.U)
  step(1)
  step(1)
  poke(c.io.axi_master.waitrequest, 0.U)
  expect(c.io.axi_master.address, "hbbbbbbbb".U)                        // acc_cache address
  poke(c.io.axi_master.readdata, "h55555555555555555555555555555555".U) // acc_cache databits
  step(1)
  expect(c.io.acc_cache.readdata, "h55555555555555555555555555555555".U)
  poke(c.io.acc_cache.read, 0.U)
  poke(c.io.axi_master.waitrequest, 1.U)
  step(1)
  step(1)
  step(1)

  }

  test_alu()
  test_alu()
}

class MemArbiterTester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "MemArbiter"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new MemArbiter(), backend)((c) => new MemArbiterTests(c)) should be (true)
    }
  }
}

