// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class VTATests(c: VTA)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  step(1)
  poke(c.io.h2f_lite.write, 1.U)
  poke(c.io.h2f_lite.writedata, 0.U)
  step(1)

}

class VTATester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "VTA"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new VTA(), backend)((c) => new VTATests(c)) should be (true)
    }
  }
}

