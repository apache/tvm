// See LICENSE.txt for license details.
package vta

import chisel3.iotesters.{Driver, TesterOptionsManager}
import utils.TutorialRunner
import freechips.rocketchip.config.Parameters

object Launcher {
  implicit val p = (new VTAConfig).toInstance
  val modules = Map(
      "Fetch" -> { (manager: TesterOptionsManager) =>
        Driver.execute(() => new Fetch(), manager) {
          (c) => new FetchTests(c)
        }
      },
      "VTA" -> { (manager: TesterOptionsManager) =>
        Driver.execute(() => new VTA(), manager) {
          (c) => new VTATests(c)
        }
      },
      "MemArbiter" -> { (manager: TesterOptionsManager) =>
        Driver.execute(() => new MemArbiter(), manager) {
          (c) => new MemArbiterTests(c)
        }
      },
      "Compute" -> { (manager: TesterOptionsManager) =>
        Driver.execute(() => new Compute(), manager) {
          (c) => new ComputeTests(c)
        }
      },
      "Store" -> { (manager: TesterOptionsManager) =>
        Driver.execute(() => new Store(), manager) {
          (c) => new StoreTests(c)
        }
      }
  )
  def main(args: Array[String]): Unit = {
    TutorialRunner("vta", modules, args)
  }
}

