package accel

import chisel3._
import vta.dpi._

/** Add-by-one accelerator.
  * 
  * ___________      ___________
  * |         |      |         |
  * | HostDPI | <--> | RegFile | <->|
  * |_________|      |_________|    |
  *                                 |
  * ___________      ___________    |
  * |         |      |         |    |
  * | MemDPI  | <--> | Compute | <->|
  * |_________|      |_________|
  *    
  */
class Accel extends Module {
  val io = IO(new Bundle {
    val host = new VTAHostDPIClient
    val mem = new VTAMemDPIMaster
  })
  val rf = Module(new RegFile)
  val ce = Module(new Compute)
  rf.io.host <> io.host
  io.mem <> ce.io.mem
  ce.io.launch := rf.io.launch
  rf.io.finish := ce.io.finish
  ce.io.length := rf.io.length
  ce.io.inp_baddr := rf.io.inp_baddr
  ce.io.out_baddr := rf.io.out_baddr
}