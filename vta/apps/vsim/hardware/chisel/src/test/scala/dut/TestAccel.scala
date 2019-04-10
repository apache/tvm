package test

import chisel3._
import chisel3.experimental.{RawModule, withClockAndReset}
import vta.dpi._
import accel._

/** VTA simulation shell.
  * 
  * Instantiate Host and Memory DPI modules.
  *    
  */
class VTASimShell extends RawModule {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val host = new VTAHostDPIMaster
    val mem = new VTAMemDPIClient
  })
  val host = Module(new VTAHostDPI)
  val mem = Module(new VTAMemDPI)
  mem.io.reset := io.reset
  mem.io.clock := io.clock
  host.io.reset := io.reset
  host.io.clock := io.clock
  io.mem <> mem.io.dpi
  io.host <> host.io.dpi
}

/** Test accelerator.
  * 
  * Instantiate and connect the simulation-shell and the accelerator.
  *    
  */
class TestAccel extends RawModule {
  val clock = IO(Input(Clock()))
  val reset = IO(Input(Bool()))

  val sim_shell = Module(new VTASimShell)
  val vta_accel = withClockAndReset(clock, reset) { Module(new Accel) }

  sim_shell.io.clock := clock
  sim_shell.io.reset := reset
  vta_accel.io.host <> sim_shell.io.host
  sim_shell.io.mem <> vta_accel.io.mem
}

/** Generate TestAccel as top module */
object Elaborate extends App {
  chisel3.Driver.execute(args, () => new TestAccel)
}
