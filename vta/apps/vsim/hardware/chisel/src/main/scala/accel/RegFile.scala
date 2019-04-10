package accel

import chisel3._
import chisel3.util._
import vta.dpi._

/** Register File.
  * 
  * Six 32-bit register file.
  *
  * -------------------------------
  *  Register description    | addr
  * -------------------------|-----
  *  Control status register | 0x00
  *  Length value register   | 0x04
  *  Input pointer lsb       | 0x08
  *  Input pointer msb       | 0x0c
  *  Output pointer lsb      | 0x10
  *  Output pointer msb      | 0x14
  * -------------------------------

  * ------------------------------
  *  Control status register | bit
  * ------------------------------
  *  Launch                  | 0
  *  Finish                  | 1
  * ------------------------------
  */
class RegFile extends Module {
  val io = IO(new Bundle {
    val launch = Output(Bool())
    val finish = Input(Bool())
    val length = Output(UInt(32.W))
    val inp_baddr = Output(UInt(64.W))
    val out_baddr = Output(UInt(64.W))
    val host = new VTAHostDPIClient
  })
  val sIdle :: sRead :: Nil = Enum(2)
  val state = RegInit(sIdle)

  switch (state) {
    is (sIdle) {
      when (io.host.req.valid && !io.host.req.opcode) {
        state := sRead
      }
    }
    is (sRead) {
      state := sIdle
    }
  }

  io.host.req.deq := state === sIdle & io.host.req.valid

  val reg = Seq.fill(6)(RegInit(0.U.asTypeOf(chiselTypeOf(io.host.req.value))))
  val addr = Seq.tabulate(6)(_ * 4)
  val reg_map = (addr zip reg)  map { case (a, r) => a.U -> r }

  (reg zip addr).foreach { case(r, a) =>
    if (a == 0) { // control status register
      when (io.finish) {
        r := "b_10".U
      } .elsewhen (state === sIdle && io.host.req.valid &&
            io.host.req.opcode && a.U === io.host.req.addr) {
        r := io.host.req.value
      }
    } else {
      when (state === sIdle && io.host.req.valid &&
            io.host.req.opcode && a.U === io.host.req.addr) {
        r := io.host.req.value
      }
    }
  }

  val rdata = RegInit(0.U.asTypeOf(chiselTypeOf(io.host.req.value)))
  when (state === sIdle && io.host.req.valid && !io.host.req.opcode) {
    rdata := MuxLookup(io.host.req.addr, 0.U, reg_map)
  }

  io.host.resp.valid := state === sRead
  io.host.resp.bits := rdata

  io.launch := reg(0)(0)
  io.length := reg(1)
  io.inp_baddr := Cat(reg(3), reg(2))
  io.out_baddr := Cat(reg(5), reg(4))
}