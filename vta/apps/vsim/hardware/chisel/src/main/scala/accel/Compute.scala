package accel

import chisel3._
import chisel3.util._
import vta.dpi._

/** Compute
  * 
  * Add-by-one procedure:
  *
  * 1. Wait for launch to be asserted
  * 2. Issue a read request for 8-byte value at inp_baddr address
  * 3. Wait for the value
  * 4. Issue a write request for 8-byte value at out_baddr address
  * 5. Increment read-address and write-address for next value
  * 6. Check if counter (cnt) is equal to length to assert finish,
  *    otherwise go to step 2.
  */
class Compute extends Module {
  val io = IO(new Bundle {
    val launch = Input(Bool())
    val finish = Output(Bool())
    val length = Input(UInt(32.W))
    val inp_baddr = Input(UInt(64.W))
    val out_baddr = Input(UInt(64.W))
    val mem = new VTAMemDPIMaster
  })
  val sIdle :: sReadReq :: sReadData :: sWriteReq :: sWriteData :: Nil = Enum(5)
  val state = RegInit(sIdle)
  val reg = Reg(chiselTypeOf(io.mem.rd.bits))
  val cnt = Reg(chiselTypeOf(io.length))
  val raddr = Reg(chiselTypeOf(io.inp_baddr))
  val waddr = Reg(chiselTypeOf(io.out_baddr))

  switch (state) {
    is (sIdle) {
      when (io.launch) {
        state := sReadReq
      }
    }
    is (sReadReq) {
      state := sReadData
    }
    is (sReadData) {
      when (io.mem.rd.valid) {
        state := sWriteReq
      }
    }
    is (sWriteReq) {
      state := sWriteData
    }
    is (sWriteData) {
      when (cnt === (io.length - 1.U)) {
        state := sIdle
      } .otherwise {
        state := sReadReq
      }
    }
  }

  // calculate next address
  when (state === sIdle) {
    raddr := io.inp_baddr
    waddr := io.out_baddr
  } .elsewhen (state === sWriteData) { // increment by 8-bytes
    raddr := raddr + 8.U
    waddr := waddr + 8.U
  }

  // create request
  io.mem.req.valid := state === sReadReq | state === sWriteReq
  io.mem.req.opcode := state === sWriteReq
  io.mem.req.len := 0.U // one-word-per-request
  io.mem.req.addr := Mux(state === sReadReq, raddr, waddr)

  // read
  when (state === sReadData && io.mem.rd.valid) {
    reg := io.mem.rd.bits + 1.U
  }
  io.mem.rd.ready := state === sReadData

  // write
  io.mem.wr.valid := state === sWriteData
  io.mem.wr.bits := reg

  // count read/write
  when (state === sIdle) {
    cnt := 0.U
  } .elsewhen (state === sWriteData) {
    cnt := cnt + 1.U
  }

  // done when read/write are equal to length
  io.finish := state === sWriteData && cnt === (io.length - 1.U)
}