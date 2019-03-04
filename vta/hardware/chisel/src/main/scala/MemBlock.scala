// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.Parameters

class MemBlockIO(val addrBits : Int, val dataBits : Int) extends Bundle {
  val waitrequest = Output(Bool())
  val address = Input(UInt(addrBits.W))
  val read  = Input(Bool())
  val readdata = Output(UInt(dataBits.W))
  val write  = Input(UInt(1.W))
  val writedata = Input(UInt(dataBits.W))
}

class MemBlock(val addrBits : Int, val dataBits : Int) extends Module {
  val io = IO(new MemBlockIO(addrBits, dataBits))
  val mem = Mem(1 << addrBits, UInt(dataBits.W))

  // write
  when (io.write === 1.U) {
    mem(io.address) := io.writedata
  }

  // read
  val readdata_reg = RegNext(mem(io.address))
  io.readdata := readdata_reg
  io.waitrequest := 0.U

  // force read during write behavior
  when (RegNext(io.write) === 1.U && RegNext(io.read) === 1.U) {
    io.readdata := RegNext(io.writedata)
  }
}

