// See LICENSE for license details.

package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.Parameters

class MemArbiterIO(implicit p: Parameters) extends CoreBundle()(p) {
  val ins_cache = new AvalonSlaveIO(dataBits = 128, addrBits = 32)
  val inp_cache = new AvalonSlaveIO(dataBits = 128, addrBits = 32)
  val wgt_cache = new AvalonSlaveIO(dataBits = 128, addrBits = 32)
  val uop_cache = new AvalonSlaveIO(dataBits = 128, addrBits = 32)
  val acc_cache = new AvalonSlaveIO(dataBits = 128, addrBits = 32)
  val out_cache = new AvalonSlaveIO(dataBits = 128, addrBits = 32)
  val axi_master = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
}

class MemArbiter(implicit val p: Parameters) extends Module {
  val io = IO(new MemArbiterIO())

  val s_IDLE :: s_INS_CACHE_READ :: s_INP_CACHE_READ :: s_WGT_CACHE_READ :: s_UOP_CACHE_READ :: s_ACC_CACHE_READ :: s_OUT_CACHE_WRITE :: s_OUT_CACHE_ACK :: Nil = Enum(8)
  val state = RegInit(s_IDLE)
  val idle            = state === s_IDLE
  val ins_cache_read  = state === s_INS_CACHE_READ
  val inp_cache_read  = state === s_INP_CACHE_READ
  val wgt_cache_read  = state === s_WGT_CACHE_READ
  val uop_cache_read  = state === s_UOP_CACHE_READ
  val acc_cache_read  = state === s_ACC_CACHE_READ
  val out_cache_write = state === s_OUT_CACHE_WRITE
  val out_cache_ack   = state === s_OUT_CACHE_ACK

  // common
  io.axi_master.address := MuxLookup(state, s_IDLE,
    List(s_INS_CACHE_READ -> io.ins_cache.address,
         s_INP_CACHE_READ -> io.inp_cache.address,
         s_WGT_CACHE_READ -> io.wgt_cache.address,
         s_UOP_CACHE_READ -> io.uop_cache.address,
         s_ACC_CACHE_READ -> io.acc_cache.address,
         s_OUT_CACHE_WRITE -> io.out_cache.address,
         s_OUT_CACHE_ACK -> io.out_cache.address, s_IDLE -> 0.U))

  // write
  io.axi_master.writedata  := io.out_cache.writedata
  io.axi_master.write := io.out_cache.write && out_cache_write
  io.out_cache.waitrequest := io.axi_master.waitrequest && (out_cache_write || out_cache_ack || idle)
  io.out_cache.readdata <> DontCare

  // read
  val axi_master_read = MuxLookup(state, s_IDLE,
    List(s_INS_CACHE_READ -> io.ins_cache.read,
         s_INP_CACHE_READ -> io.inp_cache.read,
         s_WGT_CACHE_READ -> io.wgt_cache.read,
         s_UOP_CACHE_READ -> io.uop_cache.read,
         s_ACC_CACHE_READ -> io.acc_cache.read, s_IDLE -> 0.U))
  io.axi_master.read := axi_master_read
  io.ins_cache.readdata := io.axi_master.readdata
  io.inp_cache.readdata := io.axi_master.readdata
  io.wgt_cache.readdata := io.axi_master.readdata
  io.uop_cache.readdata := io.axi_master.readdata
  io.acc_cache.readdata := io.axi_master.readdata
  io.ins_cache.waitrequest := (io.axi_master.waitrequest && (ins_cache_read || idle)) || (!ins_cache_read && io.ins_cache.read)
  io.inp_cache.waitrequest := (io.axi_master.waitrequest && (inp_cache_read || idle)) || (!inp_cache_read && io.inp_cache.read)
  io.wgt_cache.waitrequest := (io.axi_master.waitrequest && (wgt_cache_read || idle)) || (!wgt_cache_read && io.wgt_cache.read)
  io.uop_cache.waitrequest := (io.axi_master.waitrequest && (uop_cache_read || idle)) || (!uop_cache_read && io.uop_cache.read)
  io.acc_cache.waitrequest := (io.axi_master.waitrequest && (acc_cache_read || idle)) || (!acc_cache_read && io.acc_cache.read)

  switch(state) {
    is (s_IDLE) {
      when (io.ins_cache.read) {
        state := s_INS_CACHE_READ
      } .elsewhen (io.out_cache.write) {
        state := s_OUT_CACHE_WRITE
      } .elsewhen (io.inp_cache.read) {
        state := s_INP_CACHE_READ
      } .elsewhen (io.inp_cache.read) {
        state := s_WGT_CACHE_READ
      } .elsewhen (io.wgt_cache.read) {
        state := s_WGT_CACHE_READ
      } .elsewhen (io.uop_cache.read) {
        state := s_UOP_CACHE_READ
      } .elsewhen (io.acc_cache.read) {
        state := s_ACC_CACHE_READ
      } .otherwise {
        state := s_IDLE
      }
    }
    is (s_INS_CACHE_READ) {
      when (!io.axi_master.waitrequest) {
        state := s_IDLE
      }
    }
    is (s_INP_CACHE_READ) {
      when (!io.axi_master.waitrequest) {
        state := s_IDLE
      }
    }
    is (s_WGT_CACHE_READ) {
      when (!io.axi_master.waitrequest) {
        state := s_IDLE
      }
    }
    is (s_UOP_CACHE_READ) {
      when (!io.axi_master.waitrequest) {
        state := s_IDLE
      }
    }
    is (s_ACC_CACHE_READ) {
      when (!io.axi_master.waitrequest) {
        state := s_IDLE
      }
    }
    is (s_OUT_CACHE_WRITE) {
      when (!io.axi_master.waitrequest) {
        state := s_IDLE
        // state := s_OUT_CACHE_ACK
      }
    }
    is (s_OUT_CACHE_ACK) {
      // when (!io.axi_master.waitrequest) {
        state := s_IDLE
      // }
    }
  }

}
