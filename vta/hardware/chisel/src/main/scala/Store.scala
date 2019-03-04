// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.{Parameters, Field}

class StoreIO(implicit p: Parameters) extends CoreBundle()(p) {
  val outputs = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val store_queue = new AvalonSinkIO(dataBits = 128)
  val s2g_dep_queue = Flipped(new AvalonSinkIO(dataBits = 1))
  val g2s_dep_queue = new AvalonSinkIO(dataBits = 1)
  val out_mem = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 17))
}

class Store(implicit val p: Parameters) extends Module with CoreParams {
  val io = IO(new StoreIO())

  val started = !reset.toBool

  val insn            = Reg(UInt(128.W))
  val insn_valid      = (insn =/= 0.U) && started

  val g2s_dep_queue_valid = Reg(Bool())
  val g2s_dep_queue_wait = Reg(Bool())
  val s2g_dep_queue_done = Reg(Bool())

  // Decode
  val pop_prev_dep = insn(insn_mem_1)
  val pop_next_dep = insn(insn_mem_2)
  val push_prev_dep = insn(insn_mem_3)
  val push_next_dep = insn(insn_mem_4)
  val memory_type = insn(insn_mem_5_1, insn_mem_5_0)
  val sram_base   = insn(insn_mem_6_1, insn_mem_6_0)
  val dram_base   = insn(insn_mem_7_1, insn_mem_7_0)
  val y_size      = insn(insn_mem_8_1, insn_mem_8_0)
  val x_size      = insn(insn_mem_9_1, insn_mem_9_0)
  val x_stride    = insn(insn_mem_a_1, insn_mem_a_0)
  val y_pad_0     = insn(insn_mem_b_1, insn_mem_b_0)
  val y_pad_1     = insn(insn_mem_c_1, insn_mem_c_0)
  val x_pad_0     = insn(insn_mem_d_1, insn_mem_d_0)
  val x_pad_1     = insn(insn_mem_e_1, insn_mem_e_0)

  val y_size_total = y_pad_0 + y_size + y_pad_1
  val x_size_total = x_pad_0 + x_size + x_pad_1
  val y_offset = x_size_total * y_pad_0

  val sram_idx = (sram_base + y_offset) + x_pad_0
  val dram_idx = dram_base

  // fifo buffer
  val out_queue = Module(new Queue(UInt(128.W), 8))

  // status registers
  val state = Reg(UInt(3.W))
  val s_IDLE :: s_DUMP :: s_BUSY :: s_PUSH :: s_DONE :: Nil = Enum(5)
  val idle = state === s_IDLE
  val dump = state === s_DUMP
  val busy = state === s_BUSY
  val push = state === s_PUSH
  val done = state === s_DONE

  // counters
  val enq_cntr_max = x_size * y_size
  val enq_cntr_en = insn_valid
  val enq_cntr_wait = !out_queue.io.enq.ready || io.out_mem.waitrequest
  val enq_cntr_val = Reg(UInt(16.W))
  val enq_cntr_wrap = (enq_cntr_val === enq_cntr_max)

  val deq_cntr_max = x_size * y_size
  val deq_cntr_en = insn_valid
  val deq_cntr_wait = io.outputs.waitrequest || !out_queue.io.deq.valid
  val deq_cntr_val = Reg(UInt(16.W))
  val deq_cntr_wrap = (deq_cntr_val === deq_cntr_max)

  val out_mem_read = RegInit(false.B)
  val out_mem_data = Reg(UInt(128.W))

  val pop_prev_dep_ready = RegInit(false.B)
  val push_prev_dep_valid = push_prev_dep && push
  val push_prev_dep_ready = RegInit(false.B)

  // setup state transitions
  when ((enq_cntr_en && !enq_cntr_wrap) && (deq_cntr_en && !deq_cntr_wrap)) {
    when (pop_prev_dep && !pop_prev_dep_ready) {
      state := s_DUMP
    } .otherwise {
      state := s_BUSY
    }
  }
  when ((enq_cntr_en && enq_cntr_wrap) && (deq_cntr_en && deq_cntr_wrap)) {
    when (push_prev_dep && !push_prev_dep_ready) {
      state := s_PUSH
    } .otherwise {
      state := s_DONE
    }
  }

  // dependency queue processing
  when (dump &&  pop_prev_dep_ready) { state := s_BUSY }
  when (push && push_prev_dep_ready) { state := s_DONE }

  // dependency queue processing
  io.g2s_dep_queue.ready := pop_prev_dep_ready && dump
  io.g2s_dep_queue.data <> DontCare
  when (pop_prev_dep && io.g2s_dep_queue.valid && dump) {
    pop_prev_dep_ready := true.B
  }
  io.s2g_dep_queue.data := 1.U
  io.s2g_dep_queue.valid := push_prev_dep_valid
  when (push_prev_dep_valid && io.s2g_dep_queue.ready && push) {
    push_prev_dep_ready := true.B
  }

  // setup counter
  when (out_mem_read && !enq_cntr_wait && busy && enq_cntr_val < enq_cntr_max) {
    enq_cntr_val := enq_cntr_val + 1.U
  }
  
  // fetch instruction
  when (io.store_queue.valid && (idle || done)) {
    enq_cntr_val := 0.U
    deq_cntr_val := 0.U
    pop_prev_dep_ready := false.B
    push_prev_dep_ready := false.B
    insn := io.store_queue.data
    when (insn_valid) {
      io.store_queue.ready := 1.U
    } .otherwise {
      io.store_queue.ready := 0.U
    }
  } .otherwise {
    insn := insn
    io.store_queue.ready := 0.U
  }

  // enqueue from out_mem to fifo
  val out_sram_addr = (sram_idx * batch.U + enq_cntr_val) << 4.U
  out_mem_read := enq_cntr_en && !enq_cntr_wrap && busy
  io.out_mem.read := out_mem_read
  io.out_mem.address := out_sram_addr
  io.out_mem.write <> DontCare
  io.out_mem.writedata <> DontCare
  out_queue.io.enq.bits := io.out_mem.readdata
  when (out_mem_read && !enq_cntr_wait && busy) {
    out_queue.io.enq.valid := 1.U
  } .otherwise {
    out_queue.io.enq.valid := 0.U
  }

  // dequeue fifo and send to outputs
  io.outputs.writedata := out_queue.io.deq.bits
  io.outputs.address := (dram_idx * batch.U + deq_cntr_val) << 4.U
  io.outputs.read <> DontCare
  io.outputs.readdata <> DontCare
  when (deq_cntr_en && out_queue.io.deq.valid && busy) {
    io.outputs.write := 1.U
    when (!io.outputs.waitrequest) {
      deq_cntr_val := deq_cntr_val + 1.U
      out_queue.io.deq.ready := 1.U
    } .otherwise {
      out_queue.io.deq.ready := 0.U
    }
  } .otherwise {
    io.outputs.write := 0.U
    out_queue.io.deq.ready := 0.U
  }
}

