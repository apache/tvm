// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.{Parameters, Field}

class LoadIO(implicit p: Parameters) extends CoreBundle()(p) {
  val inputs = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val weights = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val load_queue = new AvalonSinkIO(dataBits = 128)
  val g2l_dep_queue = new AvalonSinkIO(dataBits = 1)
  val l2g_dep_queue = Flipped(new AvalonSinkIO(dataBits = 1))
  val inp_mem = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val wgt_mem = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
}

/*
 * Operating sequence:
 * 1. set insn_count via memory map interface
 * 2. start instruction counter and enqueue instructions into insn_queue
 * 3. dequeue insn_queue and stream into corresponding (load/gemm/store) queue
 */
class Load(implicit val p: Parameters) extends Module with CoreParams {
  val io = IO(new LoadIO())

  val started = !reset.toBool

  val insn            = Reg(UInt(128.W))
  val insn_valid      = (insn =/= 0.U) && started

  val g2l_dep_queue_valid = Reg(Bool())
  val g2l_dep_queue_wait = Reg(Bool())
  val l2g_dep_queue_done = Reg(Bool())

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
  val inp_queue = Module(new Queue(UInt(128.W), 8))
  val wgt_queue = Module(new Queue(UInt(128.W), 8))

  // status registers
  val state = Reg(UInt(3.W))
  val s_IDLE :: s_DUMP :: s_BUSY :: s_PUSH :: s_DONE :: Nil = Enum(5)
  val idle = state === s_IDLE
  val dump = state === s_DUMP
  val busy = state === s_BUSY
  val push = state === s_PUSH
  val done = state === s_DONE

  // inputs counters
  val inp_enq_cntr_max = x_size * y_size
  val inp_enq_cntr_en = insn_valid && (memory_type === mem_id_inp.U)
  val inp_enq_cntr_wait = !inp_queue.io.enq.ready || io.inputs.waitrequest
  val inp_enq_cntr_val = Reg(UInt(16.W))
  val inp_enq_cntr_wrap = (inp_enq_cntr_val === inp_enq_cntr_max)

  val inp_deq_cntr_max = x_size * y_size
  val inp_deq_cntr_en = insn_valid && (memory_type === mem_id_wgt.U)
  val inp_deq_cntr_wait = io.inp_mem.waitrequest || !inp_queue.io.deq.valid
  val inp_deq_cntr_val = Reg(UInt(16.W))
  val inp_deq_cntr_wrap = (inp_deq_cntr_val === inp_deq_cntr_max)

  val inputs_read = RegInit(false.B)
  val inputs_data = Reg(UInt(128.W))

  // weights counters
  val wgt_enq_cntr_max = x_size * y_size
  val wgt_enq_cntr_en = insn_valid
  val wgt_enq_cntr_wait = !wgt_queue.io.enq.ready || io.weights.waitrequest
  val wgt_enq_cntr_val = Reg(UInt(16.W))
  val wgt_enq_cntr_wrap = (wgt_enq_cntr_val === wgt_enq_cntr_max)

  val wgt_deq_cntr_max = x_size * y_size
  val wgt_deq_cntr_en = insn_valid
  val wgt_deq_cntr_wait = io.wgt_mem.waitrequest || !wgt_queue.io.deq.valid
  val wgt_deq_cntr_val = Reg(UInt(16.W))
  val wgt_deq_cntr_wrap = (wgt_deq_cntr_val === wgt_deq_cntr_max)

  val weights_read = RegInit(false.B)
  val weights_data = Reg(UInt(128.W))

  // push and pop signals
  val pop_next_dep_ready = RegInit(false.B)
  val push_next_dep_valid = push_next_dep && push
  val push_next_dep_ready = RegInit(false.B)

  // setup state transitions
  when (((inp_enq_cntr_en && !inp_enq_cntr_wrap) && (inp_deq_cntr_en && !inp_deq_cntr_wrap)) ||
        ((wgt_enq_cntr_en && !wgt_enq_cntr_wrap) && (wgt_deq_cntr_en && !wgt_deq_cntr_wrap))) {
    when (pop_next_dep && !pop_next_dep_ready) {
      state := s_DUMP
    } .otherwise {
      state := s_BUSY
    }
  }
  when (((inp_enq_cntr_en && inp_enq_cntr_wrap) && (inp_deq_cntr_en && inp_deq_cntr_wrap)) ||
        ((wgt_enq_cntr_en && wgt_enq_cntr_wrap) && (wgt_deq_cntr_en && wgt_deq_cntr_wrap))) {
    when (push_next_dep && !push_next_dep_ready) {
      state := s_PUSH
    } .otherwise {
      state := s_DONE
    }
  }

  // dependency queue processing
  when (dump &&  pop_next_dep_ready) { state := s_BUSY }
  when (push && push_next_dep_ready) { state := s_DONE }

  // dependency queue processing
  io.g2l_dep_queue.ready := pop_next_dep_ready && dump
  io.g2l_dep_queue.data <> DontCare
  when (pop_next_dep && io.g2l_dep_queue.valid && dump) {
    pop_next_dep_ready := true.B
  }
  io.l2g_dep_queue.data := 1.U
  io.l2g_dep_queue.valid := push_next_dep_valid
  when (push_next_dep_valid && io.l2g_dep_queue.ready && push) {
    push_next_dep_ready := true.B
  }

  // setup counter
  when (inputs_read && !inp_enq_cntr_wait && busy && inp_enq_cntr_val < inp_enq_cntr_max) {
    inp_enq_cntr_val := inp_enq_cntr_val + 1.U
  }
  when (weights_read && !wgt_enq_cntr_wait && busy && wgt_enq_cntr_val < wgt_enq_cntr_max) {
    wgt_enq_cntr_val := wgt_enq_cntr_val + 1.U
  }
  
  // fetch instruction
  when (io.load_queue.valid && (idle || done)) {
    // clean up local variables
    inp_enq_cntr_val := 0.U
    inp_deq_cntr_val := 0.U
    wgt_enq_cntr_val := 0.U
    wgt_deq_cntr_val := 0.U
    pop_next_dep_ready := false.B
    push_next_dep_ready := false.B
    // dequeue
    insn := io.load_queue.data
    when (insn_valid) {
      io.load_queue.ready := 1.U
    } .otherwise {
      io.load_queue.ready := 0.U
    }
  } .otherwise {
    insn := insn
    io.load_queue.ready := 0.U
  }

  // enqueue from inputs to fifo
  val inp_sram_addr = (sram_idx * batch.U + inp_enq_cntr_val) << 4.U
  inputs_read := inp_enq_cntr_en && !inp_enq_cntr_wrap && busy
  io.inputs.read := inputs_read
  io.inputs.address := inp_sram_addr
  io.inputs.write <> DontCare
  io.inputs.writedata <> DontCare
  inp_queue.io.enq.bits := io.inputs.readdata
  when (inputs_read && !inp_enq_cntr_wait && busy) {
    inp_queue.io.enq.valid := 1.U
  } .otherwise {
    inp_queue.io.enq.valid := 0.U
  }

  // dequeue fifo and send to inp_mem
  io.inp_mem.writedata := inp_queue.io.deq.bits
  io.inp_mem.address := (dram_idx * batch.U + inp_deq_cntr_val) << 4.U
  io.inp_mem.read <> DontCare
  io.inp_mem.readdata <> DontCare
  when (inp_deq_cntr_en && inp_queue.io.deq.valid && busy) {
    io.inp_mem.write := 1.U
    when (!io.inp_mem.waitrequest) {
      inp_deq_cntr_val := inp_deq_cntr_val + 1.U
      inp_queue.io.deq.ready := 1.U
    } .otherwise {
      inp_queue.io.deq.ready := 0.U
    }
  } .otherwise {
    io.inp_mem.write := 0.U
    inp_queue.io.deq.ready := 0.U
  }

  // enqueue from weights to fifo
  val wgt_sram_addr = (sram_idx * block_out.U + wgt_enq_cntr_val) << 4.U
  weights_read := wgt_enq_cntr_en && !wgt_enq_cntr_wrap && busy
  io.weights.read := weights_read
  io.weights.address := wgt_sram_addr
  io.weights.write <> DontCare
  io.weights.writedata <> DontCare
  wgt_queue.io.enq.bits := io.weights.readdata
  when (weights_read && !wgt_enq_cntr_wait && busy) {
    wgt_queue.io.enq.valid := 1.U
  } .otherwise {
    wgt_queue.io.enq.valid := 0.U
  }

  // dequeue fifo and send to wgt_mem
  io.wgt_mem.writedata := wgt_queue.io.deq.bits
  io.wgt_mem.address := (dram_idx * block_out.U + wgt_deq_cntr_val) << 4.U
  io.wgt_mem.read <> DontCare
  io.wgt_mem.readdata <> DontCare
  when (wgt_deq_cntr_en && wgt_queue.io.deq.valid && busy) {
    io.wgt_mem.write := 1.U
    when (!io.wgt_mem.waitrequest) {
      wgt_deq_cntr_val := wgt_deq_cntr_val + 1.U
      wgt_queue.io.deq.ready := 1.U
    } .otherwise {
      wgt_queue.io.deq.ready := 0.U
    }
  } .otherwise {
    io.wgt_mem.write := 0.U
    wgt_queue.io.deq.ready := 0.U
  }

}

