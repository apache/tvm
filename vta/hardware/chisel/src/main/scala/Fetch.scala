// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.{Parameters, Field}

class FetchIO(implicit p: Parameters) extends CoreBundle()(p) {
  val insn_count = new AvalonSlaveIO(dataBits = 16)
  val insns = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val load_queue = Flipped(new AvalonSinkIO(dataBits = 128))
  val gemm_queue = Flipped(new AvalonSinkIO(dataBits = 128))
  val store_queue = Flipped(new AvalonSinkIO(dataBits = 128))
}

/*
 * Operating sequence:
 * 1. set insn_count via memory map interface
 * 2. start instruction counter and enqueue instructions into insn_queue
 * 3. dequeue insn_queue and stream into corresponding (load/gemm/store) queue
 */
class Fetch(implicit val p: Parameters) extends Module with CoreParams {
  val io = IO(new FetchIO())

  val insn_count = Reg(UInt(16.W))

  // create instruction buffer
  val insns_queue = Module(new InstructionQueue(UInt(128.W), 4))

  // output registers
  val insn = Wire(UInt(128.W))
  val opcode = insn(insn_mem_0_1, insn_mem_0_0)
  val memory_type = insn(insn_mem_5_1, insn_mem_5_0)

  // status registers
  val state = Reg(UInt(2.W))
  val s_IDLE :: s_BUSY :: s_DONE :: Nil = Enum(3)
  val idle = state === s_IDLE
  val busy = state === s_BUSY
  val done = state === s_DONE

  // counters
  val insn_cntr_en = insn_count > 0.U
  val insn_cntr_wait = !insns_queue.io.enq.ready || io.insns.waitrequest
  val insn_cntr_val = Reg(UInt(16.W))
  val insn_cntr_wrap = (insn_cntr_val === insn_count) && busy

  // update operating status
  when (busy && insn_cntr_wrap && !insns_queue.io.deq.valid) { state := s_DONE }

  // update counter
  when (insn_cntr_en && !insn_cntr_wait && (insn_cntr_val < insn_count)) {
    insn_cntr_val := insn_cntr_val + 1.U;
  }

  // fetch insn_count
  io.insn_count.readdata <> DontCare
  io.insn_count.waitrequest := 0.U
  when (done) {
    insn_count := 0.U
  }
  when (io.insn_count.write) {
    when (busy) {
      io.insn_count.waitrequest := 1.U
    } .otherwise {
      insn_count := io.insn_count.writedata
      insn_cntr_val := 0.U
      state := s_BUSY
    }
  }

  // fetch instructions
  val insns_read = insn_cntr_en && busy && insns_queue.io.enq.ready && (insn_cntr_val < insn_count)
  io.insns.address := insn_cntr_val << 4.U
  io.insns.read := insns_read
  io.insns.write := 0.U
  io.insns.writedata <> DontCare
  insns_queue.io.enq.bits := io.insns.readdata
  insns_queue.io.enq.valid := 0.U
  when (insns_read && !io.insns.waitrequest) {
    insns_queue.io.enq.valid := 1.U
  }

  // set default behavior
  insn := insns_queue.io.deq.bits
  io.store_queue.data := insn
  io.load_queue.data := insn
  io.gemm_queue.data := insn
  io.load_queue.valid := 0.U
  io.gemm_queue.valid := 0.U
  io.store_queue.valid := 0.U
  insns_queue.io.deq.ready := 0.U

  // enqueue instructions into corresponding queue
  // (implements blocking write to streams)
  when (busy) {
  when (opcode === opcode_store.U) {
    io.store_queue.valid := insns_queue.io.deq.valid
    insns_queue.io.deq.ready := io.store_queue.ready
  } .elsewhen (opcode === opcode_store.U && (memory_type === mem_id_inp.U || memory_type === mem_id_wgt.U)) {
    io.load_queue.valid := insns_queue.io.deq.valid
    insns_queue.io.deq.ready := io.load_queue.ready
  } .otherwise {
    io.gemm_queue.valid := insns_queue.io.deq.valid
    insns_queue.io.deq.ready := io.gemm_queue.ready
  }
  }

}

