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

class Fetch(implicit val p: Parameters) extends Module with CoreParams {
  val io = IO(new FetchIO())

  val insn_count = Reg(UInt(16.W))

  // create instruction buffer
  val insns_queue = Module(new Queue(UInt(128.W), 8))

  // counters
  val insn_cntr_en = !io.insns.waitrequest
  val insn_cntr_wait = !io.load_queue.ready && !io.gemm_queue.ready && !io.store_queue.ready
  // val (insn_cntr_val, insn_cntr_wrap) = ICounter(insn_cntr_en && !insn_cntr_wait, insn_count)
  val insn_cntr_val = RegInit(0.U)
  val insn_cntr_wrap = RegInit(0.U)

  // status registers
  val busy = Mux(insn_cntr_val <= insn_count, 1.U, 0.U)

  // output registers
  val insn = insns_queue.io.deq.bits
  val opcode = insn(insn_mem_0_1, insn_mem_0_0)
  val memory_type = insn(insn_mem_5_1, insn_mem_5_0)

  // update counter
  when (insn_cntr_en && !insn_cntr_wait) {
    when (insn_cntr_val < (insn_count - 1.U)) {
      insn_cntr_val := insn_cntr_val + 1.U;
      insn_cntr_wrap := 0.U
    } .otherwise {
      insn_cntr_val := insn_cntr_val
      insn_cntr_wrap := 1.U
    }
  } .otherwise {
    insn_cntr_val := insn_cntr_val
    insn_cntr_wrap := 0.U
  }

  // fetch insn_count
  io.insn_count.readdata <> DontCare
  when (io.insn_count.write && !busy) {
    insn_count := io.insn_count.writedata
    io.insn_count.waitrequest := 0.U
  } .otherwise {
    insn_count := insn_count
    io.insn_count.waitrequest := 1.U
  }

  // fetch instructions
  io.insns.address := insn_cntr_val
  io.insns.read := insn_cntr_en
  io.insns.write := 0.U
  io.insns.writedata <> DontCare
  insns_queue.io.enq.bits := io.insns.readdata
  insns_queue.io.enq.valid := insn_cntr_en

  // set default behavior
  io.store_queue.data := insn
  io.load_queue.data := insn
  io.gemm_queue.data := insn
  io.load_queue.valid := 0.U
  io.gemm_queue.valid := 0.U
  io.store_queue.valid := 0.U
  insns_queue.io.deq.ready := 0.U

  // enqueue instructions
  when (opcode === opcode_store.U) {
    io.store_queue.valid := 1.U
    insns_queue.io.deq.ready := 1.U
  } .elsewhen (opcode === opcode_store.U && (memory_type === mem_id_inp.U || memory_type === mem_id_wgt.U)) {
    io.load_queue.valid := 1.U
    insns_queue.io.deq.ready := 1.U
  } .otherwise {
    io.gemm_queue.valid := 1.U
    insns_queue.io.deq.ready := 1.U
  }

}

