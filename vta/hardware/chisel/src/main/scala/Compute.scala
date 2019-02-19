// See LICENSE for license details.

package vta

import chisel3._
import chisel3.util._
import freechips.rocketchip.config.Parameters

class ComputeIO(implicit p: Parameters) extends CoreBundle()(p) {
  val done = new AvalonSlaveIO(dataBits = 1, addrBits = 1)
  val uops = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val biases = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 32))
  val gemm_queue = new AvalonSinkIO(dataBits = 128)
  val l2g_dep_queue = new AvalonSinkIO(dataBits = 1)
  val s2g_dep_queue = new AvalonSinkIO(dataBits = 1)
  val g2l_dep_queue = Flipped(new AvalonSinkIO(dataBits = 1))
  val g2s_dep_queue = Flipped(new AvalonSinkIO(dataBits = 1))
  val inp_mem = Flipped(new AvalonSlaveIO(dataBits = 64, addrBits = 15))
  val wgt_mem = Flipped(new AvalonSlaveIO(dataBits = 64, addrBits = 18))
  val out_mem = Flipped(new AvalonSlaveIO(dataBits = 128, addrBits = 17))
}

class DepQueue[T <: Data](gen: T, entries: Int) extends Queue (gen, entries) {}
class OutQueue[T <: Data](gen: T, entries: Int) extends Queue (gen, entries) {}

class Compute(implicit val p: Parameters) extends Module with CoreParams {
  val io = IO(new ComputeIO)

  io.inp_mem <> DontCare
  io.wgt_mem <> DontCare

  val started = !reset.toBool

  val acc_mem = Mem(1 << 8, UInt(512.W))
  val uop_mem = Mem(1 << 10, UInt(32.W))

  val insn            = Reg(UInt(128.W))
  val insn_valid      = (insn =/= 0.U) && started

  val opcode          = insn(insn_mem_0_1, insn_mem_0_0)
  val pop_prev_dep    = insn(insn_mem_1)
  val pop_next_dep    = insn(insn_mem_2)
  val push_prev_dep   = insn(insn_mem_3)
  val push_next_dep   = insn(insn_mem_4)

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

  // write to out_mem
  val uop_bgn = Reg(UInt(16.W))
  uop_bgn := insn(insn_gem_6_1, insn_gem_6_0)
  val uop_end = Reg(UInt(16.W))
  uop_end := insn(insn_gem_7_1, insn_gem_7_0)
  val iter_out = insn(insn_gem_8_1, insn_gem_8_0)
  val iter_in = insn(insn_gem_9_1, insn_gem_9_0)
  val dst_factor_out = 1.U(16.W)
  val dst_factor_in  = 1.U(16.W)
  val src_factor_out = 1.U(16.W)
  val src_factor_in  = 1.U(16.W)
  val wgt_factor_out = 1.U(16.W)
  val wgt_factor_in  = 1.U(16.W)
  val alu_opcode = insn(insn_alu_e_1, insn_alu_e_0)
  val use_imm = insn(insn_alu_f)
  val imm_raw = insn(insn_alu_g_1, insn_alu_g_0)
  val imm = Mux(imm_raw.asSInt < 0.S, Cat("hffff".U, imm_raw), Cat("h0000".U, imm_raw)).asSInt

  val sram_idx = sram_base
  val dram_idx = dram_base
  val y_size_total = y_pad_0 + y_size + y_pad_1
  val x_size_total = x_pad_0 + x_size + x_pad_1
  val y_offset = x_size_total * y_pad_0

  // control and status registers (csr)
  val opcode_finish_en = (opcode === opcode_finish.U)
  val opcode_load_en = (opcode === opcode_load.U || opcode === opcode_store.U)
  val opcode_gemm_en = (opcode === opcode_gemm.U)
  val opcode_alu_en = (opcode === opcode_alu.U)

  val memory_type_uop_en = memory_type === mem_id_uop.U
  val memory_type_acc_en = memory_type === mem_id_acc.U

  // status
  val state = RegInit(0.U(3.W))
  val s_IDLE :: s_DUMP :: s_BUSY :: s_PUSH :: s_DONE :: Nil = Enum(5)
  val idle = state === s_IDLE
  val dump = state === s_DUMP
  val busy = state === s_BUSY
  val push = state === s_PUSH
  val done = state === s_DONE

  // uops / biases / out_mem
  val uops_read   = Reg(Bool())
  val uops_data   = Reg(UInt((32 + 128).W))

  val biases_read = Reg(Bool())
  val biases_bits = block_out * acc_width
  val biases_beats = biases_bits / 128
  val biases_data = Reg(Vec(biases_beats + 1, UInt(128.W)))

  val out_mem_write  = RegInit(false.B)

  // counters
  val uop_cntr_max_val = x_size >> log2Ceil(128 / uop_width).U
  val uop_cntr_max = Mux(uop_cntr_max_val === 0.U, 1.U, uop_cntr_max_val)
  val uop_cntr_en = (opcode_load_en && memory_type_uop_en && insn_valid)
  val uop_cntr_wait = io.uops.waitrequest
  val uop_cntr_val = Reg(UInt(16.W))
  val uop_cntr_wrap = ((uop_cntr_val === uop_cntr_max) && uop_cntr_en && busy)

  val acc_cntr_max = x_size * biases_beats.U + 1.U
  val acc_cntr_en = (opcode_load_en && memory_type_acc_en && insn_valid)
  val acc_cntr_wait = io.biases.waitrequest
  val acc_cntr_val = Reg(UInt(16.W))
  val acc_cntr_wrap = ((acc_cntr_val === acc_cntr_max) && acc_cntr_en && busy)

  val upc_cntr_max_val = uop_end - uop_bgn
  val upc_cntr_max = Mux(upc_cntr_max_val <= 0.U, 1.U, upc_cntr_max_val)
  val out_cntr_max_val = iter_in * iter_out * upc_cntr_max
  val out_cntr_max = out_cntr_max_val + 2.U
  val out_cntr_en = ((opcode_alu_en || opcode_gemm_en) && insn_valid)
  val out_cntr_wait = io.out_mem.waitrequest
  val out_cntr_val = Reg(UInt(16.W))
  val out_cntr_wrap = ((out_cntr_val === out_cntr_max) && out_cntr_en && busy)

  // dependency queue status
  val pop_prev_dep_ready = RegInit(false.B)
  val pop_next_dep_ready = RegInit(false.B)
  val push_prev_dep_valid = push_prev_dep && push
  val push_next_dep_valid = push_next_dep && push
  val push_prev_dep_ready = RegInit(false.B)
  val push_next_dep_ready = RegInit(false.B)

  val gemm_queue_ready = RegInit(false.B)

  // update busy status
  val finish_wrap = RegInit(false.B)
  when (opcode_finish_en) {
    when ( pop_prev_dep)      { finish_wrap :=  pop_prev_dep_ready && busy }
    .elsewhen ( pop_next_dep) { finish_wrap :=  pop_next_dep_ready && busy }
    .elsewhen (push_prev_dep) { finish_wrap := push_prev_dep_ready && busy }
    .elsewhen (push_next_dep) { finish_wrap := push_next_dep_ready && busy }
    .otherwise { finish_wrap := false.B }
  } .otherwise { finish_wrap := false.B }
  when (uop_cntr_wrap || acc_cntr_wrap || out_cntr_wrap || finish_wrap) {
    when (push_prev_dep || push_next_dep) {
      state := s_PUSH
    } .otherwise {
      state := s_DONE
    }
  }

  // dependency queue processing
  when (busy && (!pop_prev_dep_ready && !pop_next_dep_ready) && (pop_prev_dep || pop_next_dep)) { state := s_DUMP }
  when (dump && ( pop_prev_dep_ready ||  pop_next_dep_ready)) { state := s_BUSY }
  when (push && (push_prev_dep_ready || push_next_dep_ready)) { state := s_DONE }

  // dependency queue processing
  io.l2g_dep_queue.ready := pop_prev_dep_ready && dump
  io.s2g_dep_queue.ready := pop_next_dep_ready && dump
  io.l2g_dep_queue.data <> DontCare
  io.s2g_dep_queue.data <> DontCare
  when (pop_prev_dep && dump && io.l2g_dep_queue.valid) {
    pop_prev_dep_ready := true.B
  }
  when (pop_next_dep && dump && io.s2g_dep_queue.valid) {
    pop_next_dep_ready := true.B
  }
  io.g2l_dep_queue.data := 1.U
  io.g2s_dep_queue.data := 1.U
  io.g2l_dep_queue.valid := push_prev_dep_valid
  io.g2s_dep_queue.valid := push_next_dep_valid
  when (push_prev_dep_valid && io.g2l_dep_queue.ready && push) {
    push_prev_dep_ready := true.B
  }
  when (push_next_dep_valid && io.g2s_dep_queue.ready && push) {
    push_next_dep_ready := true.B
  }

  // setup counters
  when (uops_read && !uop_cntr_wait && busy && uop_cntr_val < uop_cntr_max) {
    uop_cntr_val := uop_cntr_val + 1.U
  }
  when (biases_read && !acc_cntr_wait && busy && acc_cntr_val < acc_cntr_max) {
    acc_cntr_val := acc_cntr_val + 1.U
  }
  // when (out_mem_write && !out_cntr_wait && busy && out_cntr_val < out_cntr_max) {
  //   out_cntr_val := out_cntr_val + 1.U
  // }
  when (out_mem_write && busy && out_cntr_val < out_cntr_max) {
    out_cntr_val := out_cntr_val + 1.U
  }

  // fetch instruction
  when (gemm_queue_ready) {
    insn := io.gemm_queue.data
    uop_cntr_val := 0.U
    acc_cntr_val := 0.U
    out_cntr_val := 0.U
    pop_prev_dep_ready := false.B
    pop_next_dep_ready := false.B
    push_prev_dep_ready := false.B
    push_next_dep_ready := false.B
    state := s_BUSY
  } .otherwise {
    insn := insn
  }
  gemm_queue_ready := io.gemm_queue.valid && (idle || done)
  io.gemm_queue.ready := gemm_queue_ready
  when (gemm_queue_ready) { gemm_queue_ready := false.B }

  // response to done signal
  io.done.waitrequest := 0.U
  io.done.write <> DontCare
  io.done.writedata <> DontCare
  io.done.readdata := opcode_finish_en.asUInt

  // fetch uops
  val uop_mem_write_en = (opcode_load_en && (memory_type === mem_id_uop.U))
  val uop_dram_addr = (dram_idx + uop_cntr_val) << log2Ceil(128 / 8).U
  val uop_sram_addr = Wire(UInt(32.W))
  uop_sram_addr := 0.U
  uop_sram_addr := (sram_idx + uop_cntr_val) << log2Ceil(128 / uop_width).U
  uops_read := uop_cntr_en && !uop_cntr_wrap && busy
  io.uops.read := uops_read
  io.uops.address := uop_dram_addr
  io.uops.write <> DontCare
  io.uops.writedata <> DontCare
  val uops_read_en = Reg(Bool())
  when (uops_read && !uop_cntr_wait) {
    uops_data := Cat(uop_sram_addr, io.uops.readdata) // TODO: copy to uops_data from io.uops.readdata
    uops_read_en := true.B
    when (uop_cntr_val === (uop_cntr_max - 1.U)) { uops_read := 0.U }
  } .otherwise { uops_read_en := false.B }
  when (uops_read_en) {
    val _uop_sram_addr = uops_data(128 + 31, 128)
    for (i <- 0 to 3) {
      uop_mem(_uop_sram_addr + i.U) := uops_data(31 + i * 32, i * 32)
    }
  }

  // fetch biases
  val acc_dram_addr = ((((dram_idx + y_offset + x_pad_0) << 2.U) * batch.U + acc_cntr_val) << log2Ceil(128 / 8).U)
  val acc_sram_addr = ((((sram_idx + y_offset + x_pad_0) << 2.U) * batch.U + acc_cntr_val) >> log2Ceil(biases_beats).U) - 1.U
  biases_read := acc_cntr_en && !done
  io.biases.address := acc_dram_addr
  io.biases.read := biases_read
  io.biases.write <> DontCare
  io.biases.writedata <> DontCare
  when (biases_read && !acc_cntr_wait) {
    biases_data(acc_cntr_val % biases_beats.U) := io.biases.readdata
    // There is a delay between putting biases_readdata into biases_data,
    // and putting concatenated biases_data into acc_mem,
    // therefore, acc_cntr_max = x_size * beats + 1.
    when ((acc_cntr_val % biases_beats.U) === 0.U) {
      acc_mem(acc_sram_addr) := Cat(biases_data.init.reverse)
    }
  }

  // read from uop_mem
  val upc = out_cntr_val % upc_cntr_max
  val uop = RegNext(uop_mem(upc)) // TODO: construct uop as register, and copy from uop_mem block ram
  val dst_offset_out = 0.U(16.W) // it_in
  val src_offset_out = 0.U(16.W) // it_in
  val it_in = RegNext(out_cntr_val) % (iter_in * iter_out)
  val dst_offset_in = dst_offset_out + (it_in * dst_factor_in)(15, 0)
  val src_offset_in = src_offset_out + (it_in * src_factor_in)(15, 0)
  val dst_idx = uop(uop_alu_0_1, uop_alu_0_0) + dst_offset_in
  val src_idx = uop(uop_alu_1_1, uop_alu_1_0) + src_offset_in

  // build alu
  val dst_vector = Reg(UInt(biases_bits.W))
  val src_vector = Reg(UInt(biases_bits.W))
  // when (out_mem_write && !out_cntr_wait) {
  when (out_mem_write) {
    dst_vector := acc_mem(dst_idx)
    src_vector := acc_mem(src_idx)
  }
  val cmp_res       = Wire(Vec(block_out + 1, SInt(acc_width.W)))
  val short_cmp_res = Wire(Vec(block_out + 1, UInt(out_width.W)))
  val add_res       = Wire(Vec(block_out + 1, SInt(acc_width.W)))
  val short_add_res = Wire(Vec(block_out + 1, UInt(out_width.W)))
  val shr_res       = Wire(Vec(block_out + 1, SInt(acc_width.W)))
  val short_shr_res = Wire(Vec(block_out + 1, UInt(out_width.W)))
  val src_0         = Wire(Vec(block_out + 1, SInt(acc_width.W)))
  val src_1         = Wire(Vec(block_out + 1, SInt(acc_width.W)))

  val mix_val       = Wire(Vec(block_out + 1, SInt(acc_width.W)))
  val add_val       = Wire(Vec(block_out + 1, SInt(acc_width.W)))
  val shr_val       = Wire(Vec(block_out + 1, SInt(acc_width.W)))

  val alu_opcode_min_en = alu_opcode === alu_opcode_min.U
  val alu_opcode_max_en = alu_opcode === alu_opcode_max.U

  // set default value
  for (i <- 0 to (block_out)) {
    cmp_res(i) := 0.S
    short_cmp_res(i) := 0.U
    add_res(i) := 0.S
    short_add_res(i) := 0.U
    shr_res(i) := 0.S
    short_shr_res(i) := 0.U

    mix_val(i) := 0.S
    add_val(i) := 0.S
    shr_val(i) := 0.S
    src_0(i) := 0.S
    src_1(i) := 0.S
  }

  // loop unroll
  when (insn_valid && out_cntr_en) {
    when (alu_opcode_max_en) {
      for (b <- 0 to (block_out - 1)) {
        src_0(b) := src_vector((b + 1) * acc_width - 1, b * acc_width).asSInt
        src_1(b) := dst_vector((b + 1) * acc_width - 1, b * acc_width).asSInt
      }
    } .otherwise {
      for (b <- 0 to (block_out - 1)) {
        src_0(b) := dst_vector((b + 1) * acc_width - 1, b * acc_width).asSInt
        src_1(b) := src_vector((b + 1) * acc_width - 1, b * acc_width).asSInt
      }
    }
    when (use_imm) {
      for (b <- 0 to (block_out - 1)) { src_1(b) := imm }
    }
    val block_out_val = block_out - 1
    for (b <- 0 to block_out_val) {
      mix_val(b) := Mux(src_0(b) < src_1(b), src_0(b), src_1(b))
      cmp_res(b) := mix_val(b)
      short_cmp_res(b) := mix_val(b)(out_width - 1, 0)
      add_val(b) := (src_0(b)(acc_width - 1, 0) + src_1(b)(acc_width - 1, 0)).asSInt
      add_res(b) := add_val(b)
      short_add_res(b) := add_res(b)(out_width - 1, 0)
      shr_val(b) := (src_0(b)(acc_width - 1, 0) >> src_1(b)(log_acc_width - 1, 0)).asSInt
      shr_res(b) := shr_val(b)
      short_shr_res(b) := shr_res(b)(out_width - 1, 0)
    }
  }

  // write to out_mem_fifo
  val alu_opcode_minmax_en = alu_opcode_min_en || alu_opcode_max_en
  val alu_opcode_add_en = (alu_opcode === alu_opcode_add.U)
  val out_mem_address = Reg(UInt(32.W))
  val out_mem_writedata = Reg(UInt(128.W))
  val out_mem_fifo = Module(new OutQueue(UInt((32 + 128).W), 32))
  val out_mem_enq_bits = Mux(alu_opcode_minmax_en, Cat(short_cmp_res.init.reverse),
                         Mux(alu_opcode_add_en, Cat(short_add_res.init.reverse), Cat(short_shr_res.init.reverse)))
  out_mem_write := opcode_alu_en && busy && (out_cntr_val <= out_cntr_max_val)
  out_mem_fifo.io.enq.ready <> DontCare
  out_mem_fifo.io.enq.valid := out_mem_write && (out_cntr_val >= 2.U) && (out_cntr_val <= (out_cntr_max - 1.U))
  out_mem_fifo.io.enq.bits := Cat(RegNext(dst_idx), out_mem_enq_bits)

  // write to out_mem interface
  io.out_mem.address := out_mem_fifo.io.deq.bits(128 + 31 , 128) << 4.U
  io.out_mem.write := out_mem_fifo.io.deq.valid
  out_mem_fifo.io.deq.ready := !io.out_mem.waitrequest
  io.out_mem.read <> DontCare
  io.out_mem.writedata := out_mem_fifo.io.deq.bits(127, 0) // out_mem_writedata
}
