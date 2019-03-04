package vta

import chisel3._
import freechips.rocketchip.config.{Parameters, Field}

case object LOG_INP_WIDTH extends Field[Int]
case object LOG_WGT_WIDTH extends Field[Int]
case object LOG_ACC_WIDTH extends Field[Int]
case object LOG_OUT_WIDTH extends Field[Int]
case object LOG_BATCH extends Field[Int]
case object LOG_BLOCK_IN extends Field[Int]
case object LOG_BLOCK_OUT extends Field[Int]
case object LOG_UOP_BUFF_SIZE extends Field[Int]
case object LOG_INP_BUFF_SIZE extends Field[Int]
case object LOG_WGT_BUFF_SIZE extends Field[Int]
case object LOG_ACC_BUFF_SIZE extends Field[Int]

class AvalonSlaveIO(val dataBits: Int = 32, val addrBits: Int = 1)(implicit p: Parameters) extends CoreBundle()(p) {
  val waitrequest = Output(Bool())
  val address = Input(UInt(addrBits.W))
  val read = Input(Bool())
  val readdata = Output(UInt(dataBits.W))
  val write = Input(Bool())
  val writedata = Input(UInt(dataBits.W))
}
  
class AvalonSinkIO(val dataBits: Int = 32)(implicit p: Parameters) extends CoreBundle()(p) {
  val ready = Output(Bool())
  val valid = Input(Bool())
  val data = Input(UInt(dataBits.W))
}

abstract trait CoreParams {
  implicit val p: Parameters

  val log_uop_buff_size = p(LOG_UOP_BUFF_SIZE)
  val log_inp_buff_size = p(LOG_INP_BUFF_SIZE)
  val log_wgt_buff_size = p(LOG_WGT_BUFF_SIZE)
  val log_acc_buff_size = p(LOG_ACC_BUFF_SIZE)

  val log_batch = p(LOG_BATCH)
  val log_block_in = p(LOG_BLOCK_IN)
  val log_block_out = p(LOG_BLOCK_OUT)
  val batch = 1 << p(LOG_BATCH)
  val block_in = 1 << p(LOG_BLOCK_IN)
  val block_out = 1 << p(LOG_BLOCK_OUT)

  val log_uop_width = 5
  val log_inp_width = p(LOG_INP_WIDTH)
  val log_wgt_width = p(LOG_WGT_WIDTH)
  val log_acc_width = p(LOG_ACC_WIDTH)
  val log_out_width = p(LOG_OUT_WIDTH)
  val uop_width = 1 << log_uop_width
  val inp_width = 1 << p(LOG_INP_WIDTH)
  val wgt_width = 1 << p(LOG_WGT_WIDTH)
  val acc_width = 1 << p(LOG_ACC_WIDTH)
  val out_width = 1 << p(LOG_OUT_WIDTH)

  val opcode_bit_width = 3
  val alu_opcode_bit_width = 2
  val memop_id_bit_width = 2
  val memop_sram_addr_bit_width = 16
  val memop_dram_addr_bit_width = 32
  val memop_size_bit_width = 16
  val memop_stride_bit_width = 16
  val memop_pad_bit_width = 4
  val memop_pad_val_bit_width = 2
  val aluop_imm_bit_width = 16
  val loop_iter_width = 14

  val mem_id_uop = 0
  val mem_id_wgt = 1
  val mem_id_inp = 2
  val mem_id_acc = 3
  val mem_id_out = 4

  val opcode_load = 0
  val opcode_store = 1
  val opcode_gemm = 2
  val opcode_finish = 3
  val opcode_alu = 4

  val alu_opcode_min = 0
  val alu_opcode_max = 1
  val alu_opcode_add = 2
  val alu_opcode_shr = 3

  val log_uop_buff_depth = log_uop_buff_size - log_uop_width + 3
  val log_wgt_buff_depth = log_wgt_buff_size - log_block_out - log_block_in - log_wgt_width + 3
  val log_inp_buff_depth = log_inp_buff_size - log_batch - log_block_in - log_inp_width + 3
  val log_acc_buff_depth = log_acc_buff_size - log_batch - log_block_out - log_acc_width + 3

  val insn_mem_0_0 = 0
  val insn_mem_0_1 = opcode_bit_width - 1
  val insn_mem_1 = insn_mem_0_1 + 1
  val insn_mem_2 = insn_mem_1 + 1
  val insn_mem_3 = insn_mem_2 + 1
  val insn_mem_4 = insn_mem_3 + 1
  val insn_mem_5_0 = insn_mem_4 + 1
  val insn_mem_5_1 = insn_mem_5_0 + memop_id_bit_width - 1
  val insn_mem_6_0 = insn_mem_5_1 + 1
  val insn_mem_6_1 = insn_mem_6_0 + memop_sram_addr_bit_width - 1
  val insn_mem_7_0 = insn_mem_6_1 + 1
  val insn_mem_7_1 = insn_mem_7_0 + memop_dram_addr_bit_width - 1
  val insn_mem_8_0 = 64
  val insn_mem_8_1 = insn_mem_8_0 + memop_size_bit_width - 1
  val insn_mem_9_0 = insn_mem_8_1 + 1
  val insn_mem_9_1 = insn_mem_9_0 + memop_size_bit_width - 1
  val insn_mem_a_0 = insn_mem_9_1 + 1
  val insn_mem_a_1 = insn_mem_a_0 + memop_stride_bit_width - 1
  val insn_mem_b_0 = insn_mem_a_1 + 1
  val insn_mem_b_1 = insn_mem_b_0 + memop_pad_bit_width - 1
  val insn_mem_c_0 = insn_mem_b_1 + 1
  val insn_mem_c_1 = insn_mem_c_0 + memop_pad_bit_width - 1
  val insn_mem_d_0 = insn_mem_c_1 + 1
  val insn_mem_d_1 = insn_mem_d_0 + memop_pad_bit_width - 1
  val insn_mem_e_0 = insn_mem_d_1 + 1
  val insn_mem_e_1 = insn_mem_e_0 + memop_pad_bit_width - 1

  val insn_gem_0_0 = 0
  val insn_gem_0_1 = opcode_bit_width - 1
  val insn_gem_1 = insn_gem_0_1 + 1
  val insn_gem_2 = insn_gem_1 + 1
  val insn_gem_3 = insn_gem_2 + 1
  val insn_gem_4 = insn_gem_3 + 1
  val insn_gem_5 = insn_gem_4 + 1
  val insn_gem_6_0 = insn_gem_5 + 1
  val insn_gem_6_1 = insn_gem_6_0 + log_uop_buff_depth - 1
  val insn_gem_7_0 = insn_gem_6_1 + 1
  val insn_gem_7_1 = insn_gem_7_0 + log_uop_buff_depth
  val insn_gem_8_0 = insn_gem_7_1 + 1
  val insn_gem_8_1 = insn_gem_8_0 + loop_iter_width - 1
  val insn_gem_9_0 = insn_gem_8_1 + 1
  val insn_gem_9_1 = insn_gem_9_0 + loop_iter_width - 1
  val insn_gem_a_0 = 64
  val insn_gem_a_1 = insn_gem_a_0 + log_acc_buff_depth - 1
  val insn_gem_b_0 = insn_gem_a_1 + 1
  val insn_gem_b_1 = insn_gem_b_0 + log_acc_buff_depth - 1
  val insn_gem_c_0 = insn_gem_b_1 + 1
  val insn_gem_c_1 = insn_gem_c_0 + log_inp_buff_depth - 1
  val insn_gem_d_0 = insn_gem_c_1 + 1
  val insn_gem_d_1 = insn_gem_d_0 + log_inp_buff_depth - 1
  val insn_gem_e_0 = insn_gem_d_1 + 1
  val insn_gem_e_1 = insn_gem_e_0 + log_wgt_buff_depth - 1
  val insn_gem_f_0 = insn_gem_e_1 + 1
  val insn_gem_f_1 = insn_gem_f_0 + log_wgt_buff_depth - 1

  val insn_alu_e_0 = insn_gem_d_1 + 1
  val insn_alu_e_1 = insn_alu_e_0 + alu_opcode_bit_width - 1
  val insn_alu_f = insn_alu_e_1 + 1
  val insn_alu_g_0 = insn_alu_f + 1
  val insn_alu_g_1 = insn_alu_g_0 + aluop_imm_bit_width - 1

  val uop_alu_0_0 = 0
  val uop_alu_0_1 = uop_alu_0_0 + log_acc_buff_depth - 1
  val uop_alu_1_0 = uop_alu_0_1 + 1
  val uop_alu_1_1 = uop_alu_1_0 + log_inp_buff_depth - 1
}

abstract class CoreBundle(implicit val p: Parameters) extends Bundle with CoreParams

