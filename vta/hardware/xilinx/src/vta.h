/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta.h
 * \brief Type definitions and prototype for VTA HLS design.
 */
#ifndef VTA_VTA_H_
#define VTA_VTA_H_

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <assert.h>
#include <hls_stream.h>

#include <vta/hw_spec.h>

/* \typedef uop_T Micro-op datatype*/
typedef ap_uint<VTA_UOP_WIDTH> uop_T;

/* \typedef inp_T Input datatype*/
typedef ap_int<VTA_INP_WIDTH> inp_T;

/* \typedef wgt_T Weight datatype*/
typedef ap_int<VTA_WGT_WIDTH> wgt_T;

/* \typedef out_T Output datatype*/
typedef ap_int<VTA_OUT_WIDTH> out_T;

/* \typedef acc_T Accumulator datatype*/
typedef ap_int<VTA_ACC_WIDTH> acc_T;

/* \typedef mul_T Multiplier output datatype*/
typedef ap_int<VTA_WGT_WIDTH+VTA_INP_WIDTH+1> mul_T;

/* \typedef sum_T GEMM accumulator datatype*/
typedef ap_int<VTA_WGT_WIDTH+VTA_INP_WIDTH+VTA_LOG_BLOCK_IN+1> sum_T;

/* \typedef inp_vec_T Input vector datatype*/
typedef ap_uint<VTA_INP_WIDTH*VTA_BLOCK_IN> inp_vec_T;

/* \typedef wgt_vec_T Weight vector datatype*/
typedef ap_uint<VTA_WGT_WIDTH*VTA_BLOCK_IN> wgt_vec_T;

/* \typedef acc_vec_T Accumulator vector datatype*/
typedef ap_uint<VTA_ACC_WIDTH*VTA_BLOCK_OUT> acc_vec_T;

/* \typedef out_vec_T Output vector datatype*/
typedef ap_uint<VTA_OUT_WIDTH*VTA_BLOCK_OUT> out_vec_T;

/* \typedef uop_idx_T Micro-op SRAM index datatype*/
typedef ap_uint<VTA_LOG_UOP_BUFF_DEPTH+1> uop_idx_T;

/* \typedef inp_idx_T Input SRAM index datatype*/
typedef ap_uint<VTA_LOG_INP_BUFF_DEPTH+1> inp_idx_T;

/* \typedef wgt_idx_T Weight SRAM index datatype*/
typedef ap_uint<VTA_LOG_WGT_BUFF_DEPTH+1> wgt_idx_T;

/* \typedef acc_idx_T Accumulator SRAM index datatype*/
typedef ap_uint<VTA_LOG_ACC_BUFF_DEPTH+1> acc_idx_T;

/* \typedef opcode_T Opcode datatype*/
typedef ap_uint<VTA_OPCODE_BIT_WIDTH> opcode_T;

/* \typedef insn_T Instruction datatype*/
typedef ap_uint<VTA_INS_WIDTH> insn_T;

/* \typedef loop_T Loop bound datatype*/
typedef ap_uint<VTA_LOOP_ITER_WIDTH> loop_T;

/* \typedef memop_id_T Memory operation ID datatype*/
typedef ap_uint<VTA_MEMOP_ID_BIT_WIDTH> memop_id_T;

/* \typedef memop_sram_T Memory operation SRAM index datatype*/
typedef ap_uint<VTA_MEMOP_SRAM_ADDR_BIT_WIDTH> memop_sram_T;

/* \typedef memop_dram_T Memory operation DRAM index datatype*/
typedef ap_uint<VTA_MEMOP_DRAM_ADDR_BIT_WIDTH> memop_dram_T;

/* \typedef memop_size_T Memory operation range datatype*/
typedef ap_uint<VTA_MEMOP_SIZE_BIT_WIDTH> memop_size_T;

/* \typedef memop_stride_T Memory operation stride datatype*/
typedef ap_uint<VTA_MEMOP_STRIDE_BIT_WIDTH> memop_stride_T;

/* \typedef memop_pad_T Memory operation pad width datatype*/
typedef ap_uint<VTA_MEMOP_PAD_BIT_WIDTH> memop_pad_T;

/* \typedef aluop_opcode_T ALU operation opcode datatype*/
typedef ap_uint<VTA_ALU_OPCODE_BIT_WIDTH> aluop_opcode_T;

/* \typedef aluop_opcode_T ALU operation immediate datatype*/
typedef ap_int<VTA_ALUOP_IMM_BIT_WIDTH> aluop_imm_T;

/* \typedef aluop_opcode_T ALU operation shift immediate datatype*/
typedef ap_int<VTA_LOG_ACC_WIDTH> aluop_sh_imm_T;

/*!
* \brief Fetch module.
*   Reads in \a insn_count instructions via DMA and pushes them to the
*   appropriate load, gemm or store queue.
* \param insns Instruction data base address in DRAM. AXI-4 master port.
* \param insn_count Total instruction count. AXI-lite memory mapped register.
* \param load_queue Load instruction queue. AXI-stream FIFO.
* \param gemm_queue GEMM instruction queue. AXI-stream FIFO.
* \param store_queue Store instruction queue. AXI-stream FIFO.
*/
void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue);

/*!
* \brief Load module.
*   Reads in load instructions from the load queue, and performs appropriate
*   DMA load operation to the \a wgt_mem and \a inp_mem SRAM buffers from DRAM.
*   Updates dependence queues accordingly.
* \param inputs Input data base address in DRAM. AXI-4 master port.
* \param weights Weight data base address in DRAM. AXI-4 master port.
* \param load_queue Load instruction queue. AXI-stream FIFO.
* \param g2l_dep_queue Dependence queue from GEMM to load stage.
*   AXI-stream FIFO.
* \param l2g_dep_queue Dependence queue from load to GEMM stage.
*   AXI-stream FIFO.
* \param inp_mem Local input SRAM buffer. Write only single port BRAM.
* \param wgt_mem Local weight SRAM buffer. Write only single port BRAM.
*/
void load(
  volatile inp_vec_T *inputs,
  volatile wgt_vec_T *weights,
  hls::stream<insn_T> &load_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &l2g_dep_queue,
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH],
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT]);

/*!
* \brief Compute module.
*   Reads in GEMM instructions from the gemm queue, and performs appropriate
*   GEMM/ALU instructions. Reads in data from the \a wgt_mem and \a inp_mem,
*   and writes computation results into the \a out_mem. Updates dependence
*   queues accordingly.
* \param done Signal that indicates that VLA is done.  AXI-lite memory mapped
*   register.
* \param uops Micro-op data base address in DRAM. AXI-4 master port.
* \param biases Bias data base address in DRAM. AXI-4 master port.
* \param gemm_queue GEMM instruction queue. AXI-stream FIFO.
* \param l2g_dep_queue Dependence queue from load to gemm stage.
*   AXI-stream FIFO.
* \param s2g_dep_queue Dependence queue from store to gemm stage.
*   AXI-stream FIFO.
* \param g2l_dep_queue Dependence queue from gemm to load stage.
*   AXI-stream FIFO.
* \param g2s_dep_queue Dependence queue from gemm to store stage.
*   AXI-stream FIFO.
* \param inp_mem Local input SRAM buffer. Read only single port BRAM.
* \param wgt_mem Local weight SRAM buffer. Read only single port BRAM.
* \param out_mem Local output SRAM buffer. Write only single port BRAM.
*/
void compute(
  volatile uint32_t &done,
  volatile uop_T *uops,
  volatile acc_vec_T *biases,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<bool> &l2g_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &g2s_dep_queue,
  out_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH],
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT],
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH]);

/*!
* \brief Store module.
*   Reads in store instructions from the store queue, and performs appropriate
*   store instructions from the output buffer in SRAM to DRAM. Updates dependence
*   queues accordingly.
* \param outputs Output data base address in DRAM. AXI-4 master port.
* \param store_queue Store instruction queue. AXI-stream FIFO.
* \param g2s_dep_queue Dependence queue from gemm to store stage.
*   AXI-stream FIFO.
* \param s2g_dep_queue Dependence queue from store to gemm stage.
*   AXI-stream FIFO.
* \param out_mem Local output SRAM buffer. Read only single port BRAM.
*/
void store(
  volatile out_vec_T *outputs,
  hls::stream<insn_T> &store_queue,
  hls::stream<bool> &g2s_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH]);

/*!
* \brief VTA wrapper for simulation purpose only.
*   Orchestrates dataflow execution of the fetch, load, GEMM and store stages.
* \param insn_count Total instruction count. AXI-lite memory mapped register.
* \param insns Instruction data base address in DRAM. AXI-4 master port.
* \param uops Micro-op data base address in DRAM. AXI-4 master port.
* \param inputs Input data base address in DRAM. AXI-4 master port.
* \param weights Weight data base address in DRAM. AXI-4 master port.
* \param biases Bias data base address in DRAM. AXI-4 master port.
* \param outputs Output data base address in DRAM. AXI-4 master port.
*/
void vta(
  uint32_t insn_count,
  volatile insn_T *insns,
  volatile uop_T *uops,
  volatile inp_vec_T *inputs,
  volatile wgt_vec_T *weights,
  volatile acc_vec_T *biases,
  volatile out_vec_T *outputs);

#endif  // VTA_VTA_H_
