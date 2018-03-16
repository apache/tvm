/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta.h
 * \brief Type definitions and prototype for VTA HLS design.
 */
#ifndef VTA_MAIN_H_
#define VTA_MAIN_H_

#include <assert.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "vta_typedefs.h"
#include "vta_params.h"

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
void fetch (
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
void load (
  volatile inp_vec_T *inputs,
  volatile wgt_vec_T *weights,
  hls::stream<insn_T> &load_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &l2g_dep_queue,
  inp_vec_T inp_mem[INP_BUFF_DEPTH][BATCH],
  wgt_vec_T wgt_mem[WGT_BUFF_DEPTH][BLOCK_OUT]
  );

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
void compute (
  volatile uint32_t &done,
  volatile uop_T *uops,
  volatile acc_vec_T *biases,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<bool> &l2g_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &g2s_dep_queue,
  out_vec_T inp_mem[INP_BUFF_DEPTH][BATCH],
  wgt_vec_T wgt_mem[WGT_BUFF_DEPTH][BLOCK_OUT],
  out_vec_T out_mem[ACC_BUFF_DEPTH][BATCH]
  );

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
void store (
  volatile out_vec_T *outputs,
  hls::stream<insn_T> &store_queue,
  hls::stream<bool> &g2s_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  out_vec_T out_mem[ACC_BUFF_DEPTH][BATCH]
  );

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
void vta (
  uint32_t insn_count,
  volatile insn_T *insns,
  volatile uop_T *uops,
  volatile inp_vec_T *inputs,
  volatile wgt_vec_T *weights,
  volatile acc_vec_T *biases,
  volatile out_vec_T *outputs);

#endif  // VTA_MAIN_H_