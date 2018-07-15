/*!
 *  Copyright (c) 2018 by Contributors
 * \file test_lib.cpp
 * \brief Test library for the VTA design simulation and driver tests.
 */

#ifndef TESTS_HARDWARE_COMMON_TEST_LIB_H_
#define TESTS_HARDWARE_COMMON_TEST_LIB_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vta/hw_spec.h>

#ifdef NO_SIM

#include <vta/driver.h>

#ifdef VTA_TARGET_PYNQ
#include "../../../src/pynq/pynq_driver.h"
#endif  // VTA_TARGET_PYNQ

typedef uint64_t axi_T;
typedef uint32_t uop_T;
typedef int8_t wgt_T;
typedef int8_t inp_T;
typedef int8_t out_T;
typedef int32_t acc_T;

uint64_t vta(
  uint32_t insn_count,
  VTAGenericInsn *insns,
  VTAUop *uops,
  inp_T *inputs,
  wgt_T *weights,
  acc_T *biases,
  inp_T *outputs);

#else  // NO_SIM

#include "../../../hardware/xilinx/src/vta.h"

#endif  // NO_SIM

/*!
* \brief Returns opcode string.
* \param opcode Opcode parameter (defined in vta_defines.h).
* \param use_imm Boolean that indicates if the operation uses an immediate value.
* \return The opcode string.
*/
const char* getOpcodeString(int opcode, bool use_imm);

/*!
* \brief Performs buffer data packing and tiling.
* \param dst Pointer to the packed, and tiled destination 1D array (flattened).
* \param src Pointer to the unpacked source 2D array.
* \param y_size Number of rows.
* \param x_size Number of columns.
* \param y_block Inner tiling along row dimension.
* \param x_block Inner tiling along column dimension.
*/
template <typename T, int T_WIDTH>
void packBuffer(T *dst, T **src, int y_size, int x_size, int y_block, int x_block);

/*!
* \brief Performs buffer data unpacking.
* \param dst Pointer to the unpacked destination 2D array.
* \param src Pointer to the packed, and tiled source 1D array (flattened).
* \param y_size Number of rows.
* \param x_size Number of columns.
* \param y_block Inner tiling along row dimension.
* \param x_block Inner tiling along column dimension.
*/
template <typename T, int T_WIDTH>
void unpackBuffer(T **dst, T *src, int y_size, int x_size, int y_block, int x_block);

/*!
* \brief Allocates and initializes a 2D array in the heap.
* \param rows Number of rows.
* \param cols Number of columns.
* \return Pointer to the 2D array.
*/
template <typename T, int T_WIDTH>
T ** allocInit2dArray(int rows, int cols);

/*!
* \brief Allocates a 2D array in the heap.
* \param rows Number of rows.
* \param cols Number of columns.
* \return Pointer to the 2D array.
*/
template <typename T>
T ** alloc2dArray(int rows, int cols);

/*!
* \brief Frees a 2D array.
* \param array Pointer to the 2D array to be freed.
* \param rows Number of rows.
* \param cols Number of columns.
*/
template <typename T>
void free2dArray(T **array, int rows, int cols);

/*!
* \brief Allocates a 3D array in the heap.
* \param rows Number of rows (dim 0).
* \param cols Number of columns (dim 1).
* \param depth Depth of the array (dim 2).
* \return Pointer to the 3D array.
*/
template <typename T>
T *** alloc3dArray(int rows, int cols, int depth);

/*!
* \brief Frees a 3D array.
* \param array Pointer to the 3D array.
* \param rows Number of rows (dim 0).
* \param cols Number of columns (dim 1).
* \param depth Depth of the array (dim 2).
*/
template <typename T>
void free3dArray(T *** array, int rows, int cols, int depth);

/*!
* \brief Performs memory allocation in a physically contiguous region of memory.
* \param num_bytes Size of the buffer in bytes.
* \return Pointer to the allocated buffer.
*/
void * allocBuffer(size_t num_bytes);

/*!
* \brief Frees buffer allocated in a physically contiguous region of memory.
* \param buffer Pointer to the buffer to free.
*/
void freeBuffer(void * buffer);

/*!
* \brief Returns a VTA reset instruction on a 2D patch of the register file.
* \param type On-chip memory target.
* \param sram_offset Offset in SRAM.
* \param y_size Number of rows to reset (y axis).
* \param x_size Number of elements per row to reset (x axis).
* \param x_stride Stride along the x axis.
* \param pop_prev_dep Pop dependence from previous stage.
* \param pop_next_dep Pop dependence from next stage.
* \param push_prev_dep Push dependence to previous stage.
* \param push_next_dep Push dependence to next stage.
* \return A VTAGenericInsn for a reset op.
*/
VTAGenericInsn reset2DInsn(int type, int sram_offset, int y_size, int x_size, int x_stride,
  int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep);

/*!
* \brief Returns a VTA 2D load or store instruction.
* \param opcode Type of operation.
* \param type On-chip memory target.
* \param sram_offset Offset in SRAM.
* \param dram_offset Offset in DRAM.
* \param y_size Number of rows to load/store (y axis).
* \param x_size Number of elements per row to load/store (x axis).
* \param x_stride Stride along the x axis.
* \param y_pad Padding along the y axis.
* \param x_pad Padding along the x axis.
* \param pop_prev_dep Pop dependence from previous stage.
* \param pop_next_dep Pop dependence from next stage.
* \param push_prev_dep Push dependence to previous stage.
* \param push_next_dep Push dependence to next stage.
* \return A VTAGenericInsn for a 2D load or store op.
*/
VTAGenericInsn get2DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset,
  int y_size, int x_size, int x_stride, int y_pad, int x_pad, int pop_prev_dep, int pop_next_dep,
  int push_prev_dep, int push_next_dep);

/*!
* \brief Returns a VTA 1D load or store instruction.
* \param opcode Type of operation.
* \param type On-chip memory target.
* \param sram_offset Offset in SRAM.
* \param dram_offset Offset in DRAM.
* \param size Number of elements to load/store.
* \param pop_prev_dep Pop dependence from previous stage.
* \param pop_next_dep Pop dependence from next stage.
* \param push_prev_dep Push dependence to previous stage.
* \param push_next_dep Push dependence to next stage.
* \return A VTAGenericInsn for a 1D load or store op.
*/
VTAGenericInsn get1DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset, int size,
  int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep);

/*!
* \brief Returns a VTA matrix multiplication instruction of size (a, b) x (b, c).
* \param uop_offset Offset of the micro-op in SRAM.
* \param batch Batch size (a).
* \param in_feat Input features (b).
* \param out_feat Output features (c).
* \param uop_compression Apply micro-op compression.
* \param pop_prev_dep Pop dependence from previous stage.
* \param pop_next_dep Pop dependence from next stage.
* \param push_prev_dep Push dependence to previous stage.
* \param push_next_dep Push dependence to next stage.
* \return A VTAGenericInsn for a GEMM op.
*/
VTAGenericInsn getGEMMInsn(int uop_offset, int batch, int in_feat, int out_feat,
  bool uop_compression, int pop_prev_dep, int pop_next_dep, int push_prev_dep,
  int push_next_dep);

/*!
* \brief Returns a VTA ALU instruction for map type operation.
* \param opcode Opcode of the ALU instruction.
* \param use_imm Use immediate.
* \param imm Immediate value (int16).
* \param vector_size Vector size of the ALU operation size.
* \param uop_compression Apply micro-op compression.
* \param pop_prev_dep Pop dependence from previous stage.
* \param pop_next_dep Pop dependence from next stage.
* \param push_prev_dep Push dependence to previous stage.
* \param push_next_dep Push dependence to next stage.
* \return A VTAGenericInsn for a ALU op.
*/
VTAGenericInsn getALUInsn(int opcode, bool use_imm, int imm, int vector_size, bool uop_compression,
  int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep);

/*!
* \brief Returns a VTA finish instruction.
* \param pop_prev Pop dependence from previous stage.
* \param pop_next Pop dependence from next stage.
* \return A VTAGenericInsn for a finish op.
*/
VTAGenericInsn getFinishInsn(bool pop_prev, bool pop_next);

/*!
* \brief Returns an allocated buffer of VTA micro-ops to implement a copy operation.
* \param y_size Number of rows to load/store (y axis).
* \param x_size Number of elements per row to load/store (x axis).
* \param uop_compression Apply micro-op compression.
* \return A VTAUop pointer to an allocated micro-op buffer.
*/
VTAUop * getCopyUops(int y_size, int x_size, int uop_compression);

/*!
* \brief Returns an allocated buffer of VTA micro-ops to implement a matrix multiplication
*   of size (a, b) x (b, c).
* \param batch Batch size (a).
* \param in_feat Input features (b).
* \param out_feat Output features (c).
* \param uop_compression Apply micro-op compression.
* \param multi_threaded Generate micro-ops for two virtual execution threads.
* \return A VTAUop pointer to an allocated micro-op buffer.
*/
VTAUop * getGEMMUops(int batch, int in_feat, int out_feat, bool uop_compression,
  bool multi_threaded);

/*!
* \brief Returns an allocated buffer of VTA micro-ops to implement a vector-vector map operation.
* \param vector_size Vector size.
* \param uop_compression Apply micro-op compression.
* \return A VTAUop pointer to an allocated micro-op buffer.
*/
VTAUop * getMapALUUops(int vector_size, bool uop_compression);

/*!
* \brief Print out parameters of the VTA design (for debugging purposes).
*/
void printParameters();

/*!
* \brief Print out instruction information (for debugging purposes).
* \param num_insn Number of instructions.
* \param insns Pointer to the instruction buffer.
*/
void printInstruction(int num_insn, VTAGenericInsn *insns);

/*!
* \brief Print out micro-op information (for debugging purposes).
* \param num_insn Number of micro-ops.
* \param insns Pointer to the micro-op buffer.
*/
void printMicroOp(int num_uop, VTAUop *uops);

/*!
* \brief VTA ALU unit test.
* \param opcode The ALU opcode.
* \param use_imm Use immediate.
* \param batch Batch size.
* \param vector_size Vector length of the ALU operation.
* \param uop_compression Apply micro-op compression.
* \return Number of errors from the test run.
*/
int alu_test(int opcode, bool use_imm, int batch, int vector_size, bool uop_compression);

/*!
* \brief VTA blocked GEMM unit test.
* \param batch Batch size.
* \param channels Channel width.
* \param block Blocking size.
* \param uop_compression Apply micro-op compression.
* \return Number of errors from the test run.
*/
int blocked_gemm_test(int batch, int channels, int block, bool uop_compression,
  int virtual_threads);

/*!
* \brief VTA GEMM unit test.
* \param batch Batch size.
* \param in_channels Input channels.
* \param out_channels Output channels.
* \param uop_compression Apply micro-op compression.
* \return Number of errors from the test run.
*/
int gemm_test(int batch, int in_channels, int out_channels, bool uop_compression);

#endif  //  TESTS_HARDWARE_COMMON_TEST_LIB_H_
