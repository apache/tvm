/*!
 *  Copyright (c) 2018 by Contributors
 * \file driver_test.cpp
 * \brief Bare-metal test to test driver and VTA design.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vta_test_lib.h"
#include "vta_pynq_driver.h"

// VTA invocation (present the same abstraction as in the simulation tests)
uint64_t vta (
    uint32_t insn_count,
    VTAGenericInsn *insns,
    VTAUop *uops,
    inp_T *inputs,
    wgt_T *weights,
    acc_T *biases,
    inp_T *outputs) {

    // Performance counter variables
    uint64_t t_fpga;
    struct timespec start, stop;

    // Derive bitstream file
    char bitstream[64];
    char str_batch_size[4];
    char str_block_out_size[4];
    char str_block_in_size[4];
    char str_block_bit_width[4];
    sprintf(str_batch_size, "%d", BATCH);
    sprintf(str_block_out_size, "%d", BLOCK_OUT);
    sprintf(str_block_in_size, "%d", BLOCK_IN);
    sprintf(str_block_bit_width, "%d", WGT_WIDTH);
    strcpy(bitstream, "vta.bit");

#if DEBUG==1
    printf("INFO - Programming FPGA: %s!\n", bitstream);
#endif

    // Program VTA
    ProgramVTA(bitstream);
    // Get VTA handles
    VTAHandle vta_fetch_handle = MapRegister(VTA_FETCH_ADDR, VTA_RANGE);
    VTAHandle vta_load_handle = MapRegister(VTA_LOAD_ADDR, VTA_RANGE);
    VTAHandle vta_compute_handle = MapRegister(VTA_COMPUTE_ADDR, VTA_RANGE);
    VTAHandle vta_store_handle = MapRegister(VTA_STORE_ADDR, VTA_RANGE);

    // Physical address pointers
    uint32_t insn_phy = insns ? cma_get_phy_addr(insns) : 0;
    uint32_t uop_phy = uops ? cma_get_phy_addr(uops) : 0;
    uint32_t weight_phy = weights ? cma_get_phy_addr(weights) : 0;
    uint32_t input_phy = inputs ? cma_get_phy_addr(inputs) : 0;
    uint32_t bias_phy = biases ? cma_get_phy_addr(biases) : 0;
    uint32_t output_phy = outputs ? cma_get_phy_addr(outputs) : 0;

#if DEBUG==1
    printf("INFO - Starting FPGA!\n");
#endif

    clock_gettime(CLOCK_REALTIME, &start);

    // FETCH @ 0x10 : Data signal of insn_count_V
    WriteMappedReg(vta_fetch_handle, 0x10, insn_count);
    // FETCH @ 0x18 : Data signal of insns_V
    if (insns) WriteMappedReg(vta_fetch_handle, 0x18, insn_phy);
    // LOAD @ 0x10 : Data signal of weight_V
    if (weights) WriteMappedReg(vta_load_handle, 0x10, weight_phy);
    // LOAD @ 0x18 : Data signal of inputs_V
    if (inputs) WriteMappedReg(vta_load_handle, 0x18, input_phy);
    // COMPUTE @ 0x20 : Data signal of uops_V
    if (uops) WriteMappedReg(vta_compute_handle, 0x20, uop_phy);
    // COMPUTE @ 0x28 : Data signal of biases_V
    if (biases) WriteMappedReg(vta_compute_handle, 0x28, bias_phy);
    // STORE @ 0x10 : Data signal of outputs_V
    if (outputs) WriteMappedReg(vta_store_handle, 0x10, output_phy);

    // VTA start
    WriteMappedReg(vta_fetch_handle, 0x0, 0x1);
    WriteMappedReg(vta_load_handle, 0x0, 0x81);
    WriteMappedReg(vta_compute_handle, 0x0, 0x81);
    WriteMappedReg(vta_store_handle, 0x0, 0x81);

    int flag = 0, t = 0;
    for (t = 0; t < 10000000; ++t) {
      flag = ReadMappedReg(vta_compute_handle, 0x18);
      if (flag & VTA_DONE) break;
    }

    if (t==10000000) {
        printf("\tWARNING: VTA TIMEOUT!!!!\n");
    }
#if DEBUG==1
    else {
        printf("INFO - FPGA Finished!\n");
    }
#endif

    clock_gettime(CLOCK_REALTIME, &stop);
    t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

    // Unmap VTA register
    UnmapRegister(vta_fetch_handle, VTA_RANGE);
    UnmapRegister(vta_load_handle, VTA_RANGE);
    UnmapRegister(vta_compute_handle, VTA_RANGE);
    UnmapRegister(vta_store_handle, VTA_RANGE);

    return t_fpga;
};

int main(void)
{

#if DEBUG==1
    printParameters();
#endif

    int status = 0;

    // Run ALU test (vector-scalar operators)
    status |= alu_test(ALU_OPCODE_MAX, true, 16, 128, true);
    status |= alu_test(ALU_OPCODE_MAX, true, 16, 128, false);
    status |= alu_test(ALU_OPCODE_ADD, true, 16, 128, true);
    status |= alu_test(ALU_OPCODE_ADD, true, 16, 128, false);
    status |= alu_test(ALU_OPCODE_SHR, true, 16, 128, true);
    status |= alu_test(ALU_OPCODE_SHR, true, 16, 128, false);

    // Run ALU test (vector-vector operators)
    status |= alu_test(ALU_OPCODE_MAX, false, 16, 128, true);
    status |= alu_test(ALU_OPCODE_MAX, false, 16, 128, false);
    status |= alu_test(ALU_OPCODE_ADD, false, 16, 128, true);
    status |= alu_test(ALU_OPCODE_ADD, false, 16, 128, false);

    // Run blocked GEMM test
    status |= blocked_gemm_test(256, 256, BLOCK_OUT*4, true, 2);
    status |= blocked_gemm_test(256, 256, BLOCK_OUT*4, false, 2);
    status |= blocked_gemm_test(256, 256, BLOCK_OUT*4, true, 1);
    status |= blocked_gemm_test(256, 256, BLOCK_OUT*4, false, 1);

    if (status==0) {
        printf("\nINFO - Unit tests successful!\n");
    } else {
        printf("\nINTO - Unit tests failed!\n");
    }

    return status;

}
