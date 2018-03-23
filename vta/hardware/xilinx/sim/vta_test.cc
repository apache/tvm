/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_test.cpp
 * \brief Simulation tests for the VTA design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../src/vta.h"
#include "../../../tests/hardware/common/test_lib.h"

int main(void) {
#if DEBUG == 1
    printParameters();
#endif

    // Buffer indexing
    assert(VTA_LOG_ACC_BUFF_DEPTH >= VTA_LOG_INP_BUFF_DEPTH);
    // Micro op bound
    assert(VTA_UOP_GEM_3_1 < VTA_UOP_WIDTH);
    assert(VTA_UOP_ALU_3_1 < VTA_UOP_WIDTH);
    // Instruction alignment checks
    assert(VTA_INSN_MEM_7_1 < VTA_INSN_MEM_8_0);
    assert(VTA_INSN_GEM_8_1 < VTA_INSN_GEM_9_0);
    // Instruction bounds
    assert(VTA_INSN_MEM_E_1 < VTA_INS_WIDTH);
    assert(VTA_INSN_GEM_E_1 < VTA_INS_WIDTH);
    assert(VTA_INSN_ALU_F_1 < VTA_INS_WIDTH);

    int status = 0;

    // Run ALU test (vector-scalar operators)
    status |= alu_test(VTA_ALU_OPCODE_MIN, true, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MIN, true, 16, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MAX, true, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MAX, true, 16, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_ADD, true, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_ADD, true, 16, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_SHR, true, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_SHR, true, 16, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_SHL, true, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_SHL, true, 16, 128, false);

    // Run ALU test (vector-vector operators)
    status |= alu_test(VTA_ALU_OPCODE_MIN, false, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MIN, false, 16, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MAX, false, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MAX, false, 16, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_ADD, false, 16, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_ADD, false, 16, 128, false);

    // Run blocked GEMM test
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 2);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 2);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 1);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 1);

    return status;
}
