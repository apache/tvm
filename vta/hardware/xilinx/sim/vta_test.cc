/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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

    // Micro op bound
    assert(VTA_UOP_GEM_2_1 < VTA_UOP_WIDTH);
    assert(VTA_UOP_ALU_1_1 < VTA_UOP_WIDTH);
    // Make sure there is no misaligment
    assert(VTA_INSN_GEM_9_1 < VTA_INSN_GEM_A_0);
    assert(VTA_INSN_MEM_7_1 < VTA_INSN_MEM_8_0);
    // Instruction bounds
    assert(VTA_INSN_MEM_E_1 < VTA_INS_WIDTH);
    assert(VTA_INSN_GEM_F_1 < VTA_INS_WIDTH);
    assert(VTA_INSN_ALU_G_1 < VTA_INS_WIDTH);

    int status = 0;

    // Run ALU test (vector-scalar operators)
    status |= alu_test(VTA_ALU_OPCODE_MIN, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MIN, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MAX, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MAX, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_ADD, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_ADD, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_SHR, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_SHR, true, VTA_BLOCK_OUT, 128, false);

    // Run ALU test (vector-vector operators)
    status |= alu_test(VTA_ALU_OPCODE_MIN, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MIN, false, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MAX, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MAX, false, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_ADD, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_ADD, false, VTA_BLOCK_OUT, 128, false);

    // Run blocked GEMM test
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 2);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 2);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 1);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 1);

    // Simple GEMM unit test
    status |= gemm_test(64, 64, 64, true);

    return status;
}
