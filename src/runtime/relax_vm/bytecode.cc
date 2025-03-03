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
 * \file src/runtime/relax_vm/bytecode.cc
 * \brief The bytecode for Relax virtual machine.
 */

#include <tvm/runtime/logging.h>
#include <tvm/runtime/relax_vm/bytecode.h>

#include <functional>
#include <sstream>

namespace tvm {
namespace runtime {
namespace relax_vm {

Instruction Instruction::Call(Index func_idx, Index num_args, Instruction::Arg* args, RegName dst) {
  Instruction instr;
  instr.op = Opcode::Call;
  instr.dst = dst;
  instr.func_idx = func_idx;
  instr.num_args = num_args;
  instr.args = args;
  return instr;
}

Instruction Instruction::Ret(RegName result) {
  Instruction instr;
  instr.op = Opcode::Ret;
  instr.result = result;
  return instr;
}

Instruction Instruction::Goto(Index pc_offset) {
  Instruction instr;
  instr.op = Opcode::Goto;
  instr.pc_offset = pc_offset;
  return instr;
}

Instruction Instruction::If(RegName cond, Index false_offset) {
  Instruction instr;
  instr.op = Opcode::If;
  instr.cond = cond;
  instr.false_offset = false_offset;
  return instr;
}
}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
