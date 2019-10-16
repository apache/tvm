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
 *  Copyright (c) 2019 by Contributors
 * \file tvm/runtime/vm/executable.cc
 * \brief The implementation of a virtual machine executable APIs.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/vm.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include "serializer.h"

namespace tvm {
namespace runtime {
namespace vm {

PackedFunc Executable::GetFunction(const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "get_lib") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetLib();
    });
  } else if (name == "get_bytecode") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetBytecode();
    });
  } else if (name == "get_stats") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->Stats();
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc(nullptr);
  }
}

std::string Executable::GetBytecode() const {
  std::ostringstream oss;

  for (const auto& func : functions) {
    // Print the header of the function format.
    oss << "# func name, reg file size, param count, inst count:"
        << std::endl;
    oss << func.name << " "
        << func.register_file_size << " "
        << func.params.size() << " "
        << func.instructions.size() << std::endl;

    // Print pramams of a `VMFunction`.
    oss << "# Parameters: "<< std::endl;
    for (const auto& param : func.params) {
      oss << param << " ";
    }
    oss << std::endl;

    // Print the instructions of a `VMFunction`.
    // The part after ";" is the instruction in text format.
    oss << "hash, opcode, fields # inst(text):"<< std::endl;
    for (const auto& instr : func.instructions) {
      const auto& serialized_instr = SerializeInstruction(instr);
      oss << std::hex << "0x" << serialized_instr.Hash() << " "
          << std::dec << serialized_instr.opcode << " ";
      for (auto it : serialized_instr.fields) {
        oss << it << " ";
      }
      oss << "  # " << instr;
      if (oss.str().back() != '\n') oss << std::endl;
    }
  }

  return oss.str();
}

std::string Executable::Stats() const {
  std::ostringstream oss;
  oss << "Relay VM executable statistics:" << std::endl;

  // Get the number of constants and the shape of each of them.
  oss << "  Constant shapes (# " << constants.size() << "): [";
  for (const auto& it : constants) {
    const auto* cell = it.as<TensorObj>();
    CHECK(cell);
    runtime::NDArray data = cell->data;
    const auto& shape = data.Shape();

    // Scalar
    if (shape.empty()) {
      oss << "scalar, ";
      continue;
    }

    oss << "[";
    for (auto s : shape) {
      oss << s << ", ";
    }
    oss.seekp(-2, oss.cur);
    oss << "], " << std::endl;
  }
  if (!constants.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of globals and the name of each of them.
  oss << "  Globals (#" << global_map.size() << "): [";
  for (const auto& it : global_map) {
    oss << "(\"" << it.first << "\", " << it.second << ")" << ", ";
  }
  if (!global_map.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of primitive ops and the name of each of them.
  oss << "  Primitive ops (#" << primitive_map.size() << "): [";
  std::vector<std::string> prim_ops;
  for (const auto& it : primitive_map) {
    auto packed_index = static_cast<size_t>(it.second);
    if (prim_ops.size() <= packed_index) {
      prim_ops.resize(packed_index + 1);
    }
    prim_ops[packed_index] = it.first;
  }
  for (const auto& it : prim_ops) {
    oss << it << ", ";
  }
  if (!prim_ops.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  return oss.str();
}

TVM_REGISTER_GLOBAL("relay._vm.GetNumOfGlobals")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec);
  *rv = static_cast<int>(exec->global_map.size());
});

TVM_REGISTER_GLOBAL("relay._vm.GetGlobalFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec);
  int idx = args[1];
  std::vector<std::pair<std::string, Index> > globals(exec->global_map.begin(),
                                                      exec->global_map.end());
  auto comp = [](const std::pair<std::string, Index>& a,
                 const std::pair<std::string, Index>& b) {
    return a.second < b.second;
  };
  std::sort(globals.begin(), globals.end(), comp);
  CHECK_LT(idx, globals.size());
  *rv = globals[idx].first;
});

TVM_REGISTER_GLOBAL("relay._vm.GetNumOfPrimitives")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec);
  *rv = static_cast<int>(exec->primitive_map.size());
});


TVM_REGISTER_GLOBAL("relay._vm.GetPrimitiveFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec);
  int idx = args[1];
  CHECK_GE(idx, 0);
  CHECK_LT(idx, exec->primitive_map.size());

  for (const auto& it : exec->primitive_map) {
    if (idx == static_cast<int>(it.second)) {
      *rv = it.first;
      break;
    }
  }
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
