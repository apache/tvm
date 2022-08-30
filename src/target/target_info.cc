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
 * \file target/target_info.cc
 */
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>

namespace tvm {

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MemoryInfoNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MemoryInfoNode*>(node.get());
      p->stream << "mem-info("
                << "unit_bits=" << op->unit_bits << ", "
                << "max_num_bits=" << op->max_num_bits << ", "
                << "max_simd_bits=" << op->max_simd_bits << ", "
                << "head_address=" << op->head_address << ")";
    });

TVM_REGISTER_NODE_TYPE(MemoryInfoNode);

MemoryInfo GetMemoryInfo(const std::string& scope) {
  std::string fname = "tvm.info.mem." + scope;
  const runtime::PackedFunc* f = runtime::Registry::Get(fname);
  if (f == nullptr) {
    LOG(WARNING) << "MemoryInfo for scope = " << scope << " is undefined";
    return MemoryInfo();
  } else {
    return (*f)();
  }
}

}  // namespace tvm
