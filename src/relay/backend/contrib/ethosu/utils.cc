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
 * \file relay/backend/contrib/ethosu/utils.cc
 * \brief Utilities for microNPU codegen
 */

#include "utils.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/usmp/utils.h>

#include <utility>

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

BaseAddress::BaseAddress(String name, Integer primfunc_param_idx, Integer region, Integer size,
                         Bool is_runtime_allocation) {
  auto base_address_node = make_object<BaseAddressNode>();
  base_address_node->name = name;
  base_address_node->primfunc_param_idx = primfunc_param_idx;
  base_address_node->region = region;
  base_address_node->size = size;
  base_address_node->is_runtime_allocation = is_runtime_allocation;
  data_ = std::move(base_address_node);
}

TVM_REGISTER_NODE_TYPE(BaseAddressNode);
TVM_REGISTER_GLOBAL("relay.ext.ethos-u.BaseAddress")
    .set_body_typed([](String name, Integer primfunc_param_idx, Integer region, Integer size,
                       Bool is_runtime_allocation) {
      if (is_runtime_allocation.defined()) {
        return BaseAddress(name, primfunc_param_idx, region, size, is_runtime_allocation);
      } else {
        return BaseAddress(name, primfunc_param_idx, region, size);
      }
    });

CompilationArtifact::CompilationArtifact(String function_name, String command_stream,
                                         String encoded_constants,
                                         Array<BaseAddress> base_addresses) {
  auto compilation_artifact_node = make_object<CompilationArtifactNode>();
  compilation_artifact_node->function_name = function_name;
  compilation_artifact_node->command_stream = command_stream;
  compilation_artifact_node->encoded_constants = encoded_constants;
  compilation_artifact_node->base_addresses = base_addresses;
  data_ = std::move(compilation_artifact_node);
}

TVM_REGISTER_NODE_TYPE(CompilationArtifactNode);
TVM_REGISTER_GLOBAL("relay.ext.ethos-u.CompilationArtifact")
    .set_body_typed([](String function_name, String command_stream, String encoded_constants,
                       Array<BaseAddress> base_addresses) {
      return CompilationArtifact(function_name, command_stream, encoded_constants, base_addresses);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CompilationArtifactNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CompilationArtifactNode*>(ref.get());
      p->stream << "CompilationArtifactNode(\n"
                << "function_name=" << node->function_name
                << ",\n  command_stream=" << node->command_stream
                << ",\n  encoded_constants=" << node->encoded_constants
                << ",\n  base_addresses=" << node->base_addresses << ")";
    });

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
