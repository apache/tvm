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

CompilationArtifact::CompilationArtifact(String command_stream, String encoded_constants,
                                         Integer scratch_size, Integer input_size,
                                         Integer output_size, String function_name) {
  auto compilation_artifact_node = make_object<CompilationArtifactNode>();
  compilation_artifact_node->command_stream = command_stream;
  compilation_artifact_node->encoded_constants = encoded_constants;
  compilation_artifact_node->scratch_size = scratch_size;
  compilation_artifact_node->input_size = input_size;
  compilation_artifact_node->output_size = output_size;
  compilation_artifact_node->function_name = function_name;
  data_ = std::move(compilation_artifact_node);
}

TVM_REGISTER_NODE_TYPE(CompilationArtifactNode);
TVM_REGISTER_GLOBAL("relay.ext.ethos-u.CompilationArtifact")
    .set_body_typed([](String command_stream, String encoded_constants, Integer scratch_size,
                       Integer input_size, Integer output_size, String function_name) {
      return CompilationArtifact(command_stream, encoded_constants, scratch_size, input_size,
                                 output_size, function_name);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CompilationArtifactNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CompilationArtifactNode*>(ref.get());
      p->stream << "CompilationArtifactNode(\n"
                << "command_stream=" << node->command_stream
                << ",\n  encoded_constants=" << node->encoded_constants
                << ",\n  scratch_size=" << node->scratch_size
                << ",\n  input_size=" << node->input_size
                << ",\n  output_size=" << node->output_size
                << ",\n  function_name=" << node->function_name << ")";
    });

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
