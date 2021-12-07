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
 * \file relay/backend/contrib/ethosu/utils.h
 * \brief Utilities for microNPU codegen
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ETHOSU_UTILS_H_
#define TVM_RELAY_BACKEND_CONTRIB_ETHOSU_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

/*!
 * \brief Captures all the binary artifactes required to create
 * the C-source runtime module
 */
struct CompilationArtifactNode : public Object {
  /*! \brief The binary command stream (CS) in hex format */
  String command_stream;
  /*! \brief The encoded biases and weights in hex format */
  String encoded_constants;
  /*! \brief The intermediary scratch area required for the execution of the CS */
  Integer scratch_size;
  /*! \brief The size of the input tensor in bytes */
  Integer input_size;
  /*! \brief The size of the output tensor in bytes */
  Integer output_size;
  /*! \brief The name of the function */
  String function_name;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("command_stream", &command_stream);
    v->Visit("encoded_constants", &encoded_constants);
    v->Visit("scratch_size", &scratch_size);
    v->Visit("input_size", &input_size);
    v->Visit("output_size", &output_size);
    v->Visit("function_name", &function_name);
  }

  bool SEqualReduce(const CompilationArtifactNode* other, SEqualReducer equal) const {
    return equal(command_stream, other->command_stream) &&
           equal(encoded_constants, other->encoded_constants) &&
           equal(scratch_size, other->scratch_size) && equal(input_size, other->input_size) &&
           equal(output_size, other->output_size) && equal(function_name, other->function_name);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(command_stream);
    hash_reduce(encoded_constants);
    hash_reduce(scratch_size);
    hash_reduce(input_size);
    hash_reduce(output_size);
    hash_reduce(function_name);
  }

  static constexpr const char* _type_key = "relay.ext.ethos-u.CompilationArtifact";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompilationArtifactNode, Object);
};

class CompilationArtifact : public ObjectRef {
 public:
  TVM_DLL CompilationArtifact(String command_stream, String encoded_constants, Integer scratch_size,
                              Integer input_size, Integer output_size, String function_name);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CompilationArtifact, ObjectRef, CompilationArtifactNode);
};

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSU_UTILS_H_
