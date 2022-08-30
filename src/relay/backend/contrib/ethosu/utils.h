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
 * \brief Base addresses are input pointers to
 * the driver that get accessed by the command stream
 * using offsets to read/write data.
 */
struct BaseAddressNode : public Object {
  /*! \brief The identifier, usually it the param name of the PrimFunc that gets lowered */
  String name;
  /*! \brief The index in the params array of the PrimFunc. This is needed to keep aligned
   * between the PrimFunc arguments ordering and argument ordering of generated code */
  Integer primfunc_param_idx;
  /*! \brief The region used by the command stream. This needs to match with base address
   * index passed into the driver */
  Integer region;
  /*! \brief The size of the buffer accessible by this base address */
  Integer size;
  /*! \brief This is a runtime allocation that needs to be done in the function */
  Bool is_runtime_allocation{Bool(false)};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("primfunc_param_idx", &primfunc_param_idx);
    v->Visit("region", &region);
    v->Visit("size", &size);
    v->Visit("is_runtime_allocation", &is_runtime_allocation);
  }

  bool SEqualReduce(const BaseAddressNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(primfunc_param_idx, other->primfunc_param_idx) &&
           equal(region, other->region) && equal(size, other->size) &&
           equal(is_runtime_allocation, other->is_runtime_allocation);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(primfunc_param_idx);
    hash_reduce(region);
    hash_reduce(size);
    hash_reduce(is_runtime_allocation);
  }

  static constexpr const char* _type_key = "relay.ext.ethos-u.BaseAddress";
  TVM_DECLARE_FINAL_OBJECT_INFO(BaseAddressNode, Object);
};

class BaseAddress : public ObjectRef {
 public:
  TVM_DLL BaseAddress(String name, Integer primfunc_param_idx, Integer region, Integer size,
                      Bool is_runtime_allocation = Bool(false));
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BaseAddress, ObjectRef, BaseAddressNode);
};

/*!
 * \brief Captures all the binary artifactes required to create
 * the C-source runtime module
 */
struct CompilationArtifactNode : public Object {
  /*! \brief The function name for this artifact belongs to */
  String function_name;
  /*! \brief The binary command stream (CS) in hex format */
  String command_stream;
  /*! \brief The encoded biases and weights in hex format */
  String encoded_constants;
  /*! \brief The information regarding the base addresses */
  Array<BaseAddress> base_addresses;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("function_name", &function_name);
    v->Visit("command_stream", &command_stream);
    v->Visit("encoded_constants", &encoded_constants);
    v->Visit("base_addresses", &base_addresses);
  }

  bool SEqualReduce(const CompilationArtifactNode* other, SEqualReducer equal) const {
    return equal(function_name, other->function_name) &&
           equal(command_stream, other->command_stream) &&
           equal(encoded_constants, other->encoded_constants) &&
           equal(base_addresses, other->base_addresses);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(function_name);
    hash_reduce(command_stream);
    hash_reduce(encoded_constants);
    hash_reduce(base_addresses);
  }

  static constexpr const char* _type_key = "relay.ext.ethos-u.CompilationArtifact";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompilationArtifactNode, Object);
};

class CompilationArtifact : public ObjectRef {
 public:
  TVM_DLL CompilationArtifact(String function_name, String command_stream, String encoded_constants,
                              Array<BaseAddress> base_addresses);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CompilationArtifact, ObjectRef, CompilationArtifactNode);
};

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSU_UTILS_H_
