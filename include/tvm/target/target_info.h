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
 * \file tvm/target/target_info.h
 * \brief Various information about target.
 */
#ifndef TVM_TARGET_TARGET_INFO_H_
#define TVM_TARGET_TARGET_INFO_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <string>

namespace tvm {

/*!
 * \brief Memory information of special memory region.
 *  Use MemoryInfo as its container type
 */
class MemoryInfoNode : public Object {
 public:
  /*! \brief The addressable unit */
  int64_t unit_bits;
  /*! \brief Maximum number of bits supported in the memory */
  int64_t max_num_bits;
  /*! \brief maximum number of bits to be used in simd op */
  int64_t max_simd_bits;
  /*!
   * \brief head address of the buffer, if visible to CPU
   *  This address can be None.
   */
  PrimExpr head_address;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MemoryInfoNode>()
        .def_ro("unit_bits", &MemoryInfoNode::unit_bits)
        .def_ro("max_num_bits", &MemoryInfoNode::max_num_bits)
        .def_ro("max_simd_bits", &MemoryInfoNode::max_simd_bits)
        .def_ro("head_address", &MemoryInfoNode::head_address);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.MemoryInfo", MemoryInfoNode, Object);
};

/*! \brief Defines memory info */
class MemoryInfo : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MemoryInfo, ObjectRef, MemoryInfoNode);
};

/*!
 * \brief get memory info given scope
 * \param scope The scope name.
 * \return info The memory info.
 */
TVM_DLL MemoryInfo GetMemoryInfo(const std::string& scope);

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_INFO_H_
