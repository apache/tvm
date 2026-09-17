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

#ifndef TVM_S_TIR_IR_DATA_TYPE_REWRITER_H_
#define TVM_S_TIR_IR_DATA_TYPE_REWRITER_H_

#include <tvm/s_tir/stmt.h>

#include "../../tirx/ir/data_type_rewriter.h"

namespace tvm {
namespace s_tir {

// Explicit normalization for schedulable TE functions and tensor intrinsic bodies.
// Ordinary TIRX dtype passes run after block lowering.
class IndexDataTypeNormalizer : public tirx::IndexDataTypeNormalizer {
 public:
  using Parent = tirx::IndexDataTypeNormalizer;
  using Parent::Mutate;
  using Parent::Mutate_;
  explicit IndexDataTypeNormalizer(PrimType target_data_type)
      : Parent(std::move(target_data_type), GlobalVTable()) {}
  tirx::PrimFunc Rewrite(tirx::PrimFunc func);

  UnchangedOr<tirx::Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode);
  UnchangedOr<tirx::Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode);

 protected:
  static void InitVTable(VTable* table) {
    Parent::InitVTable(table);
    SetDispatch<IndexDataTypeNormalizer, SBlockNode>(table);
    SetDispatch<IndexDataTypeNormalizer, SBlockRealizeNode>(table);
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }

 private:
  ffi::Map<ffi::String, ffi::Any> VisitBlockAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations);
  tirx::IterVar VisitIterVar(const tirx::IterVar& iter_var);
  tirx::BufferRegion VisitBufferRegion(const tirx::BufferRegion& buffer_region);
};

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_IR_DATA_TYPE_REWRITER_H_
