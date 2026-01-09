/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_texture_alloc.cc
 */

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/backend/adreno/transform.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../arith/ir_mutator_with_analyzer.h"
#include "../../../runtime/texture.h"
#include "../../transforms/ir_utils.h"

namespace tvm {
namespace tir {
namespace backend {
namespace adreno {
using runtime::ApplyTexture2DFlattening;
using runtime::DefaultTextureLayoutSeparator;
using runtime::IsTextureStorage;

/*!
 * \brief Inject Texture Alloc Intrensic right after AllocateNode are realized.
 */
class TextureAllocInjector : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Inject(PrimFunc func) {
    arith::Analyzer ana;
    auto pass = TextureAllocInjector(&ana);
    auto writer = func.CopyOnWrite();
    pass.MarkBufferMapShapes(func);
    writer->body = pass.VisitStmt(func->body);
    return func;
  }

 private:
  using IRMutatorWithAnalyzer::VisitExpr;
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt;
  using IRMutatorWithAnalyzer::VisitStmt_;

  explicit TextureAllocInjector(arith::Analyzer* ana) : IRMutatorWithAnalyzer(ana) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    std::string storage_scope = GetStorageScope(op->buffer_var);
    if (IsTextureStorage(storage_scope)) {
      op = stmt.as<AllocateNode>();
      ICHECK(op->extents.size() >= 3) << "Only 2D Array RGBA texture is currently supported";
      const int data_bits = op->dtype.bits(),
                vec_length = static_cast<int>(op->extents.back().as<IntImmNode>()->value);
      const int channel_size = data_bits * vec_length;
      ICHECK(channel_size == 128 || channel_size == 64)
          << "Invalid Channel Size: " << channel_size << " bits";

      size_t axis = DefaultTextureLayoutSeparator(op->extents.size(), storage_scope);
      auto texture = ApplyTexture2DFlattening<PrimExpr>(op->extents, op->extents.size(), axis);
      ffi::Array<PrimExpr> args;
      args.push_back(StringImm(storage_scope));
      args.push_back(IntImm(DataType::Int(64), 3));  // 2d Array
      args.push_back(Call(DataType::Handle(), builtin::tvm_stack_make_shape(),
                          {texture.width, texture.height, texture.depth}));
      args.push_back(IntImm(DataType::Int(64), channel_size));
      stmt =
          LetStmt(op->buffer_var,
                  Call(op->buffer_var.dtype(), builtin::nd_mem_alloc_with_scope(), args), op->body);
    }
    return stmt;
  }

 protected:
  std::string GetStorageScope(const Var& buffer_var) {
    auto* ptr = buffer_var->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    return ptr->storage_scope;
  }
};

namespace transform {

Pass InjectTextureAlloc() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureAllocInjector::Inject(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.backend.adreno.InjectTextureAlloc", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.backend.adreno.transform.InjectTextureAlloc", InjectTextureAlloc);
}

}  // namespace transform

}  // namespace adreno
}  // namespace backend
}  // namespace tir
}  // namespace tvm
