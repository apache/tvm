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

#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/backend/adreno/transform.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>

#include "../../../backend/opencl/runtime/texture.h"
#include "../../../s_tir/ir/ir_mutator_with_analyzer.h"
#include "../../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
namespace backend {
namespace adreno {
using namespace tvm::tirx;
using runtime::ApplyTexture2DFlattening;
using runtime::DefaultTextureLayoutSeparator;
using runtime::IsTextureStorage;

/*!
 * \brief Inject Texture Alloc Intrinsic right after buffer allocations are realized.
 */
class TextureAllocInjector : public s_tir::IRMutatorWithAnalyzer {
 public:
  using s_tir::IRMutatorWithAnalyzer::Mutate;
  using s_tir::IRMutatorWithAnalyzer::Mutate_;

  static PrimFunc Inject(PrimFunc func) {
    sym::Analyzer ana;
    auto pass = ffi::make_object<TextureAllocInjector>(ana);
    auto writer = func.CopyOnWrite();
    pass->MarkBufferParamShapes(func);
    writer->body = pass->Mutate(func->body).ValueOrUnchanged(func->body);
    return func;
  }

  explicit TextureAllocInjector(const sym::Analyzer& ana) : IRMutatorWithAnalyzer(ana) {}

 private:
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::builtin::alloc_tensor())) {
      return Mutate_AllocTensor(op, call, inplace_mode);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_AllocTensor(const BindNode* op, const CallNode* call,
                                       InplaceMode inplace_mode) {
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    ffi::String scope = call->args[2].as_or_throw<StringImm>()->value;
    if (IsTextureStorage(scope)) {
      op = stmt.as<BindNode>();
      if (const auto* call = op ? op->value.as<CallNode>() : nullptr;
          !call || !call->op.same_as(tirx::builtin::alloc_tensor())) {
        TVM_FFI_THROW(InternalError) << "Expected an allocation binding after buffer mutation";
      }
      const auto* allocation = op->value.as<CallNode>();
      tvm::Tuple shape = allocation->args[0].as_or_throw<tvm::Tuple>();
      DLDataType dtype = allocation->args[1].as_or_throw<DataTypeImm>()->value;
      ffi::Array<PrimExpr> extents =
          shape->fields.Map([](const Expr& e) { return e.as_or_throw<PrimExpr>(); });
      TVM_FFI_ICHECK(extents.size() >= 3) << "Only 2D Array RGBA texture is currently supported";
      const int data_bits = dtype.bits,
                vec_length = extents.back().as<IntImmNode>()->value.as<int>().value();
      const int channel_size = data_bits * vec_length;
      TVM_FFI_ICHECK(channel_size == 128 || channel_size == 64)
          << "Invalid Channel Size: " << channel_size << " bits";

      size_t axis = DefaultTextureLayoutSeparator(extents.size(), scope);
      auto texture = ApplyTexture2DFlattening<PrimExpr>(extents, extents.size(), axis);
      ffi::Array<Expr> args;
      args.push_back(StringImm(scope));
      args.push_back(IntImm::Int64(3));
      args.push_back(Call(PointerType(PrimType::Int(64)), tirx::builtin::tvm_stack_make_shape(),
                          {texture.width, texture.height, texture.depth}));
      args.push_back(IntImm::Int64(channel_size));
      stmt = Bind(op->var.as_or_throw<BufferVar>(),
                  Call(op->var.as_or_throw<BufferVar>().type(), tirx::builtin::decl_tensor(),
                       {Call(op->var.as_or_throw<BufferVar>().DataPointerType(),
                             tirx::builtin::nd_mem_alloc_with_scope(), args),
                        tvm::Tuple(op->var.as_or_throw<BufferVar>()->shape),
                        DataTypeImm(op->var.as_or_throw<BufferVar>()->dtype->dtype),
                        StringImm(op->var.as_or_throw<BufferVar>().scope())},
                       {}, allocation->ty_args, allocation->span),
                  op->span);
    }
    return stmt;
  }

 protected:
  std::string GetStorageScope(const Var& buffer_var) {
    auto* ptr = buffer_var->ty.as<PointerTypeNode>();
    TVM_FFI_ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    return ptr->storage_scope;
  }
};

namespace transform {

Pass InjectTextureAlloc() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureAllocInjector::Inject(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "s_tir.backend.adreno.InjectTextureAlloc", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.backend.adreno.transform.InjectTextureAlloc", InjectTextureAlloc);
}

}  // namespace transform

}  // namespace adreno
}  // namespace backend
}  // namespace s_tir
}  // namespace tvm
