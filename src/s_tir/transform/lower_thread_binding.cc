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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/builtin.h>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

class ThreadBindingLowerer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

 private:
  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    // Attribute metadata can refer to an enclosing loop's lexical binding.
    auto node = Mutate(op->node, inplace_mode);
    auto value = Mutate(op->value, inplace_mode);
    auto body = Mutate(op->body, inplace_mode);
    if (node.UnchangedOrSameAs(op->node) && value.UnchangedOrSameAs(op->value) &&
        body.UnchangedOrSameAs(op->body)) {
      return ffi::Unchanged();
    }
    return AttrStmt(std::move(node).ValueOrUnchanged(op->node), op->attr_key,
                    std::move(value).ValueOrUnchanged(op->value),
                    std::move(body).ValueOrUnchanged(op->body), op->span);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) final {
    auto annotations = Mutate(op->annotations, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>();
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (annotations.UnchangedOrSameAs(op->annotations)) return result;
    SBlock block = std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<SBlock>();
    block.CopyOnWrite()->annotations = std::move(annotations).ValueUnchecked();
    return block;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    if (op->kind != ForKind::kThreadBinding) {
      auto annotations = Mutate(op->annotations, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>();
      auto result = StmtExprMutator::Mutate_(op, inplace_mode);
      if (annotations.UnchangedOrSameAs(op->annotations)) return result;
      For loop = std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<For>();
      loop.CopyOnWrite()->annotations = std::move(annotations).ValueUnchecked();
      return loop;
    }
    PrimExpr min = Mutate(op->min, inplace_mode).ValueOrUnchanged(op->min);
    PrimExpr extent = Mutate(op->extent, inplace_mode).ValueOrUnchanged(op->extent);
    TVM_FFI_ICHECK(is_zero(min)) << "Thread binding loops must start at zero";
    TVM_FFI_ICHECK(op->thread_binding.has_value());
    TVM_FFI_ICHECK(!op->annotations.count("loop_partition_hint") ||
                   op->annotations.at("loop_partition_hint") == nullptr)
        << "Run LoopPartition before LowerThreadBinding";
    PrimVar launch_var(op->loop_var->name, extent.ty());
    auto previous_remap = VarRemapGet(op->loop_var);
    VarRemapSet(op->loop_var, prim::cast(op->loop_var.ty(), launch_var));
    auto annotations = Mutate(op->annotations, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>()
                           .ValueOrUnchanged(op->annotations);
    Stmt body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    VarRemapSet(op->loop_var, previous_remap);
    if (!annotations.empty()) {
      PrimType ty = op->loop_var.ty();
      body = For(PrimVar("annotation", ty), IntImm(ty, 0), IntImm(ty, 1), ForKind::kSerial,
                 std::move(body), std::nullopt, std::move(annotations), std::nullopt);
    }
    return RegionStmt(tirx::builtin::launch_thread(),
                      {StringImm(op->thread_binding.value()->thread_tag), extent}, {launch_var},
                      DictAttrs(), std::move(body), {}, op->span);
  }
};

namespace transform {

Pass LowerThreadBinding() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto lower = ffi::make_object<ThreadBindingLowerer>();
    auto fptr = f.CopyOnWrite();
    fptr->body = lower->Mutate(fptr->body.value(), InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body).value());
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.LowerThreadBinding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerThreadBinding", LowerThreadBinding);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
