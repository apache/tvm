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
 * \file remap_thread_axis.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

// Mutator to change the read pattern
class ThreadAxisRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit ThreadAxisRewriter(const std::unordered_map<std::string, ffi::String>& tmap)
      : tmap_(tmap) {}

  Stmt Rewrite(Stmt stmt) { return Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt); }

 private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) != 0) {
      auto it = tmap_.find(op->args[0].as_or_throw<StringImm>()->value);
      if (it != tmap_.end()) {
        PrimExpr old_extent = op->args[1].as_or_throw<PrimExpr>();
        PrimExpr extent = Mutate(old_extent, inplace_mode).ValueOrUnchanged(old_extent);
        PrimVar old_var = op->body_params[0].as_or_throw<PrimVar>();
        PrimVar new_var(it->second, extent.ty());
        ffi::Any previous_remap = VarRemapGet(old_var);
        VarRemapSet(old_var, new_var);
        Stmt body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
        VarRemapSet(old_var, previous_remap);
        return RegionStmt(tirx::launch_thread_op(), {StringImm(it->second), extent}, {new_var},
                          DictAttrs(), body, {}, op->span);
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  // The thread map
  const std::unordered_map<std::string, ffi::String>& tmap_;
};

PrimFunc RemapThreadAxis(PrimFunc func, ffi::Map<ffi::String, ffi::String> thread_map) {
  std::unordered_map<std::string, ffi::String> tmap;
  for (const auto& kv : thread_map) {
    tmap[kv.first] = kv.second;
  }

  if (auto opt = func->GetAttr<ffi::Array<ffi::String>>(tirx::attr::kKernelLaunchParams)) {
    TVM_FFI_ICHECK(opt != nullptr) << "Require attribute " << tirx::attr::kKernelLaunchParams;
    auto launch_params = opt.value();
    // replace the thread axis attribute
    for (size_t i = 0; i < launch_params.size(); ++i) {
      auto it = tmap.find(launch_params[i]);
      if (it != tmap.end()) {
        launch_params.Set(i, it->second);
      }
    }

    func = WithAttr(std::move(func), tirx::attr::kKernelLaunchParams, launch_params);
  }

  if (!func->body.has_value()) return func;
  auto* n = func.CopyOnWrite();
  n->body = ffi::make_object<ThreadAxisRewriter>(tmap)->Rewrite(std::move(n->body).value());
  return func;
}

namespace transform {

Pass RemapThreadAxis(ffi::Map<ffi::String, ffi::String> thread_map) {
  auto pass_func = [thread_map](PrimFunc f, IRModule m, PassContext ctx) {
    return RemapThreadAxis(std::move(f), thread_map);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.RemapThreadAxis", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.RemapThreadAxis", RemapThreadAxis);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
