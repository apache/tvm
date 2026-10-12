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
 * \file ir_utils.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_TIR_TRANSFORM_IR_UTILS_H_
#define TVM_TIR_TRANSFORM_IR_UTILS_H_

#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/ir/with_context.h>
#include <tvm/runtime/device_api.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Get the type that is passed around TVM ffi::Function API.
 * \param t The original type.
 * \return The corresponding API type.
 */
inline PrimType APIType(const PrimType& t) {
  TVM_FFI_ICHECK(!t.IsVoid()) << "Cannot pass void type through packed API.";
  TVM_FFI_ICHECK_EQ(t.lanes(), 1) << "Cannot pass vector type through packed API.";
  if (t.MatchesCode(DLDataTypeCode::kDLBool, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt)) {
    return PrimType::Int(64);
  }
  TVM_FFI_ICHECK(t.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat));
  return PrimType::Float(64);
}

/*!
 * \brief Rule to get allocation alignment requirement for a given const array.
 * \param type The type of allocation.
 * \param const_size The constant size of the array.
 * \return the alignment
 */
inline int GetTempAllocaAlignment(const PrimType& type, int64_t const_size) {
  int align = runtime::kTempAllocaAlignment;
  if (const_size > 0) {
    int64_t element_bytes = type.StorageBytes();
    // Only compute the total size when it can reduce the alignment. This also avoids
    // overflowing for very large allocations.
    if (element_bytes > 0 && const_size <= (align - 1) / element_bytes) {
      int64_t const_s = const_size * element_bytes;
      while (align > const_s) {
        align = align / 2;
      }
    }
  }
  return align;
}

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
Stmt ConvertSSA(Stmt stmt);

/*! \brief Shared SSA renaming algorithm; dialects extend statement dispatch explicitly. */
class IRConvertSSA : public StmtExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(IRConvertSSA, StmtExprMutator)
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  Function VisitFunction(Function func);
  IRModule VisitIRModule(IRModule mod);

 protected:
  explicit IRConvertSSA(const VTable* table) : StmtExprMutator(table) {}
  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final;
  Stmt WithScope(const std::function<Stmt()>& body);
  Var DefineVar(Var var);
  Var GetRemappedVar(Var var);
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) final;
  static Var MakeNewVar(const Var& old_var);
  void PushVarRemap(const Var& old_var, const Var& new_var);
  void PopAllRemapsInCurrentScope();

 private:
  struct VarRemap {
    Var old_var;
    Var new_var;
  };
  /*! \brief Scope stack: each scope level holds the remaps introduced in that scope.
   *
   * When a body-carrying statement (For, Allocate, or a dialect statement) calls
   * scope_.WithNewScope([&]{...}), a new scope level is pushed.
   * Bind statements push their remaps to the current scope.
   * On scope exit, the destructor of std::vector<VarRemap> triggers,
   * and we undo all remaps in that level.
   *
   * Note: ScopeStack<T>::WithNewScope calls T's destructor on exit.
   * std::vector's destructor destroys elements but does NOT call custom
   * cleanup.  So we wrap the vector in ScopeLevel which handles cleanup.
   */
  struct ScopeLevel {
    std::vector<VarRemap> remaps;
    IRConvertSSA* parent{nullptr};

    void push_back(VarRemap remap) { remaps.push_back(std::move(remap)); }
    size_t size() const { return remaps.size(); }
    VarRemap& back() { return remaps.back(); }
    void pop_back() { remaps.pop_back(); }

    void Clear() {
      if (!parent) return;
      while (!remaps.empty()) {
        parent->scoped_var_remap_[remaps.back().old_var.get()].pop_back();
        remaps.pop_back();
      }
    }
    ~ScopeLevel() { Clear(); }
    ScopeLevel() = default;
    ScopeLevel(const ScopeLevel&) = delete;
    ScopeLevel& operator=(const ScopeLevel&) = delete;
    ScopeLevel(ScopeLevel&& other) noexcept
        : remaps(std::move(other.remaps)), parent(other.parent) {
      other.parent = nullptr;
    }
    ScopeLevel& operator=(ScopeLevel&& other) noexcept {
      if (this != &other) {
        Clear();
        remaps = std::move(other.remaps);
        parent = other.parent;
        other.parent = nullptr;
      }
      return *this;
    }
  };

  std::unordered_map<const VarNode*, std::vector<Var>> scoped_var_remap_;
  std::unordered_set<const VarNode*> defined_;
  ScopeStack<ScopeLevel> scope_;
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_IR_UTILS_H_
