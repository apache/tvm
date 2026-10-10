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
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Fill empty loop/conditional bodies and prepend statement prefixes.
 * \param nest Loops, conditionals with empty bodies, or Bind/Assert/SeqStmt prefixes.
 * \param body body
 * \return The combined Stmt
 */
Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body);

/*!
 * \brief Fill empty loop/conditional bodies and prepend statement prefixes.
 * \param nest Groups of loops, conditionals with empty bodies, or statement prefixes.
 * \param body body
 * \return The combined Stmt
 */
Stmt MergeNest(const std::vector<std::vector<Stmt>>& nest, Stmt body);

/*!
 * \brief update array with an unary function
 * \param arr array
 * \param fupdate an unary function
 * \tparam T type of array element
 * \tparam F type of the unary function
 * \return if update happens, return the new array, else return the
 *  original array
 */
template <typename T, typename F>
inline ffi::Array<T> UpdateArray(ffi::Array<T> arr, F fupdate) {
  std::vector<T> new_arr(arr.size());
  bool changed = false;
  for (size_t i = 0; i < arr.size(); ++i) {
    T old_elem = arr[i];
    T new_elem = fupdate(old_elem);
    if (!new_elem.same_as(old_elem)) changed = true;
    new_arr[i] = new_elem;
  }
  if (!changed) {
    return arr;
  } else {
    return ffi::Array<T>(new_arr);
  }
}

/*!
 * \brief Get construct from struct
 * \param dtype The data type.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \return the get expression.
 */
inline Expr TVMStructGet(Type type, Var handle, int index, tirx::TVMStructFieldKind kind) {
  ffi::Array<Expr> args = {handle, IntImm::Int32(index), IntImm::Int32(static_cast<int>(kind))};
  return Call(std::move(type), tirx::abi_field_get_op(), args);
}

inline PrimExpr TVMStructGet(PrimType type, Var handle, int index, tirx::TVMStructFieldKind kind) {
  return TVMStructGet(Type(type), std::move(handle), index, kind).as_or_throw<PrimExpr>();
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline Call AddressOffset(Var handle, PrimType dtype, int offset) {
  PrimExpr offset_expr = IntImm::Int32(offset * dtype.lanes());
  ffi::Array<PrimExpr> shape = {offset_expr + 1};
  auto pointer_type = handle->ty.as_or_throw<PointerType>();
  TensorVar dummy_buf(handle->name,
                      TensorType(pointer_type->storage_scope, dtype, shape, {}, 0, 0, 0));
  TensorLoad buf_load = MakeTensorLoad(dummy_buf, {offset_expr});

  return Call(handle->ty, tirx::address_of_op(), {buf_load});
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline Call AddressOffset(Var handle, PrimType dtype, PrimExpr offset) {
  if (dtype.lanes() != 1) {
    PrimType offset_ty = offset.ty();
    offset = offset * IntImm(offset_ty, dtype.lanes());
    offset = prim::Ramp(offset, IntImm(offset_ty, 1), dtype.lanes());
  }

  ffi::Array<PrimExpr> shape = {offset + 1};
  auto pointer_type = handle->ty.as_or_throw<PointerType>();
  TensorVar dummy_buf(handle->name, TensorType(pointer_type->storage_scope, dtype.WithLanes(1),
                                               shape, {}, 0, 0, 0));
  TensorLoad buf_load = MakeTensorLoad(dummy_buf, {offset});

  return Call(handle->ty, tirx::address_of_op(), {buf_load});
}

/*!
 * \brief Set value into struct.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \param value The value to be set.
 * \return the set stmt.
 */
inline Stmt TVMStructSet(Var handle, int index, tirx::TVMStructFieldKind kind, Expr value) {
  ffi::Array<Expr> args = {handle, IntImm::Int32(index), IntImm::Int32(static_cast<int>(kind)),
                           value};
  return Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(), args).as_or_throw<PrimExpr>());
}

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
  TVM_FFI_ICHECK_EQ(t.code(), DLDataTypeCode::kDLFloat);
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
 * \brief Create an int32 constant
 * \param index the value of the constant
 * \return the PrimExpr that represents the constant
 */
inline PrimExpr ConstInt32(size_t index) {
  TVM_FFI_ICHECK_LE(index, std::numeric_limits<int>::max());
  return IntImm::Int32(static_cast<int>(index));
}

/*!
 * \brief Allocate TVMValues on the stack
 * \param ret_type exact pointer type returned by the allocation
 * \param type type of allocation
 * \param num number of TVMValues to allocate
 * \return Call representing the allocated pointer
 */
inline Call StackAlloca(Type ret_type, std::string type, size_t num) {
  ffi::Array<Expr> args = {StringImm(type), ConstInt32(num)};
  return Call(std::move(ret_type), tirx::stack_alloca_op(), args);
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

/*!
 * \brief Return the storage scope associated with a buffer variable.
 * \param buffer_var The input buffer variable.
 * \return A string representing the storage scope of this buffer variable.
 */
ffi::String GetPtrStorageScope(Var buffer_var);

/*!
 * \brief Get stride aware buffer allocation shape from buffer.
 * \param buffer The buffer object.
 * \return shape The shape considering buffer strides.
 */
ffi::Array<PrimExpr> GetBufferAllocationShape(const TensorVar& buffer);

/*!
 * \brief Split string separated by "," to get wmma fragment dimension size.
 * \param  shape_str The string to split.
 * \param  scope The scope to match.
 * \return The result pair of fragment dimension size.
 */
std::pair<int32_t, int32_t> GetWmmaFragmentDimSize(const std::string& shape_str,
                                                   const std::string& scope);

/*! \brief Check if a Function is a host function
 *
 * \param func The function to be inspected
 *
 * \return True if the function is known to run on the host, false if
 * the function is known to run on the device.  If it cannot be
 * determined (e.g. a function without a tvm::attr::kTarget
 * attribute), returns std::nullopt.
 */
std::optional<bool> IsHostFunc(const Function& func);

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_IR_UTILS_H_
