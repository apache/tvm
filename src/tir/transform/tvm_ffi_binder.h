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
 * \file tvm_ffi_binder.h
 * \brief Helper utility to match and bind arguments.
 */
#ifndef TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
#define TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Helper utility to generate match and bind of arguments.
 *
 * \note There is many places in TVM IR where we need argument bindings.
 *
 *  Consider a function f(tA(shape=var(n)), tB(shape=3), tC(shape=(n+2)).
 *  Here n is a undefined variable that is decided by the outside, tB imposes
 *  a constraint such that it can only take tensor with shape 3, tC imposes
 *  another constraint that it's shape must equals n + 2.
 *  So if we call it with f(bufferA, bufferB, bufferC), we need to generate
 *  the following binding sequence:
 *  - define n = bufferA.shape[0]
 *  - assert bufferB.shape[0] == 3
 *  - assert bufferB.shape[1] == n + 3
 *
 *  In general, this is a constraint solving problem. We have simplified assumption
 *  over the binding declaration, such that we require the variable occurred in
 *  constraint must be declared in argument list. So it is illegal to have signature
 *  f(tA(shape=(n+3))) without any argument variable corresponds to n, even though
 *  it is already enough to derive n from the input argument.
 */
class ArgBinder {
 public:
  /*!
   * \brief Constructor
   * \param def_map A definition map that contains definition of known variables.
   *   ArgBinder will update this def_map when adding new definitions.
   */
  explicit ArgBinder(std::unordered_map<const VarNode*, PrimExpr>* def_map) : def_map_(def_map) {}
  /*!
   * \brief Constructor with function signature info for rich error messages.
   *
   * When provided, assertion failures will include the function signature
   * in the error message to help users identify which function failed.
   * The signature is formatted as:
   *   func_name(param0: Tensor([dim0, dim1], dtype), param1: int32)
   *
   * Also records a mapping from param_index to param_name so that
   * AccessPath rendering replaces array indices with human-readable names.
   *
   * \param def_map A definition map that contains definition of known variables.
   * \param func_name The function name.
   * \param params The function parameters.
   * \param buffer_map The buffer map from parameters to buffers.
   */
  ArgBinder(std::unordered_map<const VarNode*, PrimExpr>* def_map, const std::string& func_name,
            const ffi::Array<Var>& params, const ffi::Map<Var, Buffer>& buffer_map);
  /*!
   * \brief Try to bind arg to value, generate constraint if necessary.
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   * \param with_let Whether add lets during bind
   */
  void Bind(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
            bool with_let = false);
  /*!
   * \brief Bind array to array
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   */
  void BindArray(const ffi::Array<PrimExpr>& arg, const ffi::Array<PrimExpr>& value,
                 const std::string& arg_name);
  /*!
   * \brief Bind symbolic buffer to another symbolic buffer
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   * \param fuzzy_match If enabled, we allow value's dimension to be smaller than arg, as long as
   * arg's higher dimensions are of 1.
   */
  void BindBuffer(const Buffer& arg, const Buffer& value, const std::string& arg_name,
                  bool fuzzy_match);
  /*!
   * \brief Bind symbolic buffer to a DLTensor handle.
   *
   * Creates bindings for the buffer's shape, strides, dtype, byte offset,
   * device, and data pointer from the DLTensor handle. When function signature
   * is available, generates rich error messages using AccessPath to describe
   * the location of mismatches (e.g. "a.shape[0]").
   *
   * \param buffer The argument buffer to be binded.
   * \param device_type The device id to be binded.
   * \param device_id The device id to be binded.
   * \param handle The DLTensor handle.
   * \param arg_name argument name.
   * \param base_path The AccessPath for this parameter (e.g. Root()->ArrayItem(i)->Attr(name)).
   */
  void BindDLTensor(const Buffer& buffer, const PrimExpr& device_type, const PrimExpr& device_id,
                    const Var& handle, const std::string& arg_name,
                    ffi::reflection::AccessPath base_path);

  /*! \return The defs generated in binding. */
  const std::vector<Var>& defs() const { return defs_; }

  /*! \return The asserts generated in binding
   *
   * This contains statements that assert the correct value has been
   * bound.  For example, `binder.Bind(var, expr_1)` will produce an
   * entry mapping `var` to `expr_1` in the `binder.defs()`.  If
   * `binder.Bind(var, expr_2)` is called later, then this will
   * produce an assert statemtn that `expr_1 == expr_2`.
   *
   * Note: Some assert statements produced by BindDLTensor are located
   * in `binder.init_nest()`, not within `binder.asserts()`.  This is
   * deliberate, as some values may require checks prior to
   * initialization.  (e.g. Intializing `m = dl_tensor->shape[3]`
   * requires first asserting that `3 < dl_tensor->ndim`.)
   */
  const std::vector<Stmt>& asserts() const { return asserts_; }

  /*!
   * \brief Initialization nest generated
   *
   * This contains both variable bindings and any assert statements
   * that are required in order to safely produce those variable
   * bindings.
   *
   * \note Variable bindings may be implemented either as a `LetStmt`
   *     that defines the variable, or as a variable replacement.  Any
   *     bindings implemented as a `LetStmt` will be in the
   *     initialization list.  Any bindings implemented as a variable
   *     replacement will be stored in the `var_def` map.
   *
   *     A `tir::LetStmt` is usually generated when binding to a
   *     `DLTensor`.  This requires loading values from memory, which
   *     should only be performed once.  If the binding to a
   *     `DLTensor` were implemented as a variable replacement, it
   *     would load values from memory once for each usage of the
   *     variable.
   *
   * \return The initialization nest generated during binding.
   */
  const std::vector<Stmt>& init_nest() const { return init_nest_; }
  /*! \return Handle data type of the data */
  const ffi::Map<Var, PrimExpr>& def_handle_dtype() const { return def_handle_dtype_; }

 private:
  /*!
   * \brief Internal bind function.
   *
   * Binds \p arg to \p value. On first bind for a Var, records the definition.
   * On duplicate bind, emits an assertion that the previous value equals
   * the new value, with rich error messages when AccessPaths are available.
   *
   * \param arg The argument expression to bind.
   * \param value The target value expression.
   * \param arg_name Textual argument name for legacy error messages.
   * \param with_lets Whether to generate LetStmt bindings.
   * \param path AccessPath of this bind site (used for rich error tracking).
   * \return True if this was the first bind (definition created), false otherwise.
   */
  bool Bind_(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
             bool with_lets, ffi::reflection::AccessPath path);

  /*!
   * \brief Render an AccessPath as a human-readable string.
   *
   * Uses parameter names from the function signature when available.
   * For example, an AccessPath representing params[0].shape[0] where
   * param 0 is named "a" will render as "a.shape[0]".
   *
   * \param path The AccessPath to render.
   * \return A human-readable string representation.
   */
  std::string RenderAccessPath(const ffi::reflection::AccessPath& path) const;

  /*! \brief The definition map, can be uses to substitute */
  std::unordered_map<const VarNode*, PrimExpr>* def_map_;
  /*! \brief defs generated in the current binder */
  std::vector<Var> defs_;
  /*! \brief Initialize nest */
  std::vector<Stmt> init_nest_;
  /*! \brief handle data type in the defintiions */
  ffi::Map<Var, PrimExpr> def_handle_dtype_;
  /*! \brief asserts generated */
  std::vector<Stmt> asserts_;
  /*! \brief internal analyzer. */
  arith::Analyzer analyzer_;
  /*! \brief function name for error messages. */
  std::string func_name_;
  /*! \brief function signature string for error messages. */
  std::string func_signature_;
  /*! \brief Map from param_index to param_name for AccessPath rendering. */
  std::unordered_map<int, std::string> param_names_;
  /*! \brief Track first-bind AccessPath for each variable, used for cross-reference messages. */
  std::unordered_map<const VarNode*, ffi::reflection::AccessPath> first_bind_path_;
};
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
