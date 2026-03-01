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
 * \brief Helper utility to match and bind packed function arguments.
 *
 * ArgBinder owns ALL error message generation and type checking for
 * packed function arguments. The caller (make_packed_api.cc) creates
 * an ArgBinder and calls BindPackedArg(i) for each parameter.
 */
#ifndef TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
#define TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Helper utility to generate match and bind of packed function arguments.
 *
 * ArgBinder consolidates all binding logic, type checking, and error message
 * generation for packed function parameters. The primary public method is
 * BindPackedArg(i), which handles everything for one packed parameter:
 * type index extraction, type checking (TypeError), value loading, buffer
 * binding (if applicable), and rich error message generation with AccessPath.
 *
 * \note Consider a function f(tA(shape=var(n)), tB(shape=3), tC(shape=(n+2)).
 *  Here n is an undefined variable decided by the outside, tB imposes
 *  a constraint such that it can only take tensor with shape 3, tC imposes
 *  another constraint that its shape must equal n + 2.
 *  So if we call it with f(bufferA, bufferB, bufferC), we need to generate
 *  the following binding sequence:
 *  - define n = bufferA.shape[0]
 *  - assert bufferB.shape[0] == 3
 *  - assert bufferB.shape[1] == n + 3
 */
class ArgBinder {
 public:
  /*!
   * \brief Constructor with function signature info for rich error messages.
   *
   * Builds a human-readable function signature string and records parameter
   * name mappings. All assertions will include the function signature.
   *
   * \param def_map A definition map that contains definition of known variables.
   *   ArgBinder will update this def_map when adding new definitions.
   * \param func_name The function name.
   * \param params The function parameters.
   * \param buffer_map The buffer map from parameters to buffers.
   * \param v_packed_args The packed args variable (used for struct_get calls).
   */
  ArgBinder(std::unordered_map<const VarNode*, PrimExpr>* def_map, const std::string& func_name,
            const ffi::Array<Var>& params, const ffi::Map<Var, Buffer>& buffer_map,
            const Var& v_packed_args);

  /*!
   * \brief Bind a packed argument: type-check, load value, bind buffer if applicable.
   *
   * Handles everything for one packed parameter:
   * 1. Extract type_index from v_packed_args
   * 2. Type-check via BindTypeCheck_ (emits TypeError assertions)
   * 3. Load arg value based on dtype (handle/bool/int/float)
   * 4. For handles: tensor-offset logic (DLTensor header offset)
   * 5. Return (arg_value, param) pair for the caller to use
   *
   * \param i Parameter index.
   * \return (arg_value, param) pair. The caller uses these for LetStmt bindings.
   */
  std::pair<PrimExpr, Var> BindPackedArg(int i);

  /*!
   * \brief Bind all scalar params and buffer params after BindPackedArg.
   *
   * This must be called after all BindPackedArg calls. It binds scalar params
   * to their loaded values and DLTensor buffers to their handles.
   *
   * \param var_defs The (arg_value, param) pairs returned by BindPackedArg.
   * \param device_type The device type expression.
   * \param device_id The device id expression.
   * \param arg_buffer_declarations Output: DeclBuffer stmts for buffer args.
   */
  void BindAllParams(const std::vector<std::pair<PrimExpr, Var>>& var_defs,
                     const PrimExpr& device_type, const PrimExpr& device_id,
                     std::vector<Stmt>* arg_buffer_declarations);

  /*! \return The function signature string. */
  const std::string& func_signature() const { return func_signature_; }

  /*! \return The defs generated in binding. */
  const std::vector<Var>& defs() const { return defs_; }

  /*! \return The asserts generated in binding.
   *
   * This contains statements that assert the correct value has been
   * bound.  For example, binding a variable twice will produce an
   * assert statement that the first value equals the second.
   *
   * Note: Some assert statements produced by BindDLTensor_ are located
   * in `init_nest()`, not within `asserts()`. This is deliberate, as
   * some values may require checks prior to initialization.
   */
  const std::vector<Stmt>& asserts() const { return asserts_; }

  /*!
   * \brief Initialization nest generated.
   *
   * This contains both variable bindings and any assert statements
   * that are required in order to safely produce those variable bindings.
   *
   * \return The initialization nest generated during binding.
   */
  const std::vector<Stmt>& init_nest() const { return init_nest_; }

  /*! \return Handle data type of the data */
  const ffi::Map<Var, PrimExpr>& def_handle_dtype() const { return def_handle_dtype_; }

 private:
  // -- Private binding submethods (all take AccessPath) --

  /*!
   * \brief Internal scalar bind with AccessPath tracking and rich error messages.
   * \return True if this was the first bind (definition created), false otherwise.
   */
  bool Bind_(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
             bool with_lets, ffi::reflection::AccessPath path);

  /*!
   * \brief Array bind: binds element-wise with AccessPath[k] for each element.
   */
  void BindArray_(const ffi::Array<PrimExpr>& arg, const ffi::Array<PrimExpr>& value,
                  const std::string& arg_name, ffi::reflection::AccessPath base_path);

  /*!
   * \brief Buffer-to-buffer bind with AccessPath.
   */
  void BindBuffer_(const Buffer& arg, const Buffer& value, const std::string& arg_name,
                   ffi::reflection::AccessPath base_path, bool fuzzy_match);

  /*!
   * \brief DLTensor bind: ndim/dtype/shape/strides/data/device assertions.
   */
  void BindDLTensor_(const Buffer& buffer, const PrimExpr& device_type, const PrimExpr& device_id,
                     const Var& handle, const std::string& arg_name,
                     ffi::reflection::AccessPath base_path);

  /*!
   * \brief Type-check a packed arg's FFI type code, emit TypeError on mismatch.
   */
  void BindTypeCheck_(int i, const Var& type_index, DataType dtype);

  // -- Error message helpers --

  /*!
   * \brief Build a rich AssertStmt with kind, detail, signature, expectation.
   *
   * Produces the standard format:
   *   <detail> when calling:
   *     `<func_signature>`,
   *     expected <expectation>
   */
  void EmitRichAssert_(const std::string& kind, PrimExpr cond, const std::string& detail,
                       const std::string& expectation, std::vector<Stmt>* target);

  /*!
   * \brief Render an AccessPath as a human-readable string (e.g. "a.shape[0]").
   */
  std::string RenderAccessPath_(const ffi::reflection::AccessPath& path) const;

  /*!
   * \brief Extract param_index from the root ArrayItem step of a path.
   * \return The param index, or -1 if not found.
   */
  int GetParamIndex_(const ffi::reflection::AccessPath& path) const;

  // -- Data members --
  /*! \brief The definition map, can be used to substitute */
  std::unordered_map<const VarNode*, PrimExpr>* def_map_;
  /*! \brief defs generated in the current binder */
  std::vector<Var> defs_;
  /*! \brief Initialize nest */
  std::vector<Stmt> init_nest_;
  /*! \brief handle data type in the definitions */
  ffi::Map<Var, PrimExpr> def_handle_dtype_;
  /*! \brief asserts generated */
  std::vector<Stmt> asserts_;
  /*! \brief internal analyzer. */
  arith::Analyzer analyzer_;

  // Function metadata
  /*! \brief function name for error messages. */
  std::string func_name_;
  /*! \brief function signature string for error messages. */
  std::string func_signature_;
  /*! \brief The function parameters. */
  ffi::Array<Var> params_;
  /*! \brief The buffer map from parameters to buffers. */
  ffi::Map<Var, Buffer> buffer_map_;
  /*! \brief The packed args variable. */
  Var v_packed_args_;
  /*! \brief Map from param_index to param_name for AccessPath rendering. */
  std::unordered_map<int, std::string> param_names_;

  // AccessPath tracking
  /*! \brief Track first-bind AccessPath for each variable, used for cross-reference messages. */
  std::unordered_map<const VarNode*, ffi::reflection::AccessPath> first_bind_path_;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
