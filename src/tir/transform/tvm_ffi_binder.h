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
 * ## Calling Protocol
 *
 * 1. Construct with function metadata (def_map, func_name, params, buffer_map, v_packed_args)
 * 2. Call BindPackedArg(i) for each parameter i in [0, num_args)
 *    - Returns (arg_value, param) pairs for LetStmt bindings
 * 3. Call BindAllParams(var_defs, device_type, device_id, &arg_buffer_declarations)
 *    - Binds scalar params to loaded values and DLTensor buffers to handles
 * 4. Build the function body using:
 *    - init_nest(): variable bindings and pre-initialization assertions
 *    - asserts(): post-initialization assertions (shape, dtype, alignment checks)
 *    - arg_buffer_declarations: DeclBuffer statements for buffer arguments
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
   * 2. Type-check and load value via per-dtype methods
   * 3. For handles: tensor-offset logic (DLTensor header offset)
   * 4. Return (arg_value, param) pair for the caller to use
   *
   * \param i Parameter index in [0, num_args).
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
   * Note: Some assert statements produced by BindDLTensor are located
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
  // ── Per-dtype type-check + value-load methods ──────────────────
  //
  // Each method combines type checking (emitting TypeError on mismatch)
  // with value loading from the packed args array. BindPackedArg
  // dispatches to one of these based on the parameter dtype.

  /*!
   * \brief Type-check and load a handle argument (DLTensor or opaque pointer).
   *
   * Example error:
   *   TypeError: Mismatched type on argument #0 when calling:
   *     `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
   *     expected Tensor
   *
   * \param i Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \return The loaded argument value.
   */
  PrimExpr BindPackedHandle(int i, const Var& type_index);

  /*!
   * \brief Type-check and load a boolean argument.
   * Accepts: kTVMFFIBool, kTVMFFIInt (int->bool coercion).
   *
   * Example error:
   *   TypeError: Mismatched type on argument #0 when calling:
   *     `func(flag: bool)`, expected boolean
   *
   * \param i Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \return The loaded argument value.
   */
  PrimExpr BindPackedBool(int i, const Var& type_index);

  /*!
   * \brief Type-check and load an integer argument.
   * Accepts: kTVMFFIInt, kTVMFFIBool (bool->int coercion).
   *
   * Example error:
   *   TypeError: Mismatched type on argument #0 when calling:
   *     `func(n: int32)`, expected int
   *
   * \param i Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \param dtype The expected data type for this parameter.
   * \return The loaded argument value.
   */
  PrimExpr BindPackedInt(int i, const Var& type_index, DataType dtype);

  /*!
   * \brief Type-check and load a float argument.
   * Accepts: kTVMFFIFloat, kTVMFFIInt, kTVMFFIBool (int/bool->float promotion).
   *
   * Example error:
   *   TypeError: Mismatched type on argument #0 when calling:
   *     `func(x: float32)`, expected float
   *
   * \param i Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \param dtype The expected data type for this parameter.
   * \return The loaded argument value.
   */
  PrimExpr BindPackedFloat(int i, const Var& type_index, DataType dtype);

  // ── Private binding submethods (all take AccessPath) ───────────

  /*!
   * \brief Internal scalar bind with AccessPath tracking and rich error messages.
   *
   * Binds \p arg to \p value. If arg is a Var not yet in def_map_, creates a
   * new definition; otherwise emits a rich assertion that the existing value
   * matches the new one.
   *
   * \param arg The argument expression to bind (typically a Var or constant).
   * \param value The value expression to bind to the argument.
   * \param arg_name Human-readable name for error messages.
   * \param with_lets If true, emit LetStmt bindings into init_nest_.
   * \param path AccessPath for rich error message rendering.
   * \return True if this was the first bind (definition created), false otherwise.
   */
  bool BindScalar(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                  bool with_lets, ffi::reflection::AccessPath path);

  /*!
   * \brief Array bind: binds element-wise with AccessPath[k] for each element.
   *
   * \param arg The expected array of expressions.
   * \param value The actual array of expressions to bind against.
   * \param arg_name Human-readable base name for error messages.
   * \param base_path Base AccessPath; each element appends ArrayItem(k).
   */
  void BindArray(const ffi::Array<PrimExpr>& arg, const ffi::Array<PrimExpr>& value,
                 const std::string& arg_name, ffi::reflection::AccessPath base_path);

  /*!
   * \brief Buffer-to-buffer bind with AccessPath.
   *
   * Binds data, elem_offset, shape, and strides of \p arg against \p value,
   * emitting assertions for any mismatches.
   *
   * \param arg The expected buffer definition.
   * \param value The actual buffer to bind against.
   * \param arg_name Human-readable name for error messages.
   * \param base_path Base AccessPath for the buffer parameter.
   * \param fuzzy_match If true, allow value to have more dimensions than arg.
   */
  void BindBuffer(const Buffer& arg, const Buffer& value, const std::string& arg_name,
                  ffi::reflection::AccessPath base_path, bool fuzzy_match);

  /*!
   * \brief DLTensor bind: ndim/dtype/shape/strides/data/device assertions.
   *
   * Generates all the checks and bindings for a DLTensor packed argument,
   * including null-pointer check, ndim, dtype, shape elements, strides,
   * byte offset, alignment, device, and data pointer.
   *
   * \param buffer The buffer definition to bind against.
   * \param device_type The expected device type expression.
   * \param device_id The expected device id expression.
   * \param handle The variable holding the DLTensor handle.
   * \param arg_name Human-readable name for error messages.
   * \param base_path Base AccessPath for the tensor parameter.
   */
  void BindDLTensor(const Buffer& buffer, const PrimExpr& device_type, const PrimExpr& device_id,
                    const Var& handle, const std::string& arg_name,
                    ffi::reflection::AccessPath base_path);

  // ── DLTensor sub-helpers ───────────────────────────────────────

  /*!
   * \brief Extract a DLTensor array field and declare a buffer for element access.
   *
   * Creates a buffer declaration for the array, extracts the pointer
   * from the DLTensor handle, and appends the LetStmt + DeclBuffer to init_nest_.
   *
   * \param handle The DLTensor handle variable.
   * \param field_kind kArrShape or kArrStrides.
   * \param field_name "shape" or "strides" (used for buffer naming).
   * \param num_elements Number of elements in the array.
   * \param arg_name Human-readable base name for the buffer.
   * \return The declared buffer for element access.
   */
  Buffer ExtractDLTensorArray(const Var& handle, int field_kind, const std::string& field_name,
                              int num_elements, const std::string& arg_name);

  /*!
   * \brief Assert strides form a compact (C-contiguous) layout.
   * Skipped if strides pointer is NULL.
   *
   * Example error:
   *   ValueError: Mismatched a.strides on argument #0 when calling:
   *     `add_one(a: Tensor([4, 8], float32), ...)`,
   *     expected to be compact array
   *
   * \param buffer The expected buffer definition.
   * \param buf_strides The strides buffer extracted from the DLTensor.
   * \param v_strides_is_null Expression checking if strides pointer is NULL.
   * \param param_path AccessPath for the tensor parameter.
   */
  void BindCompactStrides(const Buffer& buffer, const Buffer& buf_strides,
                          PrimExpr v_strides_is_null,
                          const ffi::reflection::AccessPath& param_path);

  /*!
   * \brief Bind strides for auto-broadcast buffers: stride=0 for shape==1 dims.
   *
   * \param buffer The expected buffer definition.
   * \param buf_strides The strides buffer extracted from the DLTensor.
   * \param v_strides_is_null Expression checking if strides pointer is NULL.
   * \param param_path AccessPath for the tensor parameter.
   */
  void BindAutoBroadcastStrides(const Buffer& buffer, const Buffer& buf_strides,
                                PrimExpr v_strides_is_null,
                                const ffi::reflection::AccessPath& param_path);

  /*!
   * \brief Bind strides with C-contiguous fallback when strides pointer is NULL.
   *
   * \param buffer The expected buffer definition.
   * \param buf_strides The strides buffer extracted from the DLTensor.
   * \param buf_shape The shape buffer (for computing C-contiguous strides).
   * \param v_strides_is_null Expression checking if strides pointer is NULL.
   * \param param_path AccessPath for the tensor parameter.
   */
  void BindRegularStrides(const Buffer& buffer, const Buffer& buf_strides, const Buffer& buf_shape,
                          PrimExpr v_strides_is_null,
                          const ffi::reflection::AccessPath& param_path);

  // ── Error message helpers ──────────────────────────────────────

  /*!
   * \brief Build a rich AssertStmt with kind, detail, signature, expectation.
   *
   * Produces the standard format:
   *   <detail> when calling:
   *     `<func_signature>`,
   *     expected <expectation>
   *
   * Uses cached StringImm values for the signature and "when calling:" prefix
   * to enable string sharing across assertions.
   *
   * \param kind The error kind string (e.g. "TypeError", "ValueError").
   * \param cond The boolean condition; assertion fails when cond is false.
   * \param detail Human-readable detail string (unique per assertion).
   * \param expectation Human-readable expectation string (e.g. "Tensor", "128").
   * \param target Output vector to append the AssertStmt to.
   */
  void EmitRichAssert(const std::string& kind, PrimExpr cond, const std::string& detail,
                      const std::string& expectation, std::vector<Stmt>* target);

  /*!
   * \brief Render an AccessPath as a human-readable string (e.g. "a.shape[0]").
   *
   * \param path The AccessPath to render.
   * \return A human-readable string representation of the path.
   */
  std::string RenderAccessPath(const ffi::reflection::AccessPath& path) const;

  /*!
   * \brief Extract param_index from the root ArrayItem step of a path.
   *
   * \param path The AccessPath to extract the index from.
   * \return The param index, or -1 if not found.
   */
  int GetParamIndex(const ffi::reflection::AccessPath& path) const;

  // ── Data members ───────────────────────────────────────────────
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

  // Cached StringImm for string sharing across assertions
  /*! \brief Cached StringImm of func_signature_. */
  StringImm sig_imm_;
  /*! \brief Cached StringImm of " when calling:\n  `" (prefix before signature). */
  StringImm when_calling_imm_;

  // AccessPath tracking
  /*! \brief Track first-bind AccessPath for each variable, used for cross-reference messages. */
  std::unordered_map<const VarNode*, ffi::reflection::AccessPath> first_bind_path_;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
