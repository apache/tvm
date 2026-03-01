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
 * TVMFFIABIBuilder owns ALL error message generation and type checking for
 * packed function arguments. The caller (make_packed_api.cc) creates
 * an TVMFFIABIBuilder and calls DecodeParam(i) for each parameter.
 */
#ifndef TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
#define TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Helper utility to generate match and bind of packed function arguments.
 *
 * TVMFFIABIBuilder consolidates all binding logic, type checking, and error message
 * generation for packed function parameters. The primary public method is
 * DecodeAllParams(), which handles everything: type index extraction,
 * type checking (TypeError), value loading, scalar binding, buffer
 * binding, and rich error message generation with AccessPath.
 *
 * ## Calling Protocol
 *
 * 1. Construct with function metadata (func_name, params, buffer_map, v_packed_args,
 *    v_num_packed_args). The constructor emits arg count and null-pointer checks.
 * 2. Call DecodeAllParams(device_type, device_id)
 *    - Decodes, type-checks, and binds all packed arguments
 *    - Binds DLTensor buffers (shape, strides, dtype, device checks)
 * 3. Build the function body using init_nest() which contains all generated
 *    statements: variable bindings, assertions, and DeclBuffer statements.
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
class TVMFFIABIBuilder {
 public:
  /*!
   * \brief Constructor with function signature info for rich error messages.
   *
   * Builds a human-readable function signature string, records parameter
   * name mappings, and emits argument count and null-pointer checks into init_nest_.
   *
   * \param func_name The function name.
   * \param params The function parameters.
   * \param buffer_map The buffer map from parameters to buffers.
   * \param v_packed_args The packed args variable (used for struct_get calls).
   * \param v_num_packed_args The variable holding the actual number of packed args.
   * \param device_type The expected device type expression.
   * \param device_id The device id variable (may be defined during buffer binding).
   */
  TVMFFIABIBuilder(const ffi::String& func_name, const ffi::Array<Var>& params,
                   const ffi::Map<Var, Buffer>& buffer_map, const Var& v_packed_args,
                   const Var& v_num_packed_args, const PrimExpr& device_type,
                   const PrimExpr& device_id);

  /*!
   * \brief Decode all packed arguments: type-check, load values, bind buffers.
   *
   * This is the primary public method after construction. It:
   * 1. Calls DecodeParam(i) for each parameter to type-check, load values,
   *    and bind scalar params
   * 2. Binds DLTensor buffers to handles (shape, strides, dtype, device checks)
   *
   * All generated statements are appended to init_nest.
   */
  void DecodeAllParams();

  /*!
   * \brief Finalize and consume the binder, returning generated statements and definitions.
   *
   * Moves init_nest (all generated statements: variable bindings, assertions,
   * DeclBuffers) and def_map (Var -> bound value mappings) out of the binder.
   * The binder should not be used after this call.
   *
   * \return A pair of (def_map, init_nest).
   */
  std::pair<ffi::Map<Var, PrimExpr>, std::vector<Stmt>> Finalize() {
    return {std::move(def_map_), std::move(init_nest_)};
  }

 private:
  // ── Assert helpers ────────────────────────────────────────────

  /*! \brief Convert various string types to StringImm for EmitAssert. */
  static StringImm ToMsgPart(StringImm s) { return s; }
  static StringImm ToMsgPart(const char* s) { return StringImm(s); }
  static StringImm ToMsgPart(std::string s) { return StringImm(std::move(s)); }
  static StringImm ToMsgPart(ffi::String s) { return StringImm(std::move(s)); }

  /*!
   * \brief Emit an assertion into init_nest_ with auto-converted message parts.
   *
   * Each variadic argument is converted to StringImm automatically.
   * Accepts StringImm, const char*, std::string, or ffi::String.
   */
  template <typename... Args>
  void EmitAssert(const PrimExpr& cond, const char* error_kind, Args&&... args) {
    ffi::Array<StringImm> parts;
    (parts.push_back(ToMsgPart(std::forward<Args>(args))), ...);
    init_nest_.emplace_back(AssertStmt(cond, StringImm(error_kind), parts));
  }

  // ── Binding submethods ─────────────────────────────────────────

  /*! \brief Type-check, load value, and bind one packed param. */
  void DecodeParam(int param_index);

  /*! \brief Load the i-th packed argument as the given type from the union value. */
  static PrimExpr LoadTVMFFIAnyUnionValue(const Var& v_packed_args, int param_index,
                                          DataType arg_type);

  // ── Per-dtype type-check + value-load methods ──────────────────
  //
  // Each method combines type checking (emitting TypeError on mismatch)
  // with value loading from the packed args array. DecodeParam
  // dispatches to one of these based on the parameter dtype.

  /*!
   * \brief Type-check and load a handle argument (DLTensor or opaque pointer).
   * \param param_index Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \return The loaded argument value.
   */
  PrimExpr DecodeParamOpaqueHandle(int param_index, const Var& type_index);

  /*!
   * \brief Type-check and load a boolean argument.
   * \param param_index Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \return The loaded argument value.
   */
  PrimExpr DecodeParamBool(int param_index, const Var& type_index);

  /*!
   * \brief Type-check and load an integer argument.
   * \param param_index Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \param dtype The expected data type for this parameter.
   * \return The loaded argument value.
   */
  PrimExpr DecodeParamInt(int param_index, const Var& type_index, DataType dtype);

  /*!
   * \brief Type-check and load a float argument.
   * \param param_index Parameter index.
   * \param type_index The variable holding the FFI type index.
   * \param dtype The expected data type for this parameter.
   * \return The loaded argument value.
   */
  PrimExpr DecodeParamFloat(int param_index, const Var& type_index, DataType dtype);

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
   * \param with_lets If true, emit LetStmt bindings into init_nest_.
   * \param path AccessPath for rich error message rendering.
   * \return True if this was the first bind (definition created), false otherwise.
   */
  bool BindScalar(const PrimExpr& arg, const PrimExpr& value,
                  const ffi::reflection::AccessPath& path, bool with_lets);

  /*!
   * \brief Array bind: binds element-wise with AccessPath[k] for each element.
   *
   * \param arg The expected array of expressions.
   * \param value The actual array of expressions to bind against.
   * \param base_path Base AccessPath; each element appends ArrayItem(k).
   */
  void BindArray(const ffi::Array<PrimExpr>& arg, const ffi::Array<PrimExpr>& value,
                 const ffi::reflection::AccessPath& base_path);

  /*!
   * \brief Buffer-to-buffer bind with AccessPath.
   *
   * Binds data, elem_offset, shape, and strides of \p arg against \p value,
   * emitting assertions for any mismatches.
   *
   * \param arg The expected buffer definition.
   * \param value The actual buffer to bind against.
   * \param base_path Base AccessPath for the buffer parameter.
   * \param fuzzy_match If true, allow value to have more dimensions than arg.
   */
  void BindBuffer(const Buffer& arg, const Buffer& value, ffi::reflection::AccessPath base_path,
                  bool fuzzy_match);

  /*!
   * \brief DLTensor bind: ndim/dtype/shape/strides/data/device assertions.
   *
   * \param buffer The buffer definition to bind against.
   * \param device_type The expected device type expression.
   * \param device_id The expected device id expression.
   * \param handle The variable holding the DLTensor handle.
   * \param arg_name Human-readable name for error messages.
   * \param base_path Base AccessPath for the tensor parameter.
   */
  void DecodeParamDLTensor(const Buffer& buffer, const PrimExpr& device_type,
                           const PrimExpr& device_id, const Var& handle,
                           const std::string& arg_name, ffi::reflection::AccessPath base_path);

  // ── DLTensor sub-helpers ───────────────────────────────────────

  /*!
   * \brief Extract shape or strides array from a DLTensor and declare a buffer for element access.
   *
   * \param handle The DLTensor handle variable.
   * \param field_kind kArrShape or kArrStrides.
   * \param field_name "shape" or "strides" (used for buffer naming).
   * \param num_elements Number of elements in the array.
   * \param arg_name Human-readable base name for the buffer.
   * \return The declared buffer for element access.
   */
  Buffer DLTensorExtractShapeOrStrides(const Var& handle, int field_kind,
                                       const std::string& field_name, int num_elements,
                                       const std::string& arg_name);

  /*!
   * \brief Assert strides form a compact (C-contiguous) layout.
   *
   * \param buffer The expected buffer definition.
   * \param buf_strides The strides buffer extracted from the DLTensor.
   * \param v_strides_is_null Expression checking if strides pointer is NULL.
   * \param param_path AccessPath for the tensor parameter.
   */
  void BindCompactStrides(const Buffer& buffer, const Buffer& buf_strides,
                          const PrimExpr& v_strides_is_null,
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
                                const PrimExpr& v_strides_is_null,
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
                          const PrimExpr& v_strides_is_null,
                          const ffi::reflection::AccessPath& param_path);

  // ── Error message helpers ──────────────────────────────────────

  /*!
   * \brief Emit a type-mismatch TypeError for a given parameter.
   *
   * Emits an assertion with kind "TypeError" and message parts:
   * "Mismatched type on argument #N when calling:\n  `<sig>`,\n  expected <type>"
   *
   * \param param_index The parameter index.
   * \param cond The boolean condition; assertion fails when cond is false.
   * \param expected_type Human-readable expected type (e.g. "Tensor", "int").
   */
  void EmitTypeIndexCheck(int param_index, const PrimExpr& cond, const std::string& expected_type);

  /*!
   * \brief Render an AccessPath as a human-readable string (e.g. "a.shape[0]").
   * \param path The AccessPath to render.
   * \return A human-readable string representation of the path.
   */
  ffi::String RenderAccessPath(const ffi::reflection::AccessPath& path) const;

  /*!
   * \brief Extract param_index from the root ArrayItem step of a path.
   * \param path The AccessPath to extract the index from.
   * \return The param index, or -1 if not found.
   */
  int GetParamIndex(const ffi::reflection::AccessPath& path) const;

  // ── Data members ───────────────────────────────────────────────
  /*! \brief The definition map: Var -> its bound value. Uses Var keys to avoid dangling pointers.
   */
  ffi::Map<Var, PrimExpr> def_map_;
  /*! \brief Track first-bind AccessPath for each variable, used for cross-reference messages. */
  ffi::Map<Var, ffi::reflection::AccessPath> first_bind_path_;
  /*! \brief All generated statements: bindings, assertions, DeclBuffers. */
  std::vector<Stmt> init_nest_;
  /*! \brief internal analyzer. */
  arith::Analyzer analyzer_;

  // Function metadata
  /*! \brief function name for error messages. */
  ffi::String func_name_;
  /*! \brief function signature string for error messages. */
  std::string func_signature_;
  /*! \brief The function parameters. */
  ffi::Array<Var> params_;
  /*! \brief The buffer map from parameters to buffers. */
  ffi::Map<Var, Buffer> buffer_map_;
  /*! \brief The packed args variable. */
  Var v_packed_args_;
  /*! \brief The expected device type expression. */
  PrimExpr device_type_;
  /*! \brief The device id variable. */
  PrimExpr device_id_;
  /*! \brief Map from param_index to param_name for AccessPath rendering. */
  std::unordered_map<int, std::string> param_names_;

  // Pre-cached common message fragments for string sharing across assertions
  StringImm sig_imm_;  // func_signature_ (set in constructor)
  StringImm when_calling_imm_ = StringImm(" when calling:\n  `");
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_TVM_FFI_BINDER_H_
