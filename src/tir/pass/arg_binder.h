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
 * \file arg_binder.h
 * \brief Helper utility to match and bind arguments.
 */
#ifndef TVM_TIR_PASS_ARG_BINDER_H_
#define TVM_TIR_PASS_ARG_BINDER_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/buffer.h>
#include <tvm/arith/analyzer.h>

#include <string>
#include <vector>
#include <unordered_map>

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
  explicit ArgBinder(
      std::unordered_map<const VarNode*, PrimExpr>* def_map)
      : def_map_(def_map) {
  }
  /*!
   * \brief Try to bind arg to value, generate constraint if necessary.
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   * \param with_let Whether add lets during bind
   */
  void Bind(const PrimExpr& arg,
            const PrimExpr& value,
            const std::string& arg_name,
            bool with_let = false);
  /*!
   * \brief Bind array to array
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   */
  void BindArray(const Array<PrimExpr>& arg,
                 const Array<PrimExpr>& value,
                 const std::string& arg_name);
  /*!
   * \brief Bind symbolic buffer to another symbolic buffer
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   * \param fuzzy_match If enabled, we allow value's dimension to be smaller than arg, as long as arg's higher dimensions are of 1.
   */
  void BindBuffer(const Buffer& arg,
                  const Buffer& value,
                  const std::string& arg_name,
                  bool fuzzy_match);
  /*!
   * \brief Bind symbolic buffer to a DLTensor handle.
   * \param buffer The argument buffer to be binded.
   * \param device_type The device id to be binded.
   * \param device_id The device id to be binded.
   * \param handle The DLTensor handle.
   * \param arg_name argument name.
   */
  void BindDLTensor(const Buffer& buffer,
                    const PrimExpr& device_type,
                    const PrimExpr& device_id,
                    const Var& handle,
                    const std::string& arg_name);

  /*! \return The defs generated in binding. */
  const std::vector<Var>& defs() const {
    return defs_;
  }
  /*! \return The asserts generated in binding */
  const std::vector<Stmt>& asserts() const {
    return asserts_;
  }
  /*!
   * \brief Initialization nest generated
   *  This is only non-empty when BindDLTensor is called.
   *
   * \note The binder may choose to generate a let statement
   *  and simply put def_map to map Variable to itself,
   *  or update def_map to directly map to new value and not generate let statement.
   *
   *  Let statement is usually generated when bind to DLTensor and memory load is involved.
   * \return The initialization nest generated during binding.
   */
  const std::vector<Stmt>& init_nest() const {
    return init_nest_;
  }
  /*! \return Handle data type of the data */
  const Map<Var, PrimExpr>& def_handle_dtype() const {
    return def_handle_dtype_;
  }

 private:
  // Internal bind function
  bool Bind_(const PrimExpr& arg,
             const PrimExpr& value,
             const std::string& arg_name,
             bool with_lets);
  /*! \brief The definition map, can be uses to substitute */
  std::unordered_map<const VarNode*, PrimExpr>* def_map_;
  /*! \brief defs generated in the current binder */
  std::vector<Var> defs_;
  /*! \brief Initialize nest */
  std::vector<Stmt> init_nest_;
  /*! \brief handle data type in the defintiions */
  Map<Var, PrimExpr> def_handle_dtype_;
  /*! \brief asserts generated */
  std::vector<Stmt> asserts_;
  /*! \brief internal analyzer. */
  arith::Analyzer analyzer_;
};
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_PASS_ARG_BINDER_H_
