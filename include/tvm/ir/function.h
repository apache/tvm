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
 * \file tvm/ir/function.h
 * \brief Function nodes.
 */
#ifndef TVM_IR_FUNCTION_H_
#define TVM_IR_FUNCTION_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>

#include <string>
#include <type_traits>

namespace tvm {

/*!
 * \brief Possible Calling conventions.
 *
 *  NOTE: The calling convention also implies
 *  the way we implement the function during lowering.
 */
enum class CallingConv : int {
  /*!
   * \brief Default calling convention.
   *
   * - Uses the native calling convention of the target.
   * - Implementation: specified by the native target.
   */
  kDefault = 0,
  /*!
   * \brief PackedFunc that exposes a CPackedFunc signature.
   *
   * - Calling by PackedFunc calling convention.
   * - Implementation: Expose a function with the CPackedFunc signature.
   */
  kCPackedFunc = 1,
  /*!
   * \brief Device kernel launch
   *
   * - Call by PackedFunc calling convention.
   * - Implementation: defined by device runtime(e.g. runtime/cuda)
   */
  kDeviceKernelLaunch = 2,
};

/*!
 * \brief Base node of all functions.
 *
 * We support several variants of functions throughout the stack.
 * All of the functions share the same type system(via checked_type)
 * to support cross variant calls.
 *
 * \sa BaseFunc
 */
class BaseFuncNode : public RelayExprNode {
 public:
  /*! \brief Additional attributes storing the meta-data */
  DictAttrs attrs;

  static constexpr const char* _type_key = "BaseFunc";
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseFuncNode, RelayExprNode);
};

/*!
 * \brief Managed reference to BaseFuncNode.
 * \sa BaseFuncNode
 */
class BaseFunc : public RelayExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseFunc, RelayExpr, BaseFuncNode);
};

/*!
 * \brief Generic attribute names that can be attached to any function.
 *
 * \sa tvm::tir::attr, tvm::relay::attr
 */
namespace attr {
/*!
 * \brief Indicates the special calling convention.
 *
 * Type: Integer
 *
 * \sa tvm::CallingConv
 */
constexpr const char* kCallingConv = "calling_conv";

/*!
 * \brief Compilation target of the function.
 *
 * Type: Target
 *
 * \sa tvm::Target
 */
constexpr const char* kTarget = "target";

/*!
 * \brief Global linker symbol of the function in generated code.
 *
 *  This option forces the code generator to name the
 *  function with the given.
 *
 *  For example, we could set a global_symbol of a function
 *  early to make sure that we can always refer to it by
 *  the symbol name in the generated DLL.
 *
 *  We should not set the attribute for local functions,
 *  so that the compiler can freely rename them.
 *
 *  A unique global symbol will be automatically assigned
 *  to each function in the module before the target code
 *  generation phase.
 *
 * Type: String
 */
constexpr const char* kGlobalSymbol = "global_symbol";
}  // namespace attr
}  // namespace tvm
#endif  // TVM_IR_FUNCTION_H_
