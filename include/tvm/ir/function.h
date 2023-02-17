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
 * \brief Supported linkage types.
 */
enum class LinkageType : int {
  /*!
   * \brief Internal linkage.
   */
  kInternal = 0,
  /*!
   * \brief External linkage.
   - Function with external linkage should have a global symbol attached to it.
   */
  kExternal = 1
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

  /*!
   * \brief Get a function attribute.
   *
   * \param attr_key The attribute key.
   * \param default_value The default value if the key does not exist, defaults to nullptr.
   *
   * \return The result
   *
   * \tparam TOBjectRef the expected object type.
   * \throw Error if the key exists but the value does not match TObjectRef
   *
   * \code
   *
   *  void GetAttrExample(const BaseFunc& f) {
   *    auto value = f->GetAttr<Integer>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    return attrs.GetAttr(attr_key, default_value);
  }
  // variant that uses TObjectRef to enable implicit conversion to default value.
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
  }

  /*!
   * \brief Check whether the function has an non-zero integer attr.
   *
   * This function can be used to check whether an optional
   * attribute mark(e.g. inline) exists.
   *
   * \param attr_key The key to the attribute.
   * \return The check result.
   *
   * \code
   *
   *  void HasNonzeroAttrExample(const BaseFunc& f) {
   *    if (f->HasNonzeroAttr(attr::kInline)) {
   *      // inline the function.
   *    }
   *  }
   *
   * \endcode
   */
  bool HasNonzeroAttr(const std::string& attr_key) const { return attrs.HasNonzeroAttr(attr_key); }
  /*!
   * \brief Get the type of the linkage.
   *
   * Currently, we only consider external/internal linkage.
   * This can be extended in the future when necessary.
   *
   * \return Linkage type.
   *
   * \code
   *
   *  void Example(const BaseFunc& f) {
   *    if (f->GetLinkageType() == tvm::LinkageType::kExternal) {
   *      // Do not remove a function with external linkage
   *    }
   *  }
   *
   * \endcode
   */

  LinkageType GetLinkageType() const {
    if (GetAttr<String>(attr::kGlobalSymbol))
      return LinkageType::kExternal;
    else
      return LinkageType::kInternal;
  }

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

}  // namespace tvm
#endif  // TVM_IR_FUNCTION_H_
