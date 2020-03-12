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

#include <tvm/ir/expr.h>
#include <tvm/ir/attrs.h>
#include <type_traits>
#include <string>


namespace tvm {

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
   *    Integer value = f->GetAttr<Integer>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template<typename TObjectRef>
  TObjectRef GetAttr(const std::string& attr_key,
                     TObjectRef default_value = NullValue<TObjectRef>()) const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Can only call GetAttr with ObjectRef types.");
    if (!attrs.defined()) return default_value;
    auto it = attrs->dict.find(attr_key);
    if (it != attrs->dict.end()) {
      return Downcast<TObjectRef>((*it).second);
    } else {
      return default_value;
    }
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
  bool HasNonzeroAttr(const std::string& attr_key) const {
    return GetAttr<Integer>(attr_key, 0)->value != 0;
  }

  static constexpr const char* _type_key = "BaseFunc";
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
