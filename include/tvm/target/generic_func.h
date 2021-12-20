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
 * \file tvm/target/generic_func.h
 * \brief Generic function that can be specialzied on a per target basis.
 */
#ifndef TVM_TARGET_GENERIC_FUNC_H_
#define TVM_TARGET_GENERIC_FUNC_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {

class GenericFuncNode;

/*!
 * \brief Generic function that can be specialized on a per-target basis.
 */
class GenericFunc : public ObjectRef {
 public:
  GenericFunc() {}
  explicit GenericFunc(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*!
   * \brief Set the default function implementaiton.
   * \param value The default function
   * \param allow_override If true, this call may override a previously registered function. If
   * false, an error will be logged if the call would override a previously registered function.
   * \return reference to self.
   */
  TVM_DLL GenericFunc& set_default(const runtime::PackedFunc value, bool allow_override = false);
  /*!
   * \brief Register a specialized function
   * \param tags The tags for this specialization
   * \param value The specialized function
   * \param allow_override If true, this call may override previously registered tags. If false,
   * an error will be logged if the call would override previously registered tags.
   * \return reference to self.
   */
  TVM_DLL GenericFunc& register_func(const std::vector<std::string>& tags,
                                     const runtime::PackedFunc value, bool allow_override = false);
  /*!
   * \brief Call generic function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call generic function
   *   void CallGeneric(GenericFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template <typename... Args>
  inline runtime::TVMRetValue operator()(Args&&... args) const;
  /*!
   * \brief Invoke the relevant function for the current target context, set by set_target_context.
   * Arguments are passed in packed format.
   * \param args The arguments to pass to the function.
   * \param ret The return value
   */
  TVM_DLL void CallPacked(runtime::TVMArgs args, runtime::TVMRetValue* ret) const;
  /*!
   * \brief Get the packed function specified for the current target context.
   */
  TVM_DLL PackedFunc GetPacked() const;
  /*!
   * \brief Find or register the GenericFunc instance corresponding to the give name
   * \param name The name of the registered GenericFunc
   * \return The GenericFunc instance
   */
  TVM_DLL static GenericFunc Get(const std::string& name);

  /*!
   * \brief Add a GenericFunc instance to the registry
   * \param func The GenericFunc instance
   * \param name The name of the registered GenericFunc
   */
  TVM_DLL static void RegisterGenericFunc(GenericFunc func, const std::string& name);

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline GenericFuncNode* operator->();

  // declare container type
  using ContainerType = GenericFuncNode;

  // Internal class.
  struct Manager;

 private:
  friend struct Manager;
};

template <typename... Args>
inline runtime::TVMRetValue GenericFunc::operator()(Args&&... args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  runtime::detail::for_each(runtime::TVMArgsSetter(values, type_codes),
                            std::forward<Args>(args)...);
  runtime::TVMRetValue rv;
  CallPacked(runtime::TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

/*!
 * \brief Represents a generic function that can be specialized on a per-target basis.
 */
class GenericFuncNode : public Object {
 public:
  /*! \brief name of the function */
  std::string name_;
  /* \brief the generic builder */
  runtime::PackedFunc generic_func_;
  /* \brief map from keys to registered functions */
  std::unordered_map<std::string, runtime::PackedFunc> dispatch_dict_;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "GenericFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(GenericFuncNode, Object);
};

inline GenericFuncNode* GenericFunc::operator->() {
  return static_cast<GenericFuncNode*>(get_mutable());
}

#define TVM_GENERIC_FUNC_REG_VAR_DEF static TVM_ATTRIBUTE_UNUSED ::tvm::GenericFunc& __mk_##TVM

/*!
 * \def TVM_REGISTER_GENERIC_FUNC
 * \brief Register a new generic function, or set a device-specific variant
 * of the corresponding function.
 *
 * \param name The name of the function
 */
#define TVM_REGISTER_GENERIC_FUNC(name) \
  TVM_STR_CONCAT(TVM_GENERIC_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::GenericFunc::Get(#name)

}  // namespace tvm
#endif  // TVM_TARGET_GENERIC_FUNC_H_
