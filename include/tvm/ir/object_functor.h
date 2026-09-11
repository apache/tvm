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
 * \file tvm/ir/object_functor.h
 * \brief Defines the Functor data structures.
 */
#ifndef TVM_IR_OBJECT_FUNCTOR_H_
#define TVM_IR_OBJECT_FUNCTOR_H_

#include <tvm/ffi/error.h>

#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief A dynamically dispatched functor on the type of the first argument.
 *
 * Dispatch is based on the runtime type of the first object argument.
 *
 * \code
 *   ObjectFunctor<std::string(const ffi::ObjectRef& n, std::string prefix)> tostr;
 *   tostr.SetDispatch<prim::AddNode>([](const ffi::ObjectRef& op, std::string prefix) {
 *     return prefix + "Add";
 *   });
 *   tostr.SetDispatch<IntImmNode>([](const ffi::ObjectRef& op, std::string prefix) {
 *     return prefix + "IntImm";
 *   });
 *
 *   tirx::PrimVar x("x");
 *   PrimExpr y = x + 1;
 *   // dispatch to IntImm, outputs "MyIntImm"
 *   LOG(INFO) << tostr(IntImm::Int32(1), "My");
 *   // dispatch to Add, outputs "MyAdd"
 *   LOG(INFO) << tostr(y, "My");
 * \endcode
 *
 * A shared table can register dispatch functions during static initialization:
 *
 * \code
 *   class Printer {
 *    public:
 *     using FType = ObjectFunctor<std::string(const ffi::ObjectRef&)>;
 *     static FType& vtable();
 *   };
 *
 *   Printer::FType& Printer::vtable() {
 *     static FType inst;
 *     return inst;
 *   }
 *
 *   TVM_FFI_STATIC_INIT_BLOCK() {
 *     Printer::vtable()
 *         .SetDispatch<prim::AddNode>([](const ffi::ObjectRef&) { return std::string("Add"); })
 *         .SetDispatch<IntImmNode>([](const ffi::ObjectRef&) { return std::string("IntImm"); });
 *   }
 * \endcode
 *
 * \tparam FType Function signature, with const ffi::ObjectRef& as its first argument.
 */
template <typename FType>
class ObjectFunctor;

template <typename R, typename... Args>
class ObjectFunctor<R(const ffi::ObjectRef& n, Args...)> {
 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*!
   * \brief Whether a dispatch function is registered for the exact runtime type.
   * \param n The object to be dispatched.
   * \return Whether a dispatch function is registered for n's type, excluding ancestors.
   */
  bool CanDispatch(const ffi::ObjectRef& n) const {
    uint32_t type_index = n->type_index();
    if (type_index < begin_type_index_) return false;
    type_index -= begin_type_index_;
    return type_index < func_.size() && func_[type_index] != nullptr;
  }
  /*!
   * \brief invoke the functor, dispatch on type of n
   * \param n The object argument
   * \param args The additional arguments
   * \return The result.
   */
  R operator()(const ffi::ObjectRef& n, Args... args) const {
    uint32_t type_index = n->type_index();
    if (type_index >= begin_type_index_) {
      uint32_t index = type_index - begin_type_index_;
      if (index < func_.size() && func_[index] != nullptr) {
        return (*func_[index])(n, std::forward<Args>(args)...);
      }
    }

    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    for (int32_t i = type_info->type_depth - 1; i >= 0; --i) {
      type_index = type_info->type_ancestors[i]->type_index;
      if (type_index >= begin_type_index_) {
        uint32_t index = type_index - begin_type_index_;
        if (index < func_.size() && func_[index] != nullptr) {
          return (*func_[index])(n, std::forward<Args>(args)...);
        }
      }
    }
    TVM_FFI_THROW(InternalError) << "ObjectFunctor calls un-registered function on type "
                                 << n->GetTypeKey();
    throw;
  }
  /*!
   * \brief set the dispatcher for type TNode
   * \param f The function to be set.
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename TNode>
  ObjectFunctor& SetDispatch(R (*f)(const ffi::ObjectRef& n, Args...)) {
    uint32_t tindex = TNode::RuntimeTypeIndex();
    if (func_.size() <= tindex) {
      func_.resize(tindex + 1, nullptr);
    }
    TVM_FFI_ICHECK(func_[tindex] == nullptr)
        << "Dispatch for " << TNode::_type_key << " is already set";
    TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << " Cannot call SetDispatch after calling Finalize";
    func_[tindex] = f;
    return *this;
  }
  /*!
   * \brief unset the dispatcher for type TNode
   *
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename TNode>
  ObjectFunctor& ClearDispatch() {
    uint32_t tindex = TNode::RuntimeTypeIndex();
    TVM_FFI_ICHECK_LT(tindex, func_.size()) << "ClearDispatch: index out of range";
    TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << " Cannot call ClearDispatch after calling Finalize";
    func_[tindex] = nullptr;
    return *this;
  }
  /*!
   * \brief Finalize the functor after calling sequence of SetDispatch
   * This function will attempt to find the min type index that is not null
   * and optimize the space of the func table so it is more compact
   */
  void Finalize() {
    TVM_FFI_ICHECK_EQ(begin_type_index_, 0) << "Can only call Finalize once";
    while (begin_type_index_ < func_.size() && func_[begin_type_index_] == nullptr) {
      ++begin_type_index_;
    }
    // shift up the function value
    size_t new_ftable_size = func_.size() - begin_type_index_;
    if (begin_type_index_ != 0) {
      std::memmove(func_.data(), func_.data() + begin_type_index_,
                   new_ftable_size * sizeof(FPointer));
    }
    func_.resize(new_ftable_size);
    func_.shrink_to_fit();
  }

 private:
  /*! \brief internal function pointer type */
  using FPointer = R (*)(const ffi::ObjectRef& n, Args...);
  /*! \brief internal function table */
  std::vector<FPointer> func_;
  /*! \brief start range of func index */
  uint32_t begin_type_index_{0};
};

}  // namespace tvm
#endif  // TVM_IR_OBJECT_FUNCTOR_H_
