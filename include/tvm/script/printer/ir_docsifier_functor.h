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
#ifndef TVM_SCRIPT_PRINTER_IR_DOCSIFIER_FUNCTOR_H_
#define TVM_SCRIPT_PRINTER_IR_DOCSIFIER_FUNCTOR_H_

#include <tvm/node/node.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>

#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

/*!
 * \brief Dynamic dispatch functor based on ObjectPath.
 *
 * This functor dispatches based on the type of object and the input dispatch token.
 */
template <typename R, typename... Args>
class IRDocsifierFunctor {
 private:
  using TSelf = IRDocsifierFunctor<R, Args...>;

  template <class TObjectRef, class TCallable>
  using IsDispatchFunction =
      typename std::is_convertible<TCallable, std::function<R(TObjectRef, Args...)>>;

 public:
  /*!
   * \brief Call the dispatch function.
   * \param token The dispatch token.
   * \param obj The object.
   * \param args Other args.
   *
   * \return The return value of the dispatch function
   *
   * If the TObjectRef isn't registered with the token, it will try to find
   * dispatch function for TObjectRef with the default dispatch token (empty string).
   */
  template <class TObjectRef>
  R operator()(const String& token, TObjectRef obj, Args... args) const {
    uint32_t type_index = obj.defined() ? obj->type_index() : 0;
    const runtime::PackedFunc* pf = nullptr;
    if ((pf = LookupDispatchTable(token, type_index)) != nullptr) {
      return (*pf)(obj, args...);
    }
    if ((pf = LookupDispatchTable("", type_index)) != nullptr) {
      return (*pf)(obj, args...);
    }
    if ((pf = LookupFallback()) != nullptr) {
      return (*pf)(obj, args...);
    }

    LOG(WARNING) << "ObjectFunctor calls un-registered function on type: "
                 << runtime::Object::TypeIndex2Key(type_index) << " (token: " << token << ")"
                 << ". ObjectType: " << obj->GetTypeKey() << ". Object: " << obj;
    ICHECK(false) << "ObjectFunctor calls un-registered function on type: "
                  << runtime::Object::TypeIndex2Key(type_index) << " (token: " << token << ")"
                  << ". ObjectType: " << obj->GetTypeKey() << ". Object: " << obj;
  }

  /*!
   * \brief Set the dispatch function
   * \param token The dispatch token.
   * \param type_index The TVM object type index for this dispatch function.
   * \param f The dispatch function.
   *
   * This takes a type-erased packed function as input. It should be used
   * through FFI boundary, for example, registering dispatch function from Python.
   */
  TSelf& set_dispatch(String token, uint32_t type_index, runtime::PackedFunc f) {
    std::vector<runtime::PackedFunc>* table = &dispatch_table_[token];
    if (table->size() <= type_index) {
      table->resize(type_index + 1, nullptr);
    }
    runtime::PackedFunc& slot = (*table)[type_index];
    if (slot != nullptr) {
      ICHECK(false) << "Dispatch for type is already registered: "
                    << runtime::Object::TypeIndex2Key(type_index);
    }
    slot = f;
    return *this;
  }

  TSelf& set_fallback(runtime::PackedFunc f) {
    ICHECK(!dispatch_fallback_.has_value()) << "Fallback is already defined";
    dispatch_fallback_ = f;
    return *this;
  }

  void remove_fallback() { dispatch_fallback_ = std::nullopt; }

  /*!
   * \brief Set the dispatch function
   * \param token The dispatch token.
   * \param f The dispatch function.
   */
  template <typename TObjectRef, typename TCallable,
            typename = std::enable_if_t<IsDispatchFunction<TObjectRef, TCallable>::value>>
  TSelf& set_dispatch(String token, TCallable f) {
    return set_dispatch(token, TObjectRef::ContainerType::RuntimeTypeIndex(),
                        runtime::TypedPackedFunc<R(TObjectRef, Args...)>(f));
  }

  template <typename TCallable,
            typename = std::enable_if_t<IsDispatchFunction<ObjectRef, TCallable>::value>>
  TSelf& set_fallback(TCallable f) {
    runtime::PackedFunc func = runtime::TypedPackedFunc<R(ObjectRef, Args...)>(f);
    return set_fallback(func);
  }

  /*!
   * \brief Remove dispatch function
   * \param token The dispatch token.
   * \param type_index The TVM object type index for the dispatch function to be removed.
   *
   * This is useful when dispatch function comes from other language's runtime, and
   * those function should be removed before that language runtime shuts down.
   */
  void remove_dispatch(String token, uint32_t type_index) {
    std::vector<runtime::PackedFunc>* table = &dispatch_table_[token];
    if (table->size() <= type_index) {
      return;
    }
    (*table)[type_index] = nullptr;
  }

 private:
  /*!
   * \brief Look up the dispatch table for the given token and type_index.
   * \param token The dispatch token.
   * \param type_index The TVM object type index.
   * \return Returns the functor if the lookup succeeds, nullptr otherwise.
   */
  const runtime::PackedFunc* LookupDispatchTable(const String& token, uint32_t type_index) const {
    auto it = dispatch_table_.find(token);
    if (it == dispatch_table_.end()) {
      return nullptr;
    }
    const std::vector<runtime::PackedFunc>& tab = it->second;
    if (type_index >= tab.size()) {
      return nullptr;
    }
    const PackedFunc* f = &tab[type_index];
    if (f->defined()) {
      return f;
    } else {
      return nullptr;
    }
  }

  /*!
   * \brief Look up the fallback to be used if no handler is registered
   */
  const runtime::PackedFunc* LookupFallback() const {
    if (dispatch_fallback_.has_value()) {
      return &*dispatch_fallback_;
    } else {
      return nullptr;
    }
  }

  /*
   * This type alias and the following free functions are created to reduce the binary bloat
   * from template and also hide implementation details from this header
   */
  using DispatchTable = std::unordered_map<std::string, std::vector<runtime::PackedFunc>>;
  /*! \brief The dispatch table. */
  DispatchTable dispatch_table_;
  std::optional<runtime::PackedFunc> dispatch_fallback_;
};

}  // namespace printer
}  // namespace script
}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_IR_DOCSIFIER_FUNCTOR_H_
