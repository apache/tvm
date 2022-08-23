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
#ifndef TVM_SCRIPT_PRINTER_TRACED_OBJECT_FUNCTOR_H_
#define TVM_SCRIPT_PRINTER_TRACED_OBJECT_FUNCTOR_H_

#include <tvm/node/node.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/script/printer/traced_object.h>

#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

namespace {

namespace detail {
/*!
 * \brief Helper template class to extract the type of first argument of a function
 * \tparam FType The function type.
 */
template <typename FType>
struct FirstArgTypeGetter;

template <typename R, typename ArgOne, typename... OtherArgs>
struct FirstArgTypeGetter<R(ArgOne, OtherArgs...)> {
  using T = ArgOne;
};

/*!
 * \brief Template alias for the type of first argument of a function
 * \tparam FType The function type.
 *
 * The name of public functions are in snake case to be consistent with
 * tvm/node/functor.h
 */
template <typename FType>
using FirstArgType = typename detail::FirstArgTypeGetter<
    typename tvm::runtime::detail::function_signature<FType>::FType>::T;
}  // namespace detail

}  // namespace

/*
 * This type alias and the following free functions are created to reduce the binary bloat
 * from template and also hide implementation details from this header
 */
using DispatchTable = std::unordered_map<std::string, std::vector<runtime::PackedFunc>>;

/*!
 * \brief Get function from dispatch table.
 * \param dispatch_table The dispatch table.
 * \param token The dispatch token.
 * \param type_index The type index of the Object type to be dispatched.
 *
 * \return The dispatch function.
 */
const runtime::PackedFunc& GetDispatchFunction(const DispatchTable& dispatch_table,
                                               const String& token, uint32_t type_index);

/*!
 * \brief Set function in dispatch table.
 * \param dispatch_table The dispatch table.
 * \param token The dispatch token.
 * \param type_index The type index of the Object type to be dispatched.
 * \param f The dispatch function.
 */
void SetDispatchFunction(DispatchTable* dispatch_table, const String& token, uint32_t type_index,
                         runtime::PackedFunc f);

/*!
 * \brief Remove function from dispatch table.
 * \param dispatch_table The dispatch table.
 * \param token The dispatch token.
 * \param type_index The TVM object type index for the dispatch function to be removed.
 */
void RemoveDispatchFunction(DispatchTable* dispatch_table, const String& token,
                            uint32_t type_index);

constexpr const char* kDefaultDispatchToken = "";

/*!
 * \brief Dynamic dispatch functor based on TracedObject.
 *
 * This functor dispatches based on the type of object ref inside the input TracedObject,
 * and the input dispatch token.
 */
template <typename R, typename... Args>
class TracedObjectFunctor {
 private:
  using TSelf = TracedObjectFunctor<R, Args...>;

  template <class TObjectRef, class TCallable>
  using IsDispatchFunction =
      typename std::is_convertible<TCallable, std::function<R(TracedObject<TObjectRef>, Args...)>>;

 public:
  /*!
   * \brief Call the dispatch function.
   * \param token The dispatch token.
   * \param traced_object The traced object.
   * \param args Other args.
   *
   * \return The return value of the dispatch function
   *
   * If the TObjectRef isn't registered with the token, it will try to find
   * dispatch function for TObjectRef with kDefaultDispatchToken.
   */
  template <class TObjectRef>
  R operator()(const String& token, TracedObject<TObjectRef> traced_object, Args... args) const {
    const runtime::PackedFunc& dispatch_function =
        GetDispatchFunction(dispatch_table_, token, traced_object.Get()->type_index());
    return dispatch_function(traced_object.Get(), traced_object.GetPath(), args...);
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
    SetDispatchFunction(&dispatch_table_, token, type_index, std::move(f));
    return *this;
  }

  /*!
   * \brief Set the dispatch function
   * \param token The dispatch token.
   * \param f The dispatch function.
   *
   * The diaptch function should have signature `R(TracedObject<TObjectRef>, Args...)`.
   */
  template <typename TCallable,
            typename TObjectRef = typename detail::FirstArgType<TCallable>::ObjectRefType,
            typename = std::enable_if_t<IsDispatchFunction<TObjectRef, TCallable>::value>>
  TSelf& set_dispatch(String token, TCallable f) {
    return set_dispatch(
        token,                                          //
        TObjectRef::ContainerType::RuntimeTypeIndex(),  //
        runtime::TypedPackedFunc<R(TObjectRef, ObjectPath, Args...)>(
            [f = std::move(f)](TObjectRef object, ObjectPath path, Args... args) -> R {
              return f(MakeTraced(object, path), args...);
            }));
  }
  /*!
   * \brief Set the default dispatch function
   * \param f The dispatch function.
   *
   * Default dispatch function will be used if there is no function registered
   * with the requested dispatch token.
   *
   * Default dispatch function has an empty string as dispatch token.
   */
  template <typename TCallable>
  TSelf& set_dispatch(TCallable&& f) {
    return set_dispatch(kDefaultDispatchToken, std::forward<TCallable>(f));
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
    RemoveDispatchFunction(&dispatch_table_, token, type_index);
  }

 private:
  DispatchTable dispatch_table_;
};

}  // namespace printer
}  // namespace script
}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_TRACED_OBJECT_FUNCTOR_H_
