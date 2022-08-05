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
struct first_arg_type_helper;

template <typename R, typename ArgOne, typename... OtherArgs>
struct first_arg_type_helper<R(ArgOne, OtherArgs...)> {
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
using first_arg_type = typename detail::first_arg_type_helper<
    typename tvm::runtime::detail::function_signature<FType>::FType>::T;
}  // namespace detail

}  // namespace

namespace dispatch_table {
/*
 * Functions in dispatch_table namespace is created to reduce the binary bloat
 * from template and also hide implementation details from this header
 */

using DispatchTable = std::unordered_map<std::string, std::vector<runtime::PackedFunc>>;

constexpr const char* kDefaultDispatchToken = "";

const runtime::PackedFunc& GetDispatchFunction(const DispatchTable& dispatch_table,
                                               const String& token, uint32_t type_index);
void SetDispatchFunction(DispatchTable* dispatch_table, const String& token, uint32_t type_index,
                         runtime::PackedFunc f);
}  // namespace dispatch_table

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
    const runtime::PackedFunc& dispatch_function = dispatch_table::GetDispatchFunction(
        dispatch_table_, token, traced_object.Get()->type_index());
    return dispatch_function(traced_object.Get(), traced_object.GetPath(),
                             std::forward<Args>(args)...);
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
    dispatch_table::SetDispatchFunction(&dispatch_table_, token, type_index, std::move(f));
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
            typename TObjectRef = typename detail::first_arg_type<TCallable>::ObjectRefType,
            typename = std::enable_if_t<IsDispatchFunction<TObjectRef, TCallable>::value>>
  TSelf& set_dispatch(String token, TCallable f) {
    return set_dispatch(token,                                          //
                        TObjectRef::ContainerType::RuntimeTypeIndex(),  //
                        runtime::TypedPackedFunc<R(TObjectRef, ObjectPath, Args...)>(
                            [f](TObjectRef object, ObjectPath path, Args... args) -> R {
                              return f(MakeTraced(object, path), std::forward<Args>(args)...);
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
  TSelf& set_dispatch(TCallable f) {
    return set_dispatch(dispatch_table::kDefaultDispatchToken, std::forward<TCallable>(f));
  }

 private:
  dispatch_table::DispatchTable dispatch_table_;
};

}  // namespace printer
}  // namespace script
}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_TRACED_OBJECT_FUNCTOR_H_
