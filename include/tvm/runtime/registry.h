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
 * \file tvm/runtime/registry.h
 * \brief This file defines the TVM global function registry.
 *
 *  The registered functions will be made available to front-end
 *  as well as backend users.
 *
 *  The registry stores type-erased functions.
 *  Each registered function is automatically exposed
 *  to front-end language(e.g. python).
 *
 *  Front-end can also pass callbacks as PackedFunc, or register
 *  then into the same global registry in C++.
 *  The goal is to mix the front-end language and the TVM back-end.
 *
 * \code
 *   // register the function as MyAPIFuncName
 *   TVM_REGISTER_GLOBAL(MyAPIFuncName)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#ifndef TVM_RUNTIME_REGISTRY_H_
#define TVM_RUNTIME_REGISTRY_H_

#include <tvm/runtime/packed_func.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Check if signals have been sent to the process and if so
 *  invoke the registered signal handler in the frontend environment.
 *
 *  When running TVM in another language (Python), the signal handler
 *  may not be immediately executed, but instead the signal is marked
 *  in the interpreter state (to ensure non-blocking of the signal handler).
 *
 *  This function can be explicitly invoked to check the cached signal
 *  and run the related processing if a signal is marked.
 *
 *  On Linux, when siginterrupt() is set, invoke this function whenever a syscall returns EINTR.
 *  When it is not set, invoke it between long-running syscalls when you will not immediately
 *  return to the frontend. On Windows, the same rules apply, but due to differences in signal
 *  processing, these are likely to only make a difference when used with Ctrl+C and socket calls.
 *
 *  Not inserting this function will not cause any correctness
 *  issue, but will delay invoking the Python-side signal handler until the function returns to
 *  the Python side. This means that the effect of e.g. pressing Ctrl+C or sending signals the
 *  process will be delayed until function return. When a C function is blocked on a syscall
 *  such as accept(), it needs to be called when EINTR is received.
 *  So this function is not needed in most API functions, which can finish quickly in a
 *  reasonable, deterministic amount of time.
 *
 * \code
 *
 * int check_signal_every_k_iter = 10;
 *
 * for (int iter = 0; iter < very_large_number; ++iter) {
 *   if (iter % check_signal_every_k_iter == 0) {
 *     tvm::runtime::EnvCheckSignals();
 *   }
 *   // do work here
 * }
 *
 * \endcode
 *
 * \note This function is a nop when no PyErr_CheckSignals is registered.
 *
 * \throws This function throws an exception when the frontend signal handler
 *         indicate an error happens, otherwise it returns normally.
 */
TVM_DLL void EnvCheckSignals();

/*! \brief Registry for global function */
class Registry {
 public:
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  TVM_DLL Registry& set_body(PackedFunc f);  // NOLINT(*)
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  template <typename TCallable,
            typename = typename std::enable_if_t<
                std::is_convertible<TCallable, std::function<void(TVMArgs, TVMRetValue*)>>::value &&
                !std::is_base_of<PackedFunc, TCallable>::value>>
  Registry& set_body(TCallable f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }
  /*!
   * \brief set the body of the function to the given function.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * int multiply(int x, int y) {
   *   return x * y;
   * }
   *
   * TVM_REGISTER_GLOBAL("multiply")
   * .set_body_typed(multiply); // will have type int(int, int)
   *
   * // will have type int(int, int)
   * TVM_REGISTER_GLOBAL("sub")
   * .set_body_typed([](int a, int b) -> int { return a - b; });
   *
   * \endcode
   *
   * \param f The function to forward to.
   * \tparam FLambda The signature of the function.
   */
  template <typename FLambda>
  Registry& set_body_typed(FLambda f) {
    using FType = typename detail::function_signature<FLambda>::FType;
    return set_body(TypedPackedFunc<FType>(std::move(f), name_).packed());
  }
  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct Example {
   *    int doThing(int x);
   * }
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method(&Example::doThing); // will have type int(Example, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename T, typename R, typename... Args>
  Registry& set_body_method(R (T::*f)(Args...)) {
    using R_ = typename std::remove_reference<R>::type;
    auto fwrap = [f](T target, Args... params) -> R_ {
      // call method pointer
      return (target.*f)(params...);
    };
    return set_body(TypedPackedFunc<R_(T, Args...)>(fwrap, name_));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct Example {
   *    int doThing(int x);
   * }
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method(&Example::doThing); // will have type int(Example, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename T, typename R, typename... Args>
  Registry& set_body_method(R (T::*f)(Args...) const) {
    auto fwrap = [f](const T target, Args... params) -> R {
      // call method pointer
      return (target.*f)(params...);
    };
    return set_body(TypedPackedFunc<R(const T, Args...)>(fwrap, name_));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Used when calling a method on a Node subclass through a ObjectRef subclass.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct ExampleNode: BaseNode {
   *    int doThing(int x);
   * }
   *
   * // noderef subclass
   * struct Example;
   *
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method<Example>(&ExampleNode::doThing); // will have type int(Example, int)
   *
   * // note that just doing:
   * // .set_body_method(&ExampleNode::doThing);
   * // wouldn't work, because ExampleNode can't be taken from a TVMArgValue.
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam TObjectRef the node reference type to call the method on
   * \tparam TNode the node type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename TObjectRef, typename TNode, typename R, typename... Args,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  Registry& set_body_method(R (TNode::*f)(Args...)) {
    auto fwrap = [f](TObjectRef ref, Args... params) {
      TNode* target = ref.operator->();
      // call method pointer
      return (target->*f)(params...);
    };
    return set_body(TypedPackedFunc<R(TObjectRef, Args...)>(fwrap, name_));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Used when calling a method on a Node subclass through a ObjectRef subclass.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct ExampleNode: BaseNode {
   *    int doThing(int x);
   * }
   *
   * // noderef subclass
   * struct Example;
   *
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method<Example>(&ExampleNode::doThing); // will have type int(Example, int)
   *
   * // note that just doing:
   * // .set_body_method(&ExampleNode::doThing);
   * // wouldn't work, because ExampleNode can't be taken from a TVMArgValue.
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam TObjectRef the node reference type to call the method on
   * \tparam TNode the node type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename TObjectRef, typename TNode, typename R, typename... Args,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  Registry& set_body_method(R (TNode::*f)(Args...) const) {
    auto fwrap = [f](TObjectRef ref, Args... params) {
      const TNode* target = ref.operator->();
      // call method pointer
      return (target->*f)(params...);
    };
    return set_body(TypedPackedFunc<R(TObjectRef, Args...)>(fwrap, name_));
  }

  /*!
   * \brief Register a function with given name
   * \param name The name of the function.
   * \param override Whether allow override existing function.
   * \return Reference to the registry.
   */
  TVM_DLL static Registry& Register(const std::string& name, bool override = false);  // NOLINT(*)
  /*!
   * \brief Erase global function from registry, if exist.
   * \param name The name of the function.
   * \return Whether function exist.
   */
  TVM_DLL static bool Remove(const std::string& name);
  /*!
   * \brief Get the global function by name.
   * \param name The name of the function.
   * \return pointer to the registered function,
   *   nullptr if it does not exist.
   */
  TVM_DLL static const PackedFunc* Get(const std::string& name);  // NOLINT(*)
  /*!
   * \brief Get the names of currently registered global function.
   * \return The names
   */
  TVM_DLL static std::vector<std::string> ListNames();

  // Internal class.
  struct Manager;

 protected:
  /*! \brief name of the function */
  std::string name_;
  /*! \brief internal packed function */
  PackedFunc func_;
  friend struct Manager;
};

#define TVM_FUNC_REG_VAR_DEF static TVM_ATTRIBUTE_UNUSED ::tvm::runtime::Registry& __mk_##TVM

/*!
 * \brief Register a function globally.
 * \code
 *   TVM_REGISTER_GLOBAL("MyPrint")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_GLOBAL(OpName) \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::runtime::Registry::Register(OpName)

#define TVM_STRINGIZE_DETAIL(x) #x
#define TVM_STRINGIZE(x) TVM_STRINGIZE_DETAIL(x)
#define TVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" TVM_STRINGIZE(__LINE__))
/*!
 * \brief Macro to include current line as string
 */
#define TVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" TVM_STRINGIZE(__LINE__)

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_REGISTRY_H_
