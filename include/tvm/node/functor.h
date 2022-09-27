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
 * \file tvm/node/functor.h
 * \brief Defines the Functor data structures.
 */
#ifndef TVM_NODE_FUNCTOR_H_
#define TVM_NODE_FUNCTOR_H_

#include <dmlc/logging.h>
#include <tvm/runtime/object.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {

using runtime::ObjectRef;

/*!
 * \brief A dynamically dispatched functor on the type of the first argument.
 *
 * This is a class that is useful to construct polymorphic dispatching
 * base on the AST/IR node's type.
 *
 * \code
 *   NodeFunctor<std::string (const ObjectRef& n, std::string prefix)> tostr;
 *   tostr.set_dispatch<Add>([](const ObjectRef& op, std::string prefix) {
 *     return prefix + "Add";
 *   });
 *   tostr.set_dispatch<IntImm>([](const ObjectRef& op, std::string prefix) {
 *     return prefix + "IntImm"
 *   });
 *
 *   Expr x = make_const(1);
 *   Expr y = x + x;
 *   // dispatch to IntImm, outputs "MyIntImm"
 *   LOG(INFO) << tostr(x, "My");
 *   // dispatch to IntImm, outputs "MyAdd"
 *   LOG(INFO) << tostr(y, "My");
 * \endcode
 *
 * \tparam FType function signiture
 *  This type if only defined for FType with function signature
 */
template <typename FType>
class NodeFunctor;

template <typename R, typename... Args>
class NodeFunctor<R(const ObjectRef& n, Args...)> {
 private:
  /*! \brief internal function pointer type */
  typedef R (*FPointer)(const ObjectRef& n, Args...);
  /*! \brief refer to itself. */
  using TSelf = NodeFunctor<R(const ObjectRef& n, Args...)>;
  /*! \brief internal function table */
  std::vector<FPointer> func_;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*!
   * \brief Whether the functor can dispatch the corresponding Node
   * \param n The node to be dispatched
   * \return Whether dispatching function is registered for n's type.
   */
  bool can_dispatch(const ObjectRef& n) const {
    uint32_t type_index = n->type_index();
    return type_index < func_.size() && func_[type_index] != nullptr;
  }
  /*!
   * \brief invoke the functor, dispatch on type of n
   * \param n The Node argument
   * \param args The additional arguments
   * \return The result.
   */
  R operator()(const ObjectRef& n, Args... args) const {
    ICHECK(can_dispatch(n)) << "NodeFunctor calls un-registered function on type "
                            << n->GetTypeKey();
    return (*func_[n->type_index()])(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief set the dispatcher for type TNode
   * \param f The function to be set.
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename TNode>
  TSelf& set_dispatch(FPointer f) {  // NOLINT(*)
    uint32_t tindex = TNode::RuntimeTypeIndex();
    if (func_.size() <= tindex) {
      func_.resize(tindex + 1, nullptr);
    }
    ICHECK(func_[tindex] == nullptr) << "Dispatch for " << TNode::_type_key << " is already set";
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
  TSelf& clear_dispatch() {  // NOLINT(*)
    uint32_t tindex = TNode::RuntimeTypeIndex();
    ICHECK_LT(tindex, func_.size()) << "clear_dispatch: index out of range";
    func_[tindex] = nullptr;
    return *this;
  }
};

#define TVM_REG_FUNC_VAR_DEF(ClsName) static TVM_ATTRIBUTE_UNUSED auto& __make_functor##_##ClsName

/*!
 * \brief Useful macro to set NodeFunctor dispatch in a global static field.
 *
 * \code
 *  // Use NodeFunctor to implement ReprPrinter similar to Visitor Pattern.
 *  // vtable allows easy patch of new Node types, without changing
 *  // interface of ReprPrinter.
 *
 *  class ReprPrinter {
 *   public:
 *    std::ostream& stream;
 *    // the dispatch function.
 *    void print(Expr e) {
 *      const static FType& f = *vtable();
 *      f(e, this);
 *    }
 *
 *    using FType = NodeFunctor<void (const ObjectRef&, ReprPrinter* )>;
 *    // function to return global function table
 *    static FType& vtable();
 *  };
 *
 *  // in cpp/cc file
 *  ReprPrinter::FType& ReprPrinter::vtable() { // NOLINT(*)
 *    static FType inst; return inst;
 *  }
 *
 *  TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
 *  .set_dispatch<Add>([](const ObjectRef& ref, ReprPrinter* p) {
 *    auto* n = static_cast<const Add*>(ref.get());
 *    p->print(n->a);
 *    p->stream << '+'
 *    p->print(n->b);
 *  });
 *
 *
 * \endcode
 *
 * \param ClsName The name of the class
 * \param FField The static function that returns a singleton of NodeFunctor.
 */
#define TVM_STATIC_IR_FUNCTOR(ClsName, FField) \
  TVM_STR_CONCAT(TVM_REG_FUNC_VAR_DEF(ClsName), __COUNTER__) = ClsName::FField()
}  // namespace tvm
#endif  // TVM_NODE_FUNCTOR_H_
