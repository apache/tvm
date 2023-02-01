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
 * \file tvm/relay/pattern_functor.h
 * \brief A more powerful visitor on ADT patterns that enables defining
 * arbitrary function signatures with type-based dispatch on first argument.
 */
#ifndef TVM_RELAY_PATTERN_FUNCTOR_H_
#define TVM_RELAY_PATTERN_FUNCTOR_H_

#include <tvm/node/functor.h>
#include <tvm/relay/error.h>

#include <string>
#include <unordered_map>
#include <utility>

#include "./adt.h"
#include "./expr.h"
#include "./op.h"

namespace tvm {
namespace relay {

/*!
 * \brief A dynamical functor on ADT patterns that dispatches on its first argument.
 *  You can use this as a more powerful visitor, since it allows you to
 *  define the types of further arguments to VisitPattern.
 *
 * \sa tvm/ir_functor.h
 *
 * \tparam FType function signiture
 *  This type is only defined for FType with function signature R(const Pattern&,
 * Args...)
 */
template <typename FType>
class PatternFunctor;

// functions to be overriden.
#define PATTERN_FUNCTOR_DEFAULT \
  { return VisitPatternDefault_(op, std::forward<Args>(args)...); }

#define RELAY_PATTERN_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {        \
    return self->VisitPattern_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class PatternFunctor<R(const Pattern& n, Args...)> {
 private:
  using TSelf = PatternFunctor<R(const Pattern& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~PatternFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Pattern& n, Args... args) {
    return VisitPattern(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitPattern(const Pattern& n, Args... args) {
    ICHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitPattern_(const PatternWildcardNode* op, Args... args) PATTERN_FUNCTOR_DEFAULT;
  virtual R VisitPattern_(const PatternVarNode* op, Args... args) PATTERN_FUNCTOR_DEFAULT;
  virtual R VisitPattern_(const PatternConstructorNode* op, Args... args) PATTERN_FUNCTOR_DEFAULT;
  virtual R VisitPattern_(const PatternTupleNode* op, Args... args) PATTERN_FUNCTOR_DEFAULT;
  virtual R VisitPatternDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_PATTERN_FUNCTOR_DISPATCH(PatternWildcardNode);
    RELAY_PATTERN_FUNCTOR_DISPATCH(PatternVarNode);
    RELAY_PATTERN_FUNCTOR_DISPATCH(PatternConstructorNode);
    RELAY_PATTERN_FUNCTOR_DISPATCH(PatternTupleNode);
    return vtable;
  }
};

/*! \brief A simple visitor wrapper around PatternFunctor.
 *
 * Exposes two visitors with default traversal strategies, one
 * which doesn't compute a result but can mutate internal state,
 * and another which functionally builds a new pattern.
 */
class PatternVisitor : public ::tvm::relay::PatternFunctor<void(const Pattern& n)> {
 public:
  void VisitPattern_(const PatternWildcardNode* op) override;
  void VisitPattern_(const PatternVarNode* op) override;
  void VisitPattern_(const PatternConstructorNode* op) override;
  void VisitPattern_(const PatternTupleNode* op) override;
  virtual void VisitType(const Type& t);
  virtual void VisitVar(const Var& v);
  virtual void VisitConstructor(const Constructor& c);
};

/*! \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
class PatternMutator : public ::tvm::relay::PatternFunctor<Pattern(const Pattern&)> {
 public:
  Pattern Mutate(const Pattern& pat);
  Pattern VisitPattern_(const PatternWildcardNode* op) override;
  Pattern VisitPattern_(const PatternVarNode* op) override;
  Pattern VisitPattern_(const PatternConstructorNode* op) override;
  Pattern VisitPattern_(const PatternTupleNode* op) override;
  /*! \brief Used to visit the types inside of patterns.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
  /*! \brief Used to visit the vars inside of patterns. */
  virtual Var VisitVar(const Var& v);
  /*! \brief Used to visit the vars inside of patterns. */
  virtual Constructor VisitConstructor(const Constructor& c);

 private:
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_map_;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PATTERN_FUNCTOR_H_
