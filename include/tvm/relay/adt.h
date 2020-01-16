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
 * \file tvm/relay/adt.h
 * \brief Algebraic data types for Relay
 */
#ifndef TVM_RELAY_ADT_H_
#define TVM_RELAY_ADT_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/adt.h>
#include <string>
#include <functional>
#include "./base.h"
#include "./type.h"
#include "./expr.h"

namespace tvm {
namespace relay {

using Constructor = tvm::Constructor;
using ConstructorNode = tvm::ConstructorNode;

using TypeData = tvm::TypeData;
using TypeDataNode = tvm::TypeDataNode;

/*! \brief Base type for declaring relay pattern. */
class PatternNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Pattern";
  TVM_DECLARE_BASE_OBJECT_INFO(PatternNode, Object);
};

/*!
 * \brief Pattern is the base type for an ADT match pattern in Relay.
 *
 * Given an ADT value, a pattern might accept it and bind the pattern variable to some value
 * (typically a subnode of the input or the input). Otherwise, the pattern rejects the value.
 *
 * ADT pattern matching thus takes a list of values and binds to the first that accepts the value.
 */
class Pattern : public ObjectRef {
 public:
  Pattern() {}
  explicit Pattern(ObjectPtr<tvm::Object> p) : ObjectRef(p) {}

  using ContainerType = PatternNode;
};

/*! \brief A wildcard pattern: Accepts all input and binds nothing. */
class PatternWildcard;
/*! \brief PatternWildcard container node */
class PatternWildcardNode : public PatternNode {
 public:
  PatternWildcardNode() {}

  TVM_DLL static PatternWildcard make();

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternWildcard";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternWildcardNode, PatternNode);
};

class PatternWildcard : public Pattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PatternWildcard, Pattern, PatternWildcardNode);
};

/*! \brief A var pattern. Accept all input and bind to a var. */
class PatternVar;
/*! \brief PatternVar container node */
class PatternVarNode : public PatternNode {
 public:
  PatternVarNode() {}

  /*! \brief Variable that stores the matched value. */
  tvm::relay::Var var;

  TVM_DLL static PatternVar make(tvm::relay::Var var);

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternVarNode, PatternNode);
};

class PatternVar : public Pattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PatternVar, Pattern, PatternVarNode);
};

/*! \brief A constructor pattern. Matches a value with the given constructor, binds recursively. */
class PatternConstructor;
/*! \brief PatternVar container node */
class PatternConstructorNode : public PatternNode {
 public:
  /*! Constructor matched by the pattern. */
  Constructor constructor;
  /*! Sub-patterns to match against each input to the constructor. */
  tvm::Array<Pattern> patterns;

  PatternConstructorNode() {}

  TVM_DLL static PatternConstructor make(Constructor constructor, tvm::Array<Pattern> var);

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("constructor", &constructor);
    v->Visit("patterns", &patterns);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternConstructor";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternConstructorNode, PatternNode);
};

class PatternConstructor : public Pattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PatternConstructor, Pattern, PatternConstructorNode);
};

/*! \brief A tuple pattern. Matches a tuple, binds recursively. */
class PatternTuple;
/*! \brief PatternVar container node */
class PatternTupleNode : public PatternNode {
 public:
  /*! Sub-patterns to match against each value of the tuple. */
  tvm::Array<Pattern> patterns;

  PatternTupleNode() {}

  TVM_DLL static PatternTuple make(tvm::Array<Pattern> var);

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("patterns", &patterns);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternTuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternTupleNode, PatternNode);
};

class PatternTuple : public Pattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PatternTuple, Pattern, PatternTupleNode);
};

/*! \brief A clause in a match expression. */
class Clause;
/*! \brief Clause container node. */
class ClauseNode : public Object {
 public:
  /*! \brief The pattern the clause matches. */
  Pattern lhs;
  /*! \brief The resulting value. */
  Expr rhs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
  }

  TVM_DLL static Clause make(Pattern lhs, Expr rhs);

  static constexpr const char* _type_key = "relay.Clause";
  TVM_DECLARE_FINAL_OBJECT_INFO(ClauseNode, Object);
};

class Clause : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Clause, ObjectRef, ClauseNode);
};

/*! \brief ADT pattern matching exression. */
class Match;
/*! \brief Match container node. */
class MatchNode : public ExprNode {
 public:
  /*! \brief The input being deconstructed. */
  Expr data;

  /*! \brief The match node clauses. */
  tvm::Array<Clause> clauses;

  /*! \brief Should this match be complete (cover all cases)?
   *  If yes, the type checker will generate an error if there are any missing cases.
   */
  bool complete;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("clauses", &clauses);
    v->Visit("complete", &complete);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Match make(Expr data, tvm::Array<Clause> pattern, bool complete = true);

  static constexpr const char* _type_key = "relay.Match";
  TVM_DECLARE_FINAL_OBJECT_INFO(MatchNode, ExprNode);
};

class Match : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Match, RelayExpr, MatchNode);
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ADT_H_
