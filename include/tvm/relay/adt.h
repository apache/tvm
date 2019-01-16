/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/adt.h
 * \brief Algebraic data types for Relay
 */
#ifndef TVM_RELAY_ADT_H_
#define TVM_RELAY_ADT_H_

#include <tvm/attrs.h>
#include <string>
#include <functional>
#include "./base.h"
#include "./type.h"
#include "./expr.h"

namespace tvm {
namespace relay {

/*! \brief Base type for declaring relay pattern. */
class PatternNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Pattern";
  TVM_DECLARE_BASE_NODE_INFO(PatternNode, Node);
};

/*!
 * \brief Pattern is the base type for an ADT match pattern in Relay.
 *
 * Given an ADT value, a pattern might accept it and bind the pattern variable to some value
 * (typically a subnode of the input or the input). Otherwise, the pattern rejects the value.
 *
 * ADT pattern matching thus takes a list of values and bings to the first that accepts the value.
 */
class Pattern : public NodeRef {
 public:
  Pattern() {}
  explicit Pattern(NodePtr<tvm::Node> p) : NodeRef(p) {}

  using ContainerType = PatternNode;
};

/*! \brief A wildcard pattern: Accepts all input and binds nothing. */
class PatternWildcard;
/*! \brief PatternWildcard container node */
class PatternWildcardNode : public PatternNode {
 public:
  PatternWildcardNode() {}

  TVM_DLL static PatternWildcard make();

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternWildcard";
  TVM_DECLARE_NODE_TYPE_INFO(PatternWildcardNode, PatternNode);
};

RELAY_DEFINE_NODE_REF(PatternWildcard, PatternWildcardNode, Pattern);

/*! \brief A var pattern. Accept all input and bind to a var. */
class PatternVar;
/*! \brief PatternVar container node */
class PatternVarNode : public PatternNode {
 public:
  PatternVarNode() {}

  tvm::relay::Var var;

  TVM_DLL static PatternVar make(tvm::relay::Var var);

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternVar";
  TVM_DECLARE_NODE_TYPE_INFO(PatternVarNode, PatternNode);
};

RELAY_DEFINE_NODE_REF(PatternVar, PatternVarNode, Pattern);

/*!
 * \brief ADT constructor.
 * Constructors compare by pointer equality.
 */
class Constructor;
/*! \brief Constructor container node. */
class ConstructorNode : public ExprNode {
 public:
  /*! \brief The name (only a hint) */
  std::string name_hint;
  /*! \brief Input to the constructor. */
  tvm::Array<Type> inp;
  /*! \brief The datatype the constructor will construct. */
  GlobalTypeVar belong_to;
  mutable int tag = -1;

  ConstructorNode() {}

  TVM_DLL static Constructor make(std::string name_hint,
                                  tvm::Array<Type> inp,
                                  GlobalTypeVar belong_to);

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name_hint", &name_hint);
    v->Visit("inp", &inp);
    v->Visit("belong_to", &belong_to);
    v->Visit("tag", &tag);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  static constexpr const char* _type_key = "relay.Constructor";
  TVM_DECLARE_NODE_TYPE_INFO(ConstructorNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Constructor, ConstructorNode, Expr);

/*! \brief A constructor pattern. Matches a value with the given constructor, binds recursively. */
class PatternConstructor;
/*! \brief PatternVar container node */
class PatternConstructorNode : public PatternNode {
 public:
  Constructor con;
  tvm::Array<Pattern> pat;

  PatternConstructorNode() {}

  TVM_DLL static PatternConstructor make(Constructor con, tvm::Array<Pattern> var);

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("con", &con);
    v->Visit("pat", &pat);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.PatternConstructor";
  TVM_DECLARE_NODE_TYPE_INFO(PatternConstructorNode, PatternNode);
};

RELAY_DEFINE_NODE_REF(PatternConstructor, PatternConstructorNode, Pattern);

/*!
 * \brief Stores all data for an Algebraic Data Type (ADT).
 */
class TypeData;
/*! \brief TypeData container node */
class TypeDataNode : public TypeNode {
 public:
  /*!
   * \brief The header is simply the name of the ADT.
   * We adopt nominal typing for ADT definitions;
   * that is, differently-named ADT definitions with same constructors
   * have different types.
   */
  GlobalTypeVar header;
  /*! \brief The type variables (to allow for polymorphism). */
  tvm::Array<TypeVar> tv;
  /*! \brief The constructors. */
  tvm::Array<Constructor> constructors;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("header", &header);
    v->Visit("tv", &tv);
    v->Visit("constructors", &constructors);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeData make(GlobalTypeVar header,
                               tvm::Array<TypeVar> tv,
                               tvm::Array<Constructor> constructors);

  static constexpr const char* _type_key = "relay.TypeData";
  TVM_DECLARE_NODE_TYPE_INFO(TypeDataNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TypeData, TypeDataNode, Type);

/*! \brief A clause in a match expression. */
class Clause;
/*! \brief Clause container node. */
class ClauseNode : public Node {
 public:
  Pattern lhs;
  Expr rhs;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
  }

  TVM_DLL static Clause make(Pattern lhs, Expr rhs);

  static constexpr const char* _type_key = "relay.Clause";
  TVM_DECLARE_NODE_TYPE_INFO(ClauseNode, Node);
};

RELAY_DEFINE_NODE_REF(Clause, ClauseNode, NodeRef);

/*! \brief ADT pattern matching exression. */
class Match;
/*! \brief Match container node. */
class MatchNode : public ExprNode {
 public:
  Expr data;

  tvm::Array<Clause> pattern;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("pattern", &pattern);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Match make(Expr data, tvm::Array<Clause> pattern);

  static constexpr const char* _type_key = "relay.Match";
  TVM_DECLARE_NODE_TYPE_INFO(MatchNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Match, MatchNode, Expr);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ADT_H_
