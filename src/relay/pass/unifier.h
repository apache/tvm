/*!
 *  Copyright (c) 2018 by Contributors
 * \file include/tvm/relay/pass/unifier.h
 * \brief The type unifier which solves a system of equations between
 * incomplete types.
 */
#ifndef TVM_RELAY_PASS_UNIFIER_H_
#define TVM_RELAY_PASS_UNIFIER_H_

#include <tvm/relay/expr.h>
#include <string>
#include "./type_functor.h"

namespace tvm {
namespace relay {

struct UnionFindError : dmlc::Error {
  explicit UnionFindError(const std::string& msg) : Error(msg) {}
};

struct UnificationError : dmlc::Error {
  explicit UnificationError(const std::string& msg) : Error(msg) {}
};

struct SubstitutionError : dmlc::Error {
  explicit SubstitutionError(const std::string& msg) : Error(msg) {}
};

/*! \brief A union-find data structure for the type-checker */
class UnionFind;

class UnionFindNode : public Node {
 public:
  /*! \brief The inernal map from incomplete types to their representatives. */
  tvm::Map<IncompleteType, Type> uf_map;

  UnionFindNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final { v->Visit("uf_map", &uf_map); }

  TVM_DLL static UnionFind make(tvm::Map<IncompleteType, Type> uf_map);

  /*! \brief Insert it into the union find.
  * \param it The type to add to the union find.
  */
  void Insert(const IncompleteType& it);

  /*! \brief Union operation, combine two equivalence classes.
  * \param it The incomplete type to unify.
  * \param ty The other type.
  */
  void Unify(const IncompleteType& it, const Type& t);

  /*! \brief Find operation, returns the representative of the argument.
  * \param it The element to lookup.
  */
  Type Find(const IncompleteType& it);

  void debug();

  void AssertAlphaEqual(const Type& l, const Type& r);

  static constexpr const char* _type_key = "relay.UnionFind";
  TVM_DECLARE_NODE_TYPE_INFO(UnionFindNode, Node);
};

class UnionFind : public NodeRef {
 public:
  UnionFind() {}
  explicit UnionFind(std::shared_ptr<tvm::Node> p) : NodeRef(p) {}

  // The union find structure is mutable so we do not use the standard macros
  // and expose the pointer via `->`.
  UnionFindNode* operator->() const {
    return static_cast<UnionFindNode*>(node_.get());
  }

  using ContainerType = UnionFindNode;
};

class TypeUnifier;
class TypeUnifierNode : public Node,
                        private TypeFunctor<Type(const Type&, const Type)> {
 public:
  UnionFind union_find;

  TypeUnifierNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final { v->Visit("union_find", &union_find); }

  TVM_DLL static TypeUnifier make(UnionFind uf);

  /*! \brief Introduces a new type var into the unifier */
  void Insert(const IncompleteType& v);

  /*! \brief Unifies two types if possible, throws a unification error if it
   * cannot  */
  Type Unify(const Type& t1, const Type& t2);

  /*! \brief Attempts to substitute all type vars in t with concrete types,
   * throws substitution error if it cannot concretize*/
  Type Subst(const Type& t);

  // /*! \brief Checks the kinds in the given type */
  // Type CheckKinds(const Type& t);

  static constexpr const char* _type_key = "relay.TypeUnifier";
  TVM_DECLARE_NODE_TYPE_INFO(TypeUnifierNode, Node);

 private:
  /*! \brief Unify incomplete type with another type. */
  Type UnifyWithIncompleteType(const Type& t1, const IncompleteType tvn2);
  /*! \brief Implements unification between two types with incomplete portions.
   */
  Type VisitType(const Type& t1, const Type t2) override;

  // Visitor Cases
  Type VisitType_(const IncompleteTypeNode* t1, const Type t2) override;
  Type VisitType_(const TensorTypeNode* t1, const Type t2) override;
  Type VisitType_(const TypeParamNode* t1, const Type t2) override;
  Type VisitType_(const FuncTypeNode* t1, const Type t2) override;
  Type VisitType_(const TupleTypeNode* t1, const Type t2) override;
  Type VisitType_(const TypeRelationNode* s1, const Type t2) override;
};

class TypeUnifier : public NodeRef {
 public:
  TypeUnifier() {}
  explicit TypeUnifier(std::shared_ptr<tvm::Node> p) : NodeRef(p) {}

  // no const so that unifier can be mutable as a member of typechecker
  inline TypeUnifierNode* operator->() const {
    return static_cast<TypeUnifierNode*>(node_.get());
  }

  using ContainerType = TypeUnifierNode;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_UNIFIER_H_
