/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/src/relay/pass/unifier.cc
 * \brief The type unifier which solves a system of equations between
 * incomplete types.
 */

#include "./unifier.h"
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/type.h>
#include "./type_subst.h"
#include "./type_visitor.h"

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

UnionFind UnionFindNode::make(tvm::Map<IncompleteType, Type> uf_map) {
  auto n = make_node<UnionFindNode>();
  n->uf_map = uf_map;
  return UnionFind(n);
}

void UnionFindNode::Insert(const IncompleteType& v) { this->uf_map.Set(v, v); }

void UnionFindNode::debug() {
  for (const auto& entry : this->uf_map) {
    RELAY_LOG(INFO) << entry.first << " = " << entry.second << std::endl;
  }
}

void UnionFindNode::AssertAlphaEqual(const Type& l, const Type& r) {
  if (!AlphaEqual(l, r)) {
    std::stringstream ss;
    ss << "Incompatible parent types in UF:" << l << " and " << r;
    throw UnionFindError(ss.str());
  }
}

void UnionFindNode::Unify(const IncompleteType& v1, const Type& t) {
  RELAY_LOG(INFO) << "UnionFindNode::Unify v1=" << v1 << ", t=" << t
                  << std::endl;
  auto parent1 = this->Find(v1);

  // if t is a type var, then unify parents
  const IncompleteTypeNode *tvn2 = t.as<IncompleteTypeNode>();
  if (tvn2) {
    auto v2 = GetRef<IncompleteType>(tvn2);
    auto parent2 = this->Find(v2);

    // if parents are exactly equal, then we're done
    if (parent1 == parent2) {
      return;
    }

    // if first parent is a type var, then can just set its union find map to
    // second parent
    if (const IncompleteTypeNode *pvn1 = parent1.as<IncompleteTypeNode>()) {
      auto pv1 = GetRef<IncompleteType>(pvn1);
      this->uf_map.Set(pv1, parent2);
      return;
    }

    // if second parent is a type var but first isn't, can set second type var
    if (const IncompleteTypeNode *pvn2 = parent2.as<IncompleteTypeNode>()) {
      auto pv2 = GetRef<IncompleteType>(pvn2);
      this->uf_map.Set(pv2, parent1);
      return;
    }

    // if both parents are not type vars themselves, check alpha-equality
    AssertAlphaEqual(parent1, parent2);
    return;
  }

  // if t is not a type var, then unify with v1's parent if parent is a type
  // var; else, check alpha-equality for compatibility
  if (const IncompleteTypeNode *pvn1 = parent1.as<IncompleteTypeNode>()) {
    auto pv1 = GetRef<IncompleteType>(pvn1);
    this->uf_map.Set(pv1, t);
    return;
  }

  AssertAlphaEqual(parent1, t);
}

Type UnionFindNode::Find(const IncompleteType& v) {
  // The node has no mapping, so its representative is just itself.
  if (this->uf_map.find(v) == this->uf_map.end()) {
    return v;
  }

  Type parent = this->uf_map.at(v);

  if (v == parent) {
    return v;
  }

  // if parent is not a type var, then it must be the representative type
  const IncompleteTypeNode *rep = parent.as<IncompleteTypeNode>();
  if (!rep) {
    return parent;
  }

  // otherwise, recurse and perform path compression
  IncompleteType pv = GetRef<IncompleteType>(rep);
  Type higher_up = this->Find(pv);
  this->uf_map.Set(v, higher_up);
  return higher_up;
}

TVM_REGISTER_API("relay._make.UnionFind")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      if (args.size() == 0) {
        *ret = UnionFindNode::make({});
      } else {
        *ret = UnionFindNode::make(args[0]);
      }
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<UnionFindNode>([](const UnionFindNode *node,
                                    tvm::IRPrinter *p) {
      p->stream << "UnionFindNode(" << node->uf_map << ")";
    });

TypeUnifier TypeUnifierNode::make(UnionFind union_find) {
  auto n = make_node<TypeUnifierNode>();
  n->union_find = union_find;
  return TypeUnifier(n);
}

void TypeUnifierNode::Insert(const IncompleteType& v) {
  this->union_find->Insert(v);
}

Type TypeUnifierNode::Unify(const Type& t1, const Type& t2) {
  RELAY_LOG(INFO) << "TypeUnifierNode::unify: t1=" << t1 << " t2=" << t2
                  << std::endl;

  Type unified = this->VisitType(t1, t2);
  // TODO(@jroesch): Restore this code when we finish kind checker.
  // if (!check_kind(unified)) {
  // throw UnificationError("Invalid kinds in unified type");
  // }
  return unified;
}

struct IncompleteTypeSubst : TypeMutator {
  const TypeUnifierNode *unifier;

  IncompleteTypeSubst(const TypeUnifierNode *unifier) : unifier(unifier) {}

  // type var: look it up in the type map and recurse
  Type VisitType_(const IncompleteTypeNode* op) override {
    auto tv = GetRef<IncompleteType>(op);
    auto parent = unifier->union_find->Find(tv);
    if (parent == tv) {
      return tv;
    }
    return this->VisitType(parent);
  }
};

Type TypeUnifierNode::Subst(const Type& t) {
  IncompleteTypeSubst tvsubst(this);
  // normalize first so substitutions in quantifiers will be correct
  Type ret = tvsubst.VisitType(t);
  // TODO(@jroesch): Restore this code when we finish kind checker.
  // if (!check_kind(ret)) {
  // std::stringstream ss;
  // ss << "Invalid Kinds in substituted type!";
  // ss << t << std::endl;
  // ss << ret << std::endl;
  // throw SubstitutionError(ss.str());
  // }
  return ret;
}

Type TypeUnifierNode::VisitType(const Type& t1, const Type t2) {
  // When the right hand size is a type variable immediately unify.
  if (const IncompleteTypeNode *tvn2 = t2.as<IncompleteTypeNode>()) {
    return this->UnifyWithIncompleteType(t1, GetRef<IncompleteType>(tvn2));
  } else {
    return TypeFunctor<Type(const Type &t1, const Type t2)>::VisitType(t1, t2);
  }
}

Type TypeUnifierNode::UnifyWithIncompleteType(const Type& t1,
                                              const IncompleteType tv2) {
  RELAY_LOG(INFO) << "unifyWithIncompleteType: t1=" << t1 << " t2=" << tv2
                  << std::endl;
  // Fix unify to return new representative
  this->union_find->Unify(tv2, t1);
  auto rep = this->union_find->Find(tv2);
  RELAY_LOG(INFO) << "unifyWithIncompleteType: rep =" << rep << std::endl;
  return rep;
}

Type TypeUnifierNode::VisitType_(const IncompleteTypeNode* t1, const Type rt2) {
  IncompleteType tv1 = GetRef<IncompleteType>(t1);
  RELAY_LOG(INFO) << "VisitType_: IncompleteTypeNode t1=" << t1 << " = " << rt2
                  << std::endl;
  this->union_find->Unify(tv1, rt2);
  auto rep = this->union_find->Find(tv1);
  RELAY_LOG(INFO) << "VisitType_: IncompleteTypeNode rep=" << rep << std::endl;
  return rep;
}

Type TypeUnifierNode::VisitType_(const TypeParamNode* t1, const Type rt2) {
  TypeParam ti1 = GetRef<TypeParam>(t1);

  if (const TypeParamNode *tin2 = rt2.as<TypeParamNode>()) {
    TypeParam ti2 = GetRef<TypeParam>(tin2);

    if (ti1 != ti2) {
      throw UnificationError("Attempting to unify non-matching TypeParams");
    }

    return ti1;
  }

  throw UnificationError("Unable to unify TypeParamNode");
}

Type TypeUnifierNode::VisitType_(const FuncTypeNode* t1, const Type rt2) {
  FuncType ft1 = GetRef<FuncType>(t1);

  if (const FuncTypeNode *tan2 = rt2.as<FuncTypeNode>()) {
    FuncType ft2 = GetRef<FuncType>(tan2);

    if (ft1->type_params.size() != ft2->type_params.size()) {
      throw UnificationError(
          "unable to unify functions with differing number of type parameters");
    }

    tvm::Map<TypeParam, Type> subst_map;

    for (size_t i = 0; i < ft1->arg_types.size(); i++) {
      subst_map.Set(ft1->type_params[i], ft2->type_params[i]);
    }

    ft1 = Downcast<FuncType>(TypeSubst(ft1, subst_map));

    if (ft1->arg_types.size() != ft2->arg_types.size()) {
      throw UnificationError("unable to unify functions of different arities");
    }

    tvm::Array<Type> unified_args;
    for (size_t i = 0; i < ft1->arg_types.size(); i++) {
      unified_args.push_back(
          this->VisitType(ft1->arg_types[i], ft2->arg_types[i]));
    }

    Type unified_ret_type = this->VisitType(ft1->ret_type, ft2->ret_type);

    return FuncTypeNode::make(unified_args, unified_ret_type, {}, {});
  }

  throw UnificationError("unable to unify function types");
}

Type TypeUnifierNode::VisitType_(const TensorTypeNode* t1, const Type rt2) {
  TensorType tt1 = GetRef<TensorType>(t1);

  if (const TensorTypeNode *ttn2 = rt2.as<TensorTypeNode>()) {
    TensorType tt2 = GetRef<TensorType>(ttn2);

    if (!AlphaEqual(tt1, tt2)) {
      throw UnificationError("dtypes do not match");
    }

    RELAY_LOG(INFO) << "Unify Tensor Shape s1=" << tt1->shape
                    << " s2= " << tt2->shape << std::endl;

    if (tt1->shape.size() != tt2->shape.size()) {
      throw UnificationError("shapes are not of the same length");
    }

    for (size_t i = 0U; i < tt1->shape.size(); i++) {
      if (!tt1->shape[i].same_as(tt2->shape[i])) {
        throw UnificationError("shapes do not match at index");
      }
    }

    return rt2;
  }

  throw UnificationError("Cannot unify TensorTypeNode");
}

Type TypeUnifierNode::VisitType_(const TupleTypeNode* t1, const Type rt2) {
  TupleType pt1 = GetRef<TupleType>(t1);

  if (const TupleTypeNode *ptn2 = rt2.as<TupleTypeNode>()) {
    TupleType pt2 = GetRef<TupleType>(ptn2);

    std::vector<Type> unified_fields;
    if (pt1->fields.size() != pt2->fields.size()) {
      throw UnificationError("Product types are of different dimensions");
    }

    for (size_t i = 0U; i < pt1->fields.size(); i++) {
      Type unified = this->VisitType(pt1->fields[i], pt2->fields[i]);
      unified_fields.push_back(unified);
    }

    return TupleTypeNode::make(unified_fields);
  }

  throw UnificationError("Cannot unify TupleTypeNode");
}

Type TypeUnifierNode::VisitType_(const TypeRelationNode* tr1, const Type t2) {
  throw InternalError("Cannot unify different type relations");
}

}  // namespace relay
}  // namespace tvm
