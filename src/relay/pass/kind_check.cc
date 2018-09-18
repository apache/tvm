/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file kindchecker.cc
 *
 * \brief Check that types are well formed by applying "kinding rules".
 *
 * This pass ensures we do not do things that violate the design of the
 * type system when writing down types.
 *
 * For example tensors are not allowed to contain functions in Relay.
 *
 * We check this by ensuring the `dtype` field of a Tensor always
 * contains a data type such as `int`, `float`, `uint`.
 */
#include <tvm/relay/pass.h>
#include "./type_visitor.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;
using Kind = TypeParamNode::Kind;

struct KindChecker : TypeVisitor<> {
  bool valid;

  KindChecker() : valid(true) {}

  // checks if t is an incomplete node of kind k or a type param of kind k
  bool matchKind(const Type& t, Kind k) {
    if (const IncompleteTypeNode *tv = t.as<IncompleteTypeNode>()) {
      return tv->kind == k;
    }

    if (const TypeParamNode *tp = t.as<TypeParamNode>()) {
      return tp->kind == k;
    }

    return false;
  }

  bool isTypeKind(const Type& t) {
    if (matchKind(t, Kind::kType)) {
      return true;
    }

    return t.as<BaseTensorTypeNode>() || t.as<TupleTypeNode>() || t.as<FuncTypeNode>();
  }

  bool isTypeListKind(const Type& t) {
    // no type list literal?
    return matchKind(t, Kind::kTypeList);
  }

  void VisitType_(const TupleTypeNode* op) override {
    // tuples should only contain normal types
    for (const Type& t : op->fields) {
      this->VisitType(t);
      valid = valid && isTypeKind(t);
      if (!valid) {
        return;
      }
    }
  }

  void VisitType_(const FuncTypeNode* op) override {
    // func types should only take normal types for arguments
    // and only return a normal type
    for (const Type& t : op->arg_types) {
      this->VisitType(t);
      valid = valid && isTypeKind(t);
      if (!valid) {
        return;
      }
    }

    this->VisitType(op->ret_type);
    valid = valid && isTypeKind(op->ret_type);
  }

  void VisitType_(const TypeRelationNode* op) override {
    // arguments to type relation should be either normal
    // types or type lists(?)
    for (const Type& t : op->args) {
      this->VisitType(t);
      valid = valid && (isTypeKind(t) || isTypeListKind(t));
      if (!valid) {
        return;
      }
    }
  }

  bool Check(const Type &t) {
    this->VisitType(t);
    return valid;
  }
};

bool KindCheck(const Environment& env, const Type &t) {
  KindChecker kc;
  return kc.Check(t);
}

}  // namespace relay
}  // namespace tvm
