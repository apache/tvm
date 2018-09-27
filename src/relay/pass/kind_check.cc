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
  bool MatchKind(const Type& t, Kind k) {
    if (const IncompleteTypeNode *tv = t.as<IncompleteTypeNode>()) {
      return tv->kind == k;
    }

    if (const TypeParamNode *tp = t.as<TypeParamNode>()) {
      return tp->kind == k;
    }

    return false;
  }

  bool IsTypeKind(const Type& t) {
    if (MatchKind(t, Kind::kType)) {
      return true;
    }

    return t.as_derived<BaseTensorTypeNode>() || t.as<TupleTypeNode>() || t.as<FuncTypeNode>();
  }

  void VisitType_(const TupleTypeNode* op) override {
    // tuples should only contain normal types
    for (const Type& t : op->fields) {
      this->VisitType(t);
      valid = valid && IsTypeKind(t);
      if (!valid) {
        return;
      }
    }
  }

  void VisitType_(const FuncTypeNode* op) override {
    // Func types should only take normal types for arguments
    // and only return a normal type. They should also have
    // well-formed constraints
    for (const Type& t : op->arg_types) {
      this->VisitType(t);
      valid = valid && IsTypeKind(t);
      if (!valid) {
        return;
      }
    }

    for (const TypeConstraint& tc : op->type_constraints) {
      this->VisitType(tc);
      if (!valid) {
        return;
      }
    }

    this->VisitType(op->ret_type);
    valid = valid && IsTypeKind(op->ret_type);
  }

  void VisitType_(const TypeRelationNode* op) override {
    // arguments to type relation should be normal types
    for (const Type& t : op->args) {
      this->VisitType(t);
      valid = valid && IsTypeKind(t);
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

TVM_REGISTER_API("relay._ir_pass.check_kind")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      if (args.size() == 1) {
        *ret = KindCheck(EnvironmentNode::make({}), args[0]);
      } else {
        *ret = KindCheck(args[0], args[1]);
      }
    });

}  // namespace relay
}  // namespace tvm
