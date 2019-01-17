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
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

struct KindChecker : TypeVisitor {
  bool valid;
  const Module& mod;

  explicit KindChecker(const Module& mod) : valid(true), mod(mod) {}

  // checks if t is an incomplete node of kind k or a type param of kind k
  bool MatchKind(const Type& t, Kind k) {
    if (const IncompleteTypeNode* tv = t.as<IncompleteTypeNode>()) {
      return tv->kind == k;
    }

    if (const TypeVarNode* tp = t.as<TypeVarNode>()) {
      return tp->kind == k;
    }

    if (const GlobalTypeVarNode* gtp = t.as<GlobalTypeVarNode>()) {
      return gtp->kind == k;
    }

    return false;
  }

  bool IsTypeKind(const Type& t) {
    if (MatchKind(t, Kind::kType)) {
      return true;
    }

    return t.as_derived<BaseTensorTypeNode>() || t.as<TupleTypeNode>() || t.as<FuncTypeNode>()
      || t.as<TypeCallNode>();
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

  void VisitType_(const RefTypeNode* op) override {
    // tuples should only contain normal types
    this->VisitType(op->value);
    valid = valid && IsTypeKind(op->value);
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

  void VisitType_(const TypeCallNode* op) override {
    // type call func should be a global type var, args should be type
    const auto* gtv = op->func.as<GlobalTypeVarNode>();
    valid = valid && gtv != nullptr && IsTypeKind(op->func);
    if (!valid) {
      return;
    }
    for (const Type& t : op->args) {
      this->VisitType(t);
      valid = valid && IsTypeKind(t);
      if (!valid) {
        return;
      }
    }

    // finally we need to check the module to check the number of type params
    auto var = GetRef<GlobalTypeVar>(gtv);
    auto data = mod->LookupDef(var);
    valid = valid && data->tv.size() == op->args.size();
  }

  void VisitType_(const TypeDataNode* op) override {
    // Constructors can reference the header var, but no other GlobalTypeVars.
    // In theory, a TypeData could be nested, so the header scope
    // should be tracked recursively, but it is unclear that we need
    // to support it.
    valid = valid && op->header->kind == Kind::kType;
    for (const auto& var : op->tv) {
      valid = valid && IsTypeKind(var);
      if (!valid) {
        return;
      }
    }
    for (const auto& con : op->constructors) {
      valid = valid && con->belong_to.same_as(op->header);
      for (const Type& t : con->inp) {
        valid = valid && IsTypeKind(t);
        if (const auto* gtv = t.as<GlobalTypeVarNode>()) {
          valid = valid && GetRef<GlobalTypeVar>(gtv).same_as(op->header);
        }
        if (!valid) {
          return;
        }
      }
    }
  }

  bool Check(const Type& t) {
    this->VisitType(t);
    return valid;
  }
};

bool KindCheck(const Type& t, const Module& mod) {
  KindChecker kc(mod);
  return kc.Check(t);
}

TVM_REGISTER_API("relay._ir_pass.check_kind")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    if (args.size() == 1) {
      *ret = KindCheck(args[0], ModuleNode::make({}, {}));
    } else {
      *ret = KindCheck(args[0], args[1]);
    }
  });

}  // namespace relay
}  // namespace tvm
