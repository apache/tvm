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

struct KindChecker : TypeFunctor<Kind(const Type&)> {
  const Module& mod;

  explicit KindChecker(const Module& mod) : mod(mod) {}

  Kind VisitType_(const IncompleteTypeNode* op) override {
    return op->kind;
  }

  Kind VisitType_(const TypeVarNode* op) override {
    return op->kind;
  }

  Kind VisitType_(const GlobalTypeVarNode* op) override {
    return op->kind;
  }

  Kind VisitType_(const TensorTypeNode* op) override {
    return Kind::kType;
  }

  Kind VisitType_(const TupleTypeNode* op) override {
    // tuples should only contain normal types
    for (const Type& t : op->fields) {
      Kind k = this->VisitType(t);
      CHECK(k == Kind::kType)
        << "All types in tuple type must be of a type kind but "
        << t << " in " << GetRef<TupleType>(op) << " is of kind " << k;
    }
    return Kind::kType;
  }

  Kind VisitType_(const FuncTypeNode* op) override {
    // Func types should only take normal types for arguments
    // and only return a normal type. They should also have
    // well-formed constraints
    FuncType ft = GetRef<FuncType>(op);
    for (const Type& t : op->arg_types) {
      Kind k = this->VisitType(t);
      CHECK(k == Kind::kType)
        << "Function parameters must be of the type kind but parameter "
        << t << " of " << ft << " is of kind " << k;
    }

    Kind ret_kind = this->VisitType(ft->ret_type);
    CHECK(ret_kind == Kind::kType)
      << "The function return type must be of the type kind but "
      << ft->ret_type << " of " << ft << " is of kind " << ret_kind;

    for (const TypeConstraint& tc : op->type_constraints) {
      Kind k = this->VisitType(tc);
      CHECK(k == Kind::kConstraint)
        << "All function type constraints are of the constraint kind but "
        << tc << " of " << ft << " is of kind " << k;
    }

    return Kind::kType;
  }

  Kind VisitType_(const RefTypeNode* op) override {
    // ref types should only contain normal types
    Kind k = this->VisitType(op->value);
    CHECK(k == Kind::kType)
      << "The value inside a ref must be of the type kind but "
      << op->value << " of " << GetRef<RefType>(op) << " is of kind " << k;
    return Kind::kType;
  }

  Kind VisitType_(const TypeRelationNode* op) override {
    // arguments to type relation should be normal types
    for (const Type& t : op->args) {
      Kind k = this->VisitType(t);
      CHECK(k == Kind::kType)
        << "All arguments to type relations must be of the type kind but "
        << t << " of " << GetRef<TypeRelation>(op) << " is of kind " << k;
    }
    return Kind::kConstraint;
  }

  Kind VisitType_(const TypeCallNode* op) override {
    // type call func should be a global type var, args should be type
    TypeCall tc = GetRef<TypeCall>(op);
    const auto* gtv = op->func.as<GlobalTypeVarNode>();
    CHECK(gtv != nullptr)
      << "Type call must be calling a global type var";

    Kind func_kind = this->VisitType(op->func);
    CHECK(func_kind == Kind::kAdtHandle)
      << "Type calls must call a global type var that is an ADT handle but "
      << op->func << " of " << tc << " is of kind " << func_kind;

    for (const Type& t : op->args) {
      Kind k = this->VisitType(t);
      CHECK(k == Kind::kType)
        << "Type call arguments must be of the type kind but "
        << t << " of " << tc << " is of kind " << k;
    }

    // finally we need to check the module to check the number of type params
    auto var = GetRef<GlobalTypeVar>(gtv);
    auto data = mod->LookupDef(var);
    CHECK(data->tv.size() == op->args.size())
      << "Incorrect arity in " << tc
      << " Expected: " << data->tv.size()
      << " Given: " << op->args.size();
    return Kind::kType;
  }

  Kind VisitType_(const TypeDataNode* op) override {
    // Constructors can reference the header var, but no other GlobalTypeVars.
    // In theory, a TypeData could be nested, so the header scope
    // should be tracked recursively, but it is unclear that we need
    // to support it.
    TypeData td = GetRef<TypeData>(op);
    Kind header_kind = this->VisitType(op->header);
    CHECK(header_kind == Kind::kAdtHandle)
      << "The header for ADT type data must be an ADT handle but "
      << op->header << " of " << td << " is of kind " << header_kind;

    for (const auto& var : op->tv) {
      Kind k = this->VisitType(var);
      CHECK(k == Kind::kType)
        << "All type params for ADT type data must be of the type kind but "
        << var << " of " << td << " is of kind " << k;
    }

    for (const auto& con : op->constructors) {
      CHECK(con->belong_to.same_as(op->header))
        << "Constructors should have same global type var as type data";

      for (const Type& t : con->inp) {
        Kind k = this->VisitType(t);
        CHECK(k == Kind::kType)
          << "All inputs to a constructor must be of the type kind but"
          << t << " of " << con << " is of kind " << k;
        if (const auto* gtv = t.as<GlobalTypeVarNode>()) {
          CHECK(GetRef<GlobalTypeVar>(gtv).same_as(op->header))
            << "A global type var taken by a constructor must be the one the constructor makes";
        }
      }
    }
    return Kind::kTypeData;
  }

  Kind Check(const Type& t) {
    return this->VisitType(t);
  }
};

Kind KindCheck(const Type& t, const Module& mod) {
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
