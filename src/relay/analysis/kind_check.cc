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
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/error.h>

namespace tvm {
namespace relay {

using namespace tvm::runtime;

struct KindChecker : TypeFunctor<Kind(const Type&)> {
  const IRModule& mod;
  Optional<DiagnosticContext> diag_ctx;

  explicit KindChecker(const IRModule& mod, Optional<DiagnosticContext> diag_ctx)
      : mod(mod), diag_ctx(diag_ctx) {}

  void EmitFatal(Diagnostic diagnostic) {
    if (this->diag_ctx) {
      this->diag_ctx.value().EmitFatal(diagnostic);
    } else {
      LOG(FATAL) << diagnostic->message;
    }
  }

  void CheckKindMatches(const Type& t, const Type& outer, Kind expected,
                        const std::string& description) {
    Kind k = this->VisitType(t);
    if (k != expected) {
      EmitFatal(Diagnostic::Error(t->span)
                << "Incorrect kind for a " << description << ". Type " << t << " inside " << outer
                << " is of kind " << k << " but was expected to be " << expected);
    }
  }

  Kind VisitType_(const IncompleteTypeNode* op) override { return op->kind; }

  Kind VisitType_(const TypeVarNode* op) override { return op->kind; }

  Kind VisitType_(const GlobalTypeVarNode* op) override { return op->kind; }

  Kind VisitType_(const TensorTypeNode* op) override { return Kind::kType; }

  Kind VisitType_(const TupleTypeNode* op) override {
    // tuples should only contain normal types
    for (const Type& t : op->fields) {
      CheckKindMatches(t, GetRef<TupleType>(op), Kind::kType, "tuple member");
    }
    return Kind::kType;
  }

  Kind VisitType_(const FuncTypeNode* op) override {
    // Func types should only take normal types for arguments
    // and only return a normal type. They should also have
    // well-formed constraints
    FuncType ft = GetRef<FuncType>(op);
    for (const Type& t : op->arg_types) {
      CheckKindMatches(t, ft, Kind::kType, "function type parameter");
    }

    CheckKindMatches(ft->ret_type, ft, Kind::kType, "function return type");

    for (const TypeConstraint& tc : op->type_constraints) {
      CheckKindMatches(tc, ft, Kind::kConstraint, "function type constraint");
    }

    return Kind::kType;
  }

  Kind VisitType_(const RelayRefTypeNode* op) override {
    // ref types should only contain normal types
    RelayRefType rt = GetRef<RelayRefType>(op);
    CheckKindMatches(op->value, rt, Kind::kType, "ref contents");
    return Kind::kType;
  }

  Kind VisitType_(const TypeRelationNode* op) override {
    // arguments to type relation should be normal types
    for (const Type& t : op->args) {
      CheckKindMatches(t, GetRef<TypeRelation>(op), Kind::kType, "argument to type relation");
    }
    return Kind::kConstraint;
  }

  Kind VisitType_(const TypeCallNode* op) override {
    // type call func should be a global type var, args should be type
    TypeCall tc = GetRef<TypeCall>(op);
    const auto* gtv = op->func.as<GlobalTypeVarNode>();
    if (gtv == nullptr) {
      EmitFatal(Diagnostic::Error(op->span)
                << "The callee in " << tc << " is not a global type var, but is " << op->func);
    }

    CheckKindMatches(op->func, tc, Kind::kAdtHandle, "type call function");

    for (const Type& t : op->args) {
      CheckKindMatches(t, tc, Kind::kType, "type call argument");
    }

    // finally we need to check the module to check the number of type params
    auto var = GetRef<GlobalTypeVar>(gtv);
    try {
      auto data = mod->LookupTypeDef(var);

      if (data->type_vars.size() != op->args.size()) {
        EmitFatal(Diagnostic::Error(op->span)
                  << "Expected " << data->type_vars.size() << "arguments for " << tc << "; got "
                  << op->args.size());
      }
    } catch (const Error& err) {
      // TODO(@jroesch): can probably relax to just emit
      EmitFatal(Diagnostic::Error(op->span)
                << "the type variable : `" << var->name_hint << "` is undefined");
    }

    return Kind::kType;
  }

  Kind VisitType_(const TypeDataNode* op) override {
    // Constructors can reference the header var, but no other GlobalTypeVars.
    // In theory, a TypeData could be nested, so the header scope
    // should be tracked recursively, but it is unclear that we need
    // to support it.
    TypeData td = GetRef<TypeData>(op);
    CheckKindMatches(op->header, td, Kind::kAdtHandle, "type data header");

    for (const auto& var : op->type_vars) {
      CheckKindMatches(var, td, Kind::kType, "ADT type var");
    }

    for (const auto& con : op->constructors) {
      if (!con->belong_to.same_as(op->header)) {
        EmitFatal(Diagnostic::Error(op->span) << con << " has header " << con->belong_to << " but "
                                              << op << " has header " << op->header);
      }

      for (const Type& t : con->inputs) {
        CheckKindMatches(t, td, Kind::kType, "ADT constructor input");
      }
    }
    return Kind::kTypeData;
  }

  Kind Check(const Type& t) { return this->VisitType(t); }
};

Kind KindCheck(const Type& t, const IRModule& mod, Optional<DiagnosticContext> diag_ctx) {
  KindChecker kc(mod, diag_ctx);
  return kc.Check(t);
}

TVM_REGISTER_GLOBAL("relay.analysis.check_kind").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 1) {
    *ret = KindCheck(args[0], IRModule({}, {}));
  } else if (args.size() == 2) {
    *ret = KindCheck(args[0], args[1], Optional<DiagnosticContext>());
  } else {
    *ret = KindCheck(args[0], args[1], args[2]);
  }
});

}  // namespace relay
}  // namespace tvm
