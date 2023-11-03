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
 * \file type_solver.cc
 * \brief Type solver implementations.
 */
#include "type_solver.h"

#include <tvm/ir/type_functor.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace tvm {
namespace relay {

class TypeSolver::Reporter : public TypeReporterNode {
 public:
  explicit Reporter(TypeSolver* solver) : solver_(solver) {}

  void Assign(const Type& dst, const Type& src) final { solver_->Unify(dst, src, span); }

  bool Assert(const IndexExpr& cond) final {
    if (const int64_t* pdiff = tir::as_const_int(cond)) {
      return pdiff[0];
    }
    return true;
  }

  bool AssertEQ(const IndexExpr& lhs, const IndexExpr& rhs) final {
    // early warning constant case.
    IndexExpr diff = lhs - rhs;
    if (const int64_t* pdiff = tir::as_const_int(diff)) {
      return pdiff[0] == 0;
    }
    return true;
  }

  TVM_DLL void SetSpan(const Span& span) final { this->span = span; }

  TVM_DLL Span GetSpan() final { return this->span; }

  TVM_DLL DiagnosticContext GetDiagCtx() final { return this->solver_->diag_ctx_; }

  // TVM_DLL void Emit(Diagnostic diagnostic) final {
  //   return this->solver_->
  // }

  TVM_DLL IRModule GetModule() final { return this->solver_->module_; }

 private:
  /*! \brief The span to report unification errors at. */
  mutable Span span;

  TypeSolver* solver_;
};

class TypeSolver::AnyChecker : public tir::ExprVisitor {
 public:
  void VisitExpr_(const AnyNode* op) final { found_ = true; }

  bool Check(const PrimExpr& expr) {
    tir::ExprVisitor::VisitExpr(expr);
    return found_;
  }

 private:
  bool found_{false};
};

class TypeSolver::OccursChecker : public TypeVisitor {
 public:
  explicit OccursChecker(TypeSolver* solver, TypeNode* var)
      : solver_(solver), var_(var), found_(false) {}

  bool Check(const Type& t) {
    VisitType(t);
    return found_;
  }

  void VisitType_(const IncompleteTypeNode* op) override {
    IncompleteType t = GetRef<IncompleteType>(op);
    TypeNode* node = solver_->GetTypeNode(t);
    found_ = found_ || (var_->FindRoot() == node->FindRoot());
  }

 private:
  TypeSolver* solver_;
  TypeNode* var_;
  bool found_;
};

class TypeSolver::Unifier : public TypeFunctor<Type(const Type&, const Type&)> {
 public:
  explicit Unifier(TypeSolver* solver, const Span& span) : solver_(solver), span(span) {}

  Type Unify(const Type& lhs_type, const Type& rhs_type, bool assign_lhs = true,
             bool assign_rhs = true) {
    // Known limitation
    // - handle shape pattern matching
    TypeNode* lhs = solver_->GetTypeNode(lhs_type);
    TypeNode* rhs = solver_->GetTypeNode(rhs_type);

    // do occur check so we don't create self-referencing structure
    if (lhs->FindRoot() == rhs->FindRoot()) {
      return lhs->resolved_type;
    }

    if (lhs->resolved_type.as<IncompleteTypeNode>()) {
      ICHECK(!OccursCheck(lhs, rhs->resolved_type))
          << "Incomplete type " << lhs->resolved_type << " occurs in " << rhs->resolved_type
          << ", cannot unify";

      solver_->MergeFromTo(lhs, rhs);
      return rhs->resolved_type;
    } else if (rhs->resolved_type.as<IncompleteTypeNode>()) {
      ICHECK(!OccursCheck(rhs, lhs->resolved_type))
          << "Incomplete type " << rhs->resolved_type << " occurs in " << lhs->resolved_type
          << ", cannot unify";
      solver_->MergeFromTo(rhs, lhs);
      return lhs->resolved_type;
    } else {
      Type resolved = this->VisitType(rhs->resolved_type, lhs->resolved_type);

      if (!resolved.defined()) {
        solver_->Emit(Diagnostic::Error(this->span)
                      << "The Relay type checker is unable to show the following types match.\n"
                      << "In particular "
                      << "`" << PrettyPrint(lhs->resolved_type) << "` does not match `"
                      << PrettyPrint(rhs->resolved_type) << "`");
        return lhs->resolved_type;
      } else {
        TypeNode* top = solver_->GetTypeNode(resolved);
        if (assign_lhs) solver_->MergeFromTo(lhs, top);
        if (assign_rhs) solver_->MergeFromTo(rhs, top);
        return resolved;
      }
    }
  }

  bool HasAny(const PrimExpr& expr) {
    AnyChecker ac;
    return ac.Check(expr);
  }

  // Checks whether lhs (taken to be a type var) occurs in t, meaning
  // there is a recursive equality constraint, which should be rejected.
  // N.b.: A tautology like ?a = ?a is okay and should be checked for
  // *before* calling this method
  //
  // See: https://en.wikipedia.org/wiki/Occurs_check
  bool OccursCheck(TypeNode* lhs, const Type& t) {
    OccursChecker rc(solver_, lhs);
    return rc.Check(t);
  }

  // default: unify only if structural-equal
  Type VisitTypeDefault_(const Object* op, const Type& tn) final {
    ObjectRef nr = GetRef<ObjectRef>(op);
    Type t1 = Downcast<Type>(nr);
    if (!tvm::StructuralEqual()(t1, tn)) {
      return Type(nullptr);
    }
    return t1;
  }

  IndexExpr GetShape(const IndexExpr& e) {
    IndexExpr ex = e;
    while (true) {
      auto it = solver_->shape_uf_.find(ex);
      if (it == solver_->shape_uf_.end()) {
        return ex;
      } else {
        ex = (*it).second;
      }
    }
  }

  IndexExpr UnifyDim(const IndexExpr& lhs, const IndexExpr& rhs) {
    auto ulhs = GetShape(lhs);
    auto urhs = GetShape(rhs);

    if (ulhs.same_as(urhs)) {
      return ulhs;
    }
    if (HasAny(ulhs) || HasAny(urhs)) {
      return Any();
    }

    auto left_index0 = ulhs.as<tvm::tir::VarNode>();
    auto right_index0 = urhs.as<tvm::IntImmNode>();
    if (left_index0 && right_index0) {
      solver_->shape_uf_.Set(ulhs, urhs);
      return urhs;
    }

    auto left_index1 = ulhs.as<tvm::IntImmNode>();
    auto right_index1 = urhs.as<tvm::tir::VarNode>();
    if (left_index1 && right_index1) {
      solver_->shape_uf_.Set(urhs, ulhs);
      return ulhs;
    }

    auto left_index2 = ulhs.as<tvm::IntImmNode>();
    auto right_index2 = urhs.as<tvm::IntImmNode>();
    if (left_index2 && right_index2 && left_index2->value == right_index2->value) {
      return ulhs;
    }

    return tvm::PrimExpr();
  }

  Type VisitType_(const TensorTypeNode* op, const Type& tn) final {
    const auto* tt_node = tn.as<TensorTypeNode>();
    if (!tt_node) {
      return Type(nullptr);
    }

    auto tt1 = GetRef<TensorType>(op);
    auto tt2 = GetRef<TensorType>(tt_node);

    if (tvm::StructuralEqual()(tt1, tt2)) {
      return std::move(tt1);
    }

    if (tt1->dtype != tt2->dtype) {
      return Type(nullptr);
    }

    tvm::Array<IndexExpr> shape;
    if (tt1->shape.size() != tt2->shape.size()) {
      this->solver_->Emit(Diagnostic::Error(this->span)
                          << "tensor type `" << PrettyPrint(tt1) << "` has " << tt1->shape.size()
                          << " dimensions, while `" << PrettyPrint(tt2) << "` has "
                          << tt2->shape.size() << " dimensions");
      return Type(nullptr);
    }

    std::vector<std::tuple<size_t, IndexExpr, IndexExpr>> mismatches;

    ICHECK_EQ(tt1->shape.size(), tt2->shape.size());
    for (size_t i = 0; i < tt1->shape.size(); i++) {
      auto dim = UnifyDim(tt1->shape[i], tt2->shape[i]);
      if (!dim.defined()) {
        // NB: We push an arbitrary dimension here so we can continue error propagation.
        shape.push_back(tt1->shape[i]);
        tvm::PrimExpr shape1 = tt1->shape[i];
        tvm::PrimExpr shape2 = tt2->shape[i];
        std::tuple<int, IndexExpr, IndexExpr> tuple = std::make_tuple(i, shape1, shape2);
        mismatches.push_back(tuple);
      } else {
        shape.push_back(dim);
      }
    }

    if (mismatches.size() != 0) {
      auto err = Diagnostic::Error(this->span);
      err << "The Relay type checker is unable to show the following types match:\n"
          << "  " << PrettyPrint(tt1) << "\n"
          << "  " << PrettyPrint(tt2) << "\n";
      err << "In particular:\n";
      for (auto mismatch : mismatches) {
        err << "  dimension " << std::get<0>(mismatch) << " conflicts: " << std::get<1>(mismatch)
            << " does not match " << std::get<2>(mismatch) << ".";
      }
      this->solver_->Emit(err);
      return Type(nullptr);
    }

    return TensorType(shape, tt1->dtype);
  }

  Type VisitType_(const TupleTypeNode* op, const Type& tn) final {
    const auto* ttn = tn.as<TupleTypeNode>();
    if (!ttn || op->fields.size() != ttn->fields.size()) {
      return Type(nullptr);
    }

    TupleType tt1 = GetRef<TupleType>(op);
    TupleType tt2 = GetRef<TupleType>(ttn);

    std::vector<Type> new_fields;
    for (size_t i = 0; i < tt1->fields.size(); i++) {
      Type field = Unify(tt1->fields[i], tt2->fields[i]);
      new_fields.push_back(field);
    }
    return TupleType(new_fields);
  }

  Type VisitType_(const FuncTypeNode* op, const Type& tn) final {
    const auto* ftn = tn.as<FuncTypeNode>();
    if (!ftn || op->arg_types.size() != ftn->arg_types.size() ||
        op->type_constraints.size() != ftn->type_constraints.size()) {
      return Type(nullptr);
    }

    // without loss of generality, suppose op->type_params.size() >= ftn->type_params.size().
    if (op->type_params.size() < ftn->type_params.size()) {
      return VisitType_(ftn, GetRef<FuncType>(op));
    }

    // remap type vars so they match
    Map<TypeVar, Type> subst_map;
    tvm::Array<TypeVar> ft_type_params;
    for (size_t i = 0; i < ftn->type_params.size(); ++i) {
      subst_map.Set(op->type_params[i], ftn->type_params[i]);
      ft_type_params.push_back(op->type_params[i]);
    }

    for (size_t i = ftn->type_params.size(); i < op->type_params.size(); ++i) {
      subst_map.Set(op->type_params[i], IncompleteType(kType));
    }

    FuncType ft = FuncType(op->arg_types, op->ret_type, ft_type_params, op->type_constraints);
    auto ft1 = Downcast<FuncType>(Bind(ft, subst_map));
    auto ft2 = GetRef<FuncType>(ftn);

    Type ret_type = Unify(ft1->ret_type, ft2->ret_type);

    std::vector<Type> arg_types;
    for (size_t i = 0; i < ft2->arg_types.size(); ++i) {
      Type arg_type = Unify(ft1->arg_types[i], ft2->arg_types[i]);
      arg_types.push_back(arg_type);
    }

    std::vector<TypeConstraint> type_constraints;
    for (size_t i = 0; i < ft1->type_constraints.size(); ++i) {
      Type unified_constraint = Unify(ft1->type_constraints[i], ft2->type_constraints[i]);
      const auto* tcn = unified_constraint.as<TypeConstraintNode>();
      ICHECK(tcn) << "Two type constraints unified into a non-constraint?"
                  << ft1->type_constraints[i] << " and " << ft2->type_constraints[i];
      type_constraints.push_back(GetRef<TypeConstraint>(tcn));
    }

    return FuncType(arg_types, ret_type, ft2->type_params, type_constraints);
  }

  Type VisitType_(const RelayRefTypeNode* op, const Type& tn) final {
    const auto* rtn = tn.as<RelayRefTypeNode>();
    if (!rtn) {
      return Type(nullptr);
    }
    return RelayRefType(Unify(op->value, rtn->value));
  }

  Type VisitType_(const TypeCallNode* op, const Type& tn) override {
    const auto* tcn = tn.as<TypeCallNode>();
    if (!tcn || tcn->args.size() != op->args.size()) {
      return Type();
    }

    Type func = Unify(op->func, tcn->func);
    tvm::Array<Type> args;
    for (size_t i = 0; i < op->args.size(); i++) {
      args.push_back(Unify(op->args[i], tcn->args[i]));
    }
    return TypeCall(func, args);
  }

 private:
  TypeSolver* solver_;
  Span span;
};

class TypeSolver::Resolver : public TypeMutator {
 public:
  explicit Resolver(TypeSolver* solver) : solver_(solver) {}

  Type Resolve(const Type& t) {
    if (!t.defined()) {
      return t;
    }
    return VisitType(t);
  }

  Type VisitType_(const IncompleteTypeNode* op) override {
    auto* node = solver_->GetTypeNode(GetRef<IncompleteType>(op));
    return node->resolved_type;
  }

 private:
  TypeSolver* solver_;
};

// It ends up being more compact to simply have TypeFunctor<void(const Type&) than
// a TypeVisitor because we can use the default case to dispense with
// most of the overrides.
class TypeSolver::Propagator : public TypeFunctor<void(const Type&)> {
 public:
  explicit Propagator(TypeSolver* solver, const std::unordered_set<RelationNode*>* rels)
      : solver_(solver), rels_(rels) {}

  // adds the relation node to t and all child types of t
  void Propagate(const Type& t) { VisitType(t); }

  void UpdateRelSet(const Type& t) {
    TypeNode* tnode = solver_->GetTypeNode(t);
    for (auto* rel : *rels_) {
      tnode->rel_set.insert(rel);
    }
  }

  void VisitTypeDefault_(const Object* op) override {
    ObjectRef nr = GetRef<ObjectRef>(op);
    Type t = Downcast<Type>(nr);
    UpdateRelSet(t);
  }

  void VisitType_(const TupleTypeNode* op) override {
    TupleType tt = GetRef<TupleType>(op);
    UpdateRelSet(tt);

    for (const Type& t : tt->fields) {
      Propagate(t);
    }
  }

  void VisitType_(const FuncTypeNode* op) override {
    FuncType ft = GetRef<FuncType>(op);
    UpdateRelSet(ft);

    Propagate(ft->ret_type);
    for (auto arg_type : ft->arg_types) {
      Propagate(arg_type);
    }

    for (auto type_param : ft->type_params) {
      Propagate(type_param);
    }

    for (auto type_cs : ft->type_constraints) {
      Propagate(type_cs);
    }
  }

  void VisitType_(const TypeCallNode* op) override {
    TypeCall tc = GetRef<TypeCall>(op);
    UpdateRelSet(tc);

    Propagate(tc->func);
    for (auto arg : tc->args) {
      Propagate(arg);
    }
  }

 private:
  TypeSolver* solver_;
  const std::unordered_set<RelationNode*>* rels_;
};

// similarly, we use TypeFunctor<void(const Type&)> so we can use
// the default visitor case to avoid more overrides
class TypeSolver::Merger : public TypeFunctor<void(const Type&)> {
 public:
  explicit Merger(TypeSolver* solver) : solver_(solver) {}

  // Merges src node to dst, ensures *all* type relations of all
  // child nodes of src are transferred to dst.
  void Merge(TypeNode* src, TypeNode* dst) {
    if (src == dst) return;
    dst_ = dst;
    VisitType(src->resolved_type);
    // set parent at the end so later calls to GetTypeNode go back to src
    src->parent = dst;

    // now propagate relations to child nodes, since change to
    // a child node should update parent too
    Propagator prop(solver_, &dst->rel_set);
    prop.Propagate(dst->resolved_type);
  }

  // Transfers any relations linked to t to the stored dst.
  // Any unresolved relations are added back to the queue, since
  // there is now new information
  void TransferLinks(const Type& t) {
    TypeNode* src = solver_->GetTypeNode(t);
    if (src == dst_) return;
    for (auto* rel : src->rel_set) {
      // if the relation is not yet resolved, add to queue
      if (!rel->resolved) {
        solver_->AddToQueue(rel);
        dst_->rel_set.insert(rel);
      }
    }
  }

  void VisitTypeDefault_(const Object* op) override {
    ObjectRef nr = GetRef<ObjectRef>(op);
    Type t = Downcast<Type>(nr);
    TransferLinks(t);
  }

  void VisitType_(const TupleTypeNode* ttn) override {
    auto tup = GetRef<TupleType>(ttn);
    TransferLinks(tup);

    for (auto field : tup->fields) {
      VisitType(field);
    }
  }

  void VisitType_(const FuncTypeNode* ftn) override {
    auto func = GetRef<FuncType>(ftn);
    TransferLinks(func);

    VisitType(func->ret_type);
    for (auto arg : func->arg_types) {
      VisitType(arg);
    }
    for (auto param : func->type_params) {
      VisitType(param);
    }
    for (auto constraint : func->type_constraints) {
      VisitType(constraint);
    }
  }

 private:
  TypeSolver* solver_;
  TypeNode* dst_;
};

// constructor
TypeSolver::TypeSolver(const GlobalVar& current_func, DiagnosticContext diag_ctx)
    : reporter_(make_object<Reporter>(this)),
      current_func_(current_func),
      diag_ctx_(diag_ctx),
      module_(diag_ctx->module) {
  ICHECK(module_.defined());
}

// destructor
TypeSolver::~TypeSolver() {
  // call destructor of all non-POD arena object
  for (TypeNode* ptr : type_nodes_) {
    ptr->~TypeNode();
  }
  for (RelationNode* ptr : rel_nodes_) {
    ptr->~RelationNode();
  }
}

// merge src type node to dst
void TypeSolver::MergeFromTo(TypeNode* src, TypeNode* dst) {
  Merger merger(this);
  merger.Merge(src, dst);
}

// Add equality constraint
Type TypeSolver::Unify(const Type& dst, const Type& src, const Span& span, bool assign_lhs,
                       bool assign_rhs) {
  Unifier unifier(this, span);
  return unifier.Unify(dst, src, assign_lhs, assign_rhs);
}

// Add type constraint to the solver.
void TypeSolver::AddConstraint(const TypeConstraint& constraint, const Span& span) {
  if (const auto* op = constraint.as<TypeRelationNode>()) {
    // create a new relation node.
    RelationNode* rnode = arena_.make<RelationNode>();
    rnode->span = span;
    rnode->rel = GetRef<TypeRelation>(op);
    rel_nodes_.push_back(rnode);
    // populate the type information.
    for (size_t i = 0; i < op->args.size(); ++i) {
      // insert link to the type list
      LinkNode<TypeNode*>* tlink = arena_.make<LinkNode<TypeNode*>>();
      TypeNode* tnode = GetTypeNode(op->args[i]);
      tlink->value = tnode;
      rnode->type_list.Push(tlink);
      // insert type->relation node
      std::unordered_set<RelationNode*> singleton{rnode};
      Propagator prop(this, &singleton);
      prop.Propagate(tnode->resolved_type);
    }
    // add the relation to the working queue.
    this->AddToQueue(rnode);
  } else {
    LOG(FATAL) << "Do not know how to handle constraint type" << constraint->GetTypeKey();
  }
}

// Resolve a type in the solver context.
Type TypeSolver::Resolve(const Type& type) {
  Resolver resolver(this);
  auto it = tmap_.find(type);
  Type t = (it != tmap_.end()) ? it->second->FindRoot()->resolved_type : type;
  return resolver.Resolve(t);
}

bool TypeSolver::Solve() {
  while (!update_queue_.empty()) {
    RelationNode* rnode = update_queue_.front();
    const auto& rel = rnode->rel;
    update_queue_.pop();
    ICHECK(!rnode->resolved);
    // update the relation with given evidence.
    Array<Type> args;
    for (auto* tlink = rnode->type_list.head; tlink != nullptr; tlink = tlink->next) {
      args.push_back(Resolve(tlink->value->FindRoot()->resolved_type));
      ICHECK_LE(args.size(), rel->args.size());
    }

    // We need to set this in order to understand where unification
    // errors generated by the error reporting are coming from.
    reporter_->SetSpan(rnode->span);

    try {
      // Call the Type Relation's function.
      bool resolved = rel->func(args, rel->num_inputs, rel->attrs, reporter_);

      if (resolved) {
        ++num_resolved_rels_;
      }

      rnode->resolved = resolved;
    } catch (const CompileError& err) {
      this->Emit(Diagnostic::Error(rnode->span) << err.what());
      rnode->resolved = false;
    }

    // Mark inqueue as false after the function call
    // so that rnode itself won't get enqueued again.
    rnode->inqueue = false;
  }

  // This criterion is not necessarily right for all the possible cases
  // TODO(tqchen): We should also count the number of in-complete types.
  return num_resolved_rels_ == rel_nodes_.size();
}

// Expose type solver only for debugging purposes.
TVM_REGISTER_GLOBAL("relay.analysis._test_type_solver")
    .set_body([](runtime::TVMArgs args, runtime::TVMRetValue* ret) {
      using runtime::PackedFunc;
      using runtime::TypedPackedFunc;
      auto module = IRModule({}, {});
      DiagnosticContext diag_ctx = DiagnosticContext::Default(module);
      auto dummy_fn_name = GlobalVar("test");
      module->Add(dummy_fn_name, Function({}, Tuple(tvm::Array<relay::Expr>({})), Type(), {}, {}));
      auto solver = std::make_shared<TypeSolver>(dummy_fn_name, diag_ctx);

      auto mod = [module, solver, diag_ctx](std::string name) -> PackedFunc {
        if (name == "Solve") {
          return TypedPackedFunc<bool()>([solver]() { return solver->Solve(); });
        } else if (name == "Unify") {
          return TypedPackedFunc<Type(Type, Type)>([module, solver, diag_ctx](Type lhs, Type rhs) {
            auto res = solver->Unify(lhs, rhs, Span());
            DiagnosticContext ctx = diag_ctx;
            ctx.Render();
            return res;
          });
        } else if (name == "Resolve") {
          return TypedPackedFunc<Type(Type)>([solver](Type t) { return solver->Resolve(t); });
        } else if (name == "AddConstraint") {
          return TypedPackedFunc<void(TypeConstraint)>([solver](TypeConstraint c) {
            Expr e = Var("dummy_var", IncompleteType(Kind::kType), Span(SourceName(), 0, 0, 0, 0));
            return solver->AddConstraint(c, e->span);
          });
        } else {
          return PackedFunc();
        }
      };
      *ret = runtime::TypedPackedFunc<runtime::PackedFunc(std::string)>(mod);
    });

}  // namespace relay
}  // namespace tvm
