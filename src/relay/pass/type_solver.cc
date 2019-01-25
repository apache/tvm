/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_solver.cc
 * \brief Type solver implementations.
 */
#include <string>
#include "type_solver.h"
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

class TypeSolver::Reporter : public TypeReporterNode {
 public:
  explicit Reporter(TypeSolver* solver)
      : solver_(solver) {}

  void Assign(const Type& dst, const Type& src) final {
    solver_->Unify(dst, src);
  }

  bool Assert(const IndexExpr& cond) final {
    if (const uint64_t* pdiff = as_const_uint(cond)) {
      return pdiff[0];
    }
    return true;
  }

  bool AssertEQ(const IndexExpr& lhs, const IndexExpr& rhs) final {
    // early warning constant case.
    IndexExpr diff = lhs - rhs;
    if (const int64_t* pdiff = as_const_int(diff)) {
      return pdiff[0] == 0;
    }
    return true;
  }

 private:
  TypeSolver* solver_;
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
  explicit Unifier(TypeSolver* solver) : solver_(solver) {}

  Type Unify(const Type& src, const Type& dst) {
    // Known limitation
    // - handle shape pattern matching
    TypeNode* lhs = solver_->GetTypeNode(dst);
    TypeNode* rhs = solver_->GetTypeNode(src);

    // do occur check so we don't create self-referencing structure
    if (lhs->FindRoot() == rhs->FindRoot()) {
      return lhs->resolved_type;
    }
    if (lhs->resolved_type.as<IncompleteTypeNode>()) {
      CHECK(!CheckOccurs(lhs, rhs->resolved_type))
        << "Incomplete type " << lhs->resolved_type << " occurs in "
        << rhs->resolved_type << ", cannot unify";
      solver_->MergeFromTo(lhs, rhs);
      return rhs->resolved_type;
    } else if (rhs->resolved_type.as<IncompleteTypeNode>()) {
      CHECK(!CheckOccurs(rhs, lhs->resolved_type))
        << "Incomplete type " << rhs->resolved_type << " occurs in "
        << lhs->resolved_type << ", cannot unify";
      solver_->MergeFromTo(rhs, lhs);
      return lhs->resolved_type;
    } else {
      Type resolved = this->VisitType(lhs->resolved_type, rhs->resolved_type);
      CHECK(resolved.defined())
        << "Unable to unify parent types: "
        << lhs->resolved_type << " and " << rhs->resolved_type;
      TypeNode* top = solver_->GetTypeNode(resolved);
      solver_->MergeFromTo(lhs, top);
      solver_->MergeFromTo(rhs, top);
      return resolved;
    }
  }

  // Checks whether lhs (taken to be a type var) occurs in t, meaning
  // there is a recursive equality constraint, which should be rejected.
  // N.b.: A tautology like ?a = ?a is okay and should be checked for
  // *before* calling this method
  bool CheckOccurs(TypeNode* lhs, const Type& t) {
    OccursChecker rc(solver_, lhs);
    return rc.Check(t);
  }

  // default: unify only if alpha-equal
  Type VisitTypeDefault_(const Node* op, const Type& tn) override {
    NodeRef nr = GetRef<NodeRef>(op);
    Type t1 = GetRef<Type>(nr.as_derived<tvm::relay::TypeNode>());
    if (!AlphaEqual(t1, tn)) {
      return Type(nullptr);
    }
    return t1;
  }

  Type VisitType_(const TupleTypeNode* op, const Type& tn) override {
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
    return TupleTypeNode::make(new_fields);
  }

  Type VisitType_(const FuncTypeNode* op, const Type& tn) override {
    const auto* ftn = tn.as<FuncTypeNode>();
    if (!ftn
        || op->arg_types.size() != ftn->arg_types.size()
        || op->type_params.size() != ftn->type_params.size()
        || op->type_constraints.size() != ftn->type_constraints.size()) {
      return Type(nullptr);
    }

    // remap type vars so they match
    Map<TypeVar, Type> subst_map;
    for (size_t i = 0; i < op->type_params.size(); i++) {
      subst_map.Set(ftn->type_params[i], op->type_params[i]);
    }

    auto ft1 = GetRef<FuncType>(op);
    auto ft2 = Downcast<FuncType>(Bind(GetRef<FuncType>(ftn), subst_map));

    Type ret_type = Unify(ft1->ret_type, ft2->ret_type);

    std::vector<Type> arg_types;
    for (size_t i = 0; i < ft1->arg_types.size(); i++) {
      Type arg_type = Unify(ft1->arg_types[i], ft2->arg_types[i]);
      arg_types.push_back(arg_type);
    }

    std::vector<TypeConstraint> type_constraints;
    for (size_t i = 0; i < ft1->type_constraints.size(); i++) {
      Type unified_constraint = Unify(ft1->type_constraints[i],
                                      ft2->type_constraints[i]);
      const auto* tcn = unified_constraint.as<TypeConstraintNode>();
      CHECK(tcn) << "Two type constraints unified into a non-constraint?"
                 << ft1->type_constraints[i] << " and " << ft2->type_constraints[i];
      type_constraints.push_back(GetRef<TypeConstraint>(tcn));
    }

    return FuncTypeNode::make(arg_types, ret_type, ft1->type_params, type_constraints);
  }

 private:
  TypeSolver* solver_;
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
  void Propagate(const Type& t) {
    VisitType(t);
  }

  void UpdateRelSet(const Type& t) {
    TypeNode* tnode = solver_->GetTypeNode(t);
    for (auto* rel : *rels_) {
      tnode->rel_set.insert(rel);
    }
  }

  void VisitTypeDefault_(const Node* op) override {
    NodeRef nr = GetRef<NodeRef>(op);
    Type t = GetRef<Type>(nr.as_derived<tvm::relay::TypeNode>());
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

  void VisitTypeDefault_(const Node* op) override {
    NodeRef nr = GetRef<NodeRef>(op);
    Type t = GetRef<Type>(nr.as_derived<tvm::relay::TypeNode>());
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
TypeSolver::TypeSolver()
  : reporter_(make_node<Reporter>(this)) {
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
Type TypeSolver::Unify(const Type& dst, const Type& src) {
  Unifier unifier(this);
  return unifier.Unify(dst, src);
}

// Add type constraint to the solver.
void TypeSolver::AddConstraint(const TypeConstraint& constraint) {
  if (auto *op = constraint.as<TypeRelationNode>()) {
    // create a new relation node.
    RelationNode* rnode = arena_.make<RelationNode>();
    rnode->rel = GetRef<TypeRelation>(op);
    rel_nodes_.push_back(rnode);
    // populate the type information.
    for (size_t i = 0; i < op->args.size(); ++i) {
      // insert link to the type list
      LinkNode<TypeNode*>* tlink = arena_.make<LinkNode<TypeNode*> >();
      TypeNode* tnode = GetTypeNode(op->args[i]);
      tlink->value = tnode;
      rnode->type_list.Push(tlink);
      // insert type->relation node
      std::unordered_set<RelationNode*> singleton { rnode };
      Propagator prop(this, &singleton);
      prop.Propagate(tnode->resolved_type);
    }
    // add the relation to the working queue.
    this->AddToQueue(rnode);
  } else {
    LOG(FATAL) << "Do not know how to handle constraint type"
               << constraint->type_key();
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
  // update until queue is empty
  while (!update_queue_.empty()) {
    RelationNode* rnode = update_queue_.front();
    const auto& rel = rnode->rel;
    update_queue_.pop();
    CHECK(!rnode->resolved);
    // update the relation with given evidence.
    Array<Type> args;
    for (auto* tlink = rnode->type_list.head; tlink != nullptr; tlink = tlink->next) {
      args.push_back(Resolve(tlink->value->FindRoot()->resolved_type));
      CHECK_LE(args.size(), rel->args.size());
    }
    // call the function
    bool resolved = rel->func(args, rel->num_inputs, rel->attrs, reporter_);
    // mark inqueue as false after the function call
    // so that rnode itself won't get enqueued again.
    rnode->inqueue = false;

    if (resolved) {
      ++num_resolved_rels_;
    }
    rnode->resolved = resolved;
  }
  // This criterion is not necessarily right for all the possible cases
  // TODO(tqchen): We should also count the number of in-complete types.
  return num_resolved_rels_ == rel_nodes_.size();
}


// Expose type solver only for debugging purposes.
TVM_REGISTER_API("relay._ir_pass._test_type_solver")
.set_body([](runtime::TVMArgs args, runtime::TVMRetValue* ret) {
    using runtime::PackedFunc;
    using runtime::TypedPackedFunc;
    auto solver = std::make_shared<TypeSolver>();

    auto mod = [solver](std::string name) -> PackedFunc {
      if (name == "Solve") {
        return TypedPackedFunc<bool()>([solver]() {
            return solver->Solve();
          });
      } else if (name == "Unify") {
        return TypedPackedFunc<Type(Type, Type)>([solver](Type lhs, Type rhs) {
            return solver->Unify(lhs, rhs);
          });
      } else if (name == "Resolve") {
        return TypedPackedFunc<Type(Type)>([solver](Type t) {
            return solver->Resolve(t);
          });
      } else if (name == "AddConstraint") {
        return TypedPackedFunc<void(TypeConstraint)>([solver](TypeConstraint c) {
            return solver->AddConstraint(c);
          });
      } else {
        return PackedFunc();
      }
    };
    *ret = runtime::TypedPackedFunc<runtime::PackedFunc(std::string)>(mod);
  });

}  // namespace relay
}  // namespace tvm
