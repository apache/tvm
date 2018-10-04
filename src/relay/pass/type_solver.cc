/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_solver.cc
 * \brief Type solver implementations.
 */
#include <string>
#include "type_solver.h"

namespace tvm {
namespace relay {

class TypeSolver::Reporter : public TypeReporterNode {
 public:
  explicit Reporter(TypeSolver* solver)
      : solver_(solver) {}

  void Assign(const Type& dst, const Type& src) final {
    solver_->Unify(dst, src);
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

// Add equality constraint
Type TypeSolver::Unify(const Type& dst, const Type& src) {
  // Known limitation
  // - handle composite types whose component can be unknown.
  // - handle shape pattern matching
  TypeNode* lhs = GetTypeNode(dst);
  TypeNode* rhs = GetTypeNode(src);
  if (lhs->resolved_type.as<IncompleteTypeNode>()) {
    MergeFromTo(lhs, rhs);
    return rhs->resolved_type;
  } else if (rhs->resolved_type.as<IncompleteTypeNode>()) {
    MergeFromTo(rhs, lhs);
    return lhs->resolved_type;
  } else {
    lhs->parent = rhs;
    CHECK(AlphaEqual(lhs->resolved_type, rhs->resolved_type))
        << "Incompatible parent types in UF:"
        << lhs->resolved_type << " and " << rhs->resolved_type;
    return rhs->resolved_type;
  }
}

// Add type constraint to the solver.
void TypeSolver::AddConstraint(const TypeConstraint& constraint) {
  if (auto *op = constraint.as<TypeRelationNode>()) {
    // create a new relation node.
    RelationNode* rnode = make<RelationNode>();
    rnode->rel = GetRef<TypeRelation>(op);
    rel_nodes_.push_back(rnode);
    // populate the type information.
    for (size_t i = 0; i < op->args.size(); ++i) {
      // insert link to the type list
      LinkNode<TypeNode*>* tlink = make<LinkNode<TypeNode*> >();
      TypeNode* tnode = GetTypeNode(op->args[i]);
      tlink->value = tnode;
      rnode->type_list.Push(tlink);
      // insert type->relation node
      LinkNode<RelationNode*>* rlink = make<LinkNode<RelationNode*> >();
      rlink->value = rnode;
      tnode->rel_list.Push(rlink);
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
  auto it = tmap_.find(type);
  if (it != tmap_.end()) {
    return it->second->FindRoot()->resolved_type;
  } else {
    return type;
  }
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
      args.push_back(tlink->value->FindRoot()->resolved_type);
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
        return TypedPackedFunc<void(Type, Type)>([solver](Type lhs, Type rhs) {
            solver->Unify(lhs, rhs);
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
