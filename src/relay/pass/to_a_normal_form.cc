/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file to_anf.cc
 *
 * \brief Turn implicit sharing into observable sharing.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include "let_list.h"
#include "../../common/arena.h"

namespace tvm {
namespace relay {

using common::LinkNode;
using common::LinkedList;

/* DependencyGraph track input and output of an Expr.
 * Additionally, dummy scope is created to model scope.
 * It allow us to traverse the graph in reverse order.
 */
class DependencyGraph {
 public:
  /*! \brief A node in the graph. */
  struct Node {
    bool new_scope = false;
    LinkedList<Node*> input;
    LinkedList<Node*> output;
  };

  /*! \brief The node map that maps node to graph */
  std::unordered_map<Expr, Node*, NodeHash, NodeEqual> expr_node;

  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*!
   * \brief create a dependency graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static DependencyGraph Create(common::Arena* arena, const Expr& body);

 private:
  class Creator;
};

// Creator of DependencyGraph
class DependencyGraph::Creator : private ExprFunctor<void(const Expr& e)> {
 public:
  explicit Creator(common::Arena* arena)
    : arena_(arena) {}

  DependencyGraph Create(const Expr& body) {
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  common::Arena* arena_;
  // The output.
  DependencyGraph graph_;
  // Update the message stored at the node.
  void Depend(DependencyGraph::Node* parent, const Expr& child) {
    VisitExpr(child);

    CHECK_NE(graph_.expr_node.count(child), 0);

    Depend(parent, graph_.expr_node[child]);
  }

  void Depend(DependencyGraph::Node* parent, DependencyGraph::Node* child) {
    auto* parent_link = arena_->make<LinkNode<DependencyGraph::Node*> >();
    parent_link->value = parent;
    child->output.Push(parent_link);

    auto* child_link = arena_->make<LinkNode<DependencyGraph::Node*> >();
    child_link->value = child;
    parent->input.Push(child_link);
  }

  std::unordered_set<Expr, NodeHash, NodeEqual> visited_;

  DependencyGraph::Node* NewNode(bool new_scope) {
    auto* ret = arena_->make<DependencyGraph::Node>();
    ret->new_scope = new_scope;
    return ret;
  }

  void VisitExpr(const Expr& e) final {
    if (visited_.count(e) == 0) {
      if (graph_.expr_node.count(e) == 0) {
        graph_.expr_node[e] = NewNode(false);
      }
      visited_.insert(e);
      ExprFunctor<void(const Expr&)>::VisitExpr(e);
      graph_.post_dfs_order.push_back(graph_.expr_node[e]);
    }
  }

  void VisitExpr_(const CallNode* c) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(c)];
    Depend(n, c->op);
    for (const auto& a : c->args) {
      Depend(n, a);
    }
  }

  void VisitExpr_(const TupleNode* t) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(t)];
    for (const auto& a : t->fields) {
      Depend(n, a);
    }
  }

  void VisitExpr_(const TupleGetItemNode* t) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(t)];
    Depend(n, t->tuple);
  }

  void VisitExpr_(const RefCreateNode* r) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(r)];
    Depend(n, r->value);
  }

  void VisitExpr_(const RefReadNode* r) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(r)];
    Depend(n, r->ref);
  }

  void VisitExpr_(const RefWriteNode* r) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(r)];
    Depend(n, r->ref);
    Depend(n, r->value);
  }

  void VisitExpr_(const IfNode* i) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(i)];
    DependencyGraph::Node* t = NewNode(true);
    DependencyGraph::Node* f = NewNode(true);
    Depend(n, i->cond);
    Depend(n, t);
    Depend(n, f);
    Depend(t, i->true_branch);
    Depend(f, i->false_branch);
    graph_.post_dfs_order.push_back(f);
    graph_.post_dfs_order.push_back(t);
  }

  void VisitExpr_(const FunctionNode* f) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(f)];
    DependencyGraph::Node* b = NewNode(true);
    Depend(n, b);
    Depend(b, f->body);
    graph_.post_dfs_order.push_back(b);
  }

  void VisitExpr_(const LetNode* l) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(l)];
    DependencyGraph::Node* b = NewNode(true);
    Depend(n, b);
    Depend(b, l->value);
    Depend(b, l->body);
    graph_.post_dfs_order.push_back(b);
  }

  void VisitExpr_(const MatchNode* m) final {
    DependencyGraph::Node* n = graph_.expr_node[GetRef<Expr>(m)];
    Depend(n, m->data);
    std::vector<DependencyGraph::Node*> v;
    for (const Clause& c : m->clauses) {
      DependencyGraph::Node* b = NewNode(true);
      Depend(n, b);
      Depend(b, c->rhs);
      v.push_back(b);
    }
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
      graph_.post_dfs_order.push_back(*it);
    }
  }

  void VisitExpr_(const VarNode* v) final { }

  void VisitExpr_(const GlobalVarNode* v) final { }

  void VisitExpr_(const ConstantNode* c) final { }

  void VisitExpr_(const OpNode* o) final { }

  void VisitExpr_(const ConstructorNode* c) final { }
};

DependencyGraph DependencyGraph::Create(common::Arena* arena, const Expr& body) {
  return Creator(arena).Create(body);
}

Expr ToANormalForm(const Expr& e, const Module& m, std::set<GlobalVar>* gv);

struct ScopeNode;
using Scope = std::shared_ptr<ScopeNode>;

/* Invariant: when parent is null level is 0
 *
 * Invariant: when parent is not null level is 1 + parent->level
 */
struct ScopeNode {
  size_t level;
  Scope parent;
  std::shared_ptr<LetList> ll = std::make_shared<LetList>();
  explicit ScopeNode(const Scope& parent) : level(1 + parent->level), parent(parent) { }
  ScopeNode() : level(0) { }
};

Scope ChildScope(const Scope& s) {
  return std::make_shared<ScopeNode>(s);
}

Scope LCA(Scope lhs, Scope rhs) {
  while (lhs != rhs) {
    if (lhs->level > rhs->level) {
      lhs = lhs->parent;
    } else if (lhs->level < rhs->level) {
      rhs = rhs->parent;
    } else {
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }
  return lhs;
}

std::unordered_map<DependencyGraph::Node*, Scope> CalcScope(const DependencyGraph& dg) {
  std::unordered_map<DependencyGraph::Node*, Scope> expr_scope;
  Scope global_scope = std::make_shared<ScopeNode>();
  for (auto it = dg.post_dfs_order.rbegin(); it != dg.post_dfs_order.rend(); ++it) {
    DependencyGraph::Node* n = *it;
    auto iit = n->output.head;
    Scope s;
    if (iit == nullptr) {
      s = global_scope;
    } else {
      s = expr_scope.at(iit->value);
      iit = iit->next;
      for (; iit != nullptr; iit = iit->next) {
        s = LCA(s, expr_scope.at(iit->value));
      }
    }
    expr_scope.insert({n, n->new_scope ? ChildScope(s) : s});
  }
  return expr_scope;
}

bool IsPrimitiveFunction(const Expr& e) {
  return e.as<FunctionNode>() && Downcast<Function>(e)->IsPrimitive();
}

/* Special care is needed to handle local recursion.
 * Fill additionally take a (possibly null) Var argument,
 * If it is not null, Fill is required to bind the transformed result to that var.
 */
class Fill : ExprFunctor<Expr(const Expr&, const Var&)> {
 public:
  static Expr ToANormalForm(const Expr& e,
                            const Module& m,
                            const DependencyGraph& dg,
                            std::unordered_map<DependencyGraph::Node*, Scope>* node_scope,
                            std::set<GlobalVar>* gv) {
    Fill fi(m, dg, node_scope, gv);
    return fi.GetScope(e)->ll->Get(fi.VisitExpr(e));
  }

 private:
  Module mod_;
  const DependencyGraph& dg_;
  std::unordered_map<DependencyGraph::Node*, Scope>* node_scope_;
  std::set<GlobalVar>* visited_;
  std::unordered_map<Expr, Expr, NodeHash, NodeEqual> memo;

  Fill(Module mod,
       const DependencyGraph& dg,
       std::unordered_map<DependencyGraph::Node*, Scope>* node_scope,
       std::set<GlobalVar>* visited) :
    mod_(mod),
    dg_(dg),
    node_scope_(node_scope),
    visited_(visited) { }

  Scope GetScope(const Expr& e) {
    return node_scope_->at(dg_.expr_node.at(e));
  }

  Scope GetSubScope(const Expr& e, size_t i) {
    DependencyGraph::Node* n = dg_.expr_node.at(e);
    auto h = n->input.head;
    while (i != 0) {
      CHECK(h);
      --i;
      h = h->next;
    }
    CHECK(h);
    return node_scope_->at(h->value);
  }

  Expr VisitExpr(const Expr& e, const Var& v) final {
    if (memo.count(e) == 0) {
      memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
    }
    return memo.at(e);
  }

  Expr VisitExpr(const Expr& e) {
    return this->VisitExpr(e, Var());
  }

  Expr Atomic(const Expr& orig, const Expr& now, const Var& v) {
    return v.defined() ? GetScope(orig)->ll->Push(v, now) : now;
  }

  Expr Compound(const Expr& orig, const Expr& now, const Var& v) {
    Var var = v.defined() ?
      v :
      VarNode::make(std::string("x"), IncompleteTypeNode::make(Kind::kType));
    return GetScope(orig)->ll->Push(var, now);
  }

  Expr VisitExpr_(const CallNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return Compound(e, CallNode::make(VisitExpr(c->op), args, c->attrs, c->type_args), v);
  }

  Expr VisitExpr_(const TupleNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    std::vector<Expr> fields;
    for (const auto& a : t->fields) {
      fields.push_back(VisitExpr(a));
    }
    return Compound(e, TupleNode::make(fields), v);
  }

  Expr VisitExpr_(const TupleGetItemNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    return Compound(e, TupleGetItemNode::make(VisitExpr(t->tuple), t->index), v);
  }

  Expr VisitExpr_(const RefCreateNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefCreateNode::make(VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const RefReadNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefReadNode::make(VisitExpr(r->ref)), v);
  }

  Expr VisitExpr_(const RefWriteNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefWriteNode::make(VisitExpr(r->ref), VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const IfNode* i, const Var& v) final {
    Expr e = GetRef<Expr>(i);
    Expr ret = IfNode::make(VisitExpr(i->cond),
                            GetSubScope(e, 1)->ll->Get(VisitExpr(i->true_branch)),
                            GetSubScope(e, 2)->ll->Get(VisitExpr(i->false_branch)));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const FunctionNode* f, const Var& v) final {
    Expr e = GetRef<Expr>(f);
    Expr ret;
    if (IsPrimitiveFunction(e)) {
      ret = e;
    } else {
      ret = FunctionNode::make(f->params,
                               GetSubScope(e, 0)->ll->Get(VisitExpr(f->body)),
                               f->ret_type,
                               f->type_params,
                               f->attrs);
    }
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const LetNode* l, const Var& v) final {
    Expr e = GetRef<Expr>(l);
    VisitExpr(l->value, l->var);
    Expr ret = GetSubScope(e, 0)->ll->Get(VisitExpr(l->body));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const ConstantNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    return Compound(e, e, v);
  }

  Expr VisitExpr_(const VarNode* vn, const Var& v) final {
    Expr e = GetRef<Expr>(vn);
    return Atomic(e, e, v);
  }

  Expr VisitExpr_(const GlobalVarNode* gvn, const Var& v) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    if (visited_->count(gv) == 0) {
      visited_->insert(gv);
      mod_->Update(gv, Downcast<Function>(relay::ToANormalForm(mod_->Lookup(gv), mod_, visited_)));
    }
    return Atomic(gv, gv, v);
  }

  Expr VisitExpr_(const OpNode* op, const Var& v) final {
    Expr e = GetRef<Expr>(op);
    return Atomic(e, e, v);
  }

  Expr VisitExpr_(const ConstructorNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    return Atomic(e, e, v);
  }

  Expr VisitExpr_(const MatchNode* m, const Var& v) final {
    Expr e = GetRef<Expr>(m);
    Expr data = VisitExpr(m->data);
    std::vector<Clause> clauses;
    for (const Clause& c : m->clauses) {
      clauses.push_back(ClauseNode::make(
        c->lhs,
        GetSubScope(e, 1 + clauses.size())->ll->Get(VisitExpr(c->rhs))));
    }
    return Compound(e, MatchNode::make(data, clauses), v);
  }
};

Expr ToANormalFormAux(const Expr& e, const Module& m, std::set<GlobalVar>* gv) {
  /* When you lift a lambda, what is inside is also being lift.
   *
   * So we must determine the scope of the lambda before determining the scope of it's body.
   *
   * To make this more principled,
   * we always determine the scope of parent before determining the scope of children.
   *
   * So we calculate all the dependency between nodes.
   */
  common::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  /* In order to model new subscopes created by lambda, if else and pattern matching,
   * we also assign scope to edge as well.
   * The scope of an edge is either the parent's scope, or a new subscope of the parent's scope.
   *
   * So, the scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   *
   * Every scope additionally contain a LetList which collect all value of that scope.
   * We do an additional pass to fill all the LetList and we are done.
   */
  std::unordered_map<DependencyGraph::Node*, Scope> node_scope = CalcScope(dg);
  return Fill::ToANormalForm(e, m, dg, &node_scope, gv);
}

Expr ToANormalForm(const Expr& e, const Module& m, std::set<GlobalVar>* gv) {
  if (const auto* f = e.as<FunctionNode>()) {
    return FunctionNode::make(f->params,
                              ToANormalFormAux(f->body, m, gv),
                              f->ret_type,
                              f->type_params,
                              f->attrs);
  } else {
    return ToANormalFormAux(e, m, gv);
  }
}

Expr ToANormalForm(const Expr& e, const Module& m) {
  std::set<GlobalVar> gv;
  return ToANormalForm(e, m, &gv);
}

TVM_REGISTER_API("relay._ir_pass.to_a_normal_form")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = ToANormalForm(args[0], args[1]);
  });

}  // namespace relay
}  // namespace tvm
