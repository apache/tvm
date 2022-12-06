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
 * \file src/relay/transforms/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

/*
  Note on Fusing algorithm:

  The main challenge of general fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  However, at the point of conv2d we do not necessarily know that all the future paths
  will merge at the elemwise add. The fusion algorithm applies post-dominator analysis.

  The immediate post-dominator of a node defined by the closest node where all the future path goes
  into. In the above case, the elemwise add is the post-dominator of conv2d. The general algorithm
  is as follows:

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.

  Note that, because we run analysis on a DAG, we use a single pass post-dominator
  tree construction algorithm via LCA, which is simpler than the full version that handles cycles.

  The fusion algorithm traverses from each node and checks if it can be fused to its
  immediate post dominator. It has to check the following things:

  - CheckPath: check all the path between a node and its immediate post-dominator
               satisfies the fuse condition.
  - Note that these intermediate node can already be fused with another nodes, the algorithm
      will still run correctly.
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/
using support::LinkedList;
using support::LinkNode;

constexpr uint32_t kMaxFusedOps = 256;

static const Op& stop_fusion_op = Op::Get("annotation.stop_fusion");

TVM_REGISTER_PASS_CONFIG_OPTION("relay.FuseOps.max_depth", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.FuseOps.link_params", Bool);

/*!
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragment and
 *  could ignore blocks like let by simply ordering each dataflow block
 *  and mark the output node as extern_ref;
 */
class IndexedForwardGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
  };
  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      os << "node[" << i << "], " << GetRef<ObjectRef>(node->ref) << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static IndexedForwardGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

// Creator of post dominator tree of the dataflow
class IndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  IndexedForwardGraph Prepare(const Expr& body) {
    this->Update(body, nullptr, kOpaque);
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  // The output.
  IndexedForwardGraph graph_;
  // attribute equal comparator
  StructuralEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern) {
    const tvm::Object* key = node.get();
    IndexedForwardGraph::Node* current;
    auto it = graph_.node_map.find(key);
    if (it != graph_.node_map.end()) {
      current = it->second;
    } else {
      current = arena_->make<IndexedForwardGraph::Node>();
      graph_.node_map[key] = current;
    }
    if (parent != nullptr) {
      auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge>>();
      link->value.node = parent;
      link->value.pattern = pattern;
      current->outputs.Push(link);
    } else {
      current->extern_ref = true;
    }
  }

  void AddNode(const tvm::Object* key) {
    auto it = graph_.node_map.find(key);
    ICHECK(it != graph_.node_map.end()) << "Cannot find node " << GetRef<ObjectRef>(key);
    IndexedForwardGraph::Node* node = it->second;
    ICHECK(node->ref == nullptr);
    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  // Post order tree
  void VisitExpr_(const FunctionNode* op) final {
    // Skip the function that should be handled by external codegen.
    if (op->GetAttr<String>(attr::kCompiler).defined()) return;

    for (auto param : op->params) {
      this->Update(param, nullptr, kOpaque);
    }
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ConstantNode* op) final {
    this->AddNode(op);
    Node* node = graph_.node_map.at(op);
    DataType dtype = DataType(op->data->dtype);
    // This rule must be consistent with code generator.
    bool is_simple_const =
        (dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::Float(32) ||
         dtype == DataType::Float(64) || dtype == DataType::Bool());
    if (op->is_scalar() && is_simple_const) {
      node->pattern = kElemWise;
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      node->pattern = kOpaque;
    }
  }

  void VisitExpr_(const CallNode* call) final {
    ICHECK(graph_.node_map.count(call));
    Node* node = graph_.node_map.at(call);
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    // Now we set the pattern of this call.
    //
    // If we see a call mentioning an operator we should mark it with its
    // annotated pattern.
    //
    // If the pattern is not annotated we will default to opaque.
    //
    // Finally if the operator position is not a call node we will
    // need to call Update, as it may be an arbitrary expression.
    OpPatternKind op_pattern = kOpaque;
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto op = GetRef<Op>(opnode);
      if (IsDynamic(call->checked_type()) && IsDataDependent(call)) {
        // output of a shape func can't be fed to a data-dependent shape func
        op_pattern = kOpaque;
      } else {
        op_pattern = static_cast<OpPatternKind>(fpattern[op]);
      }
    } else {
      this->Update(call->op, node, kOpaque);
    }

    node->pattern = op_pattern;
    this->Update(call->op, nullptr, kOpaque);
    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    // pass the analysis back to all the children it references.
    for (size_t i = 0; i < call->args.size(); ++i) {
      const auto* arg_type = call->args[i]->checked_type().as<TensorTypeNode>();
      // specifically check if result type is the same as arguments type
      OpPatternKind edge_pattern = op_pattern;
      if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
          attr_equal_(rtype->shape, arg_type->shape)) {
        edge_pattern = kElemWise;
      }
      this->Update(call->args[i], node, edge_pattern);
    }
    ExprVisitor::VisitExpr_(call);
    this->AddNode(call);
  }

  void VisitExpr_(const TupleNode* op) final {
    ICHECK(graph_.node_map.count(op));
    Node* tuple_node = graph_.node_map.at(op);
    tuple_node->pattern = kTuple;
    for (const Expr& field : op->fields) {
      if (field->checked_type().as<TensorTypeNode>()) {
        this->Update(field, tuple_node, kInjective);
      } else {
        this->Update(field, nullptr, kOpaque);
      }
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    auto tuple_type = op->tuple->checked_type().as<TupleTypeNode>();
    ICHECK(tuple_type);
    // When TVM lowers a fused function, it expects all arguments to be a Tensor or
    // a tuple containing only Tensors. But this tuple may contain a reference or
    // another tuple. To avoid modifying codegen logic, we do not allow fusing through this node
    // if the tuple contains such non Tensor fields. However, all fields will be recursively
    // visited via call to ExprVisitor::VisitExpr_(op) below and corresponding visitor methods.
    bool has_non_tensor = false;
    for (auto ty : tuple_type->fields) {
      if (!ty.as<TensorTypeNode>()) {
        has_non_tensor = true;
        break;
      }
    }
    if (has_non_tensor) {
      this->Update(op->tuple, nullptr, kOpaque);
    } else {
      ICHECK(graph_.node_map.count(op));
      Node* node = graph_.node_map.at(op);
      node->pattern = kInjective;
      this->Update(op->tuple, node, kInjective);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const VarNode* op) final { this->AddNode(op); }

  void VisitExpr_(const LetNode* op) final {
    // do not fuse through let.
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      this->Update(op->var, nullptr, kOpaque);
      this->Update(op->value, nullptr, kOpaque);
      this->Update(op->body, nullptr, kOpaque);
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
      this->AddNode(op);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const IfNode* op) final {
    // do not fuse through if.
    this->Update(op->cond, nullptr, kOpaque);
    this->Update(op->true_branch, nullptr, kOpaque);
    this->Update(op->false_branch, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefCreateNode* op) final {
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefReadNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefWriteNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const MatchNode* op) final {
    this->Update(op->data, nullptr, kOpaque);
    for (const Clause& c : op->clauses) {
      this->Update(c->rhs, nullptr, kOpaque);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }
};

IndexedForwardGraph IndexedForwardGraph::Create(support::Arena* arena, const Expr& body) {
  return Creator(arena).Prepare(body);
}

/*!
 * \brief Dominator tree that represent domination or
 *  post domination relation of the node.
 */
class DominatorTree {
 public:
  /*!
   * \brief A node in the dominator tree.
   */
  struct Node {
    /*! \brief The node in the tree */
    IndexedForwardGraph::Node* gnode{nullptr};
    /*! \brief parent of the tree */
    Node* parent{nullptr};
    /*! \brief current depth*/
    int depth{0};
    /*! \brief aggregated pattern to parent */
    OpPatternKind pattern{kOpaque};
  };
  // index -> node.
  std::vector<Node*> nodes;
  /*!
   * \brief compute a post dominator relation for a given dataflow graph.
   * \param arena The arena used for node allocation.
   * \param graph The graph to be analyzed.
   * \return The dominator tree of the graph.
   * \note This algorithm makes use of the fact that graph is DAG,
   *       and runs a single pass algorithm via LCA (Least Common Ancestor)
   */
  static DominatorTree PostDom(support::Arena* arena, const IndexedForwardGraph& graph);

 private:
  // Combine pattern together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Find the least common ancestor of the two nodes.
   * \param lhs The left node.
   * \param rhs The right node.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of the two.
   */
  static Node* LeastCommonAncestor(Node* lhs, Node* rhs, OpPatternKind* edge_pattern) {
    while (lhs != rhs) {
      if (lhs == nullptr) return nullptr;
      if (rhs == nullptr) return nullptr;
      if (lhs->depth < rhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        rhs = rhs->parent;
      } else if (rhs->depth < lhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        lhs = lhs->parent;
      } else {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    return lhs;
  }
  /*!
   * \brief Find the least common ancestor of a list of nodes.
   * \param nodes the nodes.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of all nodes.
   */
  Node* LeastCommonAncestor(const LinkedList<IndexedForwardGraph::Edge>& input_nodes,
                            OpPatternKind* edge_pattern) {
    auto link = input_nodes.head;
    if (link == nullptr) {
      return nullptr;
    }
    auto get_node = [&](const IndexedForwardGraph::Edge& edge) {
      size_t oindex = edge.node->index;
      ICHECK_LT(oindex, nodes.size());
      Node* onode = nodes[oindex];
      ICHECK(onode != nullptr);
      return onode;
    };
    Node* parent = get_node(link->value);
    *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
    link = link->next;
    for (; link != nullptr; link = link->next) {
      parent = LeastCommonAncestor(parent, get_node(link->value), edge_pattern);
      *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
    }
    return parent;
  }
  /*!
   * \brief Convert the Node from an IndexedForwardGraph Node into DomaintorTree Node.
   * \param arena The Arena.
   * \param gnode An IndexedForwardGraph Node.
   * \return The DominatorTree Node.
   */
  Node* GetNode(support::Arena* arena, IndexedForwardGraph::Node* gnode) {
    Node* tnode = arena->make<Node>();
    tnode->gnode = gnode;
    if (gnode->extern_ref) {
      tnode->depth = 1;
      tnode->parent = nullptr;
      tnode->pattern = kOpaque;
    } else {
      // find the LCAs of all outputs.
      OpPatternKind pattern = kElemWise;
      Node* parent = LeastCommonAncestor(gnode->outputs, &pattern);
      tnode->depth = parent ? parent->depth + 1 : 1;
      tnode->parent = parent;
      tnode->pattern = pattern;
    }
    return tnode;
  }
};

DominatorTree DominatorTree::PostDom(support::Arena* arena, const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    tree.nodes[index] = tree.GetNode(arena, graph.post_dfs_order[index]);
  }
  return tree;
}

/*!
 * \brief A partition of the graph marked by union find data structure.
 */
class GraphPartitioner {
 public:
  explicit GraphPartitioner(support::Arena* arena, int opt_level, size_t max_fuse_depth)
      : arena_(arena), opt_level_(opt_level), max_fuse_depth_(max_fuse_depth) {}
  /*!
   * \brief Group as a union find data structure.
   */
  struct Group {
    /*! \brief The parent in the union find data structure. */
    Group* parent{nullptr};
    /*! \brief The pattern of the group */
    OpPatternKind pattern;
    /*! \brief reference to the root node. */
    const tvm::Object* root_ref{nullptr};
    /*!
     * \brief Reference to the anchor node,
     * this field is not nullptr only if pattern is kOutEWiseFusable.
     */
    const tvm::Object* anchor_ref{nullptr};
    /*!
     * \brief Find the group root, perform path compression
     * \return The root type node.
     */
    Group* FindRoot() {
      // fast path
      if (this->parent == nullptr) return this;
      // slow path with path compression.
      Group* root = this;
      while (root->parent != nullptr) {
        root = root->parent;
      }
      for (Group* p = this; p != root;) {
        Group* parent = p->parent;
        p->parent = root;
        p = parent;
      }
      return root;
    }

    /*!
     * \brief The number of nodes belonging to this group
     */
    uint32_t num_nodes{1};
  };
  /*!
   * \brief Partition a graph.
   * \return group assignments of each node.
   */
  std::vector<Group*> Partition(const IndexedForwardGraph& graph);

 private:
  /*! \brief The internal arena for temporary space. */
  support::Arena* arena_;
  /*! \brief optimization level for fuse operation. */
  int opt_level_;
  /*! \brief The maximum number of operations in one fused function */
  size_t max_fuse_depth_;
  /*! \brief The internal groups. */
  std::vector<Group*> groups_;
  /*! \brief internal field used for deduplication */
  std::unordered_set<IndexedForwardGraph::Node*> visited_;
  // Internal implelementation of CheckPath
  template <typename F>
  bool CheckPath_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    if (visited_.count(src)) return true;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    gnode = gnode->FindRoot();
    if (!fcond(gnode->pattern, src == sink)) return false;
    if (src == sink) return true;
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  /*!
   * \brief Check all the node and edge pattern
   *  between src and sink satisfies fcond.
   *
   * src is not checked.
   *
   * \param src The source node.
   * \param sink The termination node.
   * \param fcond The condition to be checked.
   * \tparam F the condition function, with signature
   * \note sink must be a post-dominator of src.
   */
  template <typename F>
  bool CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    ICHECK(!src->extern_ref);
    visited_.clear();
    ICHECK(src != sink);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  // Combine two patterns together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > kBroadcast && rhs > kBroadcast) {
      LOG(FATAL) << "Cannot merge two complex group together";
    }
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Merge the child group to the parent.
   * \param child The child group.
   * \param parent The parent group.
   */
  void MergeFromTo(Group* child, Group* parent) {
    child = child->FindRoot();
    parent = parent->FindRoot();
    if (child == parent) return;
    // update the number of nodes of the parent group
    parent->num_nodes += child->num_nodes;
    child->parent = parent;
    // update anchor ref and pattern
    if (child->anchor_ref != nullptr) {
      ICHECK(parent->anchor_ref == nullptr);
      parent->anchor_ref = child->anchor_ref;
      parent->pattern = CombinePattern(child->pattern, parent->pattern);
    }
  }
  // Internal implelementation of CommitFuse
  void CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, Group* target) {
    if (src == sink) return;
    if (visited_.count(src)) return;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    // merge the current group to the parent if possible.
    MergeFromTo(gnode, target);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      CommitFuse_(link->value.node, sink, target);
    }
  }
  /*!
   * \brief Commit fusion operation.
   * \param src The source node.
   * \param sink The termination node.
   * \note sink must be a post-dominator of src.
   */
  void CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
    Group* target = groups_[sink->index];
    visited_.clear();
    ICHECK(src != sink);
    CommitFuse_(src, sink, target);
  }

  size_t CountNodesUptoSink_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
    if (src == sink || visited_.count(src)) return 0;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    auto sum = gnode->num_nodes;
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      sum += CountNodesUptoSink_(link->value.node, sink);
    }
    return sum;
  }

  // Count the number of nodes in a fused subgraph if child is additionaly fused.
  // dom_parent is already known to be a part of the subgraph.
  // For a diamond structure, there can be multiple paths connecting child and dom_parent.
  // All intermediate nodes between child and dom_parent are taken into account.
  // Since dom_parent can itself be an intermediate node in the subgraph, calling FindRoot()
  // is important for correct calculation.
  size_t CountFusedNodesWithNewChild(IndexedForwardGraph::Node* child,
                                     IndexedForwardGraph::Node* dom_parent) {
    Group* target = groups_[dom_parent->index];
    visited_.clear();
    ICHECK(child != dom_parent);
    return target->FindRoot()->num_nodes + CountNodesUptoSink_(child, dom_parent);
  }

  // Initialize the groups.
  void InitGroups(const IndexedForwardGraph& graph) {
    groups_.resize(graph.post_dfs_order.size());
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      const auto* graph_node = graph.post_dfs_order[nid];
      auto* group_node = arena_->make<Group>();
      group_node->pattern = graph_node->pattern;
      group_node->root_ref = graph_node->ref;
      // set anchor ref if necessary.
      if (group_node->pattern == kOutEWiseFusable) {
        group_node->anchor_ref = graph_node->ref;
      }
      groups_[nid] = group_node;
    }
  }

  // execute the fusion algorithm.
  void RunFuse(const IndexedForwardGraph& graph, const DominatorTree& post_dom_tree, int phase) {
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      // the group of current node has been specified already.
      auto* graph_node = graph.post_dfs_order[nid];
      auto* dom_node = post_dom_tree.nodes[nid];
      Group* group_node = groups_[nid];
      ICHECK(group_node != nullptr);
      // no actions for opaque nodes
      if (group_node->pattern == kOpaque) continue;
      // no actions needed if the current node have no dominator
      if (dom_node->parent == nullptr) continue;
      ICHECK(!graph_node->extern_ref);
      size_t dom_parent_gindex = dom_node->parent->gnode->index;

      // refuse the fusion if too many ops are going to be fused together
      if (CountFusedNodesWithNewChild(graph_node, dom_node->parent->gnode) > max_fuse_depth_)
        continue;

      if (phase == 2) {
        // Fuse injective ops into intermediate tuples, if any
        if (group_node->pattern > kInjective) continue;
        Group* dom_parent_group = groups_[dom_parent_gindex];
        Group* dom_root_group = dom_parent_group->FindRoot();
        // If dom node group has a tuple as its root, we do not fuse tuple fields into it
        if (dom_root_group->pattern == kTuple) continue;
        if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= kInjective) {
          // Now we know the tuple has been fused into subsequent injective ops
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
          // dom_root_group can also be tuple, as in inception layers
          // CheckPath is needed to avoid fusing two intermediate tuples
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
        continue;
      }

      // Skip if current node is already fused to the parent.
      if (groups_[dom_parent_gindex] != nullptr &&
          group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
        continue;
      }
      // Do not fuse into tuple for now
      if (groups_[dom_parent_gindex]->pattern == kTuple) continue;
      // Try to fuse current node to its post-dominator.
      if (group_node->pattern == kOutEWiseFusable) {
        if (phase != 0) continue;
        // Path for OutEWiseFusable: conv2d
        // Check if the dominator relation is elemwise.
        if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
          ICHECK(dom_node->parent->gnode != nullptr);
          // The fuse can be executed if all the intermediate ops are still broadcast.
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern <= kBroadcast) {
        // Pre-condition: can only be fused to parent which is injective or reduction.
        if (dom_node->parent != nullptr &&
            (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce)) {
          // Check if all the intermediate ops are still broadcast.
          // The final terminal node can already be fused to a OutEWiseFusable group.
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            if (!is_sink) {
              // Elemwise, broadcast, and injective ops on the parallel branches
              // are allowed be fused to the elemwise/broadcast anchor.
              return kind <= kInjective;
            } else {
              return (kind <= kBroadcast || kind == kCommReduce || kind == kInjective ||
                      kind == kOutEWiseFusable);
            }
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern == kInjective || group_node->pattern == kTuple) {
        // defer injective fusion to second phase.
        // so conv2d always finishes fusing.
        if (phase != 1) continue;
        // Check if all path are injective.
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      } else {
        // do nothing.
        ICHECK(group_node->pattern == kCommReduce);
      }
    }
  }
};

std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);
  // get post dominator tree
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);
  // run fusion algorithm.
  for (int phase = 0; phase < 3; ++phase) {
    this->RunFuse(graph, post_dom_tree, phase);
  }
  return std::move(groups_);
}

class FuseMutator : private MixedModeMutator {
 public:
  FuseMutator(int fuse_opt_level, size_t max_fuse_depth, bool link_params)
      : fuse_opt_level_(fuse_opt_level),
        max_fuse_depth_(max_fuse_depth),
        link_params_(link_params) {}

  // Run the transform
  Expr Transform(const Expr& body) {
    return Transform(body, fuse_opt_level_, max_fuse_depth_, link_params_);
  }

 protected:
  // Run the transform
  Expr Transform(const Expr& body, int fuse_opt_level, size_t max_fuse_depth, bool link_params) {
    // setup the group map.
    auto graph = IndexedForwardGraph::Create(&arena_, body);
    auto groups = GraphPartitioner(&arena_, fuse_opt_level, max_fuse_depth).Partition(graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }
    // The following line can be used for debug.
    // this->DebugDumpGroup(body);
    return this->Mutate(body);
  }

 private:
  int fuse_opt_level_;
  size_t max_fuse_depth_;
  bool link_params_;

  using MixedModeMutator::VisitExpr_;

  /*! \brief Temporary information from each group. */
  struct GroupInfo {
   public:
    // The parameters of the function.
    Array<Var> params;
    // The arguments to call the functions.
    Array<Expr> arguments;
    // Get a new parameter or allocate an old one
    Var GetOrAllocParam(const Expr& expr, const Type& type) {
      // run linear scan as most fused groups contain only a few inputs.
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (expr.same_as(arguments[i])) return params[i];
      }
      // create a new parameter.
      std::ostringstream os;
      os << "p" << params.size();
      auto var = Var(os.str(), type);
      params.push_back(var);
      arguments.push_back(expr);
      return var;
    }
  };
  /*! \brief Internal arena. */
  support::Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Object*, GraphPartitioner::Group*> gmap_;
  /* \brief Internal group information map. */
  std::unordered_map<GraphPartitioner::Group*, GroupInfo> ginfo_;

  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node) {
    if (fn_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Expr>(fn_node);
    } else {
      return ExprMutator::VisitExpr_(fn_node);
    }
  }

  // Transform calls.
  Expr Rewrite_(const CallNode* call, const Expr& post) {
    if (call->op.as<OpNode>()) {
      static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
      static auto fqnncanonicalize = Op::GetAttrMap<FTVMLegalize>("FTVMQnnCanonicalize");

      Op op = Downcast<Op>(call->op);
      if (fnoncomputational.get(op, false) && !fqnncanonicalize.count(op)) {
        return ExprMutator::VisitExpr_(call);
      }

      // If it is a primitive op call
      // then we must have a group assignment for it already.
      ICHECK(gmap_.count(call));
      if (call->op == stop_fusion_op) {
        return ExprMutator::VisitExpr(call->args[0]);
      }
      auto* ret_group = gmap_.at(call)->FindRoot();
      Array<Expr> new_args = GetNewArguments(call->args, ret_group);

      auto new_call = Call(call->op, new_args, call->attrs, call->type_args, call->span);

      if (ret_group->root_ref == call) {
        // This is the root of the group
        // create the new call node.
        return MakeNewFunction(ret_group, call->checked_type(), new_call);
      } else {
        // This is an intermediate node of a fused function
        // simply return the new call.
        return std::move(new_call);
      }
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) {
    auto* ret_group = gmap_.at(tuple_node)->FindRoot();
    if (ret_group->root_ref == tuple_node) {
      return ExprMutator::VisitExpr_(tuple_node);
    }
    // This tuple is an intermediate node in the group
    Array<Expr> new_fields = GetNewArguments(tuple_node->fields, ret_group);
    return WithFields(GetRef<Tuple>(tuple_node), new_fields);
  }

  Expr Rewrite_(const TupleGetItemNode* tuple_get, const Expr& post) {
    auto* ret_group = gmap_.at(tuple_get)->FindRoot();
    auto new_tuple = GetNewArguments({tuple_get->tuple}, ret_group)[0];
    auto new_node = TupleGetItem(new_tuple, tuple_get->index);
    if (ret_group->root_ref == tuple_get) {
      if (gmap_.at(tuple_get->tuple.get())->FindRoot() != ret_group) {
        // Isolated. This case occurs when tuple is created by an Opaque op
        // e.g. multibox_transform_loc
        return ExprMutator::VisitExpr_(tuple_get);
      }
      // A new function whose output is a tuple field access
      return MakeNewFunction(ret_group, tuple_get->checked_type(), new_node);
    }
    // This is an intermediate node in the group
    return std::move(new_node);
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->VisitExpr(op->value);
      // Visit body and cache the op
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(var, value, body);
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr MakeNewFunction(GraphPartitioner::Group* group, Type ret_type, Expr body) {
    // Quickly check special properties of the fused function.
    // A pass to check if the fused op contains only reshape ops.
    class CheckReshapeOnly : public ExprVisitor {
     public:
      void VisitExpr_(const CallNode* cn) final {
        this->has_call = true;
        static auto freshape_op = Op::GetAttrMap<TReshapeOp>("TReshapeOp");

        if (!freshape_op.get(cn->op, false)) {
          this->reshape_only = false;
        }

        if (!this->reshape_only) return;
        ExprVisitor::VisitExpr_(cn);
      }

      void VisitExpr_(const VarNode* vn) final {
        if (!vn->type_annotation.defined() || !vn->type_annotation->IsInstance<TensorTypeNode>()) {
          this->reshape_only = false;
        }
      }

      bool reshape_only = true;
      bool has_call = false;
    } visitor;

    visitor(body);
    const GroupInfo& ginfo = ginfo_[group];
    auto func = Function(ginfo.params, body, ret_type, {});
    func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(visitor.has_call));
    // TODO(mbs): "reshape" cleanup.
    if (visitor.has_call && visitor.reshape_only) {
      func = WithAttr(std::move(func), attr::kReshapeOnly, tvm::Integer(visitor.reshape_only));
    }
    return Call(func, ginfo.arguments, Attrs());
  }

  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args,
                              GraphPartitioner::Group* current_group) {
    Array<Expr> new_args;
    for (auto arg : args) {
      auto* arg_group = gmap_.at(arg.get())->FindRoot();
      auto type = arg->checked_type();
      Expr new_arg = this->Mutate(arg);
      if (current_group != arg_group) {
        if (!link_params_ || new_arg.as<ConstantNode>() == nullptr) {
          Var param = ginfo_[current_group].GetOrAllocParam(new_arg, type);
          new_args.push_back(param);
        } else {
          new_args.push_back(new_arg);
        }
      } else {
        new_args.push_back(new_arg);
      }
    }
    return new_args;
  }

  // Debug function, dump the group assignment in text.
  void DebugDumpGroup(const Expr& body) {
    std::string text = AsText(body, false, [this](const ObjectRef& expr) -> std::string {
      auto it = gmap_.find(expr.get());
      if (it == gmap_.end()) return "";
      std::ostringstream os;
      auto* group = it->second->FindRoot();
      os << " /* group=" << group << " */";
      return os.str();
    });
    LOG(INFO) << "Dump of group info:\n" << text;
  }
};

Expr FuseOps(const Expr& expr, int fuse_opt_level, size_t max_fuse_depth, bool link_params,
             const IRModule& module) {
  return FuseMutator(fuse_opt_level, max_fuse_depth, link_params).Transform(expr);
}

namespace transform {

Pass FuseOps(int fuse_opt_level) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        bool link_params = false;
        Executor executor =
            m->GetAttr<Executor>(tvm::attr::kExecutor).value_or(NullValue<Executor>());
        link_params = executor.defined()
                          ? executor->attrs.GetAttr<Bool>("link-params").value_or(Bool(link_params))
                          : link_params;
        link_params = pc->GetConfig("relay.FuseOps.link_params", Bool(link_params)).value();
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth = pc->GetConfig("relay.FuseOps.max_depth", Integer(kMaxFusedOps));
        return Downcast<Function>(
            FuseOps(f, opt_level, max_fuse_depth.value().IntValue(), link_params, m));
      };
  return CreateFunctionPass(pass_func, 0, "FuseOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseOps").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
