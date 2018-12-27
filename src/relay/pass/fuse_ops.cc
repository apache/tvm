/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file src/tvm/relay/pass/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/ir_operator.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include "../../common/arena.h"


namespace tvm {
namespace relay {

/*
  Note on Fusing algorithm:

  The main challenge of genenral fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  However, at the point of conv2d we do not necessarily know that all its future path
  will merge at the elemwise add. The new fusor algorithm applies post-dominator analysis.
  The immediate post-dominator of a node defined by the closest node where all the future path goes into.
  In the above case, the elemwise add is the post-dominator of conv2d. The general algorithm is as follows:

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.

  Note that, because we run analysis on a DAG, we use a single pass post-dominator
  tree construction algorithm via LCA, which is simpler than the full version that handles cycles.

  The fusion algorithm traverses from each node and checks if it can be fused to its
  immediate post dominator. It has to check the following things:

  - CheckPath: check all the path between a node and its immediate post-dominator
               satiesfies the fuse condition.
  - Note that these intermediate node can already be fused with another nodes, the algorithm
      will still run correctly.
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/
using common::LinkNode;
using common::LinkedList;

/*!
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragement and
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
    const tvm::Node* ref{nullptr};
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
  std::unordered_map<const tvm::Node*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      os << "node[" << i << "], "
         << GetRef<NodeRef>(node->ref)
         << " outputs=[";
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
  static IndexedForwardGraph Create(common::Arena* arena, const Expr& body);

 private:
  class Creator;
};

// Creator of post dominator tree of the dataflow
class IndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(common::Arena* arena)
      : arena_(arena) {}

  IndexedForwardGraph Prepare(const Expr& body) {
    this->Update(body, nullptr, kOpaque);
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  common::Arena* arena_;
  // The output.
  IndexedForwardGraph graph_;
  // attribute equal comparator
  AttrsEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node,
              IndexedForwardGraph::Node* parent,
              OpPatternKind pattern) {
    const tvm::Node* key = node.get();
    IndexedForwardGraph::Node* current;
    auto it = graph_.node_map.find(key);
    if (it != graph_.node_map.end()) {
      current = it->second;
    } else {
      current = arena_->make<IndexedForwardGraph::Node>();
      graph_.node_map[key] = current;
    }
    if (parent != nullptr) {
      auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge> >();
      link->value.node = parent;
      link->value.pattern = pattern;
      current->outputs.Push(link);
    } else {
      current->extern_ref = true;
    }
  }
  void AddNode(const tvm::Node* key) {
    auto it = graph_.node_map.find(key);
    CHECK(it != graph_.node_map.end())
        << "Cannot find node " << GetRef<NodeRef>(key);
    IndexedForwardGraph::Node* node = it->second;
    CHECK(node->ref == nullptr);
    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  // Post order tree
  void VisitExpr_(const FunctionNode* op) {
    for (auto param : op->params) {
      this->Update(param, nullptr, kOpaque);
    }
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ConstantNode* op) {
    this->AddNode(op);
    Node* node = graph_.node_map.at(op);
    DataType dtype = TVMType2Type(op->data->dtype);
    // This rule must be consistent with code generator.
    bool is_simple_const = (
        dtype == Int(32) ||
        dtype == Int(64) ||
        dtype == Float(32) ||
        dtype == Float(64) ||
        dtype == Bool());
    if (op->is_scalar() && is_simple_const) {
      node->pattern = kElemWise;
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      node->pattern = kOpaque;
    }
  }

  void VisitExpr_(const CallNode* call) {
    CHECK(graph_.node_map.count(call));
    Node* node = graph_.node_map.at(call);
    static auto fpattern =
        Op::GetAttr<TOpPattern>("TOpPattern");
    // setup pattern.
    OpPatternKind op_pattern = kOpaque;
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      op_pattern = static_cast<OpPatternKind>(fpattern[GetRef<Op>(opnode)]);
    }
    node->pattern = op_pattern;
    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    // pass the message back to all the children it references.
    for (size_t i = 0; i < call->args.size(); ++i) {
      const auto* arg_type =
          call->args[i]->checked_type().as<TensorTypeNode>();
      // specifically check if result type
      OpPatternKind edge_pattern = op_pattern;
      if (edge_pattern == kBroadcast &&
          arg_type != nullptr &&
          rtype != nullptr &&
          attr_equal_(rtype->shape, arg_type->shape)) {
        edge_pattern = kElemWise;
      }
      this->Update(call->args[i], node, edge_pattern);
    }
    ExprVisitor::VisitExpr_(call);
    this->AddNode(call);
  }

  void VisitExpr_(const TupleNode* op) {
    CHECK(graph_.node_map.count(op));
    Node* tuple_node = graph_.node_map.at(op);
    tuple_node->pattern = kInjective;
    for (const Expr& field : op->fields) {
      this->Update(field, tuple_node, kInjective);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const TupleGetItemNode* op) {
    CHECK(graph_.node_map.count(op));
    Node* node = graph_.node_map.at(op);
    this->Update(op->tuple, node, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const VarNode* op) {
    this->AddNode(op);
  }

  void VisitExpr_(const LetNode* op) {
    // do not fuse through let.
    this->Update(op->var, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const IfNode* op) {
    // do not fuse through if.
    this->Update(op->cond, nullptr, kOpaque);
    this->Update(op->true_branch, nullptr, kOpaque);
    this->Update(op->false_branch, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }
};

IndexedForwardGraph IndexedForwardGraph::Create(
    common::Arena* arena, const Expr& body) {
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
   * \param graph The graph to be analyze.
   * \return The dominator tree of the graph.
   * \note This algorithm makes use of the fact that graph is DAG,
   *       and runs a single pass algorithm via LCA.
   */
  static DominatorTree PostDom(common::Arena* arena,
                               const IndexedForwardGraph& graph);

 private:
  // Combine pattern together.
  static OpPatternKind CombinePattern(
      OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Find the least common acenstor of the two nodes.
   * \param lhs The left node.
   * \param rhs The right node.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of thw two.
   */
  static Node* LeastCommonAncestor(
      Node* lhs,
      Node* rhs,
      OpPatternKind* edge_pattern) {
    while (lhs != rhs) {
      if (lhs == nullptr) return nullptr;
      if (rhs == nullptr) return nullptr;
      if (lhs->depth < rhs->depth) {
        edge_pattern[0] = CombinePattern(
            edge_pattern[0], rhs->pattern);
        rhs = rhs->parent;
      } else if (rhs->depth < lhs->depth) {
        edge_pattern[0] = CombinePattern(
            edge_pattern[0], lhs->pattern);
        lhs = lhs->parent;
      } else {
        edge_pattern[0] = CombinePattern(
            edge_pattern[0], lhs->pattern);
        edge_pattern[0] = CombinePattern(
            edge_pattern[0], rhs->pattern);
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    return lhs;
  }
};

DominatorTree DominatorTree::PostDom(common::Arena* arena,
                                     const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    Node* tnode = arena->make<Node>();
    auto* gnode = graph.post_dfs_order[index];
    tnode->gnode = gnode;
    if (gnode->extern_ref) {
      tnode->depth = 1;
      tnode->parent = nullptr;
      tnode->pattern = kOpaque;
    } else {
      // find the LCAs of all outputs.
      OpPatternKind pattern = kElemWise;
      Node* parent = nullptr;
      for (auto link = gnode->outputs.head; link != nullptr; link= link->next) {
        size_t oindex = link->value.node->index;
        CHECK_LT(oindex, tree.nodes.size());
        Node* onode = tree.nodes[oindex];
        CHECK(onode != nullptr);
        if (parent != nullptr) {
          parent = LeastCommonAncestor(parent, onode, &pattern);
        } else {
          parent = onode;
        }
        pattern = CombinePattern(pattern, link->value.pattern);
      }
      tnode->depth = parent ? parent->depth + 1 : 1;
      tnode->parent = parent;
      tnode->pattern = pattern;
    }
    tree.nodes[index] = tnode;
  }
  return tree;
}

/*!
 * \brief A partition of the graph marked by union find data structure.
 */
class GraphPartitioner {
 public:
  explicit GraphPartitioner(common::Arena* arena, int opt_level)
      : arena_(arena), opt_level_(opt_level) {}
  /*!
   * \brief Group as a union find data structure.
   */
  struct Group {
    /*! \brief The parent in the union find data structure. */
    Group* parent{nullptr};
    /*! \brief The pattern of the group */
    OpPatternKind pattern;
    /*! \brief reference to the root node. */
    const tvm::Node* root_ref{nullptr};
    /*!
     * \brief Reference to the master node,
     * this field is not nullptr only if pattern is kOutEWiseFusable.
     */
    const tvm::Node* master_ref{nullptr};
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
  };
  /*!
   * \brief Partition a graph.
   * \return group assignments of each node.
   */
  std::vector<Group*> Partition(const IndexedForwardGraph& graph);

 private:
  /*! \brief The internal arena for temporary space. */
  common::Arena* arena_;
  /*! \brief optimization level for fuse operation. */
  int opt_level_;
  /*! \brief The internal groups. */
  std::vector<Group*> groups_;
  /*! \brief internal field used for deduplication */
  std::unordered_set<IndexedForwardGraph::Node*> visited_;
  // Internal implelementation of CheckPath
  template<typename F>
  bool CheckPath_(IndexedForwardGraph::Node* src,
                  IndexedForwardGraph::Node* sink,
                  F fcond) {
    if (visited_.count(src)) return true;
    visited_.insert(src);
    Group* gnode =  groups_[src->index];
    CHECK(gnode != nullptr);
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
  template<typename F>
  bool CheckPath(IndexedForwardGraph::Node* src,
                 IndexedForwardGraph::Node* sink,
                 F fcond) {
    CHECK(!src->extern_ref);
    visited_.clear();
    CHECK(src != sink);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  // Combine two patterns together.
  static OpPatternKind CombinePattern(
      OpPatternKind lhs, OpPatternKind rhs) {
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
    child->parent = parent;
    // update master ref and pattern
    if (child->master_ref != nullptr) {
      CHECK(parent->master_ref == nullptr);
      parent->master_ref = child->master_ref;
      parent->pattern = CombinePattern(
          child->pattern, parent->pattern);
    }
  }
  // Internal implelementation of CommitFuse
  void CommitFuse_(IndexedForwardGraph::Node* src,
                   IndexedForwardGraph::Node* sink,
                   Group* target) {
    if (src == sink) return;
    if (visited_.count(src)) return;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    CHECK(gnode != nullptr);
    // merge the current group to the parent if possible.
    MergeFromTo(gnode, target);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      CommitFuse_(link->value.node, sink, target);;
    }
  }
  /*!
   * \brief Commit fusion operation.
   * \param src The source node.
   * \param sink The termination node.
   * \note sink must be a post-dominator of src.
   */
  void CommitFuse(IndexedForwardGraph::Node* src,
                  IndexedForwardGraph::Node* sink) {
    Group* target = groups_[sink->index];
    visited_.clear();
    CHECK(src != sink);
    CommitFuse_(src, sink, target);
  }

  // Initialize the groups.
  void InitGroups(const IndexedForwardGraph& graph) {
    groups_.resize(graph.post_dfs_order.size());
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      const auto* graph_node = graph.post_dfs_order[nid];
      auto* group_node = arena_->make<Group>();
      group_node->pattern = graph_node->pattern;
      group_node->root_ref = graph_node->ref;
      // set master ref if necessary.
      if (group_node->pattern == kOutEWiseFusable) {
        group_node->master_ref = graph_node->ref;
      }
      groups_[nid] = group_node;
    }
  }

  // execute the fusion algorithm.
  void RunFuse(const IndexedForwardGraph& graph,
               const DominatorTree& post_dom_tree,
               int phase) {
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      // the group of current node has been specified already.
      auto* graph_node = graph.post_dfs_order[nid];
      auto* dom_node = post_dom_tree.nodes[nid];
      Group* group_node = groups_[nid];
      CHECK(group_node != nullptr);
      // no actions for opaque nodes
      if (group_node->pattern == kOpaque) continue;
      // no actions needed if the current node have no dominator
      if (dom_node->parent == nullptr) continue;
      CHECK(!graph_node->extern_ref);
      // Skip if current node is already fused to the parent.
      size_t dom_parent_gindex = dom_node->parent->gnode->index;
      if (groups_[dom_parent_gindex] != nullptr &&
          group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
        continue;
      }
      // Try to fuse current node to its post-dominator.
      if (group_node->pattern == kOutEWiseFusable) {
        if (phase != 0) continue;
        // Path for OutEWiseFusable: conv2d
        // Check if the dominator relation is elemwise.
        if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
          CHECK(dom_node->parent->gnode != nullptr);
          // The fuse can be executed if all the intermediate ops are still broadcast.
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            return kind <= kBroadcast;
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern <= kBroadcast) {
        // Pre-condition: can only be fused to parent which is injective or reduction.
        if (dom_node->parent != nullptr &&
            (dom_node->pattern <= kInjective ||
             dom_node->pattern == kCommReduce)) {
          // Check if all the intermediate ops are still broadcast.
          // The final terminal node can already be fused to a OutEWiseFusable group.
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            if (!is_sink) {
              return kind <= kBroadcast;
            } else {
              return (kind <= kBroadcast ||
                      kind == kCommReduce ||
                      kind == kOutEWiseFusable);
            }
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern == kInjective) {
        // defer injective fusion to second phase.
        // so conv2d always finishes fusing.
        if (phase != 1) continue;
        // Check if all path are injective.
        auto fcond = [](OpPatternKind kind, bool is_sink) {
          return kind <= kInjective;
        };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      } else {
        // do nothing.
        CHECK(group_node->pattern == kCommReduce);
      }
    }
  }
};

std::vector<GraphPartitioner::Group*>
GraphPartitioner::Partition(const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);
  // get post dominator tree
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);
  // run fusion algorithm.
  for (int phase = 0; phase < 2; ++phase) {
    this->RunFuse(graph, post_dom_tree, phase);
  }
  return std::move(groups_);
}

class FuseMutator : private ExprMutator {
 public:
  // Run the transform
  Expr Transform(const Expr& body, int fuse_opt_level) {
    // setup the group map.
    auto graph = IndexedForwardGraph::Create(&arena_, body);
    auto groups = GraphPartitioner(&arena_, fuse_opt_level).Partition(
        graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      CHECK(graph.post_dfs_order[nid]->ref != nullptr);
      gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }
    // The following line can be used for debug.
    // this->DebugDumpGroup(body);
    return this->Mutate(body);
  }


 private:
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
      auto var = VarNode::make(os.str(), type);
      params.push_back(var);
      arguments.push_back(expr);
      return var;
    }
  };
  /*! \brief Internal arena. */
  common::Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Node*, GraphPartitioner::Group*> gmap_;
  /* \brief Internal group information map. */
  std::unordered_map<GraphPartitioner::Group*, GroupInfo> ginfo_;
  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node) {
    if (fn_node->IsPrimitive()) {
      return GetRef<Expr>(fn_node);
    } else {
      return ExprMutator::VisitExpr_(fn_node);
    }
  }
  // Transform calls.
  Expr VisitExpr_(const CallNode* call) {
    if (call->op.as<OpNode>()) {
      // If it is a primitive op call
      // then we must have a group assignment for it already.
      CHECK(gmap_.count(call));
      auto* ret_group = gmap_.at(call)->FindRoot();
      Array<Expr> new_args = GetNewArguments(call->args, ret_group);

      auto new_call = CallNode::make(
          call->op, new_args, call->attrs, call->type_args);

      if (ret_group->root_ref == call) {
        // This is the root of the group
        // create the new call node.
        return MakeNewFunction(ret_group, call->checked_type(), new_call);
      } else {
        // This is an intermediate node of a fused function
        // simply return the new call.
        return new_call;
      }
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr VisitExpr_(const TupleNode* tuple) {
    auto* ret_group = gmap_.at(tuple)->FindRoot();
    Array<Expr> new_fields = GetNewArguments(tuple->fields, ret_group);
    Tuple new_tuple = TupleNode::make(new_fields);
    if (ret_group == gmap_.at(tuple)) {
      bool isolated = true;
      for (size_t i = 0; i < new_fields.size(); ++i) {
        isolated &= (new_fields[i].same_as(ginfo_[ret_group].params[i]));
      }
      if (isolated) {
        // Do not put a isolated tuple into a function
        return ExprMutator::VisitExpr_(tuple);
      }
      // This tuple has been fused with other ops before it
      return MakeNewFunction(ret_group, tuple->checked_type(), new_tuple);
    }
    // This tuple is an intermediate node in the group
    return new_tuple;
  }

  Expr MakeNewFunction(GraphPartitioner::Group* group, Type ret_type, Expr body) {
    const GroupInfo& ginfo = ginfo_[group];
    auto func = FunctionNode::make(ginfo.params, body, ret_type, {});
    func = FunctionSetAttr(func, "Primitive", tvm::Integer(1));
    return CallNode::make(func, ginfo.arguments, Attrs());
  }

  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args,
                              GraphPartitioner::Group* current_group) {
    Array<Expr> new_args;
    for (auto arg : args) {
      auto* arg_group = gmap_.at(arg.get())->FindRoot();
      auto type = arg->checked_type();
      Expr new_arg = this->Mutate(arg);
      if (current_group != arg_group) {
        Var param = ginfo_[current_group].GetOrAllocParam(new_arg, type);
        new_args.push_back(param);
      } else {
        new_args.push_back(new_arg);
      }
    }
    return new_args;
  }

  // Debug function, dump the group assignment in text.
  void DebugDumpGroup(const Expr& body) {
    std::string text = RelayPrint(body, false, [this](const Expr& expr) -> std::string {
        auto it = gmap_.find(expr.get());
        if (it == gmap_.end()) return "";
        std::ostringstream os;
        auto *group = it->second->FindRoot();
        os << "group=" << group;
        return os.str();
      });
    LOG(INFO) << "Dump of group info:\n" << text;
  }
};


Expr FuseOps(const Expr& expr, int fuse_opt_level) {
  // First we convert all chains of fusable ops into
  // abstracted functions which we mark as primtive
  // then we convert these primtive functions into
  // new operators.
  return FuseMutator().Transform(expr, fuse_opt_level);
}

TVM_REGISTER_API("relay._ir_pass.FuseOps")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = FuseOps(args[0], args[1]);
});
}  // namespace relay
}  // namespace tvm
