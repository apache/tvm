/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_deep_compare.cc
 * \brief Deep compare two graph structure
 */
#include <dmlc/common.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <tvm/ir.h>
#include <tvm/runtime/packed_func.h>
#include <functional>
#include "./node_attr.h"
#include "./graph_hash.h"

namespace nnvm {
namespace compiler {

using namespace tvm;
using tvm::ir::IntImm;

size_t HashPlaceHolder(const Tensor& t) {
  size_t key = t->shape.size();
  key = dmlc::HashCombine(key, (t->dtype.code() << 8) | t->dtype.bits());
  for (Expr s : t->shape) {
    if (const IntImm* op = s.as<IntImm>()) {
      key = dmlc::HashCombine(key, op->value);
    }
  }
  return key;
}

bool PlaceHolderEqual(const Tensor& a, const Tensor& b) {
  if (a->shape.size() != b->shape.size()) return false;
  if (a->dtype != b->dtype) return false;
  for (size_t i = 0; i < a->shape.size(); ++i) {
    const IntImm* a_value = a->shape[i].as<IntImm>();
    const IntImm* b_value = b->shape[i].as<IntImm>();
    if (a_value && b_value == nullptr) return false;
    if (b_value && a_value == nullptr) return false;
    if (a_value == nullptr && b_value == nullptr) {
      continue;
    }
    if (a_value->value != b_value->value) return false;
  }
  return true;
}

size_t GraphKeyHash::Hash(const GraphKey& gkey)  {
  if (gkey->cache_hash_key_ != 0) return gkey->cache_hash_key_;
  size_t key = dmlc::HashCombine(GraphHash(gkey->graph), gkey->target);
  key = dmlc::HashCombine(key, gkey->inputs.size());
  for (size_t i = 0; i < gkey->inputs.size(); ++i) {
    key = dmlc::HashCombine(key, HashPlaceHolder(gkey->inputs[i]));
  }
  if (key == 0) key = 1;
  gkey->cache_hash_key_ = key;
  return key;
}

bool GraphKeyEqual::Equal(const GraphKey& a,
                          const GraphKey& b) {
  if (a->target != b->target) return false;
  if (a->inputs.size() != b->inputs.size()) return false;
  for (size_t i = 0; i < a->inputs.size(); ++i) {
    if (!PlaceHolderEqual(a->inputs[i], b->inputs[i])) return false;
  }
  if (GraphDeepCompare(a->graph, b->graph, false).length() != 0) return false;
  return true;
}

GraphKey GraphKeyNode::make(Graph graph,
                            tvm::Array<Tensor> inputs,
                            std::string target) {
  std::shared_ptr<GraphKeyNode> n
      = std::make_shared<GraphKeyNode>();
  n->graph = std::move(graph);
  n->inputs = inputs;
  n->target = std::move(target);
  return GraphKey(n);
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<GraphKeyNode>([](const GraphKeyNode *op, IRPrinter *p) {
    p->stream << "GraphKeyNode("<< op << ")";
});


// Run graph hash
size_t GraphHash(const Graph& graph) {
  const IndexedGraph& idx = graph.indexed_graph();
  size_t key = 0;
  // Combine a linearized sequence of ops in subgraph
  key = dmlc::HashCombine(key, idx.num_nodes());
  std::hash<std::string> str_hash;
  std::vector<size_t> hash_temp;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const IndexedGraph::Node& inode = idx[nid];
    // Use name instad op address so it is deterministic across runs
    if (inode.source->is_variable()) continue;
    key = dmlc::HashCombine(key, inode.source->op()->name);
    hash_temp.clear();
    for (const auto& kv : GetAttrDict(inode.source->attrs)) {
      hash_temp.push_back(dmlc::HashCombine(str_hash(kv.first), kv.second));
    }
    // to make sure it is deterministic
    // since unordered_map is not deterministic
    std::sort(hash_temp.begin(), hash_temp.end());
    for (size_t value : hash_temp) {
      key = dmlc::HashCombine(key, value);
    }
  }
  return key;
}

// deep compare the graph structure
// not considering the graph attributes
// return non-empty error message if the graph mismatch.
// the comparator won't match name of intermediate node.
// compare_var_attr
std::string GraphDeepCompare(const Graph& a,
                             const Graph& b,
                             bool compare_variable_attr) {
  const IndexedGraph& idxa = a.indexed_graph();
  const IndexedGraph& idxb = b.indexed_graph();
  std::ostringstream err;
  if (idxa.num_nodes() != idxb.num_nodes()) {
    err << "Number of nodes mismatch";
    return err.str();
  }
  if (idxa.num_node_entries() != idxb.num_node_entries()) {
    err << "Number of node entry mismatch";
    return err.str();
  }
  if (idxa.outputs().size() != idxb.outputs().size()) {
    err << "Number of outputs mismatch";
    return err.str();
  }
  for (size_t i = 0; i < idxa.outputs().size(); ++i) {
    if (idxa.outputs()[i].node_id != idxb.outputs()[i].node_id ||
        idxa.outputs()[i].index != idxb.outputs()[i].index) {
      err << "Output entry mismatch";
      return err.str();
    }
  }
  if (idxa.input_nodes().size() != idxb.input_nodes().size()) {
    err << "Number of inputs mismatch";
    return err.str();
  }

  for (uint32_t nid = 0; nid < idxa.num_nodes(); ++nid) {
    const IndexedGraph::Node& anode = idxa[nid];
    const IndexedGraph::Node& bnode = idxb[nid];
    if (anode.source->op() != bnode.source->op()) {
      err << "Node mismatch ";
      return err.str();
    }
    if (anode.source->is_variable()) {
      CHECK(bnode.source->is_variable());
      if (!compare_variable_attr) continue;
    }
    AttrDict adict = GetAttrDict(anode.source->attrs);
    AttrDict bdict = GetAttrDict(bnode.source->attrs);

    auto fmatch = [&err, &anode](const AttrDict& adict, const AttrDict& bdict) {
      for (const auto& kv : adict) {
        auto it = bdict.find(kv.first);
        if (it != bdict.end()) {
          if (it->second != kv.second) {
            err << "Node attr mismatch, op=" << anode.source->attrs.name
                << " attr_key=" << kv.first << " " << it->second
                << " v.s. " << kv.second;
            return false;
          }
        } else {
          err << "One attr_key=" << kv.first << " is missing in another "
               << "op=" << anode.source->attrs.name;
          return false;
        }
      }
      return true;
    };
    if (!fmatch(adict, bdict)) return err.str();
    if (adict.size() != bdict.size()) {
      CHECK(!fmatch(bdict, adict));
      return err.str();
    }
    if (anode.inputs.size() != bnode.inputs.size()) {
      err << "Node input mismatch, op=" << anode.source->attrs.name;
      return err.str();
    }
    if (anode.control_deps.size() != bnode.control_deps.size()) {
      err << "Node control_deps mistach, op=" << anode.source->attrs.name;
      return err.str();
    }
    for (size_t i = 0; i < anode.inputs.size(); ++i) {
      const IndexedGraph::NodeEntry& ae = anode.inputs[i];
      const IndexedGraph::NodeEntry& be = bnode.inputs[i];
      if (ae.node_id != be.node_id ||
          ae.index != be.index ||
          ae.version != be.version) {
        err << "Node input mismatch on, op=" << anode.source->attrs.name;
        return err.str();
      }
    }
    for (size_t i = 0; i < anode.control_deps.size(); ++i) {
      if (anode.control_deps[i] != bnode.control_deps[i]) {
        err << "Node control_dep mismatch on, op=" << anode.source->attrs.name;
        return err.str();
      }
    }
  }
  return "";
}

TVM_REGISTER_GLOBAL("nnvm.graph.DeepCompare")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = GraphDeepCompare(args[0], args[1], args[2]);
  });
}  // namespace compiler
}  // namespace nnvm
