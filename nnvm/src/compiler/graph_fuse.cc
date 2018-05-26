/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_fuse.cc
 * \brief Fuse the operators together.
 */
#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/lowered_func.h>
#include "./compile_engine.h"
#include "../../tvm/src/runtime/graph/graph_runtime.h"

namespace nnvm {
namespace compiler {

using tvm::runtime::TVMOpParam;

// parser
inline void TVMOpParamParser(nnvm::NodeAttrs* attrs) {
  TVMOpParam param;
  param.Init(attrs->dict);
  attrs->parsed = std::move(param);
}

NNVM_REGISTER_OP(tvm_op)
.set_attr_parser(TVMOpParamParser)
.set_num_inputs([](const NodeAttrs& attrs) {
    const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
    return param.num_inputs;
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
    return param.num_outputs;
  });

using namespace tvm;

// The single fuse rule.
enum class FuseRule {
  kUknown,
  kFuseToMaster,
  kRealize
};

/*!
 * \brief Get DLDataType from dtype flag.
 *
 * \param type_flag The data type flag
 * \return corresponding DLDataType
 */
DLDataType GetDLType(int type_flag) {
  if (type_flag == 0) return tvm::Type2TVMType(tvm::Float(32));
  LOG(FATAL) << "unknown type_flag=" << type_flag;
  return Type2TVMType(Float(32));
}

// Partition the graph into segments
// Each segment will be compiled into one operator.
// Need also mark the property of the segment.
nnvm::Graph GraphFusePartition(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  int opt_level = 2;
  if (g.attrs.count("opt_level") != 0) {
    opt_level = g.MoveCopyAttr<int>("opt_level");
  }

  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");

  // Reference counter of each op node
  // For now, always store result when an op is referred more than once.
  std::vector<uint32_t> ref_count(idx.num_nodes(), 0);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      ++ref_count[e.node_id];
    }
  }
  for (const auto& e : idx.outputs()) {
    // this line will realize all the outputs
    ref_count[e.node_id] += 2;
  }
  // Pattern for the subgraph
  std::vector<TOpPattern> pattern_vec(idx.num_nodes(),  kOpaque);
  // Whether node can be fused to parent.
  std::vector<FuseRule> fuse_vec(idx.num_nodes(), FuseRule::kUknown);
  // Master node id of fusion segment.
  std::vector<int> master_vec(idx.num_nodes(), -1);
  // Operator pattern
  static auto& op_pattern = nnvm::Op::GetAttr<TOpPattern>("TOpPattern");

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      fuse_vec[nid] = FuseRule::kRealize; continue;
    }
    TOpPattern pt = op_pattern.get(inode.source->op(), kOpaque);

    if (pt <= kBroadcast) {
      // Try to check if we can fuse to the master.
      int chosen_master = -1;
      bool ewise = inode.source->num_outputs() == 1;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          TOpPattern ipt = pattern_vec[e.node_id];
          if (ipt != kElemWise) ewise = false;
          if (ipt <= kInjective) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else if (ipt == kOutEWiseFusable &&
                     chosen_master == -1 &&
                     shape_vec[idx.entry_id(nid, 0)] == shape_vec[idx.entry_id(e)]) {
            chosen_master = master_vec[e.node_id];
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else {
            fuse_vec[e.node_id] = FuseRule::kRealize;
          }
        }
        if (ewise) {
          if (shape_vec[idx.entry_id(nid, 0)] != shape_vec[idx.entry_id(e)]) {
            ewise = false;
          }
        }
      }
      master_vec[nid] = chosen_master;
      if (chosen_master != -1) {
        pt = kOutEWiseFusable;
      } else {
        pt = ewise ? kElemWise : kBroadcast;
      }
    } else if (pt == kInjective || pt == kCommReduce) {
      // fuse to the comm reduce or injective
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          TOpPattern ipt = pattern_vec[e.node_id];
          if (ipt <= kInjective) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else {
            fuse_vec[e.node_id] = FuseRule::kRealize;
          }
        }
      }
      if (pt == kCommReduce) {
        master_vec[nid] = nid;
      }
    } else {
      // realize
      master_vec[nid] = nid;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          fuse_vec[e.node_id] = FuseRule::kRealize;
          if (master_vec[e.node_id] == -1) {
            master_vec[e.node_id] = e.node_id;
          }
        }
      }
    }

    pattern_vec[nid] = pt;
    if (ref_count[nid] > 1 || opt_level < 1) {
      fuse_vec[nid] = FuseRule::kRealize;
      if (master_vec[nid] == -1) {
        master_vec[nid] = nid;
      }
    }
  }

  // point to the group root id of each node
  std::vector<int> group_vec(idx.num_nodes(), -1);
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    if (group_vec[nid] == -1) {
      group_vec[nid] = nid;
    }
    // propagate the group id.
    for (const auto& e : inode.inputs) {
      if (fuse_vec[e.node_id] == FuseRule::kFuseToMaster) {
        CHECK(group_vec[e.node_id] == -1||
              group_vec[e.node_id] == group_vec[nid]);
        group_vec[e.node_id] = group_vec[nid];
      }
    }
  }
  g.attrs["group_root"] = std::make_shared<any>(std::move(group_vec));
  g.attrs["group_master"] = std::make_shared<any>(std::move(master_vec));
  g.attrs["pattern"] = std::make_shared<any>(std::move(pattern_vec));
  return g;
}


NNVM_REGISTER_PASS(GraphFusePartition)
.set_body(GraphFusePartition)
.depend_graph_attr("shape")
.depend_graph_attr("dtype");

struct INodeEntryHash {
  size_t operator()(const IndexedGraph::NodeEntry& e) const {
    return e.node_id;
  }
};

struct INodeEntryEqual {
  size_t operator()(const IndexedGraph::NodeEntry& a,
                    const IndexedGraph::NodeEntry& b) const {
    return a.node_id == b.node_id && a.index == b.index;
  }
};

// Auxiliary data structure for representing fused op.
struct FuseEntry {
  // subgraph of the fragement
  Graph subgraph;
  // The input map
  std::unordered_map<IndexedGraph::NodeEntry, nnvm::NodeEntry,
                     INodeEntryHash, INodeEntryEqual> imap;
  // reverse map to the old input entry
  std::unordered_map<const Node*, IndexedGraph::NodeEntry> reverse_imap;
  // TVM Placeholder for inputs
  std::unordered_map<const Node*, Tensor> input_info;
  // Whether we can flatten data
  bool flatten_data;
  // The corresponding function.
  GraphFunc compiled_func;
};

// Fuse the partitioned graph into segments.
// Create a new graph with fused noded.
// Also inheritate attribute shape, dltype from previous graph.
nnvm::Graph GraphFuseCompile(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const std::vector<int>& group_vec = g.GetAttr<std::vector<int> >("group_root");
  const std::vector<int>& master_vec = g.GetAttr<std::vector<int> >("group_master");
  const std::vector<TOpPattern>& pattern_vec =
      g.GetAttr<std::vector<TOpPattern> >("pattern");
  std::string target = g.GetAttr<std::string>("target");
  std::vector<FuseEntry> fuse_vec(idx.num_nodes());
  // setup inputs and placeholder.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    CHECK_GE(group_vec[nid], 0);
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    fe.flatten_data = (pattern_vec[root_id] == kElemWise);
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id && fe.imap.count(e) == 0) {
        Array<Expr> shape;
        if (fe.flatten_data) {
          // elementwise support flatten
          int64_t prod = 1;
          for (int64_t x : shape_vec[idx.entry_id(e)]) {
            prod *= x;
          }
          CHECK_LE(prod, static_cast<int64_t>(std::numeric_limits<int>::max()));
          shape.push_back(make_const(Int(32), prod));
        } else {
          for (int64_t x : shape_vec[idx.entry_id(e)]) {
            CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
            shape.push_back(make_const(Int(32), x));
          }
        }
        std::ostringstream os_name;
        os_name << "input" << fe.imap.size();
        Tensor data = placeholder(
            shape, TVMType2Type(GetDLType(dtype_vec[idx.entry_id(e)])),
            os_name.str());
        NodeEntry garg = Symbol::CreateVariable(os_name.str()).outputs[0];
        fe.imap[e] = garg;
        fe.reverse_imap[garg.node.get()] = e;
        fe.input_info[garg.node.get()] = std::move(data);
      }
    }
  }
  // Setup the Subgraph
  std::vector<NodeEntry> subgraph_vec(idx.num_node_entries());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    // copy and create subgraph node.
    NodePtr gnode = Node::Create();
    gnode->attrs = inode.source->attrs;
    // input loading
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id) {
        auto it = fe.imap.find(e);
        CHECK(it != fe.imap.end());
        gnode->inputs.push_back(it->second);
      } else {
        const NodeEntry& ne = subgraph_vec[idx.entry_id(e)];
        CHECK(!idx[e.node_id].source->is_variable());
        CHECK(ne.node != nullptr);
        gnode->inputs.push_back(ne);
      }
    }
    // schedule on root node, and use master's schedule
    if (nid != root_id) {
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        subgraph_vec[eid] = NodeEntry{gnode, index, 0};
      }
    } else {
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        fe.subgraph.outputs.push_back(NodeEntry{gnode, index, 0});
      }
    }
  }
  // Start lowering
  Array<tvm::LoweredFunc> func_list;
  std::unordered_set<const tvm::Node*> func_set;

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    int root_id = group_vec[nid];
    if (nid != root_id) continue;
    int master = master_vec[root_id];
    FuseEntry& fe = fuse_vec[root_id];

    const IndexedGraph& subidx = fe.subgraph.indexed_graph();
    CHECK_EQ(subidx.input_nodes().size(), fe.imap.size());
    CHECK_EQ(subidx.input_nodes().size(), fe.input_info.size());

    Array<Tensor> inputs;
    for (uint32_t sub_input_id : subidx.input_nodes()) {
      auto it = fe.input_info.find(subidx[sub_input_id].source);
      inputs.push_back(it->second);
    }
    fe.compiled_func = GraphLower(fe.subgraph, inputs, target,
                                  idx[master].source->op(),
                                  idx[master].source->attrs);
    for (LoweredFunc f : fe.compiled_func->funcs) {
      if (!func_set.count(f.get())) {
        func_set.insert(f.get());
        func_list.push_back(f);
      }
    }
  }
  // Rebuild the fused graph
  const nnvm::Op* tvm_op = nnvm::Op::Get("tvm_op");
  std::unordered_map<uint32_t, nnvm::NodePtr> old_new;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      // only copy over name since that is sufficient.
      nnvm::NodePtr np = nnvm::Node::Create();
      np->attrs.name = inode.source->attrs.name;
      old_new[nid] = np;
      continue;
    }
    int root_id = group_vec[nid];
    if (nid != root_id) continue;
    FuseEntry& fe = fuse_vec[root_id];
    const IndexedGraph& subidx = fe.subgraph.indexed_graph();
    nnvm::NodePtr np = nnvm::Node::Create();
    np->attrs.op = tvm_op;
    np->attrs.name = inode.source->attrs.name;
    runtime::TVMOpParam param;
    param.func_name = fe.compiled_func->func_name;
    param.num_inputs = static_cast<uint32_t>(fe.imap.size());
    param.num_outputs = static_cast<uint32_t>(fe.subgraph.outputs.size());
    param.flatten_data = fe.flatten_data;
    param.UpdateDict(&(np->attrs.dict));
    np->attrs.parsed = std::move(param);

    for (uint32_t sub_input_id : subidx.input_nodes()) {
      // Need to make sure subgraph input order meets order of the graph input
      auto rit = fe.reverse_imap.find(subidx[sub_input_id].source);
      CHECK(rit != fe.reverse_imap.end());
      const IndexedGraph::NodeEntry& e = rit->second;
      auto it = old_new.find(e.node_id);
      CHECK(it != old_new.end())
          << "cannot find node_id=" << e.node_id;
      np->inputs.emplace_back(
          nnvm::NodeEntry{it->second, e.index, e.version});
    }
    for (const uint32_t node_id : inode.control_deps) {
      auto it = old_new.find(node_id);
      CHECK(it != old_new.end());
      np->control_deps.emplace_back(it->second);
    }
    old_new[nid] = np;
  }
  nnvm::Graph ret;
  for (const auto& e : idx.outputs()) {
    auto it = old_new.find(group_vec[e.node_id]);
    CHECK(it != old_new.end())
        << "cannot find node_id=" << e.node_id;
    ret.outputs.emplace_back(
        nnvm::NodeEntry{it->second, e.index, e.version});
  }

  const IndexedGraph& new_idx = ret.indexed_graph();
  ShapeVector new_shape_vec = ShapeVector(new_idx.num_node_entries(), TShape());
  DTypeVector new_dtype_vec = DTypeVector(new_idx.num_node_entries());
  std::vector<std::string> new_dltype_vec(new_idx.num_node_entries());
  for (const auto& kv : old_new) {
    uint32_t nid = kv.first;
    const auto& inode = idx[nid];
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      uint32_t new_eid = new_idx.entry_id(new_idx.node_id(kv.second.get()), i);
      uint32_t old_eid = idx.entry_id(nid, i);
      new_shape_vec[new_eid] = shape_vec[old_eid];
      new_dtype_vec[new_eid] = dtype_vec[old_eid];
      new_dltype_vec[new_eid] = tvm::runtime::TVMType2String(
          GetDLType(dtype_vec[old_eid]));
    }
  }
  ret.attrs["shape"] = std::make_shared<any>(std::move(new_shape_vec));
  ret.attrs["dtype"] = std::make_shared<any>(std::move(new_dtype_vec));
  ret.attrs["dltype"] = std::make_shared<any>(std::move(new_dltype_vec));
  // Setup module
  static const PackedFunc& fbuild = GetPackedFunc("nnvm.compiler.build_target");
  tvm::runtime::Module module = fbuild(func_list, target);
  ret.attrs["module"] = std::make_shared<any>(std::move(module));
  ret = nnvm::ApplyPass(ret, "PlanMemory");
  return ret;
}

NNVM_REGISTER_PASS(GraphFuseCompile)
.set_body(GraphFuseCompile);

}  // namespace compiler
}  // namespace nnvm
