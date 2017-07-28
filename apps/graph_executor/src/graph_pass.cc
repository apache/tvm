/*!
 *  Copyright (c) 2017 by Contributors
 * \file Additional optimization pass of NNVM.
 */
#include <dmlc/json.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>
#include <nnvm/pass.h>
#include <tvm/operation.h>
#include <tvm/lowered_func.h>
#include "./op_attr_types.h"

namespace tvm {
namespace contrib {

using nnvm::any;
using nnvm::IndexedGraph;

// The single fuse rule.
enum class FuseRule {
  kUknown,
  kFuseToParent,
  kRealize
};

DLDataType GetDLType(int type_flag) {
  if (type_flag == 0) return Type2TVMType(Float(32));
  LOG(FATAL) << "unknown type_flag=" << type_flag;
  return Type2TVMType(Float(32));
}

// Partition the graph into segments
// Each segment will be compiled into one operator.
// Need also mark the property of the segment.
nnvm::Graph GraphPartition(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  // Transform to dltype
  // In future, directly fo type inference in dltype.
  DLTypeVector dltype_vec = DLTypeVector(dtype_vec.size());
  for (size_t i = 0; i < dtype_vec.size(); ++i) {
    dltype_vec[i] = GetDLType(dtype_vec[i]);
  }

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
  // Pattern fo the subgraph
  std::vector<TOpPattern> pattern_vec(idx.num_nodes(),  kExtern);
  // Whether node can be fused to parent.
  std::vector<FuseRule> fuse_vec(idx.num_nodes(), FuseRule::kUknown);
  // Operator pattern
  static auto& op_pattern = nnvm::Op::GetAttr<TOpPattern>("TOpPattern");

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      fuse_vec[nid] = FuseRule::kRealize; continue;
    }
    TOpPattern pt = op_pattern.get(inode.source->op(), kExtern);
    if (pt <= kBroadcast) {
      // Looking for fusable bcast pattern
      bool ewise = inode.source->num_outputs() == 1;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          if (pattern_vec[e.node_id] == kBroadcast) {
            ewise = false;
            fuse_vec[e.node_id] = FuseRule::kFuseToParent;
          } else if (pattern_vec[e.node_id] == kElemWise) {
            fuse_vec[e.node_id] = FuseRule::kFuseToParent;
          }
        }
        if (ewise) {
          TShape oshape = shape_vec[idx.entry_id(nid, 0)];
          if (oshape != shape_vec[idx.entry_id(e)]) ewise = false;
        }
      }
      pt = ewise ? kElemWise : kBroadcast;
    } else if (pt == kComplex) {
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          if (pattern_vec[e.node_id] <= kBroadcast) {
            fuse_vec[e.node_id] = FuseRule::kFuseToParent;
          }
        }
      }
    }
    pattern_vec[nid] = pt;
    if (ref_count[nid] > 1) {
      fuse_vec[nid] = FuseRule::kRealize;
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
      if (fuse_vec[e.node_id] == FuseRule::kFuseToParent) {
        CHECK(group_vec[e.node_id] == -1||
              group_vec[e.node_id] == group_vec[nid]);
        group_vec[e.node_id] = group_vec[nid];
      }
    }
  }
  g.attrs["group_root"] = std::make_shared<any>(std::move(group_vec));
  g.attrs["pattern"] = std::make_shared<any>(std::move(pattern_vec));
  g.attrs["dltype"] = std::make_shared<any>(std::move(dltype_vec));
  return g;
}

NNVM_REGISTER_PASS(GraphPartition)
.set_body(GraphPartition)
.depend_graph_attr("shape")
.depend_graph_attr("dtype")
.provide_graph_attr("dltype");

struct NodeEntryHash {
  size_t operator()(const IndexedGraph::NodeEntry& e) const {
    return e.node_id;
  }
};

struct NodeEntryEqual {
  size_t operator()(const IndexedGraph::NodeEntry& a,
                    const IndexedGraph::NodeEntry& b) const {
    return a.node_id == b.node_id && a.index == b.index;
  }
};

// Auxiliary data structure for representing fused op.
struct FuseEntry {
  // The inputs
  std::vector<IndexedGraph::NodeEntry> inputs;
  // The input map
  std::unordered_map<IndexedGraph::NodeEntry, Tensor,
                     NodeEntryHash, NodeEntryEqual> imap;
  // Output tensors
  Array<Tensor> outputs;
  // Placeholder for inputs
  Array<Tensor> placeholder;
  // Computing schedule
  Schedule schedule;
  // Function name
  std::string func_name;
};

// Fuse the partitioned graph into segments.
// Create a new graph with fused noded.
// Also inheritate attribute shape, dltype from previous graph.
nnvm::Graph GraphFuse(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DLTypeVector& dltype_vec = g.GetAttr<DLTypeVector>("dltype");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const std::vector<int>& group_vec = g.GetAttr<std::vector<int> >("group_root");
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
    TOpPattern pt = pattern_vec[root_id];
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id && fe.imap.count(e) == 0) {
        Array<Expr> shape;
        if (pt == kElemWise) {
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
        os_name << "input" << fe.inputs.size();
        Tensor data = placeholder(
            shape, TVMType2Type(dltype_vec[idx.entry_id(e)]),
            os_name.str());
        fe.imap[e] = data;
        fe.inputs.push_back(e);
        fe.placeholder.push_back(data);
      }
    }
  }
  // Setup the Tensor
  std::vector<Tensor> tensor_vec(idx.num_node_entries());
  static auto& fcompute =
      nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");
  static auto& fschedule =
      nnvm::Op::GetAttr<FTVMSchedule>("FTVMSchedule");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    Array<Tensor> inputs;
    // input loading
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id) {
        auto it = fe.imap.find(e);
        CHECK(it != fe.imap.end());
        inputs.push_back(it->second);
      } else {
        Tensor t = tensor_vec[idx.entry_id(e)];
        CHECK(t.defined());
        inputs.push_back(t);
      }
    }
    Array<Tensor> out = fcompute[inode.source->op()](
        inode.source->attrs, inputs);
    CHECK_EQ(out.size(), inode.source->num_outputs());
    if (nid != root_id) {
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        tensor_vec[eid] = out[index];
      }
    } else {
      // Work on schedule
      fe.outputs = out;
      fe.schedule = fschedule[inode.source->op()](
          inode.source->attrs, fe.outputs, target);
      std::ostringstream os;
      os << inode.source->attrs.name + "_id" << nid;
      fe.func_name = os.str();
    }
  }
  static const PackedFunc& flower = GetPackedFunc("tvm_graph.lower");
  static const PackedFunc& fbuild = GetPackedFunc("tvm_graph.build_target");
  Array<tvm::LoweredFunc> funcs;
  for (const FuseEntry& fe : fuse_vec) {
    if (fe.schedule.defined()) {
      Array<tvm::Tensor> args = fe.placeholder;
      for (tvm::Tensor x : fe.outputs) {
        args.push_back(x);
      }
      Array<tvm::LoweredFunc> ret = flower(fe.schedule, args, fe.func_name);
      for (LoweredFunc x : ret) {
        funcs.push_back(x);
      }
    }
  }
  tvm::runtime::Module module = fbuild(funcs, target);
  // Final step: Remap the node, with given attribute
  const nnvm::Op* tvm_op = nnvm::Op::Get("tvm_op");

  std::unordered_map<uint32_t, nnvm::NodePtr> old_new;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      nnvm::NodePtr np = nnvm::Node::Create();
      np->attrs = inode.source->attrs;
      old_new[nid] = np;
    } else {
      int root_id = group_vec[nid];
      if (nid != root_id) continue;
      FuseEntry& fe = fuse_vec[root_id];
      nnvm::NodePtr np = nnvm::Node::Create();
      np->attrs.op = tvm_op;
      np->attrs.name = inode.source->attrs.name;
      np->attrs.dict["num_inputs"] = std::to_string(fe.inputs.size());
      np->attrs.dict["num_outputs"] = std::to_string(fe.outputs.size());
      np->attrs.dict["func_name"] = fuse_vec[nid].func_name;
      np->attrs.dict["flatten_data"] = std::to_string(pattern_vec[nid] == kElemWise);
      np->op()->attr_parser(&(np->attrs));
      for (const auto& e : fe.inputs) {
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
  }
  nnvm::Graph ret;
  for (const auto& e : idx.outputs()) {
    auto it = old_new.find(e.node_id);
    CHECK(it != old_new.end());
    ret.outputs.emplace_back(
        nnvm::NodeEntry{it->second, e.index, e.version});
  }
  const IndexedGraph& new_idx = ret.indexed_graph();
  ShapeVector new_shape_vec = ShapeVector(new_idx.num_node_entries(), TShape());
  DTypeVector new_dtype_vec = DTypeVector(new_idx.num_node_entries());
  DLTypeVector new_dltype_vec = DLTypeVector(new_idx.num_node_entries());
  for (const auto& kv : old_new) {
    uint32_t nid = kv.first;
    const auto& inode = idx[nid];
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      uint32_t new_eid = new_idx.entry_id(new_idx.node_id(kv.second.get()), i);
      uint32_t old_eid = idx.entry_id(nid, i);
      new_shape_vec[new_eid] = shape_vec[old_eid];
      new_dtype_vec[new_eid] = dtype_vec[old_eid];
      new_dltype_vec[new_eid] = dltype_vec[old_eid];
    }
  }
  ret.attrs["shape"] = std::make_shared<any>(std::move(new_shape_vec));
  ret.attrs["dtype"] = std::make_shared<any>(std::move(new_dtype_vec));
  ret.attrs["dltype"] = std::make_shared<any>(std::move(new_dltype_vec));
  ret.attrs["module"] = std::make_shared<any>(std::move(module));
  ret = nnvm::ApplyPass(ret, "PlanMemory");
  return ret;
}

NNVM_REGISTER_PASS(GraphFuse)
.set_body(GraphFuse);


inline bool IsIdentityLayout(const LayoutInfo& layout) {
  if (layout.src == "" && layout.dst == "") return true;
  return false;
}

inline bool IsPairedLayouts(const LayoutInfo& in,
                            const LayoutInfo& out) {
  if (in.src == out.dst && in.dst == out.src) return true;
  return false;
}

inline LayoutInfo GetLayout(const nnvm::OpMap<FTVMLayoutInfo>& layouts,
                            const nnvm::NodePtr& n, int idx) {
  return layouts[n->op()](n->attrs)[idx];
}

nnvm::NodePtr CreateLayoutTransformNode(const std::string& src,
                                        const std::string& dst) {
  static const nnvm::Op* trans_op = nnvm::Op::Get("layout_transform");
  static int count = 0;
  nnvm::NodePtr n = nnvm::Node::Create();
  n->attrs.op = trans_op;
  n->attrs.name = src + "_to_" + dst + std::to_string(count++);
  n->attrs.dict["src_layout"] = src;
  n->attrs.dict["dst_layout"] = dst;
  n->op()->attr_parser(&(n->attrs));
  return n;
}

/*!
 * \brief A simple layout transform pass that will
 *  insert layout transform nodes automatically.
 */
nnvm::Graph LayoutTransform(nnvm::Graph src) {
  static auto& ilayouts =
    nnvm::Op::GetAttr<FTVMInputsLayoutInfo>("FTVMInputsLayoutInfo");
  static auto& olayouts =
    nnvm::Op::GetAttr<FTVMOutputsLayoutInfo>("FTVMOutputsLayoutInfo");
  static auto& vec_op =
    nnvm::Op::GetAttr<FTVMVectorizedOp>("FTVMVectorizedOp");

  std::unordered_map<nnvm::Node*, nnvm::NodePtr> mirror_map;
  std::unordered_map<nnvm::Node*, std::vector<nnvm::NodePtr> > transformed;

  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
      nnvm::NodePtr new_node = nnvm::Node::Create();
      *new_node = *n;
      if (new_node->is_variable()) {
        mirror_map[n.get()] = new_node;
        return;
      }

      if (vec_op.count(n->op())) {
        new_node = vec_op[n->op()](n);
        new_node->inputs.resize(new_node->num_inputs());
      }

      if (olayouts.count(new_node->op())) {
        std::vector<nnvm::NodePtr> tnodes(n->num_outputs(), nullptr);
        std::vector<LayoutInfo> layouts = olayouts[new_node->op()](new_node->attrs);
        for (uint32_t i = 0; i < n->num_outputs(); ++i) {
          const LayoutInfo& layout = layouts[i];
          if (!IsIdentityLayout(layout)) {
            tnodes[i] = CreateLayoutTransformNode(layout.src, layout.dst);
            tnodes[i]->attrs.name = new_node->attrs.name + "_" + layout.dst;
            tnodes[i]->inputs.emplace_back(nnvm::NodeEntry{new_node, i, 0});
          }
        }
        transformed.emplace(n.get(), std::move(tnodes));
      }

      for (size_t idx = 0; idx < n->inputs.size(); ++idx) {
        const nnvm::NodeEntry& e = n->inputs[idx];
        const nnvm::NodePtr& in = mirror_map.at(e.node.get());
        new_node->inputs[idx] =
          nnvm::NodeEntry{in, e.index, e.version};

        bool otrans = olayouts.count(in->op());
        bool itrans = ilayouts.count(new_node->op());
        if (otrans && itrans) {
          LayoutInfo ilayout = GetLayout(ilayouts, new_node, idx);
          LayoutInfo olayout = GetLayout(olayouts, in, e.index);
          if (IsPairedLayouts(olayout, ilayout)) {
            continue;
          }
        }

        if (otrans) {
          nnvm::NodePtr tnode = transformed.at(in.get())[e.index];
          if (tnode.get()) {
            new_node->inputs[idx] =
              nnvm::NodeEntry{tnode, 0, 0};
          }
        }

        if (itrans) {
          LayoutInfo layout = GetLayout(ilayouts, new_node, idx);
          if (!IsIdentityLayout(layout)) {
            nnvm::NodePtr tnode =
              CreateLayoutTransformNode(layout.src, layout.dst);
            tnode->attrs.name = n->inputs[idx].node->attrs.name + "_" + layout.dst;
            tnode->inputs.emplace_back(new_node->inputs[idx]);
            new_node->inputs[idx] = nnvm::NodeEntry{tnode, 0, 0};
          }
        }
      }
      mirror_map[n.get()] = std::move(new_node);
    });

  std::vector<nnvm::NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    if (olayouts.count(e.node->op())) {
      nnvm::NodePtr tnode = transformed.at(e.node.get())[e.index];
      if (tnode.get()) {
        outputs.emplace_back(nnvm::NodeEntry{tnode, 0, 0});
        continue;
      }
    }
    outputs.emplace_back(
      nnvm::NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
  }

  nnvm::Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

NNVM_REGISTER_PASS(LayoutTransform)
.set_body(LayoutTransform);

DMLC_REGISTER_PARAMETER(LayoutTransformParam);

/*! \brief Parse keyword arguments as PType arguments and save to parsed */
template<typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

NNVM_REGISTER_OP(layout_transform)
.set_attr_parser(ParamParser<LayoutTransformParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_arguments(LayoutTransformParam::__FIELDS__());


nnvm::Graph PruneGraph(nnvm::Graph src) {
  const auto& params = src.GetAttr<std::unordered_set<std::string>>("params");

  std::unordered_set<nnvm::Node*> pruned;
  nnvm::NodeEntryMap<nnvm::NodePtr> entry_var;
  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
    bool can_be_pruned = true;
    if (n->is_variable()) {
      if (params.count(n->attrs.name)) {
        pruned.emplace(n.get());
      }
      can_be_pruned = false;
    }

    for (const auto& e : n->inputs) {
      if (!pruned.count(e.node.get())) {
        can_be_pruned = false;
      }
    }
    if (can_be_pruned) {
      pruned.emplace(n.get());
    } else {
      // scan again to find edge nodes, skip variables
      for (auto& e : n->inputs) {
        if (!e.node->is_variable() && pruned.count(e.node.get())) {
          if (!entry_var.count(e)) {
            nnvm::NodePtr var = nnvm::Node::Create();
            var->attrs.name = e.node->attrs.name + "_output" + std::to_string(e.index);
            entry_var.emplace(e, var);
          }
          e = nnvm::NodeEntry{entry_var.at(e), 0, 0};
        }
      }
    }
  });

  nnvm::Graph pre_graph;
  pre_graph.outputs.reserve(entry_var.size());
  std::vector<std::string> output_names;
  output_names.reserve(entry_var.size());
  for (auto kv : entry_var) {
    if (kv.first.node->is_variable()) continue;
    pre_graph.outputs.emplace_back(kv.first);
    output_names.emplace_back(kv.second->attrs.name);
  }

  pre_graph.attrs["pruned_params"] =
    std::make_shared<dmlc::any>(std::move(output_names));
  src.attrs["pre_graph"] =
    std::make_shared<dmlc::any>(std::move(pre_graph));
  return src;
}

NNVM_REGISTER_PASS(PruneGraph)
.set_body(PruneGraph);
}  // namespace contrib
}  // namespace tvm
