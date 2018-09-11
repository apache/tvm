/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph_compile.cc
 * \brief Compile a graph. It lowers the graph nodes into low level IR.
 */

#include <dmlc/parameter.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/tuple.h>
#include <tvm/lowered_func.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>

#include "compile_engine.h"
#include "graph_fuse.h"
#include "graph_runtime.h"
#include "pattern_util.h"

namespace nnvm {
namespace compiler {

using namespace tvm;

// Decorate the result of PlanMemory
// This function does two things:
// - Give separate memory to each variable.
// - Tie the memory of output/lhs in assign node properly
//   so the execution of assign can have side effect.
nnvm::Graph DecorateMemoryPlan(
    nnvm::Graph g,
    const std::vector<int>& assign_flag) {
  const IndexedGraph& idx = g.indexed_graph();
  StorageVector storage_vec = g.MoveCopyAttr<StorageVector>("storage_id");
  g.attrs.erase("storage_allocated_bytes");
  g.attrs.erase("storage_inplace_index");
  size_t num_not_allocated = g.MoveCopyAttr<size_t>(
      "storage_num_not_allocated");
  CHECK_EQ(num_not_allocated, 0U)
      << "Can only build inference graph with all statically allocated memory";

  // Reassign variable id so that they are different.
  int max_id = 0;
  for (size_t i = 0; i < storage_vec.size(); ++i) {
    max_id = std::max(storage_vec[i] + 1, max_id);
  }
  for (uint32_t nid : idx.input_nodes()) {
    storage_vec[idx.entry_id(nid, 0)] = max_id++;
  }
  // Tie up the assign node storage properly.
  for (uint32_t nid = 0 ; nid < idx.num_nodes(); ++nid) {
    if (assign_flag[nid] == 0) continue;
    const auto& inode = idx[nid];
    int var_storage_id = storage_vec[idx.entry_id(inode.inputs[0])];
    if (inode.source->attrs.device ==
        idx[inode.inputs[0].node_id].source->attrs.device) {
      storage_vec[idx.entry_id(nid, 0)] = var_storage_id;
    }

    if (assign_flag[nid] == 2) {
      if (inode.source->attrs.device ==
          idx[inode.inputs[0].node_id].source->attrs.device) {
        storage_vec[idx.entry_id(inode.inputs[1])] = var_storage_id;
      }
    }
  }
  g.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
  return g;
}

nnvm::Graph GraphCompile(const nnvm::Graph& g) {
  // Get attributes from the graph.
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const GroupVec& group_vec = g.GetAttr<GroupVec>("group_root");
  const MasterVec& master_vec = g.GetAttr<MasterVec>("group_master");
  const PatternVec& pattern_vec = g.GetAttr<PatternVec>("pattern");

  CHECK(g.HasAttr("fused_entry")) << "Fusion hasn't been applied yet.";
  FuseEntryVec fuse_entries = g.GetAttr<FuseEntryVec>("fused_entry");

  std::string target, target_host;
  if (g.HasAttr("target")) {
    target = g.GetAttr<std::string>("target");
  }

  if (g.HasAttr("target_host")) {
    target_host = g.GetAttr<std::string>("target_host");
  }
  // Specially handle assign.
  const nnvm::Op* assign_op = nnvm::Op::Get("_assign");

  // Start lowering.
  std::unordered_map<DLDeviceType, Array<tvm::LoweredFunc>,
                     runtime::DLDeviceTypeHash>
      func_dev_map;
  std::unordered_set<const tvm::Node*> func_set;
  const IndexedGraph& idx = g.indexed_graph();

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    int root_id = group_vec[nid];
    if (static_cast<int>(nid) != root_id) continue;
    int master = master_vec[root_id];
    FuseEntry& fe = fuse_entries[root_id];
    fe.device = inode.source->attrs.device;

    // No need to lower cross devcie copy node. The actual data copy will happen
    // at runtime.
    if (inode.source->attrs.name.rfind("__copy", 0) == 0) continue;

    const IndexedGraph& subidx = fe.subgraph.indexed_graph();
    CHECK_EQ(subidx.input_nodes().size(), fe.imap.size());
    CHECK_EQ(subidx.input_nodes().size(), fe.input_info.size());

    Array<Tensor> inputs;
    for (uint32_t sub_input_id : subidx.input_nodes()) {
      auto it = fe.input_info.find(subidx[sub_input_id].source);
      inputs.push_back(it->second);
    }
    // Find master idx in the subgraph.
    int sub_master_idx = 0;
    for (uint32_t i = 0; i < subidx.num_nodes(); i++) {
      if (subidx[i].source->op() == idx[master].source->op()) {
        sub_master_idx = i;
        break;
      }
    }

    const auto& device_name = tvm::runtime::DeviceName(fe.device);
    const auto& target_ctx = "target" + device_name;
    if (g.HasAttr(target_ctx)) {
      std::string cur_target = g.GetAttr<std::string>(target_ctx);
      fe.compiled_func =
          GraphLower(fe.subgraph, inputs, cur_target, sub_master_idx);
    } else {
      CHECK_EQ(fe.device, tvm::runtime::kDLDefaultDevice)
          << "Target is not provided for " << device_name << "\n";
      fe.compiled_func =
          GraphLower(fe.subgraph, inputs, target, sub_master_idx);
    }

    for (LoweredFunc f : fe.compiled_func->funcs) {
      if (!func_set.count(f.get())) {
        func_set.insert(f.get());
        // LOG(INFO) << "ffffffffffffff " << fe.device << "    " << f->name;
        func_dev_map[fe.device].push_back(f);
      }
    }
  }

  const nnvm::Op* tvm_op = nnvm::Op::Get("tvm_op");
  std::unordered_set<int> device_types;
  std::unordered_map<uint32_t, nnvm::NodePtr> old_new;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    device_types.emplace(static_cast<int>(inode.source->attrs.device));

    if (inode.source->is_variable()) {
      // Only copy name since that is sufficient.
      nnvm::NodePtr np = nnvm::Node::Create();
      np->attrs.name = inode.source->attrs.name;
      np->attrs.device = inode.source->attrs.device;
      old_new[nid] = np;
      continue;
    }
    int root_id = group_vec[nid];
    if (static_cast<int>(nid) != root_id) continue;

    // Handle normal op
    FuseEntry& fe = fuse_entries[root_id];
    const IndexedGraph& subidx = fe.subgraph.indexed_graph();
    nnvm::NodePtr np = nnvm::Node::Create();
    np->attrs.name = inode.source->attrs.name;
    np->attrs.device = inode.source->attrs.device;
    TVMOpParam param;
    if (inode.source->attrs.name.rfind("__copy", 0) == 0) {
      np->attrs.op = inode.source->attrs.op;
      param.func_name = "__copy";
    } else {
      np->attrs.op = tvm_op;
      param.func_name = fe.compiled_func->func_name;
    }
    param.num_inputs = static_cast<uint32_t>(fe.imap.size());
    param.num_outputs = static_cast<uint32_t>(fe.subgraph.outputs.size());
    param.flatten_data = fe.flatten_data;
    param.UpdateDict(&(np->attrs.dict));
    np->attrs.parsed = std::move(param);

    for (uint32_t sub_input_id : subidx.input_nodes()) {
      // Need to make sure subgraph input order is consistent to the order of
      // the graph input.
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

  // Reference counter of each op node.
  // For now, always store result when an op is referred more than once.
  std::vector<uint32_t> ref_count = GetNodeRefCounts(idx);
  for (const auto& e : idx.outputs()) {
    // This line will realize all the outputs.
    ref_count[e.node_id] += 1;
  }

  const IndexedGraph& new_idx = ret.indexed_graph();

  // Handling assign:
  //
  //  assign is a special operator that mutates the variable.
  //  Currently assign is implemented as output = copy(input[1])
  //  Then we run DecorateMemoryPlan to force
  //  output.storage = input[0].storage
  //
  std::vector<int> assign_flag(new_idx.num_nodes(), 0);
  ShapeVector new_shape_vec = ShapeVector(new_idx.num_node_entries(), TShape());
  DTypeVector new_dtype_vec = DTypeVector(new_idx.num_node_entries());
  std::vector<std::string> new_dltype_vec(new_idx.num_node_entries());

  for (const auto& kv : old_new) {
    uint32_t nid = kv.first;
    const auto& inode = idx[nid];
    uint32_t new_nid = new_idx.node_id(kv.second.get());

    if (inode.source->op() == assign_op) {
      // Check if rhs of assign can be computed inplace.
      // If yes, we can simply set that memory to be assign target
      // and change assign to nop.
      const IndexedGraph::NodeEntry& rhs = inode.inputs[1];
      if (ref_count[rhs.node_id] <= 1 &&
          !(idx[rhs.node_id].source->is_variable()) &&
          pattern_vec[group_vec[rhs.node_id]] <= kBroadcast) {
        assign_flag[new_nid] = 2;
        TVMOpParam& param = dmlc::get<TVMOpParam>(kv.second->attrs.parsed);
        param.func_name = "__nop";
        param.UpdateDict(&(kv.second->attrs.dict));
      } else {
        assign_flag[new_nid] = 1;
      }
    }
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
  if (device_types.size() > 1) {
    for (const auto &it : func_dev_map) {
      std::string device_name = tvm::runtime::DeviceName(it.first);
      std::string target_ctx = "target" + device_name;
      CHECK(g.HasAttr(target_ctx))
          << "Graph doesn't have the attribute with target " << device_name;
      std::string cur_target = g.GetAttr<std::string>(target_ctx);
      tvm::runtime::Module module = fbuild(it.second, cur_target, target_host);
      ret.attrs["module" + device_name] =
          std::make_shared<any>(std::move(module));
    }

    DeviceVector device_vec(new_idx.num_nodes());
    for (size_t i = 0; i < new_idx.num_nodes(); i++) {
      device_vec[i] = static_cast<int>(new_idx[i].source->attrs.device);
    }
    ret.attrs["device"] = std::make_shared<any>(std::move(device_vec));
  } else {
    const auto& it = func_dev_map.begin();
    std::string device_name = tvm::runtime::DeviceName(it->first);
    std::string target_ctx = "target" + device_name;
    std::string cur_target = target;
    if (g.HasAttr(target_ctx)) {
      cur_target = g.GetAttr<std::string>(target_ctx);
      // Only one device/context is annotated on the graph. The device name is
      // tied to returned graph to make the heterogeneous build aware which
      // device the whole graph should be schduled to.
      ret.attrs["context"] = std::make_shared<any>(std::move(device_name));
    }
    tvm::runtime::Module module = fbuild(it->second, cur_target, target_host);
    ret.attrs["module"] = std::make_shared<any>(std::move(module));
  }
  ret = nnvm::ApplyPass(ret, "PlanMemory");
  ret = DecorateMemoryPlan(ret, assign_flag);
  return ret;
}

NNVM_REGISTER_PASS(GraphCompile)
    .set_body(GraphCompile)
    .depend_graph_attr("shape")
    .depend_graph_attr("dtype")
    .depend_graph_attr("fused_entry")
    .depend_graph_attr("group_root")
    .depend_graph_attr("pattern")
    .depend_graph_attr("group_master");

}  // namespace compiler
}  // namespace nnvm
