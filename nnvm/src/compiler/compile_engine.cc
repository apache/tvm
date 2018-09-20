/*!
 *  Copyright (c) 2017 by Contributors
 * \file compile_engine.cc
 * \brief The compile engine.
 */
#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/operation.h>
#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <nnvm/pass_functions.h>
#include <nnvm/compiler/op_attr_types.h>
#include <mutex>
#include <tuple>
#include <vector>
#include <limits>
#include "graph_hash.h"
#include "compile_engine.h"

namespace nnvm {
namespace compiler {

using namespace tvm;

/*!
 * \brief Get type flag from TVM Type
 *
 * \param type the tvm type.
 * \return corresponding DLDataType
 */
int GetTypeFlag(tvm::Type type) {
  if (type == tvm::Float(32)) return 0;
  if (type == tvm::Float(64)) return 1;
  if (type == tvm::Float(16)) return 2;
  if (type == tvm::UInt(8)) return 3;
  if (type == tvm::Int(32)) return 4;
  if (type == tvm::Int(8)) return 5;
  if (type == tvm::Int(64)) return 6;
  if (type == tvm::Int(16)) return 7;
  if (type == tvm::UInt(16)) return 8;
  if (type == tvm::UInt(32)) return 9;
  if (type == tvm::UInt(64)) return 10;
  LOG(FATAL) << "cannot convert " << type;
  return 0;
}
// convert from type flag to tvm type.
Type GetTVMType(int type_flag) {
  switch (type_flag) {
    case 0:
      return tvm::Float(32);
    case 1:
      return tvm::Float(64);
    case 2:
      return tvm::Float(16);
    case 3:
      return tvm::UInt(8);
    case 4:
      return tvm::Int(32);
    case 5:
      return tvm::Int(8);
    case 6:
      return tvm::Int(64);
    case 7:
      return tvm::Int(16);
    case 8:
      return tvm::UInt(16);
    case 9:
      return tvm::UInt(32);
    case 10:
      return tvm::UInt(64);
    default:
      LOG(FATAL) << "unknown type_flag=" << type_flag;
      return Float(32);
  }
}

// internal compile engine
class CompileEngine {
 public:
  static CompileEngine* Global() {
    static CompileEngine inst;
    return &inst;
  }
  // lower graph possible get back an cached op.
  GraphFunc Lower(Graph graph,
                  const Array<tvm::Tensor>& inputs,
                  const std::string& target,
                  int master_idx) {
    GraphKey key = GraphKeyNode::make(graph, inputs, target);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      ++(it->second->use_count);
      return it->second->graph_func;
    }
    GraphFunc f = DoLower(key->graph, key->inputs, key->target, master_idx);
    auto n = tvm::make_node<GraphCacheEntryNode>();
    n->graph_func = f;
    n->use_count = 1;
    n->master_idx = master_idx;
    cache_[key] = GraphCacheEntry(n);
    return f;
  }
  // List all items in the cache.
  Array<NodeRef> ListCacheItems() {
    std::lock_guard<std::mutex> lock(mutex_);
    Array<NodeRef> items;
    for (auto& kv : cache_) {
      items.push_back(kv.first);
      auto n = tvm::make_node<GraphCacheEntryNode>(*(kv.second.operator->()));
      items.push_back(GraphCacheEntry(n));
    }
    return items;
  }
  // Find the function given graph key.
  GraphCacheEntry Find(const GraphKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    } else {
      return GraphCacheEntry();
    }
  }
  // Set the given function on given graph key.
  void Set(const GraphKey& key, GraphFunc func) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto n = tvm::make_node<GraphCacheEntryNode>();
    n->graph_func = func;
    n->use_count = 1;
    cache_[key] = GraphCacheEntry(n);
  }
    // Clear the function cache.
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
  }

  // get schedule and its args
  std::tuple<Schedule, Array<tvm::Tensor>, Graph>
  GetScheduleArgs(Graph graph,
                  const Array<tvm::Tensor> &inputs,
                  const std::string &target,
                  int master_idx,
                  std::string *readable_name,
                  Array<tvm::Tensor> *outputs) {
    // shape, type
    static auto& fcompute =
        nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");
    static auto& fschedule =
        nnvm::Op::GetAttr<FTVMSchedule>("FTVMSchedule");

    std::vector<TShape> ishape;
    std::vector<int> idtype;

    for (const tvm::Tensor t : inputs) {
      std::vector<dim_t> shape;
      for (Expr v : t->shape) {
        CHECK(v.as<tvm::ir::IntImm>());
        shape.push_back(v.as<tvm::ir::IntImm>()->value);
      }
      ishape.emplace_back(TShape(shape.begin(), shape.end()));
      idtype.emplace_back(GetTypeFlag(t->dtype));
    }
    graph = pass::InferShape(graph, ishape);
    graph = pass::InferType(graph, idtype);

    const ShapeVector& shape_vec = graph.GetAttr<ShapeVector>("shape");
    const DTypeVector& dtype_vec = graph.GetAttr<DTypeVector>("dtype");
    const IndexedGraph& idx = graph.indexed_graph();
    CHECK_EQ(inputs.size(), idx.input_nodes().size());

    std::vector<tvm::Tensor> tensor_vec(idx.num_node_entries());
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes()[i];
      tensor_vec[idx.entry_id(nid, 0)] = inputs[i];
    }

    std::ostringstream readable_name_os;
    readable_name_os << "fuse";
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) continue;
      Array<Tensor> op_inputs, out_info;
      readable_name_os << "_" << inode.source->op()->name;
      // input array
      for (const IndexedGraph::NodeEntry& e : inode.inputs) {
        const tvm::Tensor& t = tensor_vec[idx.entry_id(e)];
        CHECK(t.defined());
        op_inputs.push_back(t);
      }
      // output hint
      for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
        Array<Expr> shape;
        for (int64_t x : shape_vec[idx.entry_id(nid, i)]) {
          CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
          shape.push_back(make_const(Int(32), x));
        }
        out_info.push_back(
            placeholder(shape,
                        GetTVMType(dtype_vec[idx.entry_id(nid, i)])));
      }
      // get default
      Array<Tensor> out = fcompute[inode.source->op()](
          inode.source->attrs, op_inputs, out_info);
      CHECK_EQ(out.size(), inode.source->num_outputs());

      // check output dimentions also match
      // This check is to make sure the NNVM operator Infer match with Compute result.
      // Missing this check may pass the build but leads to runtime errors.
      for (uint32_t i = 0; i < out.size(); ++i) {
        CHECK_EQ(out[i].ndim(), out_info[i].ndim()) << inode.source->op()->name;
        tvm::Tensor inferred_tensor = out[i];
        tvm::Tensor computed_tensor = out_info[i];
        for (uint32_t j = 0; j < inferred_tensor->shape.size(); ++j) {
          if ((as_const_int(inferred_tensor->shape[j])) &&
              (as_const_int(computed_tensor->shape[j])))
            CHECK_EQ((*as_const_int(inferred_tensor->shape[j])),
                     (*as_const_int(computed_tensor->shape[j]))) << inode.source->op()->name;
        }
      }

      // schedule on root node, and use master's schedule
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        tensor_vec[eid] = out[index];
      }
    }
    // Schedule on final output.
    Array<Tensor> all_args = inputs;
    Array<Tensor> outs;
    for (const IndexedGraph::NodeEntry& e : idx.outputs()) {
      const tvm::Tensor& t = tensor_vec[idx.entry_id(e)];
      CHECK(t.defined());
      outs.push_back(t);
      all_args.push_back(t);
    }

    Schedule sch = fschedule[idx[master_idx].source->op()](
        idx[master_idx].source->attrs, outs, target);

    // store extra return values
    if (readable_name != nullptr) {
      *readable_name = readable_name_os.str();
    }
    if (outputs != nullptr) {
      *outputs = outs;
    }

    return std::make_tuple(sch, all_args, graph);
  }

  // run the actual lowering process
  GraphFunc DoLower(Graph graph,
                    const Array<tvm::Tensor>& inputs,
                    const std::string& target,
                    int master_idx) {
    std::string readable_name;
    Array<tvm::Tensor> all_args;
    Array<tvm::Tensor> outputs;
    Schedule sch;

    std::tie(sch, all_args, graph) = GetScheduleArgs(
        graph, inputs, target, master_idx,
        &readable_name, &outputs);

    auto gf = tvm::make_node<GraphFuncNode>();
    gf->target = target;
    gf->func_name = GetUniqeName(readable_name);
    gf->inputs = inputs;
    gf->outputs = outputs;
    static const PackedFunc& flower = GetPackedFunc("nnvm.compiler.lower");
    gf->funcs = flower(sch, all_args, gf->func_name, graph);
    return GraphFunc(gf);
  }

 private:
  // Get unique name
  std::string GetUniqeName(std::string name) {
    while (true) {
      auto it = name_map_.find(name);
      if (it == name_map_.end()) {
        name_map_[name] = 1;
        return name;
      } else {
        std::ostringstream os;
        os << name << "_" << it->second;
        ++(it->second);
        name = os.str();
      }
    }
    return name;
  }

  // global mutex
  std::mutex mutex_;
  // the name map
  std::unordered_map<std::string, int> name_map_;
  // the compiler cache
  std::unordered_map<GraphKey, GraphCacheEntry,
                     GraphKeyHash, GraphKeyEqual> cache_;
};

GraphFunc GraphLower(Graph graph,
                     const Array<tvm::Tensor>& inputs,
                     const std::string& target,
                     int master_idx) {
  return CompileEngine::Global()->Lower(
      graph, inputs, target, master_idx);
}

// Expose cache to front end
TVM_REGISTER_GLOBAL("nnvm.compiler.ListCacheItems")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = CompileEngine::Global()->ListCacheItems();
  });

TVM_REGISTER_GLOBAL("nnvm.compiler.ClearCache")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    CompileEngine::Global()->Clear();
  });

// NOTE: this involves graph lookup and can be slow
TVM_REGISTER_GLOBAL("nnvm.compiler.GetCacheItem")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = CompileEngine::Global()->Find(args[0]);
  });

TVM_REGISTER_GLOBAL("nnvm.compiler.SetCacheItem")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    CompileEngine::Global()->Set(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("nnvm.compiler.GraphKeyGetGraph")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = args[0].operator GraphKey()->graph;
  });

TVM_REGISTER_GLOBAL("nnvm.compiler.MakeGraphKey")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = GraphKeyNode::make(args[0], args[1], args[2]);
  });

// This can be used to extract workloads from nnvm compiler
TVM_REGISTER_GLOBAL("nnvm.compiler.CacheItem2ScheduleArgs")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    Array<tvm::NodeRef> item = args[0];

    const GraphKeyNode *key = reinterpret_cast<const GraphKeyNode *>(item[0].get());
    const GraphCacheEntryNode *value = reinterpret_cast<const GraphCacheEntryNode *>(item[1].get());

    // extract arguments from cached item
    Graph graph = key->graph;
    const Array<tvm::Tensor> &inputs = key->inputs;
    std::string target = args[1];
    int master_idx = value->master_idx;

    Schedule sch;
    Array<tvm::Tensor> all_args;
    std::tie(sch, all_args, graph) =
        CompileEngine::Global()->GetScheduleArgs(
        graph, inputs, target, master_idx, nullptr, nullptr);

    Array<tvm::NodeRef> ret;
    ret.push_back(sch);
    ret.push_back(all_args);
    *rv = ret;
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<GraphFuncNode>([](const GraphFuncNode *op, IRPrinter *p) {
    p->stream << "GraphFunc(name=" << op->func_name
              << ", addr=" << op << ")";
});

}  // namespace compiler
}  // namespace nnvm
