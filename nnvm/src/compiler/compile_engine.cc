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
#include "./graph_hash.h"
#include "./compile_engine.h"

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
  LOG(FATAL) << "cannot convert " << type;
  return 0;
}
// convert from type flag to tvm type.
Type GetTVMType(int type_flag) {
  if (type_flag == 0) return tvm::Float(32);
  LOG(FATAL) << "unknown type_flag=" << type_flag;
  return Float(32);
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
                  const Op* schedule_op_key,
                  const NodeAttrs& schedule_op_attr) {
    GraphKey key = GraphKeyNode::make(graph, inputs, target);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      ++(it->second->use_count);
      return it->second->graph_func;
    }
    GraphFunc f = DoLower(key->graph, key->inputs, key->target,
                          schedule_op_key, schedule_op_attr);
    std::shared_ptr<GraphCacheEntryNode> n = std::make_shared<GraphCacheEntryNode>();
    n->graph_func = f;
    n->use_count = 1;
    cache_[key] = GraphCacheEntry(n);
    return f;
  }
  // List all items in the cache.
  Array<NodeRef> ListCacheItems() {
    std::lock_guard<std::mutex> lock(mutex_);
    Array<NodeRef> items;
    for (auto& kv : cache_) {
      items.push_back(kv.first);
      std::shared_ptr<GraphCacheEntryNode> n =
          std::make_shared<GraphCacheEntryNode>(*(kv.second.operator->()));
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
  // Find the function given graph key.
  void Set(const GraphKey& key, GraphFunc func) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_ptr<GraphCacheEntryNode> n = std::make_shared<GraphCacheEntryNode>();
    n->graph_func = func;
    n->use_count = 1;
    cache_[key] = GraphCacheEntry(n);
  }
    // Find the function given graph key.
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
  }
  // run the actual lowering process
  GraphFunc DoLower(Graph graph,
                    const Array<tvm::Tensor>& inputs,
                    const std::string& target,
                    const Op* schedule_op_key,
                    const NodeAttrs& schedule_op_attr) {
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

    std::ostringstream readable_name;
    readable_name << "fuse";
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) continue;
      Array<Tensor> inputs, out_info;
      readable_name << "_" << inode.source->op()->name;
      // input array
      for (const IndexedGraph::NodeEntry& e : inode.inputs) {
        const tvm::Tensor& t = tensor_vec[idx.entry_id(e)];
        CHECK(t.defined());
        inputs.push_back(t);
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
          inode.source->attrs, inputs, out_info);
      CHECK_EQ(out.size(), inode.source->num_outputs());
      // schedule on root node, and use master's schedule
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        tensor_vec[eid] = out[index];
      }
    }
    // Schedule on final output.
    Array<Tensor> outputs;
    Array<Tensor> all_args = inputs;
    for (const IndexedGraph::NodeEntry& e : idx.outputs()) {
      const tvm::Tensor& t = tensor_vec[idx.entry_id(e)];
      CHECK(t.defined());
      outputs.push_back(t);
      all_args.push_back(t);
    }
    Schedule sch = fschedule[schedule_op_key](
        schedule_op_attr, outputs, target);
    std::shared_ptr<GraphFuncNode> gf = std::make_shared<GraphFuncNode>();
    gf->target = target;
    gf->func_name = GetUniqeName(readable_name.str());
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
                     const Op* schedule_op_key,
                     const NodeAttrs& schedule_op_attr) {
  return CompileEngine::Global()->Lower(
      graph, inputs, target, schedule_op_key, schedule_op_attr);
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


TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<GraphFuncNode>([](const GraphFuncNode *op, IRPrinter *p) {
    p->stream << "GraphFunc(name=" << op->func_name
              << ", addr=" << op << ")";
});

}  // namespace compiler
}  // namespace nnvm
