/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_executor.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <numeric>
#include "./graph_executor.h"

namespace nnvm {
namespace runtime {

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << TVMGetLastError();                                      \
  }

using ::tvm::runtime::PackedFunc;
using ::tvm::runtime::TVMArgs;
using ::tvm::runtime::TVMRetValue;

PackedFunc GraphExecutor::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          this->SetInput(this->GetInputIndex(args[0]), args[1]);
        } else {
          this->SetInput(args[0], args[1]);
        }
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->GetOutput(args[0], args[1]);
      });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->Run();
      });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->LoadParams(args[0].operator std::string());
      });
  } else {
    return PackedFunc();
  }
}

GraphExecutor::~GraphExecutor() {
  for (DLTensor* t : storage_pool_) {
    TVM_CCALL(TVMArrayFree(t));
  }
}

void GraphExecutor::Run() {
  // setup the array and requirements.
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
}

void GraphExecutor::Init(Graph graph,
                         tvm::runtime::Module module,
                         TVMContext ctx) {
  graph_ = std::move(graph);
  module_ = std::move(module);
  ctx_ = ctx;
  this->SetupStorage();
  this->SetupOpExecs();
}

int GraphExecutor::GetInputIndex(const std::string& name) {
  const auto& idx = graph_.indexed_graph();
  for (size_t i = 0; i< idx.input_nodes().size(); ++i) {
    if (idx[idx.input_nodes()[i]].source->attrs.name == name) {
      return static_cast<int>(i);
    }
  }
  LOG(FATAL) << "cannot find " << name << " among input";
  return -1;
}

void GraphExecutor::SetInput(int index, DLTensor* data_in) {
  const auto& idx = graph_.indexed_graph();
  CHECK_LT(static_cast<size_t>(index), idx.input_nodes().size());
  uint32_t eid = idx.entry_id(idx.input_nodes()[index], 0);
  TVM_CCALL(TVMArrayCopyFromTo(data_in, &data_entry_[eid], nullptr));
}

void GraphExecutor::GetOutput(int index, DLTensor* data_out) {
  const auto& idx = graph_.indexed_graph();
  CHECK_LT(static_cast<size_t>(index), idx.outputs().size());
  uint32_t eid = idx.entry_id(idx.outputs()[index]);
  TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
}

bool LoadDLTensor(dmlc::Stream* strm, DLTensor* tensor) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header, sizeof(header)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&reserved, sizeof(reserved)))
      << "Invalid DLTensor file format";
  CHECK(header == kTVMNDArrayMagic)
      << "Invalid DLTensor file format";

  CHECK(strm->Read(&tensor->ctx, sizeof(tensor->ctx)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&tensor->ndim, sizeof(tensor->ndim)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&tensor->dtype, sizeof(tensor->dtype)))
      << "Invalid DLTensor file format";

  int ndim = tensor->ndim;
  CHECK(strm->Read(tensor->shape, sizeof(int64_t) * ndim))
      << "Invalid DLTensor file format";

  int64_t size = 1;
  int type_size = tensor->dtype.bits / 8;
  for (int i = 0; i < ndim; ++i) {
    size *= tensor->shape[i];
  }
  int64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size, sizeof(data_byte_size)))
      << "Invalid DLTensor file format";
  CHECK(data_byte_size == type_size * size)
      << "Invalid DLTensor file format";
  CHECK(strm->Read(tensor->data, type_size * size))
      << "Invalid DLTensor file format";
  return true;
}

void GraphExecutor::LoadParams(dmlc::Stream* strm) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic)
      << "Invalid parameters file format";
  CHECK(strm->Read(&reserved))
      << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names))
      << "Invalid parameters file format";

  std::unordered_map<std::string, size_t> name_eid;
  const auto& idx = graph_.indexed_graph();
  for (int nid : idx.input_nodes()) {
    name_eid.emplace(idx[nid].source->attrs.name, idx.entry_id(nid, 0));
  }

  uint64_t sz;
  strm->Read(&sz, sizeof(sz));
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size())
      << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    auto iter = name_eid.find(names[i]);
    CHECK(iter != name_eid.end());
    CHECK(LoadDLTensor(strm, &data_entry_[iter->second]))
        << "Invalid parameters file format";
  }
}

void GraphExecutor::LoadParams(const std::string& param_blob) {
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadParams(&strm);
}

void GraphExecutor::SetupStorage() {
  const auto& idx = graph_.indexed_graph();
  // Grab saved optimization plan from graph.
  auto vstorage = graph_.MoveCopyAttr<StorageVector>("storage_id");
  std::vector<TVMType> vtype;
  for (const std::string& s_type :
           graph_.GetAttr<std::vector<std::string> >("dltype")) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }
  data_shape_ = graph_.GetAttr<ShapeVector>("shape");
  data_entry_.resize(idx.num_node_entries());
  // Find the maximum space size.
  int max_id = 0;
  for (size_t i = 0; i < data_shape_.size(); ++i) {
    max_id = std::max(vstorage[i] + 1, max_id);
  }
  for (const auto& e : idx.input_nodes()) {
    vstorage[idx.entry_id(e, 0)] = max_id++;
  }
  // size of each storage pool entry
  std::vector<size_t> pool_entry_bytes;
  // Find the maximum space size.
  for (size_t i = 0; i < data_shape_.size(); ++i) {
    int storage_id = vstorage[i];
    size_t size = data_shape_[i].Size();
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size_t bytes = (bits / 8U) * size;

    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_entry_bytes.size()) {
      pool_entry_bytes.resize(sid + 1, 0);
    }
    pool_entry_bytes[sid] = std::max(pool_entry_bytes[sid], bytes);
  }
  // Allocate the space.
  for (size_t i = 0; i < pool_entry_bytes.size(); ++i) {
    TShape shape{static_cast<int64_t>(pool_entry_bytes[i] + 3) / 4};
    DLTensor* tensor;
    TVM_CCALL(TVMArrayAlloc(
        shape.data(), 1, kFloat, 32, 1, ctx_.device_type, ctx_.device_id, &tensor));
    storage_pool_.push_back(tensor);
  }
  // Assign the pooled entries.
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = vstorage[i];
    data_entry_[i] = *storage_pool_[storage_id];
    data_entry_[i].shape = const_cast<int64_t*>(data_shape_[i].data());
    data_entry_[i].ndim = data_shape_[i].ndim();
    data_entry_[i].dtype = vtype[i];
  }
}

void GraphExecutor::SetupOpExecs() {
  static const nnvm::Op* tvm_op = nnvm::Op::Get("tvm_op");
  const auto& idx = graph_.indexed_graph();
  op_execs_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(data_entry_[idx.entry_id(e)]);
    }
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      args.push_back(data_entry_[eid]);
    }
    CHECK_EQ(inode.source->op(), tvm_op)
        << "transform the graph to tvm op";
    op_execs_[nid] = CreateTVMOp(
        inode.source->attrs, args, inode.inputs.size());
  }
}

std::function<void()> GraphExecutor::CreateTVMOp(
    const nnvm::NodeAttrs& attrs,
    const std::vector<DLTensor>& args,
    size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  // get compiled function from module.
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;
  auto fexec = [arg_ptr, pf] () {
    TVMRetValue rv;
    TVMArgs targs(arg_ptr->arg_values.data(),
                  arg_ptr->arg_tcodes.data(),
                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return fexec;
}

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

tvm::runtime::Module RuntimeCreate(std::string sym_json,
                                   tvm::runtime::Module m,
                                   int device_type,
                                   int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id   = device_id;
  // load graph from json string
  nnvm::Graph g;
  g.attrs["json"] = std::make_shared<nnvm::any>(sym_json);
  g = nnvm::ApplyPass(std::move(g), "LoadJSON");
  std::shared_ptr<GraphExecutor> exec = std::make_shared<GraphExecutor>();
  exec->Init(g, m, ctx);
  return tvm::runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("nnvm.runtime.createx")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = RuntimeCreate(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("nnvm.runtime.remote_createx")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* mhandle = args[1];
    *rv = RuntimeCreate(args[0],
                        *static_cast<tvm::runtime::Module*>(mhandle),
                        args[2], args[3]);
  });

}  // namespace runtime
}  // namespace nnvm
