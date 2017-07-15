/*!
 *  Copyright (c) 2017 by Contributors
 * \file NNVM Graph executor.
 */
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>
#include <nnvm/pass.h>
#include <numeric>
#include <fstream>
#include <string>

namespace tvm {
namespace contrib {

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;
using nnvm::StorageVector;
using nnvm::ShapeVector;
using nnvm::TShape;
using nnvm::NodeAttrs;

/*! \brief DLPack compatible data types */
using DLTypeVector = std::vector<DLDataType>;

/*! \brief The executor function */
using FOpExec = std::function<void()>;

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << TVMGetLastError();                                      \
  }

/*! \brief Graph Executor with TVM runtime */
class GraphExecutor : public runtime::ModuleNode {
 public:
  const char* type_key() const {
    return "GraphExecutor";
  }
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self);
  // Destructor
  ~GraphExecutor();
  // Setup with a given graph
  void Init(const nnvm::Graph& g, TVMContext ctx);
  // Copy data to index-th input
  void SetInput(int index, DLTensor* data_in);
  // Copy index-th output to data_out
  void GetOutput(int index, DLTensor* data_out);
  // Load parameters from stream
  void LoadParams(dmlc::Stream* strm);
  // Load parameters from file
  void LoadParams(std::string fname);
  // Execute the graph.
  void Run();

 private:
  // functions
  void SetupStorage();
  void SetupOpExecs();
  // Constructor to create TVM op
  FOpExec CreateTVMOp(const nnvm::NodeAttrs& attrs,
                      std::vector<DLTensor> inputs,
                      size_t num_inputs);
  // The graph to be executed.
  nnvm::Graph graph_;
  // The execution context
  TVMContext ctx_;
  // Common storage pool
  std::vector<DLTensor*> storage_pool_;
  // The data entry
  std::vector<DLTensor> data_entry_;
  // The operation lambda on each node
  std::vector<FOpExec> op_execs_;
  // The code module.
  tvm::runtime::Module module_;
};

PackedFunc GraphExecutor::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->SetInput(args[0], args[1]);
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

void GraphExecutor::Init(const nnvm::Graph& g, TVMContext ctx) {
  graph_ = g;
  ctx_ = ctx;
  module_ = g.GetAttr<tvm::runtime::Module>("module");
  this->SetupStorage();
  this->SetupOpExecs();
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


constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

bool SaveDLTensor(dmlc::Stream* strm, DLTensor* tensor) {
    uint64_t header = kTVMNDArrayMagic, reserved = 0;
    strm->Write(&header, sizeof(header));
    strm->Write(&reserved, sizeof(reserved));

    strm->Write(&tensor->ctx, sizeof(tensor->ctx));
    strm->Write(&tensor->ndim, sizeof(tensor->ndim));
    strm->Write(&tensor->dtype, sizeof(tensor->dtype));

    int ndim = tensor->ndim;
    strm->Write(tensor->shape, sizeof(int64_t) * ndim);

    int type_size = tensor->dtype.bits / 8;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
      size *= tensor->shape[i];
    }
    int64_t data_byte_size = type_size * size;
    strm->Write(&data_byte_size, sizeof(data_byte_size));
    strm->Write(tensor->data, data_byte_size);
    return true;
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


constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

TVM_REGISTER_GLOBAL("tvm_graph._save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string fname = args[0];
    int num_params = args[1];
    std::vector<std::string> names;
    names.reserve(num_params);
    std::vector<DLTensor*> arrays;
    arrays.reserve(num_params);
    for (int i = 2; i < (2 + 2*num_params); i += 2) {
      names.emplace_back(args[i].operator std::string());
      arrays.emplace_back(args[i+1].operator DLTensor*());
    }

    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    uint64_t header = kTVMNDArrayListMagic, reserved = 0;
    fo->Write(&header, sizeof(header));
    fo->Write(&reserved, sizeof(reserved));

    fo->Write(names);
    {
      uint64_t sz = static_cast<uint64_t>(arrays.size());
      fo->Write(&sz, sizeof(sz));
      for (size_t i = 0; i < sz; ++i) {
        SaveDLTensor(fo.get(), arrays[i]);
      }
    }
  });

void GraphExecutor::LoadParams(dmlc::Stream *strm) {
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

  nnvm::Symbol s;
  s.outputs = graph_.outputs;
  std::vector<std::string> input_names =
    s.ListInputNames(nnvm::Symbol::ListInputOption::kAll);
  std::unordered_map<std::string, size_t> name_index;
  for (size_t i = 0; i < input_names.size(); ++i) {
    name_index.emplace(input_names[i], i);
  }

  uint64_t sz;
  strm->Read(&sz, sizeof(sz));
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size())
    << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    size_t idx = name_index.at(names[i]);
    CHECK(LoadDLTensor(strm, &data_entry_[idx]))
      << "Invalid parameters file format";
  }
}

void GraphExecutor::LoadParams(std::string fname) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
  this->LoadParams(fi.get());
}

void GraphExecutor::SetupStorage() {
  const auto& idx = graph_.indexed_graph();
  // Grab saved optimization plan from graph.
  auto vstorage = graph_.MoveCopyAttr<StorageVector>("storage_id");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  const auto& vtype = graph_.GetAttr<DLTypeVector>("dltype");
  data_entry_.resize(idx.num_node_entries());

  // Find the maximum space size.
  int max_id = 0;
  for (size_t i = 0; i < vshape.size(); ++i) {
    max_id = std::max(vstorage[i] + 1, max_id);
  }
  for (const auto& e : idx.input_nodes()) {
    vstorage[idx.entry_id(e, 0)] = max_id++;
  }
  // size of each storage pool entry
  std::vector<size_t> pool_entry_bytes;
  // Find the maximum space size.
  for (size_t i = 0; i < vshape.size(); ++i) {
    int storage_id = vstorage[i];
    size_t size = vshape[i].Size();
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
    data_entry_[i].shape = const_cast<int64_t*>(vshape[i].data());
    data_entry_[i].ndim = vshape[i].ndim();
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

FOpExec GraphExecutor::CreateTVMOp(const nnvm::NodeAttrs& attrs,
                                   std::vector<DLTensor> args,
                                   size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  auto it = attrs.dict.find("func_name");
  CHECK(it != attrs.dict.end())
      << "tvm_op must need func_name attr";
  bool flatten = (attrs.dict.at("flatten_data") == "1");
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (flatten) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (flatten) {
      int64_t s = 1;
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  // get compiled function from module.
  runtime::PackedFunc pf = module_.GetFunction(it->second, false);
  CHECK(pf != nullptr) << "no such function in module: " << it->second;
  auto fexec = [arg_ptr, pf] () {
    runtime::TVMRetValue rv;
    runtime::TVMArgs targs(arg_ptr->arg_values.data(),
                           arg_ptr->arg_tcodes.data(),
                           static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return fexec;
}

// Create executor
tvm::runtime::Module CreateExecutor(nnvm::Graph g, TVMContext ctx) {
  std::shared_ptr<GraphExecutor> exec =
      std::make_shared<GraphExecutor>();
  exec->Init(g, ctx);
  return tvm::runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("tvm_graph._create_executor")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* graph_handle = args[0];
    int device_type = args[1];
    int device_id = args[2];
    TVMContext ctx{static_cast<DLDeviceType>(device_type), device_id};
    nnvm::Graph g = static_cast<nnvm::Graph*>(graph_handle)[0];
    *rv = CreateExecutor(g, ctx);
  });


TVM_REGISTER_GLOBAL("tvm_graph._get_module_from_graph")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* graph_handle = args[0];
    nnvm::Graph* g = static_cast<nnvm::Graph*>(graph_handle);
    *rv = g->MoveCopyAttr<tvm::runtime::Module>("module");
  });


TVM_REGISTER_GLOBAL("tvm_graph._load_executor")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string sym_json    = args[0];
    std::string lib_fname   = args[1];
    std::string param_blob  = args[2];
    TVMContext ctx = args[3];

    // load graph from json string
    nnvm::Graph g;
    g.attrs["json"] = std::make_shared<nnvm::any>(sym_json);
    g = nnvm::ApplyPass(std::move(g), "LoadJSON");

    // load module from file
    static const PackedFunc* fsys_load_ = nullptr;
    if (fsys_load_ == nullptr) {
      fsys_load_ = runtime::Registry::Get("tvm.contrib.rpc.server.load_module");
      CHECK(fsys_load_ != nullptr);
    }
    runtime::Module m = (*fsys_load_)(lib_fname);
    g.attrs["module"] = std::make_shared<nnvm::any>(m);

    std::shared_ptr<GraphExecutor> exec =
        std::make_shared<GraphExecutor>();
    exec->Init(g, ctx);

    // load params form stream of string
    dmlc::MemoryStringStream strm(&param_blob);
    exec->LoadParams(&strm);

    *rv = static_cast<void*>(new tvm::runtime::Module(exec));
  });
}  // namespace contrib
}  // namespace tvm
