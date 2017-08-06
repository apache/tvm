/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_executor.cc
 */
#include "./graph_executor.h"

namespace tvm {
namespace contrib {

PackedFunc GraphExecutor::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          this->SetInput(this->GetIndex(args[0]), args[1]);
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
        this->LoadParamsFromBlob(args[0]);
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
  this->SetupNameIndex();
  this->SetupStorage();
  this->SetupOpExecs();
}

int GraphExecutor::GetIndex(std::string name) {
  CHECK(name_idx_.count(name))
    << name << " is not in the graph.";
  return name_idx_.at(name);
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

void GraphExecutor::LoadParamsFromBlob(std::string param_blob) {
  dmlc::MemoryStringStream strm(&param_blob);
  this->LoadParams(&strm);
}

void GraphExecutor::SetupNameIndex() {
  nnvm::Symbol s;
  s.outputs = graph_.outputs;
  std::vector<std::string> input_names = s.ListInputNames(nnvm::Symbol::kAll);
  for (size_t i = 0; i < input_names.size(); ++i) {
    name_idx_[input_names[i]] = i;
  }
}

void GraphExecutor::SetupStorage() {
  const auto& idx = graph_.indexed_graph();
  // Grab saved optimization plan from graph.
  auto vstorage = graph_.MoveCopyAttr<StorageVector>("storage_id");
  const auto& vtype = graph_.GetAttr<DLTypeVector>("dltype");
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

DMLC_REGISTER_PARAMETER(TVMOpParam);

// ewise tvm op
NNVM_REGISTER_OP(tvm_op)
.set_attr_parser(ParamParser<TVMOpParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
    const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
    return param.num_inputs;
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
    return param.num_outputs;
  });

TVM_REGISTER_GLOBAL("tvm_graph._load_executor")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string sym_json    = args[0];
    std::string lib_fname   = args[1];
    std::string param_blob  = args[2];
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[3].operator int());
    ctx.device_id   = args[4];

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
    exec->LoadParamsFromBlob(std::move(param_blob));

    *rv = tvm::runtime::Module(exec);
  });
}  // namespace contrib
}  // namespace tvm

namespace dmlc {
namespace json {

template<>
struct Handler<DLDataType> {
  static void Write(JSONWriter *writer, const DLDataType& data) {
    std::vector<int> tmp({data.code, data.bits, data.lanes});
    writer->Write(tmp);
  }

  static void Read(JSONReader *reader, DLDataType* data) {
    std::vector<int> tmp;
    reader->Read(&tmp);
    data->code  = tmp[0];
    data->bits  = tmp[1];
    data->lanes = tmp[2];
  }
};

DMLC_JSON_ENABLE_ANY(std::vector<DLDataType>, list_dltype);

}  // namespace dmlc
}  // namespace json
