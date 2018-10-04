/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include "graph_runtime.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Run all the operations one by one.
 */
void GraphRuntime::Run() {
  // setup the array and requirements.
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
}
/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param ctxs The context of the host and devices where graph nodes will be
 * executed on.
 */
void GraphRuntime::Init(const std::string& graph_json,
                        tvm::runtime::Module module,
                        const std::vector<TVMContext>& ctxs) {
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
  std::istringstream is(graph_json);
#else
  std::string is = graph_json;
#endif
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  module_ = module;
  ctxs_ = ctxs;
  this->SetupStorage();
  this->SetupOpExecs();
}
/*!
 * \brief Get the input index given the name of input.
 * \param name The name of the input.
 * \return The index of input.
 */
int GraphRuntime::GetInputIndex(const std::string& name) {
  for (size_t i = 0; i< input_nodes_.size(); ++i) {
    uint32_t nid = input_nodes_[i];
    if (nodes_[nid].name == name) {
      return static_cast<int>(i);
    }
  }
  LOG(WARNING) << "Warning: cannot find \"" << name << "\" among input";
  return -1;
}
/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void GraphRuntime::SetInput(int index, DLTensor* data_in) {
  CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  data_entry_[eid].CopyFrom(data_in);
}
/*!
 * \brief Get the number of outputs
 *
 * \return The number of outputs from graph.
 */
int GraphRuntime::NumOutputs() const {
  return outputs_.size();
}
/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray GraphRuntime::GetInput(int index) const {
  CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  return data_entry_[eid];
}
/*!
 * \brief Return NDArray for given output index.
 * \param index The output index.
 *
 * \return NDArray corresponding to given output node index.
 */
NDArray GraphRuntime::GetOutput(int index) const {
  CHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  return data_entry_[eid];
}
/*!
 * \brief Copy index-th output to data_out.
 * \param index The output index.
 * \param data_out the output data.
 */
void GraphRuntime::CopyOutputTo(int index, DLTensor* data_out) {
  CHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);

  // Check the shapes to avoid receiving in different dimension but same size.
  const NDArray& data = data_entry_[eid];
  CHECK_EQ(data->ndim, data_out->ndim);
  for (int32_t j = 0; j < data->ndim; ++j) {
    CHECK_EQ(data->shape[j], data_out->shape[j]);
  }

  data_entry_[eid].CopyTo(data_out);
}

/*!
 * \brief Load parameters from parameter blob.
 * \param param_blob A binary blob of parameter.
 */
void GraphRuntime::LoadParams(const std::string& param_blob) {
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadParams(&strm);
}

void GraphRuntime::LoadParams(dmlc::Stream* strm) {
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
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size())
      << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    CHECK_GE(in_idx, 0) << "Found param for non-existent input: " << names[i];
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    CHECK_LT(eid, data_entry_.size());

    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    NDArray temp;
    temp.Load(strm);
    data_entry_[eid].CopyFrom(temp);
  }
}

void GraphRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entry;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(ctxs_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size_t bytes = (bits / 8U) * size;

    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entry.size()) {
      pool_entry.resize(sid + 1, {0, -1});
    } else {
      CHECK(pool_entry[sid].device_type == -1 ||
            pool_entry[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    pool_entry[sid].size = std::max(pool_entry[sid].size, bytes);
    pool_entry[sid].device_type = device_type;
  }

  // Allocate the space.
  for (const auto& pit : pool_entry) {
    std::vector<int64_t> shape;
    // This for loop is very fast since there are usually only a couple of
    // devices available on the same hardware.
    const auto& cit =
        std::find_if(ctxs_.begin(), ctxs_.end(), [&pit](const TVMContext& c) {
          return pit.device_type == static_cast<int>(c.device_type);
        });
    TVMContext ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
    shape.push_back(static_cast<int64_t>(pit.size + 3) / 4);
    storage_pool_.push_back(
        NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx));
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    CHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] =
        storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
  }
}

void GraphRuntime::SetupOpExecs() {
  op_execs_.resize(this->GetNumOfNodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(*(data_entry_[this->entry_id(e)].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    CHECK(inode.op_type == "tvm_op") << "Can only take tvm_op as op";

    op_execs_[nid] = CreateTVMOp(inode.param, args, inode.inputs.size());
  }
}

std::function<void()> GraphRuntime::CreateTVMOp(
    const TVMOpParam& param,
    const std::vector<DLTensor>& args,
    size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
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

  if (param.func_name == "__nop") {
    return [](){};
  } else if (param.func_name == "__copy") {
    // Perform cross device data copy.
    // Directly copy data from the input to the output.
    auto fexec = [arg_ptr]() {
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      TVM_CCALL(TVMArrayCopyFromTo(from, to, nullptr));
    };
    return fexec;
  }

  // Get compiled function from the module that contains both host and device
  // code.
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;

  auto fexec = [arg_ptr, pf]() {
    TVMRetValue rv;
    TVMArgs targs(arg_ptr->arg_values.data(),
                  arg_ptr->arg_tcodes.data(),
                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return fexec;
}

PackedFunc GraphRuntime::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          if (in_idx >= 0) this->SetInput(in_idx, args[1]);
        } else {
          this->SetInput(args[0], args[1]);
        }
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args.num_args == 2) {
          this->CopyOutputTo(args[0], args[1]);
        } else {
          *rv = this->GetOutput(args[0]);
        }
      });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        int in_idx = 0;
        if (args[0].type_code() == kStr) {
          in_idx = this->GetInputIndex(args[0]);
        } else {
          in_idx = args[0];
        }
        CHECK_GE(in_idx, 0);
        *rv = this->GetInput(in_idx);
      });
  } else if (name == "get_num_outputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->NumOutputs();
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

Module GraphRuntimeCreate(const std::string& sym_json,
                          const tvm::runtime::Module& m,
                          const std::vector<TVMContext>& ctxs) {
  std::shared_ptr<GraphRuntime> exec = std::make_shared<GraphRuntime>();
  exec->Init(sym_json, m, ctxs);
  return Module(exec);
}

// Get all context for the host and other runtime devices.
std::vector<TVMContext> GetAllContext(const TVMArgs& args) {
  // Reserve the first item as the fallback device.
  std::vector<TVMContext> ret;
  TVMContext ctx;
  for (int i = 2; i < args.num_args; i += 2) {
    int dev_type = args[i];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[i + 1];
    ret.push_back(ctx);
  }
  return ret;
}

// 4-argument version is currently reserved to keep support of calling
// from tvm4j and javascript, since they don't have heterogeneous
// execution support yet. For heterogenenous execution, at least 5 arguments will
// be passed in. The third one is the number of devices.
// Eventually, we will only probably pass TVMContext for all the languages.
TVM_REGISTER_GLOBAL("tvm.graph_runtime.create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4)
        << "The expected number of arguments for graph_runtime.create is "
           "at least 4, but it has "
        << args.num_args;
    const auto& contexts = GetAllContext(args);
    *rv = GraphRuntimeCreate(args[0], args[1], contexts);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.remote_create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4) << "The expected number of arguments for "
                                  "graph_runtime.remote_create is "
                                  "at least 4, but it has "
                               << args.num_args;
    void* mhandle = args[1];
    const auto& contexts = GetAllContext(args);
    *rv = GraphRuntimeCreate(
        args[0], *static_cast<tvm::runtime::Module*>(mhandle), contexts);
  });
}  // namespace runtime
}  // namespace tvm
