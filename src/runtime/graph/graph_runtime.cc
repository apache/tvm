/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc */
#include "graph_runtime.h"

#include <tvm/runtime/registry.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

namespace tvm {
namespace runtime {

void GraphRuntime::LoadDLTensor(dmlc::Stream* strm, DLTensor* dst) {
  // always use strm->Read to maintain endianness conversion
  NDArray temp;
  temp.Load(strm);
  temp.CopyTo(dst);
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
    LoadDLTensor(strm, &data_entry_[eid]);
  }
}

// Return storage id to device type map. This map will be used to help memory
// allocation for the storage pool of each device. It will be also used to
// allocate memory to each data_entry_.
StorageDeviceMap GraphRuntime::GetStorageDeviceMap() const {
  StorageDeviceMap sid_dev_map;
  for (uint32_t nid = 0; nid < this->num_nodes(); ++nid) {
    const auto &inode = nodes_[nid];
    for (const auto &e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      uint32_t sid = attrs_.storage_id[eid];
      sid_dev_map[sid] = nodes_[e.node_id].device;
    }
  }
  // Get all output entries.
  for (const auto& output : outputs_) {
    uint32_t eid = this->entry_id(output);
    uint32_t sid = attrs_.storage_id[eid];
    sid_dev_map[sid] = nodes_[output.node_id].device;
  }
  return sid_dev_map;
}

void GraphRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }

  StorageDeviceMap sid_dev_map = GetStorageDeviceMap();
  std::unordered_map<DLDeviceType, std::unordered_map<size_t, size_t>,
                     DLDeviceTypeHash>
      device_pool_entry_bytes;

  // Find the maximum space size for each device.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    uint32_t sid = static_cast<uint32_t>(attrs_.storage_id[i]);
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    CHECK_GE(sid, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size_t bytes = (bits / 8U) * size;

    DLDeviceType dev_type = sid_dev_map[sid];
    device_pool_entry_bytes[dev_type][sid] =
        std::max(device_pool_entry_bytes[dev_type][sid], bytes);
    // LOG(INFO) << "pool entry bytes  " << nodes_[i].name << "  " << i << " " << sid << "   " << pool_entry_bytes[sid];
  }

  // Allocate the space on each device.
  for (const auto& it : device_pool_entry_bytes) {
    const auto& pool_entry = it.second;
    for (const auto& pit : pool_entry) {
      int64_t shape[] = {static_cast<int64_t>(pit.second + 3) / 4};
      TVMContext ctx = runtime_host_ctx_;
      // This for loop is very fast since there are only 2 or 3 devices at most.
      for (const auto& mit : runtime_device_mod_ctx_map_) {
        if (it.first == mit.second.device_type) {
          ctx = mit.second;
          break;
        }
      }
      DLTensor *tensor;
      TVM_CCALL(TVMArrayAlloc(shape, 1, kDLFloat, 32, 1, ctx.device_type,
                              ctx.device_id, &tensor));
      device_storage_pool_[it.first][pit.first] = tensor;
    }
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool by querying the storage id to device map.
  data_entry_.resize(num_node_entries());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    uint32_t storage_id = static_cast<uint32_t>(attrs_.storage_id[i]);
    DLDeviceType dev_type = sid_dev_map[storage_id];
    CHECK(device_storage_pool_[dev_type].count(storage_id))
        << "The storage hasn't been assigned to a specific device.";
    data_entry_[i] = *device_storage_pool_[dev_type][storage_id];
    data_entry_[i].shape = const_cast<int64_t*>(attrs_.shape[i].data());
    data_entry_[i].ndim = static_cast<int>(attrs_.shape[i].size());
    data_entry_[i].dtype = vtype[i];
    // LOG(INFO) << "data entry:::  " << nodes_[i].name << "  " << i << " " << storage_id << "   " << data_entry_[i].ctx.device_type;
  }
}

void GraphRuntime::SetupOpExecs() {
  op_execs_.resize(this->num_nodes());
  std::vector<DLTensor> ids;
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < this->num_nodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
     args.push_back(data_entry_[this->entry_id(e)]);
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(data_entry_[eid]);
    }
    CHECK(inode.op_type == "tvm_op" || inode.op_type == "device_copy_op")
        << "Can only take tvm_op or device_copy_op as op";

    op_execs_[nid] =
        CreateTVMOp(inode.param, args, inode.inputs.size(), inode.device);
  }
}

// TODO(chzhi) remove ctx and params in fexec.
std::function<void()> GraphRuntime::CreateTVMOp(
    const TVMOpParam& param, const std::vector<DLTensor>& args,
    size_t num_inputs, int ctx) {
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
      // auto start = std::chrono::high_resolution_clock::now();
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      TVM_CCALL(TVMArrayCopyFromTo(from, to, nullptr));
      // auto end = std::chrono::high_resolution_clock::now();
      // std::chrono::duration<double> diff = end - start;
      // LOG(INFO) << "+++++++++ coying overhead " << from->ndim << "  "
      //           << diff.count() * 1000 << " ms\n";
      // for (int i = 0; i < from->ndim; i++) {
      //   LOG(INFO) << "dim: " << i << " size: " << from->shape[i];
      // }
    };
    return fexec;
  }

  // get compiled function from module.
  tvm::runtime::PackedFunc pf =
      runtime_host_module_.GetFunction(param.func_name, false);
  if (pf == nullptr) {
    for (const auto& it : runtime_device_mod_ctx_map_) {
      pf = it.first->GetFunction(param.func_name, false);
      if (pf != nullptr) break;
    }
  }
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;

  auto fexec = [arg_ptr, pf, param, ctx]() {
    // LOG(INFO) << "executing................." << param.func_name << "    " << ctx;
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
  // return member functions during query.
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
        this->GetOutput(args[0], args[1]);
      });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          CHECK_GE(in_idx, 0);
          this->GetInput(in_idx, args[1]);
        } else {
          this->GetInput(args[0], args[1]);
        }
      });
#ifdef TVM_GRAPH_RUNTIME_DEBUG
  } else if (name == "debug_get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          this->DebugGetNodeOutput(this->GetNodeIndex(args[0]), args[1]);
        } else {
          this->DebugGetNodeOutput(args[0], args[1]);
        }
      });
#endif
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
                          int device_type,
                          int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id   = device_id;
  std::shared_ptr<GraphRuntime> exec = std::make_shared<GraphRuntime>();
  exec->Init(sym_json, m, ctx);
  return Module(exec);
}

Module GraphRuntimeCreateHeterogeneous(
    const std::string& graph_json, const tvm::runtime::Module& runtime_host_mod,
    const TVMContext& runtime_host_ctx,
    const ModuleContextMap& runtime_device_mod_ctx_map) {
  std::shared_ptr<GraphRuntime> exec = std::make_shared<GraphRuntime>();
  exec->Init(graph_json, runtime_host_mod, runtime_host_ctx,
             runtime_device_mod_ctx_map);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = GraphRuntimeCreate(args[0], args[1], args[2], args[3]
                               /*, runtime_device_mod_ctx_map_ = {}*/);
    });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.create_heterogeneous")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 5) << "5 arguments are expected, but "
                               << args.size() << " are passed in.";
      tvm::runtime::Module** modules = args[1].ptr<Module*>();
      int* device_types = args[2].ptr<int>();
      int* device_ids = args[3].ptr<int>();
      int num_devices = args[4];

      // Setup module and context for the host and other runtime devices.
      TVMContext runtime_host_ctx, runtime_device_ctx;
      runtime_host_ctx.device_type = static_cast<DLDeviceType>(device_types[0]);
      CHECK_EQ(runtime_host_ctx.device_type, kDLCPU)
          << "CPU should be the host hardware.";
      runtime_host_ctx.device_id   = device_ids[0];
      tvm::runtime::Module runtime_host_mod = *modules[0];

      ModuleContextMap runtime_device_mod_ctx_map;
      for (int i = 1; i < num_devices; i++) {
        tvm::runtime::Module* mod = modules[i];
        runtime_device_ctx.device_type =
            static_cast<DLDeviceType>(device_types[i]);
        runtime_device_ctx.device_id = device_ids[i];
        runtime_device_mod_ctx_map[mod] = runtime_device_ctx;
      }

      *rv = GraphRuntimeCreateHeterogeneous(args[0], runtime_host_mod,
                                            runtime_host_ctx,
                                            runtime_device_mod_ctx_map);
    });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.remote_create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      void* mhandle = args[1];
      *rv = GraphRuntimeCreate(args[0],
                               *static_cast<tvm::runtime::Module*>(mhandle),
                               args[2], args[3]
                               /*, runtime_device_mod_ctx_map_ = {}*/);
    });
}  // namespace runtime
}  // namespace tvm
