/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "microtvm_graph_executor.h"

#include <dlfcn.h>

#include <cassert>
#include <string>

#include "picojson.h"

namespace tvm {
namespace micro {
namespace {

int TVMSToI(const std::string& str) {
  // For platforms (e.g. older NDK versions) where std::stoi(...) is not available.
  char* end;
  return std::strtol(str.c_str(), &end, 10);
}

void ParseOutputs(const picojson::array& joutputs, DynArray<NodeEntry>* outputs) {
  outputs->resize(joutputs.size());
  for (size_t i = 0; i < joutputs.size(); ++i) {
    const auto& joutput_i = joutputs[i].get<picojson::array>();
    (*outputs)[i] = NodeEntry{static_cast<uint32_t>(joutput_i[0].get<double>()),
                              static_cast<uint32_t>(joutput_i[1].get<double>()),
                              static_cast<uint32_t>(joutput_i[2].get<double>())};
  }
}

void ParseAttrs(const picojson::object& jattr, GraphAttr* attr) {
  // parse dltype
  for (const auto& jdltype_ : jattr.at("dltype").get<picojson::array>()) {
    if (jdltype_.is<std::string>()) {
      continue;
    }
    const auto& jdltype = jdltype_.get<picojson::array>();

    attr->dltype.resize(jdltype.size());
    for (size_t i = 0; i < jdltype.size(); ++i) {
      attr->dltype[i] = jdltype[i].get<std::string>();
    }
  }
  for (const auto& jstorage_id_ : jattr.at("storage_id").get<picojson::array>()) {
    if (jstorage_id_.is<std::string>()) {
      continue;
    }
    const auto& jstorage_id = jstorage_id_.get<picojson::array>();

    attr->storage_id.resize(jstorage_id.size());
    for (size_t i = 0; i < jstorage_id.size(); ++i) {
      attr->storage_id[i] = static_cast<int>(jstorage_id[i].get<double>());
    }
  }
  for (const auto& jshape_ : jattr.at("shape").get<picojson::array>()) {
    if (jshape_.is<std::string>()) {
      continue;
    }
    const auto& jshape = jshape_.get<picojson::array>();
    attr->shape.resize(jshape.size());
    for (size_t i = 0; i < jshape.size(); ++i) {
      const auto& jshape_i = jshape[i].get<picojson::array>();
      attr->shape[i].resize(jshape_i.size());
      for (size_t j = 0; j < jshape_i.size(); ++j) {
        attr->shape[i][j] = static_cast<int64_t>(jshape_i[j].get<double>());
      }
    }
  }
}

void ParseNodes(const picojson::array& jnodes, DynArray<Node>* nodes) {
  nodes->resize(jnodes.size());
  for (size_t i = 0; i < nodes->size(); ++i) {
    auto* n = &(*nodes)[i];
    const auto& jn = jnodes[i].get<picojson::object>();
    n->op_type = jn.at("op").get<std::string>();
    n->name = jn.at("name").get<std::string>();
    const auto jinputs = jn.at("inputs").get<picojson::array>();
    n->inputs.resize(jinputs.size());
    for (size_t i = 0; i < jinputs.size(); ++i) {
      const auto& jinput_i = jinputs[i].get<picojson::array>();
      n->inputs[i] = NodeEntry{static_cast<uint32_t>(jinput_i[0].get<double>()),
                               static_cast<uint32_t>(jinput_i[1].get<double>()),
                               static_cast<uint32_t>(jinput_i[2].get<double>())};
    }
    const auto& jattrs_ = jn.find("attrs");
    if (jattrs_ != jn.end()) {
      const auto& jattrs = jattrs_->second.get<picojson::object>();
      n->param.func_name = jattrs.at("func_name").get<std::string>();
      n->param.num_inputs = TVMSToI(jattrs.at("num_inputs").get<std::string>());
      n->param.num_outputs = TVMSToI(jattrs.at("num_outputs").get<std::string>());
      n->param.flatten_data = TVMSToI(jattrs.at("flatten_data").get<std::string>());
    }
  }
}

void ParseArgNodes(const picojson::array& jinput_nodes, DynArray<uint32_t>* input_nodes) {
  input_nodes->resize(jinput_nodes.size());
  for (size_t i = 0; i < jinput_nodes.size(); ++i) {
    (*input_nodes)[i] = static_cast<uint32_t>(jinput_nodes[i].get<double>());
  }
}
}  // namespace

NDArray::~NDArray() {}

NDArray NDArray::Empty(const DynArray<int64_t>& shape, DLDataType dtype, DLDevice dev) {
  NDArray r;
  int64_t nbytes = (dtype.bits * dtype.lanes + 7) / 8;
  for (const auto& s : shape) {
    nbytes *= s;
  }

  r.storage_ = std::shared_ptr<void>(
      TVMBackendAllocWorkspace(static_cast<int>(dev.device_type), static_cast<int>(dev.device_id),
                               nbytes, dtype.code, dtype.bits),
      [=](void* ptr) {
        if (ptr) {
          TVMBackendFreeWorkspace(dev.device_type, dev.device_id, ptr);
        }
      });
  r.shape_ = shape;
  r.dtype_ = dtype;
  r.device_ = dev;
  return r;
}

NDArray NDArray::CreateView(const DynArray<int64_t>& shape, DLDataType dtype) {
  NDArray r;
  r.storage_ = storage_;
  r.shape_ = shape;
  r.dtype_ = dtype;
  r.device_ = device_;
  return r;
}

DLTensor NDArray::ToDLTensor() {
  DLTensor r;
  r.data = storage_.get();
  assert(r.data != nullptr);
  r.device = device_;
  r.ndim = shape_.size();
  r.dtype = dtype_;
  r.shape = shape_.data();
  r.strides = nullptr;
  r.byte_offset = 0;
  return r;
}

size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (size_t i = 0; i < static_cast<size_t>(arr.ndim); ++i) {
    size *= static_cast<size_t>(arr.shape[i]);
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

void NDArray::CopyFrom(DLTensor* src) {
  std::memcpy(storage_.get(),
              reinterpret_cast<const uint8_t*>(src->data) + static_cast<size_t>(src->byte_offset),
              GetDataSize(*src));
}

void NDArray::CopyTo(DLTensor* dst) const {
  std::memcpy(reinterpret_cast<uint8_t*>(dst->data) + static_cast<size_t>(dst->byte_offset),
              storage_.get(), GetDataSize(*dst));
}

DSOModule::DSOModule(const std::string& name) {
  dlerror();
  lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  assert(!dlerror());
  assert(lib_handle_ != nullptr);

#define TVM_INIT_CONTEXT_FUNC(FuncName)                                               \
  if (auto* fp = reinterpret_cast<decltype(&FuncName)*>(GetSymbol("__" #FuncName))) { \
    *fp = FuncName;                                                                   \
  }
  // Initialize the functions
  TVM_INIT_CONTEXT_FUNC(TVMAPISetLastError);
  TVM_INIT_CONTEXT_FUNC(TVMBackendAllocWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendFreeWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelLaunch);
// TODO(tulloch): implement these functions?
// TVM_INIT_CONTEXT_FUNC(TVMFuncCall);
// TVM_INIT_CONTEXT_FUNC(TVMBackendGetFuncFromEnv);
// TVM_INIT_CONTEXT_FUNC(TVMBackendParallelBarrier);
#undef TVM_INIT_CONTEXT_FUNC
}

DSOModule::~DSOModule() {
  if (lib_handle_) {
    dlclose(lib_handle_);
  }
}

BackendPackedCFunc DSOModule::GetFunction(const std::string& name) const {
  auto faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name.c_str()));
  assert(faddr);
  return faddr;
}

void* DSOModule::GetSymbol(const char* name) const {
  dlerror();
  auto* f = dlsym(lib_handle_, name);
  assert(!dlerror());
  return f;
}

MicroGraphExecutor::MicroGraphExecutor(const std::string& graph_json, DSOModule* module) {
  assert(module);
  module_ = module;
  picojson::value v;
  picojson::parse(v, graph_json);
  ParseNodes(v.get<picojson::object>()["nodes"].get<picojson::array>(), &nodes_);
  ParseArgNodes(v.get<picojson::object>()["arg_nodes"].get<picojson::array>(), &input_nodes_);
  ParseArgNodes(v.get<picojson::object>()["node_row_ptr"].get<picojson::array>(), &node_row_ptr_);
  ParseOutputs(v.get<picojson::object>()["heads"].get<picojson::array>(), &outputs_);
  ParseAttrs(v.get<picojson::object>()["attrs"].get<picojson::object>(), &attrs_);
  SetupStorage();
  SetupOpExecs();
}

MicroGraphExecutor::~MicroGraphExecutor() {}

void MicroGraphExecutor::Run() {
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
}

void MicroGraphExecutor::SetInput(int index, DLTensor* data_in) {
  assert(static_cast<size_t>(index) < input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  data_entry_[eid].CopyFrom(data_in);
}

void MicroGraphExecutor::CopyOutputTo(int index, DLTensor* data_out) {
  assert(static_cast<size_t>(index) < outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  const NDArray& data = data_entry_[eid];
  data.CopyTo(data_out);
}

void MicroGraphExecutor::SetupStorage() {
  // Grab saved optimization plan from graph.
  DynArray<DLDataType> vtype(attrs_.dltype.size());
  for (size_t i = 0; i < attrs_.dltype.size(); ++i) {
    assert(attrs_.dltype[i] == "float32");
    DLDataType ty;
    ty.bits = 32;
    ty.lanes = 1;
    ty.code = kDLFloat;
    vtype[i] = ty;
  }

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entry;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(device_.device_type);
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    assert(storage_id >= 0);
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    assert(bits % 8U == 0U || bits == 1U);
    size_t bytes = ((bits + 7U) / 8U) * size;

    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entry.size()) {
      pool_entry.resize(sid + 1, {0, -1});
    } else {
      assert(pool_entry[sid].device_type == -1 || pool_entry[sid].device_type == device_type);
    }
    pool_entry[sid].size = std::max(pool_entry[sid].size, bytes);
    pool_entry[sid].device_type = device_type;
  }

  // Allocate the space.
  storage_pool_.resize(pool_entry.size());
  for (size_t i = 0; i < pool_entry.size(); ++i) {
    const auto& pit = pool_entry[i];
    DynArray<int64_t> shape(1);
    shape[0] = static_cast<int64_t>(pit.size + 3) / 4;
    storage_pool_[i] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, device_);
  }

  // Assign the pooled entries. A unified memory pool is used to simplify
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    assert(static_cast<size_t>(storage_id) < storage_pool_.size());
    data_entry_[i] = storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
  }
}

std::function<void()> CreateTVMOp(const DSOModule& module, const TVMOpParam& param,
                                  const DynArray<DLTensor>& args) {
  typedef union {
    void* v_handle;
  } TVMValue;
  /*typedef*/ enum {
    kTVMDLTensorHandle = 7U,
  } /*TVMArgTypeCode*/;
  struct OpArgs {
    DynArray<DLTensor> args;
    DynArray<TVMValue> arg_values;
    DynArray<int> arg_tcodes;
    DynArray<int64_t> shape_data;
  };

  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  arg_ptr->args = args;
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  arg_ptr->arg_values.resize(arg_ptr->args.size());
  arg_ptr->arg_tcodes.resize(arg_ptr->args.size());
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values[i] = v;
    arg_ptr->arg_tcodes[i] = kTVMDLTensorHandle;
    if (param.flatten_data) {
      arg_ptr->shape_data[i] =
          std::accumulate(t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }

  if (param.func_name == "__nop") {
    return []() {};
  } else if (param.func_name == "__copy") {
    // TODO(mbs): device_copy cleanup.
    assert(false);
  }

  BackendPackedCFunc pf = module.GetFunction(param.func_name);
  assert(pf != nullptr);

  auto fexec = [arg_ptr, pf]() {
    assert(pf);
    (pf)(arg_ptr->arg_values.data(), arg_ptr->arg_tcodes.data(),
         static_cast<int>(arg_ptr->arg_values.size()));
  };
  return fexec;
}

void MicroGraphExecutor::SetupOpExecs() {
  op_execs_.resize(nodes_.size());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < nodes_.size(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    DynArray<DLTensor> args(inode.inputs.size() + inode.param.num_outputs);
    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      const auto& e = inode.inputs[i];
      args[i] = data_entry_[this->entry_id(e)].ToDLTensor();
    }
    for (size_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args[index + inode.inputs.size()] = data_entry_[eid].ToDLTensor();
    }
    assert(inode.op_type == "tvm_op");
    op_execs_[nid] = CreateTVMOp(*module_, inode.param, args);
  }
}

}  // namespace micro
}  // namespace tvm
