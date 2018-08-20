/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph_runtime_debug.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <sys/time.h>
#include <chrono>
#include "../graph_runtime.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Graph runtime with debug .
 *
 *  This is the extension of GraphRuntime class used for debugging
 *  TVM runtime PackedFunc API.
 */
class GraphRuntimeDebug : public GraphRuntime {
 public:
    /*!
     * \brief Run each operation and get the output.
     * \param index The index of op which needs to be run.
     */
  double DebugRun(int index) {
    auto tbegin = std::chrono::high_resolution_clock::now();
    if (op_execs()[index]) {
      op_execs()[index]();
      DeviceAPI::Get(GetCtx())->StreamSync(GetCtx(), nullptr);
    }
    auto tend = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double> >(
        tend - tbegin).count();
    return time;
  }

  /*!
   * \brief Run each operation and get the output.
   * \param index The index of op which needs to be returned.
   * \param eid The Entry id of the op.
   */
  NDArray GetNDArray(int index, int eid) {
    DLTensor *dltensor = &data_entry()[GetEntryId(index, eid)];
    return NDArray::FromDLTensor(dltensor);
  }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self);
};


PackedFunc GraphRuntimeDebug::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "debug_run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->DebugRun(args[0]);
      });
  } else if (name == "get_ndarray") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetNDArray(args[0], args[1]);
      });
  } else {
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

Module GraphRuntimeDebugCreate(std::string sym_json,
                               tvm::runtime::Module m,
                               int device_type,
                               int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id   = device_id;
  std::shared_ptr<GraphRuntimeDebug> exec = std::make_shared<GraphRuntimeDebug>();
  exec->Init(sym_json, m, ctx);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug.create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = GraphRuntimeDebugCreate(args[0], args[1], args[2], args[3]);
  });
TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug._save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    CHECK_EQ(args.size() % 2, 0u);
    size_t num_params = args.size() / 2;
    std::vector<std::string> names;
    names.reserve(num_params);
    std::vector<DLTensor*> arrays;
    arrays.reserve(num_params);
    for (size_t i = 0; i < num_params * 2; i += 2) {
      names.emplace_back(args[i].operator std::string());
      arrays.emplace_back(args[i + 1].operator DLTensor*());
    }
    std::string bytes;
    dmlc::MemoryStringStream strm(&bytes);
    dmlc::Stream* fo = &strm;
    uint64_t header = kTVMNDArrayListMagic, reserved = 0;
    fo->Write(header);
    fo->Write(reserved);
    fo->Write(names);
    {
      uint64_t sz = static_cast<uint64_t>(arrays.size());
      fo->Write(sz);
      for (size_t i = 0; i < sz; ++i) {
        tvm::runtime::SaveDLTensor(fo, arrays[i]);
      }
    }
    TVMByteArray arr;
    arr.data = bytes.c_str();
    arr.size = bytes.length();
    *rv = arr;
  });
}  // namespace runtime
}  // namespace tvm
