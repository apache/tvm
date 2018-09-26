/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph_runtime_debug.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
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
  double DebugRun(size_t index) {
    CHECK(index < op_execs().size());
    TVMContext ctx = data_entry()[GetEntryId(index, 0)].operator->()->ctx;
    auto tbegin = std::chrono::high_resolution_clock::now();
    if (op_execs()[index]) {
      op_execs()[index]();
    }
    TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
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
  NDArray GetOutputByLayer(int index, int eid) {
    return data_entry()[GetEntryId(index, eid)];
  }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self);
};


/*!
 * \brief GetFunction Get the function based on input.
 * \param name The function which needs to be invoked.
 * \param sptr_to_self Packed function pointer.
 */
PackedFunc GraphRuntimeDebug::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "debug_run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->DebugRun(args[0]);
      });
  } else if (name == "get_output_by_layer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetOutputByLayer(args[0], args[1]);
      });
  } else {
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

/*!
 * \brief GraphRuntimeDebugCreate Get the function based on input.
 * \param sym_json The graph symbol in json format.
 * \param m Compiled module which will be loaded.
 * \param ctxs All devices contexts.
 */
  Module GraphRuntimeDebugCreate(const std::string& sym_json,
                                 const tvm::runtime::Module& m,
                                 const std::vector<TVMContext>& ctxs) {
  std::shared_ptr<GraphRuntimeDebug> exec = std::make_shared<GraphRuntimeDebug>();
  exec->Init(sym_json, m, ctxs);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug.create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4)
        << "The expected number of arguments for graph_runtime.create is "
           "at least 4, but it has "
        << args.num_args;
    *rv = GraphRuntimeDebugCreate(args[0], args[1], GetAllContext(args));
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
