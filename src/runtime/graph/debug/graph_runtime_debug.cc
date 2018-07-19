/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph_runtime_debug.cc
 */
#include <sys/time.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
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
  int DebugRun(int index) {
    struct timeval tp;
    int64_t start_time = 0;
    int64_t end_time = 0;
    gettimeofday(&tp, NULL);
    if (op_execs()[index]) {
      op_execs()[index]();
    }
    start_time = int64_t(tp.tv_sec * 1000000L + tp.tv_usec);
    gettimeofday(&tp, NULL);
    end_time = int64_t(tp.tv_sec * 1000000L + tp.tv_usec);
    for (size_t j = 0; j < NumOutputs(index); j++) {
        uint32_t eid = GetEntryId(index, j);
        TVM_CCALL(TVMArrayCopyFromTo(&data_entry()[eid], debug_buffers_[eid], nullptr));
    }
    return end_time - start_time;
  }

  /*!
   * \brief Set the debug buffer to copy the output of each operation.
   * \param data The data pointer.
   */
  void SetDebugBuffer(DLTensor* data) {
      debug_buffers_.push_back(data);
  }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self);

 private:
  /*! \brief debug buffer storage pool */
  std::vector<DLTensor*> debug_buffers_;
};


PackedFunc GraphRuntimeDebug::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "set_debug_buffer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->SetDebugBuffer(args[0]);
      });
  } else if (name == "debug_run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->DebugRun(args[0]);
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
}  // namespace runtime
}  // namespace tvm
