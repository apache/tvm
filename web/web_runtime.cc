/*!
 *  Copyright (c) 2017 by Contributors
 * \file web_runtime.cc
 */
#include <sys/stat.h>
#include <fstream>

#include "../src/runtime/c_runtime_api.cc"
#include "../src/runtime/cpu_device_api.cc"
#include "../src/runtime/workspace_pool.cc"
#include "../src/runtime/module_util.cc"
#include "../src/runtime/system_lib_module.cc"
#include "../src/runtime/module.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/file_util.cc"
#include "../src/runtime/dso_module.cc"
#include "../src/runtime/rpc/rpc_session.cc"
#include "../src/runtime/rpc/rpc_event_impl.cc"
#include "../src/runtime/rpc/rpc_server_env.cc"
#include "../src/runtime/graph/graph_runtime.cc"

namespace tvm {
namespace contrib {

struct RPCEnv {
 public:
  RPCEnv() {
    base_ = "/rpc";
    mkdir(&base_[0], 0777);
  }
  // Get Path.
  std::string GetPath(const std::string& file_name) {
    return base_ + "/" + file_name;
  }

 private:
  std::string base_;
};

TVM_REGISTER_GLOBAL("tvm.contrib.rpc.server.workpath")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    static RPCEnv env;
    *rv = env.GetPath(args[0]);
  });

TVM_REGISTER_GLOBAL("tvm.contrib.rpc.server.load_module")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string file_name = "/rpc/" + args[0].operator std::string();
    *rv = Module::LoadFromFile(file_name, "");
    LOG(INFO) << "Load module from " << file_name << " ...";
  });
}  // namespace contrib
}  // namespace tvm

// dummy parallel runtime
int TVMBackendParallelLaunch(
    FTVMParallelLambda flambda,
    void* cdata,
    int num_task) {
  TVMAPISetLastError("Parallel is not supported in Web runtime");
  return -1;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
  return 0;
}
