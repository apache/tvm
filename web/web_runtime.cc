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
}  // namespace contrib
}  // namespace tvm
