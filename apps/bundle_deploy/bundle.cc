#include <memory>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

extern unsigned char build_graph_json[];
extern unsigned int build_graph_json_len;
extern unsigned char build_params_bin[];
extern unsigned int build_params_bin_len;

#define TVM_BUNDLE_FUNCTION __attribute__((visibility("default"))) extern "C"

TVM_BUNDLE_FUNCTION void *tvm_runtime_create() {
  const std::string json_data(&build_graph_json[0],
                              &build_graph_json[0] + build_graph_json_len);
  tvm::runtime::Module mod_syslib =
      (*tvm::runtime::Registry::Get("module._GetSystemLib"))();
  int device_type = kDLCPU;
  int device_id = 0;
  tvm::runtime::Module mod =
      (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(
          json_data, mod_syslib, device_type, device_id);
  TVMByteArray params;
  params.data = reinterpret_cast<const char *>(&build_params_bin[0]);
  params.size = build_params_bin_len;
  mod.GetFunction("load_params")(params);
  return new tvm::runtime::Module(mod);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_destroy(void *handle) {
  delete reinterpret_cast<tvm::runtime::Module *>(handle);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_set_input(void *handle, const char *name,
                                               void *tensor) {
  reinterpret_cast<tvm::runtime::Module *>(handle)->GetFunction("set_input")(
      name, reinterpret_cast<DLTensor *>(tensor));
}

TVM_BUNDLE_FUNCTION void tvm_runtime_run(void *handle) {
  reinterpret_cast<tvm::runtime::Module *>(handle)->GetFunction("run")();
}

TVM_BUNDLE_FUNCTION void tvm_runtime_get_output(void *handle, int index,
                                                void *tensor) {
  reinterpret_cast<tvm::runtime::Module *>(handle)->GetFunction("get_output")(
      index, reinterpret_cast<DLTensor *>(tensor));
}
