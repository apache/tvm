#include <string.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "tvm/src/runtime/graph/graph_runtime.cc"
#ifndef _LIBCPP_SGX_CONFIG
#include <iostream>
#include <fstream>
#include <sstream>
#include "tvm/src/runtime/file_util.cc"
#endif

// the statically linked graph json and params
extern char _binary_lib_deploy_params_bin_start[];
extern char _binary_lib_deploy_params_bin_end[];
extern char _binary_lib_deploy_graph_json_start[];
extern char _binary_lib_deploy_graph_json_end[];

int RunInference(const char* img) {
  tvm::runtime::Module graph_lib =
    (*tvm::runtime::Registry::Get("module._GetSystemLib"))();

  size_t graph_json_size = ((size_t)_binary_lib_deploy_graph_json_end -
                            (size_t)_binary_lib_deploy_graph_json_start);
  size_t graph_params_size = ((size_t)_binary_lib_deploy_params_bin_end -
                              (size_t)_binary_lib_deploy_params_bin_start);
  std::string graph_json(_binary_lib_deploy_graph_json_start, graph_json_size);
  std::string graph_params(_binary_lib_deploy_params_bin_start, graph_params_size);

  int device_type = kDLCPU;
  int device_id = 0;

  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  std::shared_ptr<tvm::runtime::GraphRuntime> graph_rt =
    std::make_shared<tvm::runtime::GraphRuntime>();

  graph_rt->Init(graph_json, graph_lib, ctx);
  graph_rt->LoadParams(graph_params);

  DLTensor* input;
  DLTensor* output;
  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;

  int batch_size = 1;
  int64_t input_shape[4] = {batch_size, 3, 224, 224};
  int64_t output_shape[1] = {1000 /* num_classes */};
  TVMArrayAlloc(input_shape, 4 /* ndim */, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &input);
  TVMArrayAlloc(output_shape, 1, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &output);
  memcpy(input->data, img, sizeof(float)*batch_size*3*224*224);

  graph_rt->SetInput(graph_rt->GetInputIndex("data"), input);
  graph_rt->Run();
  graph_rt->GetOutput(0, output);

  float max_prob = 0;
  unsigned max_class = -1;
  for (int i = 0; i < 1000; ++i) {
    float p = static_cast<float*>(output->data)[i];
    if (p > max_prob) {
      max_prob = p;
      max_class = i;
    }
  }

  return max_class;
}


extern "C" {
int ecall_infer(const char* img) {
  return RunInference(img);
}
}

#ifndef _LIBCPP_SGX_CONFIG
int main(void) {
  std::ifstream f_img("bin/cat.bin", std::ios::binary);
  std::string img(static_cast<std::stringstream const&>(
                  std::stringstream() << f_img.rdbuf()).str());
  unsigned predicted_class = RunInference(img.c_str());
  if (predicted_class == 281) {
    std::cout << "It's a tabby!" << std::endl;
    return 0;
  }
  std::cerr << "Inference failed! Predicted class: " <<
    predicted_class << std::endl;
  return -1;
}
#endif
