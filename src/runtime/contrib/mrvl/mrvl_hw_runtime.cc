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

/*!
 * \file src/runtime/contrib/mrvl/mrvl_hw_runtime.cc
 * \brief runtime implementation for Marvell Target.
 */

#include <dlfcn.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../../../../src/support/base64.h"
#include "mrvl_base64.h"

#define MRVL_LIBMLDPC_DEFAULT_PATH "/usr/lib/libmldpc.so"

namespace tvm {
namespace runtime {
namespace contrib {

enum buffer_type { input_quantize = 0, input_dequantize, output_quantize, output_dequantize };
enum model_type { TVM = 0, MLIP };

struct run_args {
  int model_id;
  void* i_q_buf;
  void* o_q_buf;
  int num_batches;
  void* device;
  uint16_t layer_idx;
};

void* device_handle;
int model_id;

/* Marvell DPDK Interface library callbacks for TVMC */
extern "C" typedef int (*mrvl_tvmc_ml_init_ptr)(int argc, char* argv[]);
extern "C" typedef int (*mrvl_tvmc_ml_finish_ptr)(void);
extern "C" typedef int (*mrvl_tvmc_ml_model_load_ptr)(char* model_buffer, int model_size);
extern "C" typedef int (*mrvl_tvmc_ml_model_unload_ptr)(int model_id);
extern "C" typedef void* (*mrvl_tvmc_ml_io_alloc_ptr)(int model_id, enum buffer_type dt,
                                                      uint64_t* size);
extern "C" typedef void (*mrvl_tvmc_ml_io_free_ptr)(int model_id, enum buffer_type dt, void* addr);
extern "C" typedef int (*mrvl_tvmc_ml_model_quantize_ptr)(int model_id, void* dbuffer,
                                                          void* qbuffer);
extern "C" typedef int (*mrvl_tvmc_ml_model_dequantize_ptr)(int model_id, void* qbuffer,
                                                            void* dbuffer);
extern "C" typedef int (*mrvl_tvmc_ml_model_run_ptr)(int model_id, void* input_buffer,
                                                     void* output_buffer, int num_batches);

struct ml_tvmc_cb {
  void* handle;
  mrvl_tvmc_ml_init_ptr mrvl_tvmc_ml_init;
  mrvl_tvmc_ml_finish_ptr mrvl_tvmc_ml_finish;
  mrvl_tvmc_ml_model_load_ptr mrvl_tvmc_ml_model_load;
  mrvl_tvmc_ml_model_unload_ptr mrvl_tvmc_ml_model_unload;
  mrvl_tvmc_ml_io_alloc_ptr mrvl_tvmc_ml_io_alloc;
  mrvl_tvmc_ml_io_free_ptr mrvl_tvmc_ml_io_free;
  mrvl_tvmc_ml_model_quantize_ptr mrvl_tvmc_ml_model_quantize;
  mrvl_tvmc_ml_model_dequantize_ptr mrvl_tvmc_ml_model_dequantize;
  mrvl_tvmc_ml_model_run_ptr mrvl_tvmc_ml_model_run;
};

/* DPDK callback functions */
extern "C" typedef int (*mrvl_dpdk_glow_layer_load_cb)(void* device, uint16_t model_id,
                                                       const char* layer_name, uint8_t* buffer,
                                                       size_t size, uint16_t* index);
extern "C" typedef int (*mrvl_dpdk_glow_layer_unload_cb)(void* device, uint16_t model_id,
                                                         const char* layer_name);

extern "C" typedef int (*mrvl_dpdk_io_alloc_cb)(void* device, uint16_t model_id,
                                                const char* layer_name, uint64_t** input_qbuffer,
                                                uint64_t** output_qbuffer);
extern "C" typedef int (*mrvl_dpdk_io_free_cb)(void* device, uint16_t model_id,
                                               const char* layer_name);

extern "C" typedef int (*mrvl_dpdk_malloc_cb)(const char* name, size_t size, uint32_t align,
                                              void** addr);
extern "C" typedef int (*mrvl_dpdk_free_cb)(const char* name);

extern "C" typedef int (*mrvl_dpdk_quantize_cb)(void* device, uint16_t model_id,
                                                const char* layer_name, const DLTensor** deq_tensor,
                                                void* qbuffer);
extern "C" typedef int (*mrvl_dpdk_dequantize_cb)(void* device, uint16_t model_id,
                                                  const char* layer_name, void* qbuffer,
                                                  const DLTensor** deq_tensor);
extern "C" typedef int (*mrvl_dpdk_inference_cb)(void* device, uint16_t index, void* input,
                                                 void* output, uint16_t nb_batches);

/* Call back functions structure */
struct ml_dpdk_cb {
  mrvl_dpdk_glow_layer_load_cb mrvl_dpdk_glow_layer_load;
  mrvl_dpdk_glow_layer_unload_cb mrvl_dpdk_glow_layer_unload;
  mrvl_dpdk_io_alloc_cb mrvl_dpdk_io_alloc;
  mrvl_dpdk_io_free_cb mrvl_dpdk_io_free;
  mrvl_dpdk_malloc_cb mrvl_dpdk_malloc;
  mrvl_dpdk_free_cb mrvl_dpdk_free;
  mrvl_dpdk_quantize_cb mrvl_dpdk_quantize;
  mrvl_dpdk_dequantize_cb mrvl_dpdk_dequantize;
  mrvl_dpdk_inference_cb mrvl_dpdk_inference;
};

void get_tvmc_callbacks(const char* so_path, ml_tvmc_cb* obj) {
  obj->handle = dlopen(so_path, RTLD_LAZY);
  if (obj->handle == nullptr)
    ICHECK(false) << "Marvell-Runtime-ERROR Loading shared library failed";

  obj->mrvl_tvmc_ml_init = (mrvl_tvmc_ml_init_ptr)dlsym(obj->handle, "mrvl_ml_init");
  obj->mrvl_tvmc_ml_finish = (mrvl_tvmc_ml_finish_ptr)dlsym(obj->handle, "mrvl_ml_finish");
  obj->mrvl_tvmc_ml_model_load =
      (mrvl_tvmc_ml_model_load_ptr)dlsym(obj->handle, "mrvl_ml_model_load");
  obj->mrvl_tvmc_ml_model_unload =
      (mrvl_tvmc_ml_model_unload_ptr)dlsym(obj->handle, "mrvl_ml_model_unload");
  obj->mrvl_tvmc_ml_io_alloc = (mrvl_tvmc_ml_io_alloc_ptr)dlsym(obj->handle, "mrvl_ml_io_alloc");
  obj->mrvl_tvmc_ml_io_free = (mrvl_tvmc_ml_io_free_ptr)dlsym(obj->handle, "mrvl_ml_io_free");
  obj->mrvl_tvmc_ml_model_quantize =
      (mrvl_tvmc_ml_model_quantize_ptr)dlsym(obj->handle, "mrvl_ml_model_quantize");
  obj->mrvl_tvmc_ml_model_dequantize =
      (mrvl_tvmc_ml_model_dequantize_ptr)dlsym(obj->handle, "mrvl_ml_model_dequantize");
  obj->mrvl_tvmc_ml_model_run = (mrvl_tvmc_ml_model_run_ptr)dlsym(obj->handle, "mrvl_ml_model_run");
}

/*!
 * \brief A json runtime that compiles the serialized JSON format to a binary for Marvell
hardware and then runs the generated binary on the target hardware.
 * \param symbol_name The name of the subgraph / relay function
 * \param nodes_json The serialized JSON representation of relay function
 * \param bin_code The binary code generated by the Marvell backend compiler for the subgraph
 * \param input_count Number of subgraph inputs
 * \param output_count Number of subgraph outputs
 * \param batch_size Batch count
 *
 */

class MarvellHardwareModuleNode : public ModuleNode {
 public:
  MarvellHardwareModuleNode(const std::string& symbol_name, const std::string& nodes_json,
                            const std::string& bin_code, const int input_count,
                            const int output_count, const int batch_size)
      : symbol_name_(symbol_name),
        nodes_json_(nodes_json),
        bin_code_(bin_code),
        num_inputs_(input_count),
        num_outputs_(output_count) {
    run_arg.num_batches = batch_size;
  }

  ~MarvellHardwareModuleNode() {
    if (use_dpdk_cb) {
      int ret;

      // Deallocate input quantize and output quantize buffer
      ret = dpdk_cb_.mrvl_dpdk_io_free(device_handle, run_arg.model_id, symbol_name_.c_str());

      ICHECK(ret == 0) << "IO free failed, model_id =" << run_arg.model_id;

      // Unload model
      ret = dpdk_cb_.mrvl_dpdk_glow_layer_unload(run_arg.device, run_arg.model_id,
                                                 symbol_name_.c_str());
      ICHECK(ret == 0) << "Model layer unload failed, model_id =" << run_arg.model_id;
      num_loaded--;
    } else {
      // Clean Up
      if (tvmc_cb_.handle != nullptr) {
        // Deallocate input quantize and dequant buffer
        tvmc_cb_.mrvl_tvmc_ml_io_free(run_arg.model_id, input_quantize, run_arg.i_q_buf);
        tvmc_cb_.mrvl_tvmc_ml_io_free(run_arg.model_id, input_dequantize, i_d_buf);
        // Deallocate output quantize and dequant buffer
        tvmc_cb_.mrvl_tvmc_ml_io_free(run_arg.model_id, output_quantize, run_arg.o_q_buf);
        tvmc_cb_.mrvl_tvmc_ml_io_free(run_arg.model_id, output_dequantize, o_d_buf);
        // Unload model
        tvmc_cb_.mrvl_tvmc_ml_model_unload(run_arg.model_id);
        num_loaded--;
      }
      // All models unloaded; finish the session
      if (tvmc_cb_.handle != nullptr && num_loaded == 0) tvmc_cb_.mrvl_tvmc_ml_finish();
    }
  }

  const char* type_key() const { return "mrvl_hw"; }

  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "register_cb") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        struct ml_dpdk_cb* a = static_cast<struct ml_dpdk_cb*>(args[0].value().v_handle);
        memcpy(&dpdk_cb_, a, sizeof(struct ml_dpdk_cb));
        device_handle = args[1].value().v_handle;
        model_id = args[2];
        use_dpdk_cb = true;
      });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = Array<String>{}; });
    } else if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        RunInference(args);
        *rv = 0;
      });
    } else if ("__init_" + this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        run_arg.device = device_handle;
        run_arg.model_id = model_id;
        load_and_initialize_model();
        *rv = 0;
      });
    }
    return PackedFunc(nullptr);
  }

  virtual void SaveToBinary(dmlc::Stream* stream) {
    // Save the symbol name and other data and serialize them to
    // binary format.
    stream->Write(symbol_name_);
    stream->Write(nodes_json_);
    stream->Write(bin_code_);
    stream->Write(num_inputs_);
    stream->Write(num_outputs_);
    stream->Write(run_arg.num_batches);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string symbol_name;
    std::string nodes_json;
    std::string bin_code;
    int num_inputs, num_outputs, batch_size;

    // Load the symbol_name and other data to construct the module
    ICHECK(stream->Read(&symbol_name)) << "Loading symbol name failed";
    ICHECK(stream->Read(&nodes_json)) << "Loading nodes json failed";
    ICHECK(stream->Read(&bin_code)) << "Loading binary code failed";
    ICHECK(stream->Read(&num_inputs)) << "Loading num_inputs failed";
    ICHECK(stream->Read(&num_outputs)) << "Loading num_outputs failed";
    ICHECK(stream->Read(&batch_size)) << "Loading batch_size failed";
    auto n = make_object<MarvellHardwareModuleNode>(symbol_name, nodes_json, bin_code, num_inputs,
                                                    num_outputs, batch_size);
    return Module(n);
  }

  /*!
   * \brief Get the source generated by codegen.
   *
   * \param format the format to return.
   * \return A string of JSON.
   */
  String GetSource(const String& format = "json") override { return nodes_json_; }

 protected:
  std::string symbol_name_;
  std::string nodes_json_;
  std::string bin_code_;
  int num_inputs_;
  int num_outputs_;
  static ml_tvmc_cb tvmc_cb_;
  static ml_dpdk_cb dpdk_cb_;
  static bool initialized_model;
  static int num_loaded;
  void* i_d_buf = nullptr;
  void* o_d_buf = nullptr;
  struct run_args run_arg;
  static bool use_dpdk_cb;

  void RunInference_TVMC(TVMArgs args) {
    float* i_d_buf_float;
    float* o_d_buf_float;
    const DLTensor* tensor;

    i_d_buf_float = reinterpret_cast<float*>(i_d_buf);
    for (int in = 0; in < num_inputs_; in++) {
      if (args[in].IsObjectRef<NDArray>()) {
        NDArray arr = args[in];
        tensor = arr.operator->();
      } else {
        tensor = args[in].operator DLTensor*();
      }

      if (num_inputs_ == 1) {
        // Perform Quantization
        tvmc_cb_.mrvl_tvmc_ml_model_quantize(
            run_arg.model_id, reinterpret_cast<float*>(tensor->data) + tensor->byte_offset,
            run_arg.i_q_buf);
      } else {
        uint64_t in_tot_dim = 1;

        for (int i = 0; i < tensor->ndim; i++) {
          in_tot_dim *= tensor->shape[i];
        }

        memcpy(i_d_buf_float, tensor->data, sizeof(float) * in_tot_dim);
        i_d_buf_float += in_tot_dim;
      }
    }

    if (num_inputs_ > 1) {
      // Perform Quantization
      tvmc_cb_.mrvl_tvmc_ml_model_quantize(run_arg.model_id, i_d_buf, run_arg.i_q_buf);
    }

    tvmc_cb_.mrvl_tvmc_ml_model_run(run_arg.model_id, run_arg.i_q_buf, run_arg.o_q_buf,
                                    run_arg.num_batches);

    const DLTensor* outTensor;
    int out = num_inputs_;

    if (num_outputs_ == 1) {
      if (args[out].IsObjectRef<NDArray>()) {
        NDArray arr = args[out];
        outTensor = arr.operator->();
      } else {
        outTensor = args[out].operator DLTensor*();
      }
      tvmc_cb_.mrvl_tvmc_ml_model_dequantize(
          run_arg.model_id, run_arg.o_q_buf,
          (reinterpret_cast<float*>(outTensor->data) + outTensor->byte_offset));

    } else {
      tvmc_cb_.mrvl_tvmc_ml_model_dequantize(run_arg.model_id, run_arg.o_q_buf, o_d_buf);
      o_d_buf_float = reinterpret_cast<float*>(o_d_buf);

      for (out = num_inputs_; out < args.size(); out++) {
        int out_tot_dim = 1;
        if (args[out].IsObjectRef<NDArray>()) {
          NDArray arr = args[out];
          outTensor = arr.operator->();
        } else {
          outTensor = args[out].operator DLTensor*();
        }

        for (int i = 0; i < outTensor->ndim; i++) {
          out_tot_dim *= outTensor->shape[i];
        }

        memcpy(outTensor->data, o_d_buf_float, sizeof(float) * out_tot_dim);
        o_d_buf_float += out_tot_dim;
      }
    }
  }

  void RunInference_DPDK(TVMArgs args) {
    const DLTensor* tensor[64];

    for (int in = 0; in < num_inputs_; in++) {
      if (args[in].IsObjectRef<NDArray>()) {
        NDArray arr = args[in];
        tensor[in] = arr.operator->();
      } else {
        tensor[in] = args[in].operator DLTensor*();
      }
    }

    dpdk_cb_.mrvl_dpdk_quantize(run_arg.device, run_arg.model_id, symbol_name_.c_str(), tensor,
                                run_arg.i_q_buf);

    dpdk_cb_.mrvl_dpdk_inference(run_arg.device, run_arg.layer_idx, run_arg.i_q_buf,
                                 run_arg.o_q_buf, run_arg.num_batches);

    int i = 0;
    for (int out = num_inputs_; out < args.size(); out++) {
      if (args[out].IsObjectRef<NDArray>()) {
        NDArray arr = args[out];
        tensor[i] = arr.operator->();
      } else {
        tensor[i] = args[out].operator DLTensor*();
      }
      i++;
    }

    dpdk_cb_.mrvl_dpdk_dequantize(run_arg.device, run_arg.model_id, symbol_name_.c_str(),
                                  run_arg.o_q_buf, tensor);
  }

  void RunInference(TVMArgs args) {
    if (use_dpdk_cb)
      RunInference_DPDK(args);
    else
      RunInference_TVMC(args);
  }

  void load_and_initialize_model() {
    // Load dll and get the APIs from Library
    if (!(use_dpdk_cb) && !(initialized_model)) {
      char* libpath = getenv("MRVL_LIBMLDPC_PATH");
      if (libpath == nullptr) {
        std::string str = MRVL_LIBMLDPC_DEFAULT_PATH;
        libpath = new char[str.length() + 1];
        snprintf(libpath, str.length() + 1, "%s", str.c_str());
      }
      std::cout << "MRVL_LIBMLDPC_PATH: " << libpath << std::endl;
      get_tvmc_callbacks(const_cast<char*>(libpath), &tvmc_cb_);
      int argc = 1;
      char* argv[] = {const_cast<char*>("tvmc")};
      tvmc_cb_.mrvl_tvmc_ml_init(argc, argv);
      initialized_model = true;
    }

    // Create byte array to pass to the init function
    int num_bytes = tvm::runtime::contrib::mrvl::b64strlen(bin_code_);
    std::vector<unsigned char> byte_array(num_bytes);
    tvm::runtime::contrib::mrvl::b64decode(bin_code_, byte_array.data());

    if (use_dpdk_cb) {
      int ret;
      ret = dpdk_cb_.mrvl_dpdk_glow_layer_load(
          run_arg.device, run_arg.model_id, symbol_name_.c_str(),
          reinterpret_cast<uint8_t*>(byte_array.data()), num_bytes, &run_arg.layer_idx);
      ICHECK(ret == 0) << "Model layer load failed, model_id =" << run_arg.model_id;
      num_loaded++;

      // Allocate input quantize and output quantize buffer
      ret = dpdk_cb_.mrvl_dpdk_io_alloc(device_handle, run_arg.model_id, symbol_name_.c_str(),
                                        reinterpret_cast<uint64_t**>(&run_arg.i_q_buf),
                                        reinterpret_cast<uint64_t**>(&run_arg.o_q_buf));
      ICHECK(ret == 0) << "IO alloc failed, model_id =" << run_arg.model_id;
    } else {
      // Load the model
      run_arg.model_id =
          tvmc_cb_.mrvl_tvmc_ml_model_load(reinterpret_cast<char*>(byte_array.data()), num_bytes);
      ICHECK(run_arg.model_id >= 0) << "Failed to load model!";
      num_loaded++;
      // Allocate input quantize and dequant buffer
      run_arg.i_q_buf = tvmc_cb_.mrvl_tvmc_ml_io_alloc(run_arg.model_id, input_quantize, nullptr);
      i_d_buf = tvmc_cb_.mrvl_tvmc_ml_io_alloc(run_arg.model_id, input_dequantize, nullptr);
      // Allocate output quantize and dequant buffer
      run_arg.o_q_buf = tvmc_cb_.mrvl_tvmc_ml_io_alloc(run_arg.model_id, output_quantize, nullptr);
      o_d_buf = tvmc_cb_.mrvl_tvmc_ml_io_alloc(run_arg.model_id, output_dequantize, nullptr);
    }
  }
};

runtime::Module MarvellHardwareModuleRuntimeCreate(const String& symbol_name,
                                                   const String& nodes_json, const String& bin_code,
                                                   int num_input, int num_output, int batch_size) {
  auto n = make_object<MarvellHardwareModuleNode>(symbol_name, nodes_json, bin_code, num_input,
                                                  num_output, batch_size);
  return runtime::Module(n);
}

bool MarvellHardwareModuleNode::initialized_model = false;
int MarvellHardwareModuleNode::num_loaded = 0;
bool MarvellHardwareModuleNode::use_dpdk_cb = false;
ml_tvmc_cb MarvellHardwareModuleNode::tvmc_cb_ = {};
ml_dpdk_cb MarvellHardwareModuleNode::dpdk_cb_ = {};

TVM_REGISTER_GLOBAL("runtime.mrvl_hw_runtime_create")
    .set_body_typed(MarvellHardwareModuleRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_mrvl_hw")
    .set_body_typed(MarvellHardwareModuleNode::LoadFromBinary);
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
