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
 * \file src/runtime/contrib/clml/clml_runtime.cc
 * \brief A simple JSON runtime for CLML.
 */

#include <CL/cl.h>
#include <CL/opencl.h>
#ifdef TVM_GRAPH_EXECUTOR_CLML
#include <CL/cl_qcom_ml_ops.h>
#endif
#include <stdlib.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <map>
#include <utility>

#include "../../opencl/opencl_common.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

class CLMLRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The CLML runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit CLMLRuntime(const std::string& symbol_name, const std::string& graph_json,
                       const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~CLMLRuntime() {
#ifdef TVM_GRAPH_EXECUTOR_CLML
    cl_int result = 0;
    if (this->is_tuning_run) {
      result = h_ClmlIntf->clReleaseMLTuningCacheQCOM(this->tuning_cache);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTuningCacheQCOM:" << result;
    }
    for (auto it = this->layer_.storage_map.begin(); it != this->layer_.storage_map.end(); it++) {
      auto tensor_desc = it->second.first;
      result = h_ClmlIntf->clReleaseMLTensorQCOM(tensor_desc->tensor);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorQCOM:" << result;
      result = clReleaseMemObject(tensor_desc->memory);
      ICHECK(result == CL_SUCCESS) << "clReleaseMemObject:" << result;
    }
    for (size_t i = 0; i < this->layer_.function.size(); ++i) {
      result = h_ClmlIntf->clReleaseMLOpQCOM(this->layer_.function[i]);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLOpQCOM:" << result;
    }
    for (auto it = this->layer_.in_placeholder.begin(); it != this->layer_.in_placeholder.end();
         it++) {
      result = h_ClmlIntf->clReleaseMLTensorQCOM((*it)->tensor);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorQCOM:" << result;
    }
    for (auto it = this->layer_.out_placeholder.begin(); it != this->layer_.out_placeholder.end();
         it++) {
      result = h_ClmlIntf->clReleaseMLTensorQCOM((*it)->tensor);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorQCOM:" << result;
    }
    result = h_ClmlIntf->clReleaseMLTensorMemoryDescriptorSetQCOM(layer_.descriptorSet);
    ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorMemoryDescriptorSetQCOM:" << result;
#endif
  }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const override { return "clml"; }

  /*!
   * \brief Initialize runtime. Create CLML layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);

#ifdef TVM_GRAPH_EXECUTOR_CLML
    InitCLML();
#endif

    BuildEngine();
  }

#ifdef TVM_GRAPH_EXECUTOR_CLML
  std::vector<cl_uint> GetVectorValues(const std::vector<std::string>& val) {
    std::vector<cl_uint> array;
    for (auto i : val) {
      array.push_back((cl_uint)stoi(i));
    }
    return array;
  }

  void InitCLML() {
    // Setup CLML Context
    cl_int result = 0;

    workspace = cl::OpenCLWorkspace::Global();
    workspace->Init();
    tentry = workspace->GetThreadEntry();

    if (!ExtensionStringPresent()) {
      LOG(WARNING) << "CLML Runtime Init: Qualcomm extn not present.\n";
      return;
    }
    // Query and Get CLML Interface
    static const cl_uint MAX_VERSIONS = 256;
    cl_int majorVersions[MAX_VERSIONS];
    cl_int minorVersions[MAX_VERSIONS];
    cl_uint numVersions = 0;
    result = clQueryMLInterfaceVersionsQCOM(NULL, NULL, 0, &numVersions);
    ICHECK(result == CL_SUCCESS) << "clQueryMLInterfaceVersionsQCOM:" << result;
    ICHECK(numVersions > 0u);
    ICHECK(numVersions <= MAX_VERSIONS);

    result = clQueryMLInterfaceVersionsQCOM(majorVersions, minorVersions, numVersions, NULL);
    ICHECK(result == CL_SUCCESS) << "clQueryMLInterfaceVersionsQCOM:" << result;

    for (cl_uint i = 0; i < numVersions; ++i) {
      if (majorVersions[i] == 2) {
        LOG(WARNING) << "CLML Version Selected:" << majorVersions[i] << " : " << majorVersions[i];
        h_ClmlIntf = clGetMLInterfaceV2QCOM(0);
        ICHECK(h_ClmlIntf != NULL) << "clGetMLInterfaceV2QCOM:" << result;
        break;
      }
    }
    char* tune_flag;
    if ((tune_flag = getenv("CLML_IS_TUNNING_RUN")))
      this->is_tuning_run = std::stoi(tune_flag);
    else
      this->is_tuning_run = 0;

    if (!(tuning_file = getenv("CLML_TUNNING_CACHE"))) this->is_tuning_run = 0;
    // A Tuning run, so create the cache from scratch
    result = h_ClmlIntf->clCreateMLTuningCacheQCOM(&tuning_cache);
    ICHECK(result == CL_SUCCESS) << "clCreateMLTuningCacheQCOM:" << result;
    if (!this->is_tuning_run && this->tuning_file) {
      std::vector<unsigned char> buffer;
      buffer = readBinFile(this->tuning_file);
      result = h_ClmlIntf->clLoadMLTuningCacheQCOM(tuning_cache, buffer.size(), buffer.data());
      ICHECK(result == CL_SUCCESS) << "clLoadMLTuningCacheQCOM:" << result;
    }
  }

  std::vector<unsigned char> readBinFile(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    if (!fin.good()) {
      LOG(FATAL) << "ERROR: Could not load tuning cache file: " + filename;
    }
    ICHECK(fin.good());
    int64_t size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    char* ptr = reinterpret_cast<char*>(buffer.data());
    fin.read(ptr, size);
    ICHECK(fin.good());
    return buffer;
  }

  void CopyDataToCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                            cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_NCHW_QCOM) {
    cl_int result = 0;
    cl_event evt = NULL;
    result = h_ClmlIntf->clEnqueueWriteMLTensorDataQCOM(workspace->GetQueue(tentry->device), data,
                                                        layout, tensor->tensor, tensor->memory,
                                                        0,      // n waitlist
                                                        NULL,   // waitlist
                                                        &evt);  // event
    ICHECK((evt != NULL) && result == CL_SUCCESS) << "clEnqueueWriteMLTensorDataQCOM:" << result;
  }

  void CopyDataFromCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                              cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_NCHW_QCOM) {
    cl_int result = 0;
    cl_event readEvent = NULL;
    // Read the output tensor
    result = h_ClmlIntf->clEnqueueReadMLTensorDataQCOM(workspace->GetQueue(tentry->device),
                                                       tensor->tensor, tensor->memory, data, layout,
                                                       0,            // n waitlist
                                                       NULL,         // waitlist
                                                       &readEvent);  // event
    ICHECK(result == CL_SUCCESS) << "clEnqueueReadMLTensorDataQCOM:" << result;

    result = clWaitForEvents(1, &readEvent);
    ICHECK(result == CL_SUCCESS) << "clWaitForEvents:" << result;
  }

  /*!
   * \brief Unpack inputs and outputs and run inference on a given layer.
   *
   * \param args Access inputs and outputs.
   * \param function The layer to execute inference on.
   * \return Status of inference.
   */
  void Run() override {
    cl_int result = 0;
    cl_command_queue queue = workspace->GetQueue(tentry->device);
    std::vector<cl_event>& evts = workspace->GetEventQueue(tentry->device);
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      uint32_t eid = EntryID(nid, 0);
      if (nodes_[nid].GetOpType() == "input") {
        void* data = data_entry_[eid]->data;
        size_t isize = 1;
        for (size_t j = 0; j < data_entry_[eid]->ndim; ++j) {
          isize *= data_entry_[eid]->shape[j];
        }
        if (kDLCPU == data_entry_[eid]->device.device_type) {
          CopyDataToCLMLTensor(layer_.inputs[i], data);
        } else if (kDLOpenCL == data_entry_[eid]->device.device_type) {
          layer_.in_placeholder[i]->memory = static_cast<cl_mem>(
              ((cl::BufferDescriptor*)const_cast<DLTensor*>(data_entry_[eid])->data)->buffer);
          cl_event cpy_evt = NULL;
          result = h_ClmlIntf->clEnqueueCopyMLTensorDataQCOM(
              queue, layer_.in_placeholder[i]->tensor, layer_.in_placeholder[i]->memory,
              layer_.inputs[i]->tensor, layer_.inputs[i]->memory, 0, NULL, &cpy_evt);
          ICHECK(result == CL_SUCCESS) << "clEnqueueCopyMLTensorDataQCOM:" << result;
        } else {
          DLDataType tvm_dtype = const_cast<DLTensor*>(data_entry_[eid])->dtype;
          cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
          int dtype_size = cl_dtype == CL_FLOAT ? 4 : 2;
          void* tmpptr = reinterpret_cast<void*>(malloc(isize * dtype_size));
          TVMArrayCopyToBytes(const_cast<DLTensor*>(data_entry_[eid]), const_cast<void*>(tmpptr),
                              isize * dtype_size);
          CopyDataToCLMLTensor(layer_.inputs[i], tmpptr);
          free(tmpptr);
        }
      }
    }

    for (size_t i = 0; i < this->layer_.function.size(); ++i) {
      if (getenv("CLML_PROFILING")) {
        evts.resize(evts.size() + 1);
        cl_event* evt = &(evts.back());
        result = h_ClmlIntf->clEnqueueMLOpQCOM(queue, this->layer_.function[i],
                                               this->layer_.descriptorSet, 0, NULL, evt);
      } else {
        result = h_ClmlIntf->clEnqueueMLOpQCOM(queue, this->layer_.function[i],
                                               this->layer_.descriptorSet, 0, NULL, NULL);
      }
      ICHECK(result == CL_SUCCESS) << "clEnqueueMLOpQCOM:" << result;
    }

    if (getenv("CLML_PROFILING")) {
      cl_ulong start, end;
      cl_ulong duration = 0;
      clWaitForEvents(1, &(evts.back()));
      for (size_t i = 0; i < this->layer_.layer_names.size(); ++i) {
        clGetEventProfilingInfo(evts[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
                                nullptr);
        clGetEventProfilingInfo(evts[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        duration += (end - start);
        LOG(WARNING) << "Layer:" << this->layer_.layer_names[i] << " Duration:" << (end - start);
      }
      LOG(WARNING) << "Total Duration:" << duration;
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      void* data = data_entry_[eid]->data;

      size_t osize = 1;
      for (size_t j = 0; j < data_entry_[eid]->ndim; ++j) {
        osize *= data_entry_[eid]->shape[j];
      }
      if (kDLCPU == data_entry_[eid]->device.device_type) {
        CopyDataFromCLMLTensor(layer_.outputs[0], data);
      } else if (kDLOpenCL == data_entry_[eid]->device.device_type) {
        layer_.out_placeholder[i]->memory = static_cast<cl_mem>(
            ((cl::BufferDescriptor*)const_cast<DLTensor*>(data_entry_[eid])->data)->buffer);
        cl_event cpy_evt = NULL;
        result = h_ClmlIntf->clEnqueueCopyMLTensorDataQCOM(
            queue, layer_.outputs[i]->tensor, layer_.outputs[i]->memory,
            layer_.out_placeholder[i]->tensor, layer_.out_placeholder[i]->memory, 0, NULL,
            &cpy_evt);
        ICHECK(result == CL_SUCCESS) << "clEnqueueCopyMLTensorDataQCOM:" << result;
      } else {
        DLDataType tvm_dtype = const_cast<DLTensor*>(data_entry_[eid])->dtype;
        cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
        int dtype_size = cl_dtype == CL_FLOAT ? 4 : 2;

        void* tmpptr = reinterpret_cast<void*>(malloc(osize * dtype_size));
        CopyDataFromCLMLTensor(layer_.outputs[0], tmpptr);
        TVMArrayCopyFromBytes(const_cast<DLTensor*>(data_entry_[eid]), const_cast<void*>(tmpptr),
                              osize * dtype_size);
        free(tmpptr);
      }
    }
  }

 private:
  /*!
   * \brief Build CLML layer from JSON representation and cache.
   *
   * \note For the time being only one layer or operator is supported
   * per engine.
   */
  void BuildEngine() {
    size_t nid;
    for (nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      DLDataType tvm_dtype = node.GetOpDataType()[0];
      cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
      if (node.GetOpType() == "input") {
        auto clml_input = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
        this->layer_.storage_map.insert({nid, std::make_pair(clml_input, node)});
        this->layer_.inputs.push_back(clml_input);
        // Input copy placeholder Tensor
        this->layer_.in_placeholder.push_back(
            MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype));
      } else if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          auto out = CreateConvolution2DLayer(&layer_, node, CL_CONVOLUTION_MODE_CONVOLUTION_QCOM);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.depthwise_conv2d" == op_name) {
          auto out = CreateConvolution2DLayer(&layer_, node, CL_CONVOLUTION_MODE_DEPTHWISE_QCOM);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.relu6" == op_name) {
          auto out = CreateReLULayer(&layer_, node, CL_ACTIVATION_RELU6);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.relu" == op_name) {
          auto out = CreateReLULayer(&layer_, node, CL_ACTIVATION_RELU);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.batch_norm" == op_name) {
          auto out = CreateBatchNormLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.max_pool2d" == op_name || "nn.avg_pool2d" == op_name ||
                   "nn.l2_pool2d" == op_name) {
          auto out = CreatePoolingLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.global_max_pool2d" == op_name || "nn.global_avg_pool2d" == op_name) {
          auto out = CreateGlobalPoolingLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("reshape" == op_name) {
          auto out = CreateReshapeLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("concatenate" == op_name) {
          auto out = CreateConcatLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.dense" == op_name) {
          auto out = CreateDenseLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.softmax" == op_name) {
          auto out = CreateSoftMaxLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("nn.pad" == op_name) {
          auto out = CreatePadLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("clip" == op_name) {
          auto out = CreateClipLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else if ("add" == op_name || "subtract" == op_name || "multiply" == op_name ||
                   "minimum" == op_name || "maximum" == op_name) {
          auto out = CreateBinaryLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
          this->layer_.func_outs.push_back(out);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
        this->layer_.layer_names.push_back(op_name);
      } else if (node.GetOpType() != "const") {
        LOG(WARNING) << "Build Engine: Unknown Node:" << node.GetOpType();
      }
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      nid = outputs_[i].id_;
      DLDataType tvm_dtype = nodes_[nid].GetOpDataType()[0];
      cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
      this->layer_.outputs.push_back(this->layer_.storage_map[nid].first);
      this->layer_.out_placeholder.push_back(
          MakeCLMLTensorFromJSONNode(nodes_[nid], CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype));
    }
    // ALlocate device memories and initialize the params if any
    cl_int result = 0;
    for (auto it = this->layer_.storage_map.begin(); it != this->layer_.storage_map.end(); it++) {
      auto tensor_desc = it->second.first;
      JSONGraphNode node = it->second.second;
      void* node_data = nullptr;

      allocateTensorMemory(h_ClmlIntf, workspace->context, tensor_desc);

      if (node.GetOpType() == "const") {
        node_data = data_entry_[EntryID(it->first, 0)]->data;
        if (node_data != nullptr) {
          CopyDataToCLMLTensor(tensor_desc, node_data);
        }
      }
      this->layer_.tensorMemDescs.push_back(*tensor_desc);
    }

    // Setup descriptor set
    result = h_ClmlIntf->clCreateMLTensorMemoryDescriptorSetQCOM(&this->layer_.descriptorSet);
    ICHECK(result == CL_SUCCESS) << "clCreateMLTensorMemoryDescriptorSetQCOM:" << result;

    result = h_ClmlIntf->clUpdateMLTensorMemoryDescriptorSetQCOM(
        this->layer_.descriptorSet, static_cast<uint32_t>(this->layer_.tensorMemDescs.size()),
        this->layer_.tensorMemDescs.data());
    ICHECK(result == CL_SUCCESS) << "clUpdateMLTensorMemoryDescriptorSetQCOM:" << result;

    if (this->is_tuning_run) {
      LOG(WARNING) << "CLML Tunning In Progress:";
      for (size_t i = 0; i < this->layer_.function.size(); ++i) {
        LOG(WARNING) << "CLML Tunning:" << i;
        result = h_ClmlIntf->clTuneMLOpQCOM(workspace->GetQueue(tentry->device),
                                            this->layer_.function[i], this->layer_.descriptorSet,
                                            this->tuning_cache, NULL);
        ICHECK(result == CL_SUCCESS) << "clTuneMLOpQCOM:" << result;
      }

      size_t cacheLenBytes = 0;
      size_t lenRet = 0;
      result = h_ClmlIntf->clSaveMLTuningCacheQCOM(tuning_cache, 0, NULL, &cacheLenBytes);
      ICHECK(result == CL_SUCCESS) << "clSaveMLTuningCacheQCOM:" << result;

      std::vector<unsigned char> savedCache(cacheLenBytes, 0);
      result = h_ClmlIntf->clSaveMLTuningCacheQCOM(tuning_cache, savedCache.size(),
                                                   savedCache.data(), &lenRet);
      assert(result == CL_SUCCESS);

      std::ofstream cache_out(tuning_file, std::ios_base::binary);
      if (cache_out) {
        cache_out.write(reinterpret_cast<char*>(savedCache.data()), savedCache.size());
        cache_out.close();
      }
      LOG(WARNING) << "CLML: Tuning cache dumped to:" << tuning_file;
    }
  }

  /*!
   * \brief CLML objects we cache in order to avoid needing to construct
   * a new layer each time.
   */
  struct CachedLayer {
    std::vector<cl_ml_op_qcom> function;
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> inputs;
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> in_placeholder;
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> outputs;
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> out_placeholder;
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> func_outs;
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> func_ins;
    std::map<int, std::pair<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>, JSONGraphNode>>
        storage_map;
    std::vector<cl_ml_tensor_memory_desc_qcom> tensorMemDescs;
    std::vector<cl_ml_tensor_memory_desc_qcom> in_tensorMemDescs;
    std::vector<cl_ml_tensor_memory_desc_qcom> out_tensorMemDescs;
    cl_ml_tensor_mem_desc_set_qcom descriptorSet;
    std::vector<std::string> layer_names;
    cl_ml_tensor_qcom unusedTensor = NULL;
  };

  struct tensor_dims_t {
    uint32_t n, c, h, w;
  };

  bool ExtensionStringPresent(void) {
    cl_int result = 0;
    if (workspace->platform_id == nullptr) {
      return 0;
    }
    size_t reqd_size = 0;
    cl_device_id device_id = workspace->devices[workspace->GetThreadEntry()->device.device_id];
    result = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &reqd_size);
    ICHECK(reqd_size > 0u && result == CL_SUCCESS) << "clGetDeviceInfo:" << result;

    std::vector<char> buf(reqd_size);
    result = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, reqd_size, buf.data(), NULL);
    ICHECK(result == CL_SUCCESS) << "clGetDeviceInfo:" << result;

    std::string extensions(buf.data());
    LOG(WARNING) << "OpenCL Extensions:" << extensions;
    return (extensions.find("cl_qcom_ml_ops") != std::string::npos);
  }

  cl_ml_tensor_qcom DeviceMakeCLMLTensor(
      void* pClmlIntf, cl_context context, tensor_dims_t dims,
      cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
      cl_channel_type dtype = CL_FLOAT) {
    cl_ml_tensor_qcom tensor;
    cl_int result = CL_OUT_OF_RESOURCES;

    cl_ml_tensor_desc_qcom desc = {
        dtype, layout, dims.n, dims.c, dims.h, dims.w, 0, CL_TENSOR_DIMENSIONS_4D_QCOM, { 0 }};
    CLMLInterfaceV2QCOM* clmlIntf = reinterpret_cast<CLMLInterfaceV2QCOM*>(pClmlIntf);
    result = clmlIntf->clCreateMLTensorQCOM(workspace->context, NULL, &desc, &tensor);
    ICHECK(tensor && result == CL_SUCCESS) << "clCreateMLTensorQCOM:" << result;
    (void)result;
    return tensor;
  }

  cl_int allocateTensorMemory(void* pClmlIntf, cl_context context,
                              std::shared_ptr<cl_ml_tensor_memory_desc_qcom> pTensorMemDesc) {
    uint32_t size = 0;
    cl_int result = CL_OUT_OF_HOST_MEMORY;
    cl_mem buffer = NULL;

    CLMLInterfaceV2QCOM* clmlIntf = reinterpret_cast<CLMLInterfaceV2QCOM*>(pClmlIntf);
    result =
        clmlIntf->clGetMLTensorMemorySizeQCOM(workspace->context, pTensorMemDesc->tensor, &size);
    ICHECK(result == CL_SUCCESS) << "clGetMLTensorMemorySizeQCOM:" << result;

    buffer = clCreateBuffer(workspace->context, CL_MEM_READ_WRITE, size, NULL, &result);
    ICHECK(result == CL_SUCCESS) << "clCreateBuffer:" << result;

    pTensorMemDesc->memory = buffer;

    return result;
  }

  tensor_dims_t get_tensor_dims(const JSONGraphNode& node) {
    std::vector<int64_t> shape = node.GetOpShape()[0];
    tensor_dims_t dims;
    dims.n = shape[0];
    dims.c = shape[1];
    dims.h = shape[2];
    dims.w = shape[3];
    return dims;
  }

  cl_channel_type MakeCLDataType(const DLDataType& data_type) {
    if (data_type.code == DLDataTypeCode::kDLFloat && data_type.bits == 32) {
      return CL_FLOAT;
    } else if (data_type.code == DLDataTypeCode::kDLFloat && data_type.bits == 16) {
      return CL_HALF_FLOAT;
    } else {
      LOG(FATAL) << "Datatype " << data_type << " unsupported by CLML runtime";
      return -1;
    }
  }

  cl_arithmetic_mode_qcom MakeCLArithMode(const cl_channel_type& data_type,
                                          const cl_channel_type& acc_type = CL_FLOAT) {
    if (data_type == CL_FLOAT && acc_type == CL_FLOAT) {
      return CL_ARITHMETIC_MODE_FP32_QCOM;
    } else if (data_type == CL_HALF_FLOAT && acc_type == CL_FLOAT) {
      return CL_ARITHMETIC_MODE_FP16_ACC32_QCOM;
    } else if (data_type == CL_HALF_FLOAT && acc_type == CL_HALF_FLOAT) {
      return CL_ARITHMETIC_MODE_FP16_QCOM;
    } else {
      LOG(FATAL) << "Datatype " << data_type << " unsupported by CLML runtime";
      return CL_ARITHMETIC_MODE_FP32_QCOM;
    }
  }

  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensor(
      const JSONGraphNode& tensor_rep, void* data, std::vector<size_t> c_shape,
      cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_uint dtype = CL_FLOAT) {
    std::vector<int64_t> shape = tensor_rep.GetOpShape()[0];
    std::vector<size_t> clml_shape(shape.begin(), shape.end());
    if (c_shape.size() > 0) {
      clml_shape = c_shape;
    }
    // Make sure the tensors with dimensions less than 4 are padded with 1.
    clml_shape.push_back(1);
    clml_shape.push_back(1);
    clml_shape.push_back(1);

    tensor_dims_t dims;
    dims.n = clml_shape[0];
    dims.c = clml_shape[1];
    dims.h = clml_shape[2];
    dims.w = clml_shape[3];
    DLDataType tvm_dtype = tensor_rep.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);

    auto tensor_dsc = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    tensor_dsc->tensor =
        DeviceMakeCLMLTensor(h_ClmlIntf, workspace->context, dims, layout, cl_dtype);
    return tensor_dsc;
  }

  /*!
   * \brief Create an CLML tensor given the JSON representation. If scale
   * and offset are given, then create a quantized CLML tensor.
   *
   * \param tensor The tensor to represent.
   * \return CLML Tensor.
   */

  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensorFromJSONEntry(
      const JSONGraphNodeEntry& tensor, std::vector<size_t> shape = {},
      cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_uint dtype = CL_FLOAT) {
    JSONGraphNode node = nodes_[tensor.id_];
    if (this->layer_.storage_map.find(tensor.id_) == this->layer_.storage_map.end()) {
      void* node_data = nullptr;
      if (node.GetOpType() == "const") {
        node_data = data_entry_[EntryID(tensor)]->data;
      }
      auto clml_tensor = MakeCLMLTensorFromJSONNode(node, layout, dtype, node_data, shape);
      this->layer_.storage_map.insert({tensor.id_, std::make_pair(clml_tensor, node)});
      return clml_tensor;
    } else {
      return this->layer_.storage_map[tensor.id_].first;
    }
  }
  /*!
   * \brief Create an CLML tensor given the JSON representation. If scale
   * and offset are given, then create a quantized CLML tensor.
   *
   * \param node The tensor to represent.
   * \param data (optional) Constant data of input node.
   * \return CLML Tensor.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensorFromJSONNode(
      const JSONGraphNode& node, cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
      cl_uint dtype = CL_FLOAT, void* data = nullptr, std::vector<size_t> shape = {}) {
    return MakeCLMLTensor(node, data, shape, layout, dtype);
  }
  /*!
   * \brief Create a 2D convolution layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateConvolution2DLayer(
      CachedLayer* layer, const JSONGraphNode& node, cl_convolution_mode_qcom mode) {
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<cl_uint> clml_padding = GetVectorValues(padding);
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    if (!node.HasAttr("padding")) {
      clml_padding.resize(4);
      std::fill(clml_padding.begin(), clml_padding.end(), 0);
    }
    cl_uint clml_padding_b[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {clml_padding[0], clml_padding[1]};
    cl_uint clml_padding_a[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {clml_padding[2], clml_padding[3]};
    std::vector<cl_uint> v_strides = GetVectorValues(strides);
    std::vector<cl_uint> v_dilation = GetVectorValues(dilation);
    cl_uint clml_strides[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {v_strides[0], v_strides[1]};
    cl_uint clml_dilation[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {v_dilation[0], v_dilation[1]};
    cl_int result = 0;

    cl_uint groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    if (CL_CONVOLUTION_MODE_CONVOLUTION_QCOM == mode) {
      ICHECK(groups == 1) << "CLML convolution only supports group size of 1.";
    } else {
      groups = 1;  // Don't need to pass groups to depthwise
    }

    bool has_act = false;
    std::string activation_type;
    cl_activation_function_qcom clml_act_type = CL_ACTIVATION_RELU;
    if (node.HasAttr("activation_type")) {
      activation_type = node.GetAttr<std::vector<std::string>>("activation_type")[0];
      ICHECK(activation_type == "relu" || activation_type == "relu6")
          << "Unknown activation type:" << activation_type;
      if (activation_type == "relu") {
        clml_act_type = CL_ACTIVATION_RELU;
      } else {
        clml_act_type = CL_ACTIVATION_RELU6;
      }
      has_act = true;
    }
    cl_ml_op_activation_desc_qcom act_desc = {clml_act_type, CL_PROPAGATE_NAN_QCOM,
                                              cl_arithmetic_mode};

    // Collect inputs and outputs, handling nn.conv2d.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    bool has_bias;
    bool has_bn;
    ICHECK(num_inputs >= 2U && num_inputs <= 7U)
        << "Batchnorm fused convolution requires bax 7 arguments";
    has_bias = (num_inputs == 3) || (num_inputs == 7);
    has_bn = (num_inputs == 6) || (num_inputs == 7);
    // Input
    auto input =
        MakeCLMLTensorFromJSONEntry(inputs[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    // Weight
    auto weight =
        MakeCLMLTensorFromJSONEntry(inputs[1], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    // Bias
    auto bias = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    if (has_bias) {
      bias = MakeCLMLTensorFromJSONEntry(inputs[2], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    } else {
      cl_ml_tensor_desc_qcom desc = {};
      desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
      result =
          h_ClmlIntf->clCreateMLTensorQCOM(workspace->context, NULL, &desc, &layer_.unusedTensor);
      ICHECK(layer_.unusedTensor && result == CL_SUCCESS) << "clCreateMLTensorQCOM:" << result;
      bias->tensor = layer_.unusedTensor;
    }
    // Output
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    cl_ml_op_convolution_desc_qcom conv_desc{mode,
                                             groups,
                                             4,
                                             {clml_padding_b[0], clml_padding_b[1]},
                                             {clml_padding_a[0], clml_padding_a[1]},
                                             {clml_strides[0], clml_strides[1]},
                                             {clml_dilation[0], clml_dilation[1]},
                                             0,
                                             cl_arithmetic_mode};

    cl_ml_op_qcom op = NULL;
    if (!has_bn) {
      if (!has_act) {
        result = h_ClmlIntf->clCreateMLOpConvolutionForwardQCOM(
            workspace->context, 0, &conv_desc, input->tensor, weight->tensor, bias->tensor,
            output->tensor, &op, NULL);
        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      } else {
        result = h_ClmlIntf->clCreateMLOpFusedConvolutionActivationForwardQCOM(
            workspace->context, 0, &conv_desc, &act_desc, input->tensor, weight->tensor,
            bias->tensor, NULL, output->tensor, &op, tuning_cache);
        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      }
      layer_.func_ins.push_back(input);
      layer->function.push_back(op);
    } else {
      int bn_index = has_bias ? 3 : 2;
      int axis = std::stoi(node.GetAttr<std::vector<std::string>>("batchnorm")[0]);
      auto bn_dims = get_tensor_dims(nodes_[inputs[bn_index].id_]);
      std::vector<size_t> bn_shape = {1, 1, 1, 1};
      bn_shape[axis] = bn_dims.n;
      auto bn_mean = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
      auto bn_var = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
      auto bn_scale = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
      auto bn_bias = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
      bn_scale = MakeCLMLTensorFromJSONEntry(inputs[bn_index], bn_shape,
                                             CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
      bn_bias = MakeCLMLTensorFromJSONEntry(inputs[bn_index + 1], bn_shape,
                                            CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
      bn_mean = MakeCLMLTensorFromJSONEntry(inputs[bn_index + 2], bn_shape,
                                            CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
      bn_var = MakeCLMLTensorFromJSONEntry(inputs[bn_index + 3], bn_shape,
                                           CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

      cl_ml_op_batchnorm_desc_qcom bn_desc = {CL_BATCHNORM_MODE_SPATIAL_QCOM, cl_arithmetic_mode};
      if (!has_act) {
        result = h_ClmlIntf->clCreateMLOpFusedConvolutionBatchNormForwardQCOM(
            workspace->context, 0, &conv_desc, &bn_desc, input->tensor, weight->tensor,
            bias->tensor, output->tensor, bn_mean->tensor, bn_var->tensor, bn_scale->tensor,
            bn_bias->tensor, &op, tuning_cache);
        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      } else {
        result = h_ClmlIntf->clCreateMLOpFusedConvolutionBatchNormActivationForwardQCOM(
            workspace->context, 0, &conv_desc, &bn_desc, &act_desc, input->tensor, weight->tensor,
            bias->tensor, output->tensor, NULL, bn_mean->tensor, bn_var->tensor, bn_scale->tensor,
            bn_bias->tensor, &op, tuning_cache);

        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      }
      layer_.func_ins.push_back(input);
      layer->function.push_back(op);
    }
    return output;
  }

  /*!
   * \brief Create a ReLU(X) layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateReLULayer(
      CachedLayer* layer, const JSONGraphNode& node,
      cl_activation_function_qcom clml_act_type = CL_ACTIVATION_RELU) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    cl_ml_op_activation_desc_qcom act_desc = {clml_act_type, CL_PROPAGATE_NAN_QCOM,
                                              cl_arithmetic_mode};

    cl_ml_tensor_desc_qcom desc = {};
    desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
    result =
        h_ClmlIntf->clCreateMLTensorQCOM(workspace->context, NULL, &desc, &layer_.unusedTensor);
    ICHECK(layer_.unusedTensor && result == CL_SUCCESS) << ":" << result;

    result = h_ClmlIntf->clCreateMLOpActivationForwardQCOM(workspace->context, 0, &act_desc,
                                                           input->tensor, layer_.unusedTensor,
                                                           output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Activation Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a batch norm layer.
   *
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateBatchNormLayer(CachedLayer* layer,
                                                                      const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    auto bn_dims = get_tensor_dims(nodes_[node.GetInputs()[1].id_]);
    std::vector<size_t> bn_shape = {1, 1, 1, 1};
    bn_shape[axis] = bn_dims.n;
    auto bn_mean = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    auto bn_var = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    auto bn_scale = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    auto bn_bias = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    bn_scale = MakeCLMLTensorFromJSONEntry(node.GetInputs()[1], bn_shape,
                                           CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    bn_bias = MakeCLMLTensorFromJSONEntry(node.GetInputs()[2], bn_shape,
                                          CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    bn_mean = MakeCLMLTensorFromJSONEntry(node.GetInputs()[3], bn_shape,
                                          CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    bn_var = MakeCLMLTensorFromJSONEntry(node.GetInputs()[4], bn_shape,
                                         CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    cl_ml_op_batchnorm_desc_qcom bn_desc = {CL_BATCHNORM_MODE_SPATIAL_QCOM, cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpBatchNormForwardQCOM(
        workspace->context, 0, &bn_desc, input->tensor, bn_mean->tensor, bn_var->tensor,
        bn_scale->tensor, bn_bias->tensor, output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Batchnorm Error:" << result;

    layer->function.push_back(op);
    layer_.func_ins.push_back(input);
    return output;
  }

  /*!
   * \brief Create a creating pooling layer.
   *
   * \note Currently global_max_pool2d and global_avg_pool2d are supported.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreatePoolingLayer(CachedLayer* layer,
                                                                    const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    auto in_dims = get_tensor_dims(nodes_[node.GetInputs()[0].id_]);

    std::vector<std::string> windows = node.GetAttr<std::vector<std::string>>("pool_size");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<cl_uint> clml_window = GetVectorValues(windows);
    std::vector<cl_uint> clml_stride = GetVectorValues(strides);
    std::vector<cl_uint> clml_padding = GetVectorValues(padding);

    cl_ml_op_pooling_desc_qcom pool_desc = {
        node.GetOpName() == "nn.max_pool2d" ? CL_POOLING_MODE_MAX_QCOM
                                            : CL_POOLING_MODE_AVERAGE_EXCLUDE_PADDING_QCOM,
        4,  // reserved
        {clml_padding[0], clml_padding[1]},
        {clml_padding[2], clml_padding[3]},
        {clml_stride[0], clml_stride[1]},
        {clml_window[0], clml_window[1]},
        CL_PROPAGATE_NAN_QCOM,
        cl_arithmetic_mode,
    };

    cl_ml_tensor_desc_qcom desc = {};
    cl_ml_tensor_qcom unusedTensor = NULL;
    desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
    result = h_ClmlIntf->clCreateMLTensorQCOM(workspace->context, NULL, &desc, &unusedTensor);
    ICHECK(unusedTensor && result == CL_SUCCESS) << ":" << result;

    result =
        h_ClmlIntf->clCreateMLOpPoolingForwardQCOM(workspace->context, 0, &pool_desc, input->tensor,
                                                   unusedTensor, output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Pooling Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a global pooling layer.
   *
   * \note Currently global_max_pool2d and global_avg_pool2d are supported.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateGlobalPoolingLayer(
      CachedLayer* layer, const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    auto in_dims = get_tensor_dims(nodes_[node.GetInputs()[0].id_]);
    cl_ml_op_pooling_desc_qcom pool_desc = {
        node.GetOpName() == "nn.global_max_pool2d" ? CL_POOLING_MODE_MAX_QCOM
                                                   : CL_POOLING_MODE_AVERAGE_EXCLUDE_PADDING_QCOM,
        4,  // reserved
        {0, 0},
        {0, 0},
        {1, 1},
        {in_dims.w, in_dims.h},
        CL_PROPAGATE_NAN_QCOM,
        cl_arithmetic_mode,
    };

    cl_ml_tensor_desc_qcom desc = {};
    desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
    result =
        h_ClmlIntf->clCreateMLTensorQCOM(workspace->context, NULL, &desc, &layer_.unusedTensor);
    ICHECK(layer_.unusedTensor && result == CL_SUCCESS) << ":" << result;

    result = h_ClmlIntf->clCreateMLOpPoolingForwardQCOM(workspace->context, 0, &pool_desc,
                                                        input->tensor, layer_.unusedTensor,
                                                        output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Pooling Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a SoftMax layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateSoftMaxLayer(CachedLayer* layer,
                                                                    const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto out_dims = get_tensor_dims(nodes_[node.GetInputs()[0].id_]);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype, nullptr,
                                             {out_dims.n, out_dims.c, 1, 1});

    cl_ml_op_softmax_desc_qcom softmax_desc = {CL_SOFTMAX_ALGORITHM_ACCURATE_QCOM,
                                               CL_SOFTMAX_MODE_INSTANCE_QCOM, cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpSoftmaxQCOM(workspace->context, 0, &softmax_desc,
                                                 input->tensor, output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "SoftMax Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a Pad layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreatePadLayer(CachedLayer* layer,
                                                                const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    std::string pad_mode = node.GetAttr<std::vector<std::string>>("pad_mode")[0];
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("pad_width");
    std::vector<cl_uint> clml_padding = GetVectorValues(padding);

    cl_pad_mode_qcom clml_pad_mode = CL_PAD_MODE_CONSTANT_QCOM;
    if (pad_mode == "constant")
      clml_pad_mode = CL_PAD_MODE_CONSTANT_QCOM;
    else if (pad_mode == "edge")
      clml_pad_mode = CL_PAD_MODE_SYMMETRIC_QCOM;
    else if (pad_mode == "reflect")
      clml_pad_mode = CL_PAD_MODE_REFLECT_QCOM;
    else
      LOG(FATAL) << "Padding mode not supported by CLML:" << pad_mode;

    cl_ml_op_pad_desc_qcom pad_desc{
        clml_pad_mode,
        {0, 0},
        {clml_padding[0], clml_padding[1], clml_padding[2], clml_padding[3], 0, 0, 0, 0},
        cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpPadQCOM(workspace->context, 0, &pad_desc, input->tensor,
                                             output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Pad Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a Reshape layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateReshapeLayer(CachedLayer* layer,
                                                                    const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    result = h_ClmlIntf->clCreateMLOpReshapeQCOM(workspace->context, 0, input->tensor,
                                                 output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Reshape Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a concat layer.
   *
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateConcatLayer(CachedLayer* layer,
                                                                   const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    std::vector<JSONGraphNodeEntry> input_ = node.GetInputs();
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    int inputSize = input_.size();
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    cl_ml_tensor_qcom* concatInputs = new cl_ml_tensor_qcom[inputSize];
    for (int i = 0; i < inputSize; i++) {
      auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[i], {},
                                               CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
      concatInputs[i] = input->tensor;
    }
    cl_ml_op_concat_desc_qcom concatDesc = {1, (cl_uint)inputSize, cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpConcatQCOM(workspace->context, 0, &concatDesc, concatInputs,
                                                output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Concat Error:" << result;

    layer->function.push_back(op);

    delete[] concatInputs;
    return output;
  }

  /*!
   * \brief Create a dense layer.
   *
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateDenseLayer(CachedLayer* layer,
                                                                  const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto inp_dims = get_tensor_dims(nodes_[node.GetInputs()[0].id_]);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {1, inp_dims.c, 1, 1},
                                             CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    auto wt_dims = get_tensor_dims(nodes_[node.GetInputs()[1].id_]);
    bool has_bias = node.GetInputs().size() == 3 ? true : false;
    auto weight = MakeCLMLTensorFromJSONEntry(node.GetInputs()[1], {wt_dims.n, wt_dims.c, 1, 1},
                                              CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    auto bias = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
    if (has_bias) {
      auto bias_dims = get_tensor_dims(nodes_[node.GetInputs()[2].id_]);
      bias = MakeCLMLTensorFromJSONEntry(node.GetInputs()[2], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                         cl_dtype);
    } else {
      cl_ml_tensor_desc_qcom desc = {};
      desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
      result =
          h_ClmlIntf->clCreateMLTensorQCOM(workspace->context, NULL, &desc, &layer_.unusedTensor);
      ICHECK(layer_.unusedTensor && result == CL_SUCCESS) << "clCreateMLTensorQCOM:" << result;
      bias->tensor = layer_.unusedTensor;
    }
    // Output
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype, nullptr,
                                             {1, wt_dims.n, 1, 1});
    cl_ml_op_convolution_desc_qcom conv_desc = {CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
                                                1,
                                                4,
                                                {0, 0},
                                                {0, 0},
                                                {1, 1},
                                                {1, 1},
                                                0,
                                                cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpConvolutionForwardQCOM(
        workspace->context, 0, &conv_desc, input->tensor, weight->tensor, bias->tensor,
        output->tensor, &op, NULL);
    ICHECK(op && result == CL_SUCCESS) << "Fully Connected Error:" << result;

    layer->function.push_back(op);
    layer_.func_ins.push_back(input);
    return output;
  }

  /*!
   * \brief Create a Clip(X) layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateClipLayer(CachedLayer* layer,
                                                                 const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    cl_float a_max = std::stof(node.GetAttr<std::vector<std::string>>("a_max")[0]);
    cl_float a_min = std::stof(node.GetAttr<std::vector<std::string>>("a_min")[0]);

    cl_ml_op_clip_desc_qcom clip_desc = {
        CL_CLIP_BY_VALUE_QCOM, {{a_max}, CL_FLOAT}, {{a_min}, CL_FLOAT}, cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpClipQCOM(workspace->context, 0, &clip_desc, input->tensor,
                                              output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Clip Error:" << result;

    layer_.func_ins.push_back(input);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a Binary layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateBinaryLayer(CachedLayer* layer,
                                                                   const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = NULL;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input_a = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {},
                                               CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    auto input_b = MakeCLMLTensorFromJSONEntry(node.GetInputs()[1], {},
                                               CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    std::string op_name = node.GetOpName();
    cl_binary_op_qcom binary_op = CL_TENSOR_OP_ADD_QCOM;
    if (op_name == "subtract")
      binary_op = CL_TENSOR_OP_SUB_QCOM;
    else if (op_name == "multiply")
      binary_op = CL_TENSOR_OP_MUL_QCOM;
    else if (op_name == "minimum")
      binary_op = CL_TENSOR_OP_MIN_QCOM;
    else if (op_name == "maximum")
      binary_op = CL_TENSOR_OP_MAX_QCOM;
    cl_ml_op_binary_desc_qcom add_desc = {
        binary_op, {{1.0}, CL_FLOAT}, {{1.0}, CL_FLOAT}, {{0.0}, CL_FLOAT}, cl_arithmetic_mode};

    result = h_ClmlIntf->clCreateMLOpBinaryQCOM(workspace->context, 0, &add_desc, input_a->tensor,
                                                input_b->tensor, output->tensor, &op, tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << op_name << " Node Error:" << result;

    layer_.func_ins.push_back(input_a);
    layer_.func_ins.push_back(input_b);
    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief The network layers represented by acl functions.
   * \note Currently only supports a single layer.
   */

  CachedLayer layer_;
  // CLML Context
  CLMLInterfaceV2QCOM* h_ClmlIntf = NULL;
  cl::OpenCLWorkspace* workspace = NULL;
  cl::OpenCLThreadEntry* tentry = NULL;
  cl_ml_tuningcache_qcom tuning_cache = NULL;
  bool is_tuning_run;
  char* tuning_file;
#else
  void Run() override {
    LOG(FATAL) << "Cannot call run on CLML module without runtime enabled. "
               << "Please build with USE_CLML_GRAPH_EXECUTOR.";
  }

  void BuildEngine() {
    LOG(WARNING) << "CLML engine is not initialized. "
                 << "Please build with USE_CLML_GRAPH_EXECUTOR.";
  }
#endif
};

runtime::Module CLMLRuntimeCreate(const String& symbol_name, const String& graph_json,
                                  const Array<String>& const_names) {
  auto n = make_object<CLMLRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.clml_runtime_create").set_body_typed(CLMLRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_clml")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<CLMLRuntime>);
}  //  namespace contrib
}  //  namespace runtime
}  //  namespace tvm
