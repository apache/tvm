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
#include "clml_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_CLML
#include "clml_memory_planner.h"
#include "clml_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

#ifdef TVM_GRAPH_EXECUTOR_CLML
CLMLThreadEntry* CLMLWorkspace::GetThreadEntry() { return CLMLThreadEntry::ThreadLocal(); }

CLMLWorkspace* CLMLWorkspace::Global() {
  static CLMLWorkspace* inst = new CLMLWorkspace();
  return inst;
}

CLMLWorkspace::CLMLWorkspace() {
  cl_int result = 0;
  workspace = cl::OpenCLWorkspace::Global();
  workspace->Init();
  tentry = workspace->GetThreadEntry();

  device_id = workspace->GetCLDeviceID(tentry->device.device_id);
  platform_id = workspace->device_to_platform[device_id];

  // Print extensions
  size_t reqd_size = 0;
  result = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &reqd_size);
  ICHECK(reqd_size > 0u && result == CL_SUCCESS) << "clGetDeviceInfo:" << result;
  std::vector<char> extn_buf(reqd_size);
  result = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, reqd_size, extn_buf.data(), nullptr);
  ICHECK(result == CL_SUCCESS) << "clGetDeviceInfo:" << result;
  std::string extensions(extn_buf.data());
  LOG(WARNING) << "OpenCL Extensions:" << extensions;

  if (extensions.find("cl_qcom_ml_ops") == std::string::npos) {
    LOG(FATAL) << "CLML Runtime Init: Qualcomm extn not present.\n";
    return;
  }
  is_recordable_queue = (extensions.find("cl_qcom_recordable_queues") != std::string::npos);
  is_on_chip_memory = (extensions.find("cl_qcom_onchip_global_memory") != std::string::npos);
  LOG(WARNING) << "Recordable Queues Support :" << is_recordable_queue;
  LOG(WARNING) << "On chip Memory Support :" << is_on_chip_memory;

  if (is_on_chip_memory) {
    result = clGetDeviceInfo(device_id, CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM,
                             sizeof(onchip_mem_size), &onchip_mem_size, NULL);
    ICHECK(result == CL_SUCCESS) << "clGetDeviceInfo(CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM):"
                                 << result;
    LOG(WARNING) << "On chip memory size:" << onchip_mem_size;
  }

  // Query and Get CLML Interface
  static const cl_uint MAX_VERSIONS = 256;
  cl_int majorVersions[MAX_VERSIONS];
  cl_int minorVersions[MAX_VERSIONS];
  cl_uint numVersions = 0;
  result = clQueryMLInterfaceVersionsQCOM(nullptr, nullptr, 0, &numVersions);
  ICHECK(result == CL_SUCCESS) << "clQueryMLInterfaceVersionsQCOM:" << result;
  ICHECK(numVersions > 0u);
  ICHECK(numVersions <= MAX_VERSIONS);

  result = clQueryMLInterfaceVersionsQCOM(majorVersions, minorVersions, numVersions, nullptr);
  ICHECK(result == CL_SUCCESS) << "clQueryMLInterfaceVersionsQCOM:" << result;

  for (cl_uint i = 0; i < numVersions; ++i) {
    if (majorVersions[i] == CL_QCOM_ML_OPS_H_MAJOR_VERSION) {
      h_ClmlIntf = GET_ML_INTERFACE(0);
      LOG(WARNING) << "CLML Target version:" << majorVersions[i];
      break;
    }
  }
  ICHECK(h_ClmlIntf != nullptr)
      << "clGetMLInterfaceVxQCOM:" << result
      << " Perhaps there is mispatch between CLML SDK version to target supported version:"
      << majorVersions[numVersions - 1];
  char* tune_flag;
  if ((tune_flag = getenv("CLML_IS_TUNING_RUN")))
    is_tuning_run = std::stoi(tune_flag);
  else
    is_tuning_run = 0;

  if (!(tuning_file = getenv("CLML_TUNING_CACHE"))) this->is_tuning_run = 0;
}

typedef dmlc::ThreadLocalStore<CLMLThreadEntry> CLMLThreadStore;

CLMLThreadEntry* CLMLThreadEntry::ThreadLocal() { return CLMLThreadStore::Get(); }
#endif

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
      : JSONRuntimeBase(symbol_name, graph_json, const_names), clml_symbol(symbol_name) {}

  ~CLMLRuntime() {
#ifdef TVM_GRAPH_EXECUTOR_CLML
    cl_int result = 0;
    if (this->layer_.tuning_cache) {
      result = CLML_INTF->clReleaseMLTuningCacheQCOM(this->layer_.tuning_cache);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTuningCacheQCOM:" << result;
    }
    for (auto it = this->layer_.storage_map.begin(); it != this->layer_.storage_map.end(); it++) {
      auto tensor_desc = it->second.first;
      result = CLML_INTF->clReleaseMLTensorQCOM(tensor_desc->tensor);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorQCOM:" << result;
      if (this->layer_.ddr_storage_ref_map.find(tensor_desc->memory) !=
          this->layer_.ddr_storage_ref_map.end()) {
        ReleaseDDRMemory(tensor_desc->memory);
      } else {
        result = clReleaseMemObject(tensor_desc->memory);
        ICHECK(result == CL_SUCCESS) << "clReleaseMemObject:" << result;
      }
    }
    for (size_t i = 0; i < this->layer_.function.size(); ++i) {
      result = CLML_INTF->clReleaseMLOpQCOM(this->layer_.function[i]);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLOpQCOM:" << result;
    }
    for (auto it = this->layer_.in_placeholder.begin(); it != this->layer_.in_placeholder.end();
         it++) {
      result = CLML_INTF->clReleaseMLTensorQCOM(it->second->tensor);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorQCOM:" << result;
    }
    for (auto it = this->layer_.out_placeholder.begin(); it != this->layer_.out_placeholder.end();
         it++) {
      result = CLML_INTF->clReleaseMLTensorQCOM((*it)->tensor);
      ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorQCOM:" << result;
    }
    result = CLML_INTF->clReleaseMLTensorMemoryDescriptorSetQCOM(layer_.descriptorSet);
    ICHECK(result == CL_SUCCESS) << "clReleaseMLTensorMemoryDescriptorSetQCOM:" << result;

    if (this->layer_.recordable_queue) {
      clReleaseCommandQueue(this->layer_.recordable_queue);
    }
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
  void InitCLML() {
    // Setup CLML Context
    cl_int result = 0;
    cws = CLMLWorkspace::Global();

    if (cws->is_recordable_queue) {
      this->layer_.recordable_queue =
          clCreateCommandQueue(CLML_CTX, cws->device_id, CL_QUEUE_RECORDABLE_QCOM, &result);
      ICHECK(result == CL_SUCCESS) << "clCreateCommandQueue - Recordable:" << result;

      this->layer_.recording = clNewRecordingQCOM(this->layer_.recordable_queue, &result);
      ICHECK(result == CL_SUCCESS) << "clNewRecordingQCOM:" << result;
    }

    // A Tuning run, so create the cache from scratch
    result = CLML_INTF->clCreateMLTuningCacheQCOM(&layer_.tuning_cache);
    ICHECK(result == CL_SUCCESS) << "clCreateMLTuningCacheQCOM:" << result;
    if (!cws->is_tuning_run && cws->tuning_file) {
      std::vector<unsigned char> tune_buffer;
      std::string tune_blob;
      LoadBinaryFromFile(cws->tuning_file, &tune_blob);
      dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&tune_blob));
      dmlc::Stream* strm = &mstrm;

      uint64_t header, reserve;
      std::string tune_symbol;
      while (strm->Read(&header)) {
        if (header != kTVMCLMLTuningCacheMagic) break;
        if (!strm->Read(&reserve)) break;
        if (!strm->Read(&tune_symbol)) break;
        if (tune_symbol == clml_symbol) {
          strm->Read(&tune_buffer);
          break;
        } else {
          std::vector<unsigned char> tmp_buf;
          if (!strm->Read(&tmp_buf)) break;
        }
      }

      if (tune_buffer.size()) {
        LOG(INFO) << "Loading tuning cache for symbol:" << clml_symbol
                  << " size:" << tune_buffer.size();
        result = CLML_INTF->clLoadMLTuningCacheQCOM(layer_.tuning_cache, tune_buffer.size(),
                                                    tune_buffer.data());
        ICHECK(result == CL_SUCCESS) << "clLoadMLTuningCacheQCOM:" << result;
      } else {
        LOG(WARNING) << "Tuning cache not cound for symbol :" << clml_symbol << " in file "
                     << cws->tuning_file;
      }
    }
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
    cl_command_queue queue = CLML_QUEUE;
    std::vector<cl_event>& evts = cws->workspace->GetEventQueue(cws->tentry->device);
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
          CopyDataToCLMLTensor(layer_.inputs[nid], data);
        } else if (kDLOpenCL == data_entry_[eid]->device.device_type) {
          layer_.in_placeholder[nid]->memory = static_cast<cl_mem>(
              ((cl::BufferDescriptor*)const_cast<DLTensor*>(data_entry_[eid])->data)->buffer);
          cl_event cpy_evt = nullptr;
          cl_event* evt = &cpy_evt;
          if (cws->workspace->IsProfiling(cws->tentry->device)) {
            evts.resize(evts.size() + 1);
            evt = &(evts.back());
          }
          result = CLML_INTF->clEnqueueCopyMLTensorDataQCOM(
              queue, layer_.in_placeholder[nid]->tensor, layer_.in_placeholder[nid]->memory,
              layer_.inputs[nid]->tensor, layer_.inputs[nid]->memory, 0, NULL, evt);
          ICHECK(result == CL_SUCCESS) << "clEnqueueCopyMLTensorDataQCOM:" << result;
        } else {
          DLDataType tvm_dtype = const_cast<DLTensor*>(data_entry_[eid])->dtype;
          cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
          int dtype_size = cl_dtype == CL_FLOAT ? 4 : 2;
          void* tmpptr = reinterpret_cast<void*>(malloc(isize * dtype_size));
          TVMArrayCopyToBytes(const_cast<DLTensor*>(data_entry_[eid]), const_cast<void*>(tmpptr),
                              isize * dtype_size);
          CopyDataToCLMLTensor(layer_.inputs[nid], tmpptr);
          free(tmpptr);
        }
      }
    }

    int64_t duration = 0;
    if (cws->is_recordable_queue) {
      if (getenv("CLML_PROFILING")) {
        Timer t;
        auto f = Registry::Get(std::string("profiling.timer.opencl"));
        t = f->operator()(cws->tentry->device);
        t->Start();
        queue = CLML_QUEUE;
        evts.resize(evts.size() + 1);
        cl_event* evt = &(evts.back());
        result = CLML_INTF->clEnqueueRecordingMLOpQCOM(queue, this->layer_.recording, 0, nullptr, 0,
                                                       nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
                                                       0, nullptr, 0, nullptr, 0, nullptr, evt);
        ICHECK(result == CL_SUCCESS) << "clEnqueueRecordingMLOpQCOM:" << result;
        t->Stop();
        duration += t->SyncAndGetElapsedNanos();
      } else {
        result = CLML_INTF->clEnqueueRecordingMLOpQCOM(queue, this->layer_.recording, 0, nullptr, 0,
                                                       nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
                                                       0, nullptr, 0, nullptr, 0, nullptr, nullptr);
        ICHECK(result == CL_SUCCESS) << "clEnqueueRecordingMLOpQCOM:" << result;
      }
    } else {
      for (size_t i = 0; i < this->layer_.function.size(); ++i) {
        // Make CLML subgraphs accounted by OpenCLTimerNode.
        if (getenv("CLML_PROFILING")) {
          Timer t;
          auto f = Registry::Get(std::string("profiling.timer.opencl"));
          t = f->operator()(cws->tentry->device);
          t->Start();
          queue = CLML_QUEUE;
          evts.resize(evts.size() + 1);
          cl_event* evt = &(evts.back());
          result = CLML_INTF->clEnqueueMLOpQCOM(queue, this->layer_.function[i],
                                                this->layer_.descriptorSet, 0, nullptr, evt);
          t->Stop();
          duration += t->SyncAndGetElapsedNanos();
          LOG(WARNING) << "Layer:" << this->layer_.layer_names[i]
                       << " Duration:" << t->SyncAndGetElapsedNanos();
        } else {
          result = CLML_INTF->clEnqueueMLOpQCOM(queue, this->layer_.function[i],
                                                this->layer_.descriptorSet, 0, nullptr, nullptr);
        }
        ICHECK(result == CL_SUCCESS) << "clEnqueueMLOpQCOM:" << result;
      }
    }
    if (getenv("CLML_PROFILING")) {
      LOG(WARNING) << "Total Duration for " << clml_symbol << " is:" << duration;
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
        cl_event cpy_evt = nullptr;
        cl_event* evt = &cpy_evt;
        if (cws->workspace->IsProfiling(cws->tentry->device)) {
          evts.resize(evts.size() + 1);
          evt = &(evts.back());
        }
        result = CLML_INTF->clEnqueueCopyMLTensorDataQCOM(
            queue, layer_.outputs[i]->tensor, layer_.outputs[i]->memory,
            layer_.out_placeholder[i]->tensor, layer_.out_placeholder[i]->memory, 0, nullptr, evt);
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
   * \brief check if the nid is graph output tensor or not.
   *
   */
  bool IsOutputTensor(int nid) {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (nid == outputs_[i].id_) return true;
    }
    return false;
  }

  /*!
   * \brief Initialize memory pool.
   *
   */
  void InitMemoryPool(void) {
    layer_.on_chip_pool_size.clear();
    layer_.on_chip_pool_size.insert({0, cws->onchip_mem_size});
    layer_.on_chip_pool_alloc_info.clear();
    layer_.alloc_ping_pong = true;
    layer_.in_chip_total_free = cws->onchip_mem_size;
    layer_.in_chip_total_alloc = 0;
    layer_.on_chip_alert_fail = 0;
  }

  /*!
   * \brief Plan Memory for activations to allocate on on-chip global memory where ever possible.
   *
   */
  void PlanMemory() {
    InitMemoryPool();
    // Build the ref count table for all activation tensors.
    LOG_MEM << "Build Ref Map";
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
        for (auto& input_node : inputs) {
          if (nodes_[input_node.id_].GetOpType() != "const") {
            if (layer_.storage_ref_map.find(input_node.id_) == layer_.storage_ref_map.end()) {
              layer_.storage_ref_map.insert({input_node.id_, 1});
              layer_.life_span.insert({input_node.id_, nid});
            } else {
              layer_.storage_ref_map[input_node.id_]++;
              layer_.life_span[input_node.id_] = nid;
            }
          }
        }
      }
    }
    LOG_MEM << "Print Ref Map";

    for (auto it = layer_.storage_ref_map.begin(); it != layer_.storage_ref_map.end(); it++) {
      LOG_MEM << "RefMap:" << it->first << " Count:" << it->second
              << "Life Span:" << layer_.life_span[it->first];
    }

    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      uint32_t size = 0;
      cl_int result = CL_OUT_OF_HOST_MEMORY;
      result = CLML_INTF->clGetMLTensorMemorySizeQCOM(CLML_CTX,
                                                      layer_.storage_map[nid].first->tensor, &size);
      ICHECK(result == CL_SUCCESS) << "clGetMLTensorMemorySizeQCOM:" << result;

      if ((node.GetOpType() == "kernel") || (node.GetOpType() == "input")) {
        std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
        LOG_MEM << "Request :" << size << " Nid:" << nid;
        size_t offset = -1;
        // On-chip memory only for intermediate tensors with in recording scope.
        if ((cws->is_on_chip_memory) && (!IsOutputTensor(nid)) && (node.GetOpType() != "input")) {
          offset = RequestOnChipMemory(&this->layer_, size);
        }
        if (-1 != offset) {
          LOG_MEM << "Got On-Chip Mem:" << offset << "Nid:" << nid;
          layer_.on_chip_pool_alloc_info.insert({offset, nid});
          layer_.on_chip_alloc_plan.insert({nid, std::make_pair(size, offset)});
        } else {
          layer_.on_chip_reject.insert({nid, size});
          // DDR Allocation
          auto ddr_mem = RequestDDRMemory(&this->layer_, size);
          LOG_MEM << "Alloc DDR from global pool for nid:" << nid << " Type:" << node.GetOpType();
          layer_.ddr_alloc_plan.insert({nid, ddr_mem});
        }

        // Now free up the input tensors on-chip memory for reuse.
        for (auto& input_node : inputs) {
          if (nodes_[input_node.id_].GetOpType() != "const") {
            LOG_MEM << "Free Input Mem:" << input_node.id_;
            FreeMemory(&this->layer_, input_node.id_);
          }
        }
      }
    }

    // Stats dump
    size_t in_chip_total_alloc = 0;
    size_t total_reject = 0;
    for (auto it = layer_.on_chip_alloc_plan.begin(); it != layer_.on_chip_alloc_plan.end(); it++) {
      LOG_STATS << " On-chip Alloc:" << it->first << " Size:" << it->second.first
                << " Offset:" << it->second.second;
      in_chip_total_alloc += it->second.first;
    }

    for (auto it = layer_.on_chip_reject.begin(); it != layer_.on_chip_reject.end(); it++) {
      LOG_STATS << "Reject:" << it->first << " Size:" << it->second;
      total_reject += it->second;
    }
    LOG_STATS << "Total On-chip Alloc:" << in_chip_total_alloc + total_reject
              << " On-Chip:" << in_chip_total_alloc << " Reject:" << total_reject
              << " Alert Fail:" << layer_.on_chip_alert_fail;

    auto cws = CLMLWorkspace::Global();
    for (auto it = cws->ddr_global_pool.begin(); it != cws->ddr_global_pool.end(); it++) {
      LOG_STATS << "DDR Global pool - size:" << it->second.first << " Ref:" << it->second.second;
    }
    for (auto it = this->layer_.ddr_storage_ref_map.begin();
         it != this->layer_.ddr_storage_ref_map.end(); it++) {
      LOG_STATS << "DDR Local pool - size:" << it->second.first << " Ref cnt:" << it->second.second;
    }
  }

  /*!
   * \brief Create an CLML tensor from JSON node entry. Lookup storage map before creation.
   *
   * \param tensor The tensor as Node Entry .
   * \param shape shape information of tensor
   * \param layout the tensor layout to be used
   * \param dtype tensor data type
   * \return CLML Tensor descriptor.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensorFromJSONEntry(
      const JSONGraphNodeEntry& tensor, std::vector<size_t> shape, cl_ml_tensor_layout_qcom layout,
      cl_uint dtype) {
    JSONGraphNode node = nodes_[tensor.id_];

    if (this->layer_.storage_map.find(tensor.id_) == this->layer_.storage_map.end()) {
      void* node_data = nullptr;
      if (node.GetOpType() == "const") {
        node_data = data_entry_[EntryID(tensor)]->data;
      }
      auto clml_tensor = MakeCLMLTensorFromJSONNode(node, layout, dtype, node_data, shape);
      this->layer_.storage_map.insert({tensor.id_, std::make_pair(clml_tensor, node)});

      if ("input" == node.GetOpType()) {
        this->layer_.inputs.insert({tensor.id_, clml_tensor});
        // Input copy placeholder Tensor
        this->layer_.in_placeholder.insert(
            {tensor.id_, MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_NCHW_QCOM, dtype,
                                                    node_data, shape)});
      }

      return clml_tensor;
    } else {
      return this->layer_.storage_map[tensor.id_].first;
    }
  }

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
        // Layers may request for different layout. Differ the input allocation.
      } else if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          auto out = CreateConvolution2DLayer(&layer_, node, CL_CONVOLUTION_MODE_CONVOLUTION_QCOM);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.depthwise_conv2d" == op_name) {
          auto out = CreateConvolution2DLayer(&layer_, node, CL_CONVOLUTION_MODE_DEPTHWISE_QCOM);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.conv2d_transpose" == op_name) {
          auto out = CreateConvolution2DLayer(&layer_, node, CL_CONVOLUTION_MODE_TRANSPOSE_QCOM);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.relu6" == op_name) {
          auto out = CreateReLULayer(&layer_, node, CL_ACTIVATION_RELU6);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.relu" == op_name) {
          auto out = CreateReLULayer(&layer_, node, CL_ACTIVATION_RELU);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.batch_norm" == op_name) {
          auto out = CreateBatchNormLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.max_pool2d" == op_name || "nn.avg_pool2d" == op_name ||
                   "nn.l2_pool2d" == op_name) {
          auto out = CreatePoolingLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.global_max_pool2d" == op_name || "nn.global_avg_pool2d" == op_name) {
          auto out = CreateGlobalPoolingLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("reshape" == op_name) {
          auto out = CreateReshapeLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("concatenate" == op_name) {
          auto out = CreateConcatLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.dense" == op_name) {
          auto out = CreateDenseLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.softmax" == op_name) {
          auto out = CreateSoftMaxLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.pad" == op_name) {
          auto out = CreatePadLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.batch_flatten" == op_name) {
          auto out = CreateBatchFlattenLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("clip" == op_name) {
          auto out = CreateClipLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("add" == op_name || "subtract" == op_name || "multiply" == op_name ||
                   "minimum" == op_name || "maximum" == op_name || "divide" == op_name) {
          auto out = CreateBinaryLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.depth_to_space" == op_name) {
          auto out = CreateDepthToSpaceLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.upsampling" == op_name) {
          auto out = CreateResizeLayer(&layer_, node);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
        } else if ("nn.batch_matmul" == op_name) {
          auto out = CreateBatchMatmulLayer(&layer_, node, nid);
          this->layer_.storage_map.insert({nid, std::make_pair(out, node)});
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
      if (this->layer_.out_shapes.find(nid) != this->layer_.out_shapes.end()) {
        // Handle customized shapes here
        this->layer_.out_placeholder.push_back(
            MakeCLMLTensorFromJSONNode(nodes_[nid], CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype, nullptr,
                                       this->layer_.out_shapes[nid]));
      } else {
        this->layer_.out_placeholder.push_back(
            MakeCLMLTensorFromJSONNode(nodes_[nid], CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype));
      }
    }

    // Plan memory utilization
    PlanMemory();

    // ALlocate device memories and initialize the params if any
    cl_int result = 0;
    size_t alloc_on_chip = 0;
    size_t alloc_ddr = 0;
    size_t alloc_ddr_reuse = 0;
    for (auto it = this->layer_.storage_map.begin(); it != this->layer_.storage_map.end(); it++) {
      auto tensor_desc = it->second.first;
      uint32_t mem_size = 0;
      result = CL_OUT_OF_HOST_MEMORY;
      result = CLML_INTF->clGetMLTensorMemorySizeQCOM(CLML_CTX, tensor_desc->tensor, &mem_size);
      ICHECK(result == CL_SUCCESS) << "clGetMLTensorMemorySizeQCOM:" << result;

      JSONGraphNode node = it->second.second;
      void* node_data = nullptr;
      size_t on_chip_mem_offset = -1;
      if (layer_.on_chip_alloc_plan.find(it->first) != layer_.on_chip_alloc_plan.end()) {
        LOG_MEM << "Found GMEM Alloc:" << it->first
                << " Size:" << layer_.on_chip_alloc_plan[it->first].first
                << " Offset:" << layer_.on_chip_alloc_plan[it->first].second;
        on_chip_mem_offset = layer_.on_chip_alloc_plan[it->first].second;
        alloc_on_chip += mem_size;
        tensor_desc->memory = AllocateOnChipTensorMemory(mem_size, on_chip_mem_offset);
      } else if (layer_.ddr_alloc_plan.find(it->first) != layer_.ddr_alloc_plan.end()) {
        LOG_MEM << "DDR Alloc for nid:" << it->first << " Type:" << node.GetOpType();
        tensor_desc->memory = layer_.ddr_alloc_plan[it->first];
        alloc_ddr_reuse += mem_size;
        //} else if ((node.GetOpType() == "input") || IsOutputTensor(it->first) || (node.GetOpType()
        //== "const")) {
      } else if (node.GetOpType() == "const") {
        LOG_MEM << "DDR Alloc for Const/Input/Output";
        tensor_desc->memory = AllocateDDRTensorMemory(mem_size);
        alloc_ddr += mem_size;
      } else {
        LOG(FATAL) << "Mem allocation not found on DDR as well as On-Chip nid: " << it->first
                   << " Type:" << node.GetOpType();
      }

      if (node.GetOpType() == "const") {
        node_data = data_entry_[EntryID(it->first, 0)]->data;
        if (node_data != nullptr) {
          CopyDataToCLMLTensor(tensor_desc, node_data);
        }
      }
      this->layer_.tensorMemDescs.push_back(*tensor_desc);
    }
    LOG_STATS << "Total On-Chip Allocation  :" << alloc_on_chip;
    LOG_STATS << "Total DDR Reuse Allocation:" << alloc_ddr_reuse;
    LOG_STATS << "Total DDR fixed allocation:" << alloc_ddr;
    size_t ddr_global_pool = 0;
    size_t ddr_local_pool = 0;
    auto cws = CLMLWorkspace::Global();
    for (auto it = cws->ddr_global_pool.begin(); it != cws->ddr_global_pool.end(); it++) {
      LOG_STATS << "DDR Global pool - size:" << it->second.first << " Ref:" << it->second.second;
      ddr_global_pool += it->second.first;
    }
    LOG_STATS << "Total Global Pool:" << ddr_global_pool;
    for (auto it = this->layer_.ddr_storage_ref_map.begin();
         it != this->layer_.ddr_storage_ref_map.end(); it++) {
      LOG_STATS << "DDR Local pool - size:" << it->second.first << " Ref cnt:" << it->second.second;
      ddr_local_pool += it->second.first;
    }
    LOG_STATS << "Total Local Pool:" << ddr_local_pool;

    // Setup descriptor set
    result = CLML_INTF->clCreateMLTensorMemoryDescriptorSetQCOM(&this->layer_.descriptorSet);
    ICHECK(result == CL_SUCCESS) << "clCreateMLTensorMemoryDescriptorSetQCOM:" << result;

    result = CLML_INTF->clUpdateMLTensorMemoryDescriptorSetQCOM(
        this->layer_.descriptorSet, static_cast<uint32_t>(this->layer_.tensorMemDescs.size()),
        this->layer_.tensorMemDescs.data());
    ICHECK(result == CL_SUCCESS) << "clUpdateMLTensorMemoryDescriptorSetQCOM:" << result;

    if (cws->is_tuning_run) {
      LOG(WARNING) << "CLML Tunning In Progress:";
      // Let the command queue recreated in profiling mode.
      cl::OpenCLWorkspace::Global()->EnableQueueProfiling(cws->tentry->device, true);
      for (size_t i = 0; i < this->layer_.function.size(); ++i) {
        LOG(WARNING) << "CLML Tunning:" << this->layer_.layer_names[i];
        result = CLML_INTF->clTuneMLOpQCOM(CLML_QUEUE, this->layer_.function[i],
                                           this->layer_.descriptorSet, this->layer_.tuning_cache,
                                           nullptr);
        ICHECK(result == CL_SUCCESS) << "clTuneMLOpQCOM:" << result;
      }
      cl::OpenCLWorkspace::Global()->EnableQueueProfiling(cws->tentry->device, false);

      size_t cache_len_bytes = 0;
      size_t len_ret = 0;
      result =
          CLML_INTF->clSaveMLTuningCacheQCOM(layer_.tuning_cache, 0, nullptr, &cache_len_bytes);
      ICHECK(result == CL_SUCCESS) << "clSaveMLTuningCacheQCOM:" << result;

      std::vector<unsigned char> saved_cache(cache_len_bytes, 0);
      result = CLML_INTF->clSaveMLTuningCacheQCOM(layer_.tuning_cache, saved_cache.size(),
                                                  saved_cache.data(), &len_ret);
      ICHECK(result == CL_SUCCESS) << "clSaveMLTuningCacheQCOM" << result;

      std::string tune_str;
      dmlc::MemoryStringStream mstrm(&tune_str);
      dmlc::Stream* strm = &mstrm;
      uint64_t header = kTVMCLMLTuningCacheMagic;
      uint64_t reserved = 0x0;
      strm->Write(header);
      strm->Write(reserved);
      strm->Write(clml_symbol);
      strm->Write(saved_cache);

      std::ofstream fs(cws->tuning_file, std::ios::app | std::ios::binary);
      ICHECK(!fs.fail()) << "Cannot open " << cws->tuning_file;
      fs.write(&tune_str[0], tune_str.length());
      LOG(WARNING) << "CLML: Tuning cache dumped to:" << cws->tuning_file << " size"
                   << tune_str.length() << " with tuning blob len " << saved_cache.size();
    }
    if (cws->is_recordable_queue) {
      for (size_t i = 0; i < this->layer_.function.size(); ++i) {
        result =
            CLML_INTF->clEnqueueMLOpQCOM(this->layer_.recordable_queue, this->layer_.function[i],
                                         this->layer_.descriptorSet, 0, nullptr, nullptr);
        ICHECK(result == CL_SUCCESS) << "clEnqueueMLOpQCOM - Recordable Queue:" << result;
      }

      result = clEndRecordingQCOM(this->layer_.recording);
      ICHECK(result == CL_SUCCESS) << "clEndRecordingQCOM:" << result;
    }
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
      result = CLML_INTF->clCreateMLTensorQCOM(CLML_CTX, nullptr, &desc, &layer_.unusedTensor);
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

    cl_ml_op_qcom op = nullptr;
    if (!has_bn) {
      if (!has_act) {
        result = CLML_INTF->clCreateMLOpConvolutionForwardQCOM(
            CLML_CTX, nullptr, &conv_desc, input->tensor, weight->tensor, bias->tensor,
            output->tensor, &op, nullptr);
        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      } else {
        result = CLML_INTF->clCreateMLOpFusedConvolutionActivationForwardQCOM(
            CLML_CTX, nullptr, &conv_desc, &act_desc, input->tensor, weight->tensor, bias->tensor,
            nullptr, output->tensor, &op, layer_.tuning_cache);
        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      }
      layer->function.push_back(op);
    } else {
      int bn_index = has_bias ? 3 : 2;
      int axis = std::stoi(node.GetAttr<std::vector<std::string>>("batchnorm")[0]);
      auto bn_dims = GetTensorDims(nodes_[inputs[bn_index].id_]);
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
        result = CLML_INTF->clCreateMLOpFusedConvolutionBatchNormForwardQCOM(
            CLML_CTX, nullptr, &conv_desc, &bn_desc, input->tensor, weight->tensor, bias->tensor,
            output->tensor, bn_mean->tensor, bn_var->tensor, bn_scale->tensor, bn_bias->tensor, &op,
            layer_.tuning_cache);
        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      } else {
        result = CLML_INTF->clCreateMLOpFusedConvolutionBatchNormActivationForwardQCOM(
            CLML_CTX, nullptr, &conv_desc, &bn_desc, &act_desc, input->tensor, weight->tensor,
            bias->tensor, output->tensor, nullptr, bn_mean->tensor, bn_var->tensor,
            bn_scale->tensor, bn_bias->tensor, &op, layer_.tuning_cache);

        ICHECK(op && result == CL_SUCCESS) << "Convolution Error:" << result;
      }
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
    cl_ml_op_qcom op = nullptr;
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
    result = CLML_INTF->clCreateMLTensorQCOM(CLML_CTX, nullptr, &desc, &layer_.unusedTensor);
    ICHECK(layer_.unusedTensor && result == CL_SUCCESS) << ":" << result;

    result = CLML_INTF->clCreateMLOpActivationForwardQCOM(CLML_CTX, nullptr, &act_desc,
                                                          input->tensor, layer_.unusedTensor,
                                                          output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Activation Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    std::vector<cl_ml_op_properties_qcom> opProperties;
    opProperties.push_back(CL_ML_BATCH_NORM_OP_EPSILON_QCOM);
    opProperties.push_back(*reinterpret_cast<cl_ml_op_properties_qcom*>(&epsilon));
    opProperties.push_back(CL_ML_OP_PROPERTY_LIST_END_QCOM);

    auto bn_dims = GetTensorDims(nodes_[node.GetInputs()[1].id_]);
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

    result = CLML_INTF->clCreateMLOpBatchNormForwardQCOM(
        CLML_CTX, opProperties.data(), &bn_desc, input->tensor, bn_mean->tensor, bn_var->tensor,
        bn_scale->tensor, bn_bias->tensor, output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Batchnorm Error:" << result;

    layer->function.push_back(op);
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
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

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
    cl_ml_tensor_qcom unusedTensor = nullptr;
    desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
    result = CLML_INTF->clCreateMLTensorQCOM(CLML_CTX, nullptr, &desc, &unusedTensor);
    ICHECK(unusedTensor && result == CL_SUCCESS) << ":" << result;

    result = CLML_INTF->clCreateMLOpPoolingForwardQCOM(CLML_CTX, nullptr, &pool_desc, input->tensor,
                                                       unusedTensor, output->tensor, &op,
                                                       layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Pooling Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    auto in_dims = GetTensorDims(nodes_[node.GetInputs()[0].id_]);
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
    result = CLML_INTF->clCreateMLTensorQCOM(CLML_CTX, nullptr, &desc, &layer_.unusedTensor);
    ICHECK(layer_.unusedTensor && result == CL_SUCCESS) << ":" << result;

    result = CLML_INTF->clCreateMLOpPoolingForwardQCOM(CLML_CTX, nullptr, &pool_desc, input->tensor,
                                                       layer_.unusedTensor, output->tensor, &op,
                                                       layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Pooling Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto out_dims = GetTensorDims(nodes_[node.GetInputs()[0].id_]);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype, nullptr,
                                             {out_dims.n, out_dims.c, 1, 1});

    cl_ml_op_softmax_desc_qcom softmax_desc = {CL_SOFTMAX_ALGORITHM_ACCURATE_QCOM,
                                               CL_SOFTMAX_MODE_INSTANCE_QCOM, cl_arithmetic_mode};

    result = CLML_INTF->clCreateMLOpSoftmaxQCOM(CLML_CTX, nullptr, &softmax_desc, input->tensor,
                                                output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "SoftMax Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
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

    result = CLML_INTF->clCreateMLOpPadQCOM(CLML_CTX, nullptr, &pad_desc, input->tensor,
                                            output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Pad Error:" << result;

    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a Batch Flatten layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateBatchFlattenLayer(
      CachedLayer* layer, const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    result = CLML_INTF->clCreateMLOpReshapeQCOM(CLML_CTX, nullptr, input->tensor, output->tensor,
                                                &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Reshape Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);

    result = CLML_INTF->clCreateMLOpReshapeQCOM(CLML_CTX, nullptr, input->tensor, output->tensor,
                                                &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Reshape Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
    std::vector<JSONGraphNodeEntry> input_ = node.GetInputs();
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    int inputSize = input_.size();
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    cl_uint axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    cl_ml_tensor_qcom* concatInputs = new cl_ml_tensor_qcom[inputSize];
    for (int i = 0; i < inputSize; i++) {
      auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[i], {},
                                               CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
      concatInputs[i] = input->tensor;
    }
    cl_ml_op_concat_desc_qcom concatDesc = {axis, (cl_uint)inputSize, cl_arithmetic_mode};

    result = CLML_INTF->clCreateMLOpConcatQCOM(CLML_CTX, nullptr, &concatDesc, concatInputs,
                                               output->tensor, &op, layer_.tuning_cache);
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
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto in_dims = GetTensorDims(nodes_[node.GetInputs()[0].id_]);
    auto input =
        MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype);
    auto wt_dims = GetTensorDims(nodes_[node.GetInputs()[1].id_]);
    auto weight = MakeCLMLTensorFromJSONEntry(node.GetInputs()[1], {1, 1, wt_dims.n, wt_dims.c},
                                              CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype);
    cl_gemm_transform_qcom b_transform = CL_GEMM_TRANSFORM_NONE_QCOM;
    if (in_dims.c == wt_dims.c) {
      b_transform = CL_GEMM_TRANSFORM_TRANSPOSE_QCOM;
    }
    cl_ml_op_gemm_desc_qcom gemmDesc = {in_dims.n,                    // m
                                        wt_dims.n,                    // n
                                        wt_dims.c,                    // k
                                        CL_GEMM_TRANSFORM_NONE_QCOM,  // A transform
                                        b_transform,                  // B transform
                                        {{1.0}, CL_FLOAT},            // alpha
                                        {{0.0}, CL_FLOAT},            // beta
                                        cl_arithmetic_mode};

    result = CLML_INTF->clCreateMLOpGemmQCOM(CLML_CTX, 0, &gemmDesc, input->tensor, weight->tensor,
                                             output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Dense Error:" << result;

    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a batch_matmul layer.
   *
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML function.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateBatchMatmulLayer(CachedLayer* layer,
                                                                        const JSONGraphNode& node,
                                                                        int nid) {
    cl_int result = 0;
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto in_dims = GetTensorDims(nodes_[node.GetInputs()[0].id_]);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {in_dims.c, in_dims.h},
                                             CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype);
    auto wt_dims = GetTensorDims(nodes_[node.GetInputs()[1].id_]);
    auto weight = MakeCLMLTensorFromJSONEntry(node.GetInputs()[1], {1, 1, wt_dims.c, wt_dims.h},
                                              CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype);

    std::vector<int64_t> out_shape = node.GetOpShape()[0];
    std::vector<size_t> clml_out_shape;
    clml_out_shape.push_back(out_shape[1]);
    clml_out_shape.push_back(out_shape[2]);
    clml_out_shape.push_back(1);
    clml_out_shape.push_back(1);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_NCHW_QCOM, cl_dtype, nullptr,
                                             clml_out_shape);
    layer->out_shapes.insert({nid, clml_out_shape});

    cl_bool b_transpose = std::stoi(node.GetAttr<std::vector<std::string>>("transpose_b")[0]);
    cl_gemm_transform_qcom b_transform = CL_GEMM_TRANSFORM_NONE_QCOM;
    if (b_transpose) {
      b_transform = CL_GEMM_TRANSFORM_TRANSPOSE_QCOM;
    }
    cl_ml_op_gemm_desc_qcom gemmDesc = {in_dims.c,                    // m
                                        wt_dims.c,                    // n
                                        wt_dims.h,                    // k
                                        CL_GEMM_TRANSFORM_NONE_QCOM,  // A transform
                                        b_transform,                  // B transform
                                        {{1.0}, CL_FLOAT},            // alpha
                                        {{0.0}, CL_FLOAT},            // beta
                                        cl_arithmetic_mode};

    result = CLML_INTF->clCreateMLOpGemmQCOM(CLML_CTX, 0, &gemmDesc, input->tensor, weight->tensor,
                                             output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "BatchMatmul Error:" << result;

    layer->function.push_back(op);
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
    cl_ml_op_qcom op = nullptr;
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

    result = CLML_INTF->clCreateMLOpClipQCOM(CLML_CTX, nullptr, &clip_desc, input->tensor,
                                             output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Clip Error:" << result;

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
    cl_ml_op_qcom op = nullptr;
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
    else if (op_name == "divide")
      binary_op = CL_TENSOR_OP_DIV_QCOM;
    else if (op_name == "minimum")
      binary_op = CL_TENSOR_OP_MIN_QCOM;
    else if (op_name == "maximum")
      binary_op = CL_TENSOR_OP_MAX_QCOM;
    cl_ml_op_binary_desc_qcom add_desc = {
        binary_op, {{1.0}, CL_FLOAT}, {{1.0}, CL_FLOAT}, {{0.0}, CL_FLOAT}, cl_arithmetic_mode};

    result = CLML_INTF->clCreateMLOpBinaryQCOM(CLML_CTX, nullptr, &add_desc, input_a->tensor,
                                               input_b->tensor, output->tensor, &op,
                                               layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << op_name << " Node Error:" << result;

    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a DepthToSpace(X) layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateDepthToSpaceLayer(
      CachedLayer* layer, const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    cl_uint block_size = std::stoi(node.GetAttr<std::vector<std::string>>("block_size")[0]);

    cl_ml_op_depthtospace_desc_qcom dtos_desc = {block_size, cl_arithmetic_mode};
    result = CLML_INTF->clCreateMLOpDepthToSpaceQCOM(CLML_CTX, nullptr, &dtos_desc, input->tensor,
                                                     output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "DepthToSpace Layer Error:" << result;

    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief Create a Resize(X) layer.
   *
   * \param layer The CLML layer to build. Containing inputs, outputs and the CLML output.
   * \param node The JSON representation of the operator.
   */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CreateResizeLayer(CachedLayer* layer,
                                                                   const JSONGraphNode& node) {
    cl_int result = 0;
    cl_ml_op_qcom op = nullptr;
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    cl_channel_type cl_dtype = MakeCLDataType(tvm_dtype);
    cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(cl_dtype);
    auto input = MakeCLMLTensorFromJSONEntry(node.GetInputs()[0], {}, CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
                                             cl_dtype);
    auto output = MakeCLMLTensorFromJSONNode(node, CL_TENSOR_LAYOUT_OPTIMAL_QCOM, cl_dtype);
    cl_bool align_corners = std::stoi(node.GetAttr<std::vector<std::string>>("align_corners")[0]);

    cl_ml_op_resize_bilinear_desc_qcom resize_desc = {align_corners, false, cl_arithmetic_mode};
    result = CLML_INTF->clCreateMLOpResizeBilinearQCOM(
        CLML_CTX, nullptr, &resize_desc, input->tensor, output->tensor, &op, layer_.tuning_cache);
    ICHECK(op && result == CL_SUCCESS) << "Resize Layer Error:" << result;

    layer->function.push_back(op);
    return output;
  }

  /*!
   * \brief The network layers represented by acl functions.
   * \note Currently only supports a single layer.
   */

  // This layer instance
  CachedLayer layer_;

  // CLML Workspace
  CLMLWorkspace* cws;

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
  /*! CLML sub graph symbol in TVM main module */
  std::string clml_symbol;
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
