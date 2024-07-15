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
 * \file src/runtime/contrib/adreno_accl/adreno_accl_runtime.cc
 * \brief A simple JSON runtime for ADRENO_ACCL.
 */

#ifdef TVM_GRAPH_EXECUTOR_ADRENO_ACCL
#include <adrenoaccl.h>
#endif
#include <stdlib.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <map>
#include <utility>

#include "../../file_utils.h"
#include "../../opencl/opencl_common.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

#define OPENCL_CHECK_ERROR(e)                  \
  {                                            \
    if (e != CL_SUCCESS) {                     \
      std::cout << "OpenCL Error, code=" << e; \
      exit(EXIT_FAILURE);                      \
    }                                          \
  }

#define OPENCL_CALL(func)  \
  {                        \
    cl_int e = (func);     \
    OPENCL_CHECK_ERROR(e); \
  }

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

#ifdef TVM_GRAPH_EXECUTOR_ADRENO_ACCL
struct CachedLayer {
  AdrenoAcCLOpHandle op_hndl;
  // arg list with node id and tensor type key = [{nid, "input"},]
  std::vector<std::pair<int, std::string>> op_arg_list;
  // map of dynamic shape = {node_id, {key, shape_index}}
  std::map<int, std::pair<std::string, int>> dy_shape_map;
};
#endif

class AdrenoACCLRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The ADRENO_ACCL runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit AdrenoACCLRuntime(const std::string& symbol_name, const std::string& graph_json,
                             const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names), adreno_accl_symbol(symbol_name) {}

  ~AdrenoACCLRuntime() {
#ifdef TVM_GRAPH_EXECUTOR_ADRENO_ACCL
    AdrenoAcCLReleaseOp(layer_.op_hndl);
    AdrenoAcCLRelease(accl);
#endif
  }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const override { return "adreno_accl"; }

  /*!
   * \brief Initialize runtime. Create ADRENO_ACCL layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);

#ifdef TVM_GRAPH_EXECUTOR_ADRENO_ACCL
    InitAdrenoACCL();
#endif

    BuildEngine();
  }

#ifdef TVM_GRAPH_EXECUTOR_ADRENO_ACCL

  void InitAdrenoACCL() {
    workspace = cl::OpenCLWorkspace::Global();
    workspace->Init();
    tentry = workspace->GetThreadEntry();
    device_id = workspace->GetCLDeviceID(tentry->device.device_id);
    platform_id = workspace->device_to_platform[device_id];
    accl = AdrenoAcCLInit(workspace->contexts[platform_id], device_id);
  }

  /*!
   * \brief Unpack inputs and outputs and run inference on a given layer.
   *
   * \param args Access inputs and outputs.
   * \param function The layer to execute inference on.
   * \return Status of inference.
   */
  void Run() override {
    int argIndx = 0;
    unsigned int dy_shape;
    for (size_t i = 0; i < layer_.op_arg_list.size(); ++i) {
      uint32_t eid = EntryID(layer_.op_arg_list[i].first, 0);
      ICHECK(kDLOpenCL == data_entry_[eid]->device.device_type)
          << "data ptr is not in OPENCL device";
      cl::BufferDescriptor* cl_buf_desc =
          static_cast<cl::BufferDescriptor*>(const_cast<DLTensor*>(data_entry_[eid])->data);
      OPENCL_CALL(AdrenoAcCLSetOpArg(layer_.op_hndl, layer_.op_arg_list[i].second,
                                     static_cast<cl_mem*>(&(cl_buf_desc->buffer))));
    }
    for (auto it = layer_.dy_shape_map.begin(); it != layer_.dy_shape_map.end(); it++) {
      uint32_t eid = EntryID(it->first, 0);
      dy_shape = data_entry_[eid]->shape[it->second.second];
      OPENCL_CALL(AdrenoAcCLSetOpArg(layer_.op_hndl, it->second.first, &dy_shape));
    }
    cl_command_queue queue = workspace->GetQueue(tentry->device);
    std::vector<cl_event>& evts = workspace->GetEventQueue(tentry->device);
    cl_event cpy_evt = nullptr;
    cl_event* evt = &cpy_evt;
    if (workspace->IsProfiling(tentry->device)) {
      evts.resize(evts.size() + 1);
      evt = &(evts.back());
    }
    OPENCL_CALL(AdrenoAcCLRunOp(layer_.op_hndl, queue, evt));
  }

 private:
  /*!
   * \brief Build ADRENO_ACCL layer from JSON representation and cache.
   *
   * \note For the time being only one layer or operator is supported
   * per engine.
   */
  void BuildEngine() {
    size_t nid;
    for (nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      DLDataType tvm_dtype = node.GetOpDataType()[0];
      if (node.GetOpType() == "input") {
      } else if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        if (op_name == "adreno_accl.dequant_matmul" ||
            op_name == "adreno_accl.dequant_matmul_cast" ||
            op_name == "adreno_accl.dequant_matmul_bias" ||
            op_name == "adreno_accl.dequant_matmul_bias_cast") {
          CreateDequantQ4F16_0Matmul(&layer_, node, nid);
        }
      } else if (node.GetOpType() == "const") {
      } else {
        LOG(FATAL) << "Build Engine: Unknown Node:" << node.GetOpType();
      }
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      nid = outputs_[i].id_;
      DLDataType tvm_dtype = nodes_[nid].GetOpDataType()[0];
    }
  }

  void CreateDequantQ4F16_0Matmul(CachedLayer* layer_, const JSONGraphNode& node, size_t nid) {
    size_t num_inputs = node.GetInputs().size();
    std::vector<int64_t> input_shape = nodes_[node.GetInputs()[2].id_].GetOpShape()[0];
    std::vector<int64_t> weight_shape = nodes_[node.GetInputs()[0].id_].GetOpShape()[0];
    AdrenoAcCLOpDescriptor matmul_desc;
    bool is_bias = (num_inputs == 4);

    if (input_shape[1] == 1) {
      matmul_desc.op = ADRENO_AcCL_OP_DEQUANT_VECMATMUL;
      matmul_desc.props["M"] = 1;
    } else if (input_shape[1] > 1) {
      matmul_desc.op = ADRENO_AcCL_OP_DEQUANT_MATMUL;
      unsigned int M = input_shape[1];
      matmul_desc.props["M"] = M;
    } else {
      matmul_desc.op = ADRENO_AcCL_OP_DEQUANT_MATMUL;
      std::string M = "m";
      matmul_desc.props["M"] = M;
      layer_->dy_shape_map.insert({nid, std::make_pair(M, 1)});
    }
    DLDataType tvm_dtype = node.GetOpDataType()[0];
    if (tvm_dtype.code == DLDataTypeCode::kDLFloat && tvm_dtype.bits == 32) {
      std::string out_dtype = "float32";
      matmul_desc.props["out_dtype"] = out_dtype;
    }
    if (weight_shape[1] > 0) {
      unsigned int N = weight_shape[1];
      matmul_desc.props["N"] = N;
    } else {
      std::string N = "n";
      matmul_desc.props["N"] = N;
      layer_->dy_shape_map.insert({nid, std::make_pair(N, 2)});
    }
    unsigned int K = input_shape[2];
    matmul_desc.props["K"] = K;
    matmul_desc.props["is_bias"] = is_bias;
    matmul_desc.props["quant_type"] = "q4f16_0";

    layer_->op_hndl = AdrenoAcCLGetOp(accl, matmul_desc);

    layer_->op_arg_list.push_back(std::make_pair(node.GetInputs()[2].id_, "input"));
    layer_->op_arg_list.push_back(std::make_pair(node.GetInputs()[0].id_, "weight"));
    layer_->op_arg_list.push_back(std::make_pair(node.GetInputs()[1].id_, "quant_scale"));
    layer_->op_arg_list.push_back(std::make_pair(nid, "output"));
    if (is_bias) {
      layer_->op_arg_list.push_back(std::make_pair(node.GetInputs()[3].id_, "bias"));
    }
  }

  cl::OpenCLWorkspace* workspace = NULL;
  cl::OpenCLThreadEntry* tentry = NULL;
  cl_device_id device_id;
  cl_platform_id platform_id;
  AdrenoAcCL accl;
  CachedLayer layer_;

#else
  void Run() override {
    LOG(FATAL) << "Cannot call run on ADRENO_ACCL module without runtime enabled. "
               << "Please build with USE_ADRENO_ACCL_GRAPH_EXECUTOR.";
  }

  void BuildEngine() {
    LOG(WARNING) << "ADRENO_ACCL engine is not initialized. "
                 << "Please build with USE_ADRENO_ACCL_GRAPH_EXECUTOR.";
  }
#endif
  /*! ADRENO_ACCL sub graph symbol in TVM main module */
  std::string adreno_accl_symbol;
};

runtime::Module AdrenoACCLRuntimeCreate(const String& symbol_name, const String& graph_json,
                                        const Array<String>& const_names) {
  auto n = make_object<AdrenoACCLRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.adreno_accl_runtime_create").set_body_typed(AdrenoACCLRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_adreno_accl")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<AdrenoACCLRuntime>);
}  //  namespace contrib
}  //  namespace runtime
}  //  namespace tvm
