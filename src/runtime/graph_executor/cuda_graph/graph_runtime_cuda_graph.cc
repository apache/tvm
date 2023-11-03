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
 * \file graph_executor_cuda_graph.cc
 */

#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"
#include "../graph_executor.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Graph executor with CUDA Graph Support.
 *
 *  This is the extension of GraphExecutor class used for CUDA graph launch
 *  instead of CUDA kernel launch. CUDA graph launch requires CUDA 10.0 or
 *  above, currently there are two ways of constructing CUDA graphs:
 *  (1) Using CUDA stream capture API to capture a series of operations on
 *  CUDA stream, and automatically generates a graph (2) Building a graph
 *  using CUDA graph API manually. This implementation uses stream capture.
 */
class GraphExecutorCudaGraph : public GraphExecutor {
 public:
  /*!
   * \brief Begin CUDA graph capture on stream, the stream enters capture mode.
   */
  void StartCapture() {
    const Device& dev = data_entry_[entry_id(0, 0)]->device;

    TVMStreamCreate(dev.device_type, dev.device_id, &capture_stream_);
    TVMSetStream(dev.device_type, dev.device_id, capture_stream_);

    CUDA_CALL(cudaStreamBeginCapture(static_cast<cudaStream_t>(capture_stream_),
                                     cudaStreamCaptureModeGlobal));
  }

  /*!
   * \brief Launch the instantiated graph on stream
   */
  void RunCudaGraph() {
    cudaStream_t cuStream = static_cast<cudaStream_t>(capture_stream_);
    CUDA_CALL(cudaGraphLaunch(cuda_graph_exec_, cuStream));
    CUDA_CALL(cudaStreamSynchronize(cuStream));
  }

  /*!
   * \brief End CUDA graph capture on stream, a graph will be created and
   * instantiated.
   */
  void EndCapture() {
    cudaGraph_t graph;
    CUDA_CALL(cudaStreamEndCapture(static_cast<cudaStream_t>(capture_stream_), &graph));

    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    CUDA_CALL(cudaGraphGetNodes(graph, nodes, &numNodes));
    LOG(INFO) << "Num of nodes in the cuda graph created using stream capture API = " << numNodes;

    CUDA_CALL(cudaGraphInstantiate(&cuda_graph_exec_, graph, NULL, NULL, 0));
  }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self);

 private:
  /*! \brief The Cuda stream on which to capture a CUDA graph. */
  TVMStreamHandle capture_stream_;
  /*! \brief The captured CUDA graph will be instantiated to this. */
  cudaGraphExec_t cuda_graph_exec_;
};

PackedFunc GraphExecutorCudaGraph::GetFunction(const String& name,
                                               const ObjectPtr<Object>& sptr_to_self) {
  if (name == "run_cuda_graph") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->RunCudaGraph(); });
  } else if (name == "start_capture") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->StartCapture(); });
  } else if (name == "end_capture") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->EndCapture(); });
  } else {
    return GraphExecutor::GetFunction(name, sptr_to_self);
  }
}

Module GraphExecutorCudaGraphCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                                    const std::vector<Device>& devs,
                                    PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphExecutorCudaGraph>();
  exec->Init(sym_json, m, devs, lookup_linked_param_func);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_executor_cuda_graph.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GE(args.num_args, 4)
          << "The expected number of arguments for graph_executor.create is "
             "at least 4, but it has "
          << args.num_args;
      PackedFunc lookup_linked_param_func;
      int dev_start_arg = 2;
      if (args[2].type_code() == kTVMPackedFuncHandle) {
        lookup_linked_param_func = args[2];
        dev_start_arg++;
      }

      *rv = GraphExecutorCudaGraphCreate(args[0], args[1], GetAllDevice(args, dev_start_arg),
                                         lookup_linked_param_func);
    });
}  // namespace runtime
}  // namespace tvm
