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
 * \file graph_runtime_cugraph.cc
 */

#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"
#include "../graph_runtime.h"

namespace tvm {
namespace runtime {

class GraphRuntimeCuGraph : public GraphRuntime {
 public:
  int StartCapture() {
    const TVMContext& ctx = data_entry_[entry_id(0, 0)]->ctx;

    TVMStreamCreate(ctx.device_type, ctx.device_id, &capture_stream_);
    TVMSetStream(ctx.device_type, ctx.device_id, capture_stream_);

    CUDA_CALL(cudaStreamBeginCapture(static_cast<cudaStream_t>(capture_stream_),
                                     cudaStreamCaptureModeGlobal));
    return 0;
  }

  int RunCudaGraph() {
    cudaStream_t cuStream = static_cast<cudaStream_t>(capture_stream_);
    CUDA_CALL(cudaGraphLaunch(cu_graph_exec_, cuStream));
    CUDA_CALL(cudaStreamSynchronize(cuStream));
    return 0;
  }

  int EndCapture() {
    cudaGraph_t graph;
    CUDA_CALL(cudaStreamEndCapture(static_cast<cudaStream_t>(capture_stream_), &graph));

    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    CUDA_CALL(cudaGraphGetNodes(graph, nodes, &numNodes));
    LOG(INFO) << "Num of nodes in the cuda graph created using stream capture API = " << numNodes;

    CUDA_CALL(cudaGraphInstantiate(&cu_graph_exec_, graph, NULL, NULL, 0));
    return 0;
  }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

 private:
  TVMStreamHandle capture_stream_;
  cudaGraphExec_t cu_graph_exec_;
};

PackedFunc GraphRuntimeCuGraph::GetFunction(const std::string& name,
                                            const ObjectPtr<Object>& sptr_to_self) {
  if (name == "run_cuda_graph") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->RunCudaGraph(); });
  } else if (name == "start_capture") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->StartCapture(); });
  } else if (name == "end_capture") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->EndCapture(); });
  } else {
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

Module GraphRuntimeCuGraphCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                                 const std::vector<TVMContext>& ctxs,
                                 PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphRuntimeCuGraph>();
  exec->Init(sym_json, m, ctxs, lookup_linked_param_func);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_cugraph.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 4) << "The expected number of arguments for graph_runtime.create is "
                                 "at least 4, but it has "
                              << args.num_args;
  PackedFunc lookup_linked_param_func;
  int ctx_start_arg = 2;
  if (args[2].type_code() == kTVMPackedFuncHandle) {
    lookup_linked_param_func = args[2];
    ctx_start_arg++;
  }

  *rv = GraphRuntimeCuGraphCreate(args[0], args[1], GetAllContext(args, ctx_start_arg),
                                  lookup_linked_param_func);
});
}  // namespace runtime
}  // namespace tvm
