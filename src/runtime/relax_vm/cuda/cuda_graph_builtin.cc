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
 * \file src/runtime/relax_vm/cuda_graph_builtin.cc
 * \brief The CUDA graph related builtin functions for Relax virtual machine.
 */

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>

#include "../../cuda/cuda_common.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

/*! \brief The cache states of a CUDA graph. */
class CUDAGraphCache : public Object {
 public:
  struct CaptureResult {
    ~CaptureResult() {
      if (exec) {
        CUDA_CALL(cudaGraphExecDestroy(exec));
      }
    }
    /*!
     * \brief Tuple of intemediate tensors in the capture func that will be used outside the
     * capture func
     */
    ObjectRef states;
    /*! \brief The instantiated cuda graph */
    cudaGraphExec_t exec = nullptr;
  };

  static CUDAGraphCache* Get() { return dmlc::ThreadLocalStore<CUDAGraphCache>::Get(); }

  /*!
   * \brief Launch the cuda graph if it has been cached, otherwise execute it in capture mode.
   * \param vm The virtual machine.
   * \param capture_func The function of type (args...) -> Tuple[ObjectRef], where 'args' are the
   * static arguments that are the same for all invocations of the capture function, the returned
   * tuple contains the intermediate tensors that will be used outside the capture function.
   * \param args The static arguments of the capture function
   * \param entry_index The unique index of the capture function used for lookup.
   * \return The return value of the capture function.
   */
  ObjectRef RunOrCapture(VirtualMachine* vm, const ObjectRef& capture_func, ObjectRef args,
                         int64_t entry_index) {
    if (auto it = capture_cache_.find(entry_index); it != capture_cache_.end()) {
      // Launch CUDA graph
      const auto& [states, exec] = it->second;
      CUDA_CALL(cudaGraphLaunch(exec, CUDAThreadEntry::ThreadLocal()->stream));
      return states;
    }

    cudaStream_t capture_stream;
    CUDA_CALL(cudaStreamCreate(&capture_stream));
    CUDAGraphCache::CaptureResult entry;

    // Set up arguments for the graph execution
    Array<ObjectRef> tuple_args = Downcast<Array<ObjectRef>>(args);
    int nargs = static_cast<int>(tuple_args.size());
    std::vector<TVMValue> values(nargs);
    std::vector<int> tcodes(nargs);
    TVMArgsSetter setter(values.data(), tcodes.data());
    for (int i = 0; i < nargs; ++i) {
      ObjectRef arg = tuple_args[i];
      setter(i, arg);
    }

    TVMRetValue capture_func_rv;
    // Run the function without CUDA graph. This is a warm up step to do necessary initialization
    // of the CUDA module such as loading module data, setting kernel attributes.
    vm->InvokeClosurePacked(capture_func, TVMArgs(values.data(), tcodes.data(), nargs),
                            &capture_func_rv);

    // Run the graph in capture mode
    cudaGraph_t graph;
    std::swap(capture_stream, CUDAThreadEntry::ThreadLocal()->stream);
    CUDA_CALL(cudaStreamBeginCapture(CUDAThreadEntry::ThreadLocal()->stream,
                                     cudaStreamCaptureModeGlobal));

    vm->InvokeClosurePacked(capture_func, TVMArgs(values.data(), tcodes.data(), nargs),
                            &capture_func_rv);
    entry.states = capture_func_rv;
    CUDA_CALL(cudaStreamEndCapture(CUDAThreadEntry::ThreadLocal()->stream, &graph));
    std::swap(capture_stream, CUDAThreadEntry::ThreadLocal()->stream);

    capture_cache_[entry_index] = entry;
    CUDA_CALL(cudaGraphInstantiate(&capture_cache_[entry_index].exec, graph, NULL, NULL, 0));
    CUDA_CALL(cudaStreamDestroy(capture_stream));
    CUDA_CALL(cudaGraphDestroy(graph));
    return entry.states;
  }

  /*!
   * \brief Get the cached allocation from the cache or run the allocation function.
   * \param vm The virtual machine.
   * \param alloc_func The function of type () -> ObjectRef, where the returned object is the
   * tuple of allocated storage objects.
   * \param entry_index The unique index of the allocation function used for lookup.
   */
  ObjectRef GetCachedAllocation(VirtualMachine* vm, const ObjectRef& alloc_func,
                                int64_t entry_index) {
    if (auto it = alloc_cache_.find(entry_index); it != alloc_cache_.end()) {
      return it->second;
    }
    TVMRetValue alloc_func_rv;
    vm->InvokeClosurePacked(alloc_func, TVMArgs(nullptr, nullptr, 0), &alloc_func_rv);
    ObjectRef alloc_result = alloc_func_rv;
    alloc_cache_[entry_index] = alloc_result;
    return alloc_result;
  }

 private:
  /*!
   * \brief The cache of captured cuda graphs. The key is a unique index for the capture function.
   * The value is the result of the capture.
   */
  std::unordered_map<int64_t, CaptureResult> capture_cache_;
  /*!
   * \brief The cache of allocations. The key is a unique index for the allocation function.
   * The value is the cached allocations, which is a tuple of storages.
   */
  std::unordered_map<int64_t, ObjectRef> alloc_cache_;
};

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph.run_or_capture")
    .set_body_typed([](TVMArgValue vm_ptr, ObjectRef capture_func, ObjectRef func_args,
                       int64_t entry_index) {
      VirtualMachine* vm = VirtualMachine::GetContextPtr(vm_ptr);
      CUDAGraphCache* cache = CUDAGraphCache::Get();
      return cache->RunOrCapture(vm, capture_func, func_args, entry_index);
    });

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph.get_cached_alloc")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 3);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      ObjectRef alloc_func = args[1];
      int64_t entry_index = args[2];
      CUDAGraphCache* cache = CUDAGraphCache::Get();
      *rv = cache->GetCachedAllocation(vm, alloc_func, entry_index);
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
