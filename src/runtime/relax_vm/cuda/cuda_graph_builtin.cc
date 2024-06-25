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

#include "../../../support/utils.h"
#include "../../cuda/cuda_common.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

namespace {

struct CUDAGraphCaptureKey {
  // The unique index of the capture function within the module
  int64_t index;
  // The symbolic variables the capture function depends on. When the capture function is ran with
  // different symbolic variable values, the CUDA graph will be re-captured as a different version,
  // identified by this shape tuple. This is default constructed as an empty tuple.
  ShapeTuple shape_expr;

  CUDAGraphCaptureKey(int64_t index, const Optional<ShapeTuple>& shape_expr) : index(index) {
    if (shape_expr) {
      this->shape_expr = shape_expr.value();
    }
  }
};

struct CUDAGraphCaptureKeyHash {
  size_t operator()(const CUDAGraphCaptureKey& key) const {
    std::hash<int64_t> hash_fn;
    size_t hash = hash_fn(key.index);
    for (const auto& shape : key.shape_expr) {
      support::HashCombine(hash, hash_fn(shape));
    }
    return hash;
  }
};

struct CUDAGraphCaptureKeyEqual {
  bool operator()(const CUDAGraphCaptureKey& lhs, const CUDAGraphCaptureKey& rhs) const {
    return lhs.index == rhs.index && std::equal(lhs.shape_expr.begin(), lhs.shape_expr.end(),
                                                rhs.shape_expr.begin(), rhs.shape_expr.end());
  }
};

/*! \brief The captured state of a CUDA graph */
struct CUDAGraphCapturedState {
  CUDAGraphCapturedState() {}

  CUDAGraphCapturedState(const CUDAGraphCapturedState&) = delete;
  CUDAGraphCapturedState(CUDAGraphCapturedState&& other) { *this = std::move(other); }

  CUDAGraphCapturedState& operator=(const CUDAGraphCapturedState&) = delete;
  CUDAGraphCapturedState& operator=(CUDAGraphCapturedState&& other) {
    std::swap(states, other.states);
    std::swap(exec, other.exec);
    return *this;
  }

  ~CUDAGraphCapturedState() {
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

class ScopedCUDAStream {
 public:
  ScopedCUDAStream() { CUDA_CALL(cudaStreamCreate(&stream_)); }
  ~ScopedCUDAStream() { cudaStreamDestroy(stream_); }
  ScopedCUDAStream(const ScopedCUDAStream&) = delete;
  ScopedCUDAStream(ScopedCUDAStream&&) = delete;
  ScopedCUDAStream& operator=(const ScopedCUDAStream&) = delete;
  ScopedCUDAStream& operator=(ScopedCUDAStream&&) = delete;

  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_;
};

class CUDACaptureStream {
 public:
  explicit CUDACaptureStream(cudaGraph_t* graph)
      : prev_default_stream_(CUDAThreadEntry::ThreadLocal()->stream), output_graph_(graph) {
    CUDAThreadEntry::ThreadLocal()->stream = capture_stream_;

    CUDA_CALL(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));
  }
  ~CUDACaptureStream() {
    cudaStreamEndCapture(capture_stream_, output_graph_);
    CUDAThreadEntry::ThreadLocal()->stream = prev_default_stream_;
  }

 private:
  cudaStream_t prev_default_stream_;
  ScopedCUDAStream capture_stream_;

  cudaGraph_t* output_graph_;
};

}  // namespace

/*! \brief The VM extension of CUDA graph. */
class CUDAGraphExtensionNode : public VMExtensionNode {
 public:
  TVM_DECLARE_FINAL_OBJECT_INFO(CUDAGraphExtensionNode, VMExtensionNode);

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
                         int64_t entry_index, Optional<ShapeTuple> shape_expr) {
    CUDAGraphCaptureKey entry_key{entry_index, shape_expr};
    if (auto it = capture_cache_.find(entry_key); it != capture_cache_.end()) {
      // Launch CUDA graph
      const auto& [states, exec] = it->second;
      CUDA_CALL(cudaGraphLaunch(exec, CUDAThreadEntry::ThreadLocal()->stream));
      return states;
    }

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

    {
      CUDACaptureStream capture_stream(&graph);
      vm->InvokeClosurePacked(capture_func, TVMArgs(values.data(), tcodes.data(), nargs),
                              &capture_func_rv);
    }

    CUDAGraphCapturedState entry;
    entry.states = capture_func_rv;
    CUDA_CALL(cudaGraphInstantiate(&entry.exec, graph, NULL, NULL, 0));
    CUDA_CALL(cudaGraphDestroy(graph));

    ObjectRef states = entry.states;

    capture_cache_[entry_key] = std::move(entry);

    return states;
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

  static constexpr const char* _type_key = "relax_vm.CUDAGraphExtension";

 private:
  /*!
   * \brief The cache of captured cuda graphs. The key is a unique index for the capture function.
   * The value is the result of the capture.
   */
  std::unordered_map<CUDAGraphCaptureKey, CUDAGraphCapturedState, CUDAGraphCaptureKeyHash,
                     CUDAGraphCaptureKeyEqual>
      capture_cache_;
  /*!
   * \brief The cache of allocations. The key is a unique index for the allocation function.
   * The value is the cached allocations, which is a tuple of storages.
   */
  std::unordered_map<int64_t, ObjectRef> alloc_cache_;
};

/*! Managed reference to CUDAGraphExtensionNode */
class CUDAGraphExtension : public VMExtension {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CUDAGraphExtension, VMExtension, CUDAGraphExtensionNode);
  static CUDAGraphExtension Create() {
    auto data_ = make_object<CUDAGraphExtensionNode>();
    return CUDAGraphExtension(std::move(data_));
  }
};

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph.run_or_capture")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK(args.size() == 5 || args.size() == 4);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      auto extension = vm->GetOrCreateExtension<CUDAGraphExtension>();
      ObjectRef capture_func = args[1];
      ObjectRef func_args = args[2];
      int64_t entry_index = args[3];
      Optional<ShapeTuple> shape_expr = NullOpt;
      if (args.size() == 5) {
        shape_expr = args[4].AsObjectRef<ShapeTuple>();
      }
      *rv = extension->RunOrCapture(vm, capture_func, func_args, entry_index, shape_expr);
    });

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph.get_cached_alloc")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 3);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      auto extension = vm->GetOrCreateExtension<CUDAGraphExtension>();
      ObjectRef alloc_func = args[1];
      int64_t entry_index = args[2];
      *rv = extension->GetCachedAllocation(vm, alloc_func, entry_index);
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
