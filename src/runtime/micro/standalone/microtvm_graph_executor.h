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

#ifndef TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_GRAPH_EXECUTOR_H_
#define TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_GRAPH_EXECUTOR_H_

#include <dlpack/dlpack.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "microtvm_runtime_api.h"
#include "minimal_vector.h"

namespace tvm {
namespace micro {

typedef int (*BackendPackedCFunc)(void* args, int* type_codes, int num_args);

// dlopen/dlsym/dlclose abstraction.
class DSOModule {
 public:
  explicit DSOModule(const std::string& name);
  ~DSOModule();
  BackendPackedCFunc GetFunction(const std::string& name) const;

 private:
  void* GetSymbol(const char* name) const;
  void* lib_handle_{nullptr};
};

// The graph attribute fields.
struct GraphAttr {
  DynArray<int> storage_id;
  DynArray<std::string> dltype;
  DynArray<DynArray<int64_t>> shape;
};

// Memory pool entry.
struct PoolEntry {
  size_t size;
  int device_type;
};

// Node entry
struct NodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
};

// Operator attributes about TVMOp
struct TVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
};

// Node
struct Node {
  // operator type in string
  std::string op_type;
  // name of the op
  std::string name;
  // parameters
  TVMOpParam param;
  // inputs
  DynArray<NodeEntry> inputs;
};

// Minimal NDArray abstraction
class NDArray {
 public:
  // initialize NDArray with shape/dtype/device
  static NDArray Empty(const DynArray<int64_t>& shape, DLDataType dtype, DLDevice dev);
  // create a view of the NDArray storage, with the given shape/dtype
  NDArray CreateView(const DynArray<int64_t>& shape, DLDataType dtype);
  // Copy into the internal storage.
  void CopyFrom(DLTensor* src);
  // Copy out of the internal storage
  void CopyTo(DLTensor* dst) const;
  // View `this` as a DLTensor
  DLTensor ToDLTensor();
  ~NDArray();

 private:
  // reference-counted storage
  std::shared_ptr<void> storage_;
  // tensor shape
  DynArray<int64_t> shape_;
  // tensor dtype
  DLDataType dtype_;
  // tensor device
  DLDevice device_;
};

// Minimal GraphExecutor implementation
class MicroGraphExecutor {
 public:
  // Construct a GraphExecutor with the given graph and DSOModule.
  MicroGraphExecutor(const std::string& graph_json, DSOModule* module);
  ~MicroGraphExecutor();
  // Run the graph
  void Run();
  // Set the input at `index` to a copy of the tensor `data_in`
  void SetInput(int index, DLTensor* data_in);
  // Copy the output at `index` into `data_out`
  void CopyOutputTo(int index, DLTensor* data_out);

 private:
  void SetupStorage();
  void SetupOpExecs();

  uint32_t num_node_entries() const { return node_row_ptr_.back(); }
  uint32_t entry_id(uint32_t nid, uint32_t index) const { return node_row_ptr_[nid] + index; }
  uint32_t entry_id(const NodeEntry& e) const { return entry_id(e.node_id, e.index); }

  DSOModule* module_;

  // TODO(tulloch): these are essentially unused after construction.
  // The graph nodes
  DynArray<Node> nodes_;
  // The argument noes
  DynArray<uint32_t> input_nodes_;
  // Used for quick entry indexing
  DynArray<uint32_t> node_row_ptr_;
  // Output entries
  DynArray<NodeEntry> outputs_;
  // Additional graph attributes
  GraphAttr attrs_;
  // Execution device
  DLDevice device_{kDLCPU, 0};

  // Common storage pool
  DynArray<NDArray> storage_pool_;
  // Data entry for each node
  DynArray<NDArray> data_entry_;
  // Operator for each node
  DynArray<std::function<void()>> op_execs_;
};

}  // namespace micro
}  // namespace tvm

#endif  // TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_GRAPH_EXECUTOR_H_
