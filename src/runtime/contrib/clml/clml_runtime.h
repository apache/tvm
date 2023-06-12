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
 * \file src/runtime/contrib/clml/clml_runtime.h
 * \brief CLML header
 */
#ifndef TVM_RUNTIME_CONTRIB_CLML_CLML_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_CLML_CLML_RUNTIME_H_
#include <CL/cl.h>
#include <CL/opencl.h>
#include <stdlib.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../file_utils.h"
#include "../../opencl/opencl_common.h"
#include "../../thread_storage_scope.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_CLML
#include <CL/cl_qcom_ml_ops.h>

#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)
#define GET_ML_INTERFACE CAT(CAT(clGetMLInterfaceV, CL_QCOM_ML_OPS_H_MAJOR_VERSION), QCOM)
#define GET_ML_API_INTERFACE CAT(CAT(CLMLInterfaceV, CL_QCOM_ML_OPS_H_MAJOR_VERSION), QCOM)

/*! \brief Magic number for CLML Tuning cache entry */
static const uint64_t kTVMCLMLTuningCacheMagic = 0x434C4D4C54554E45;

#define DEBUG_MEMORY_ALLOC false
#define DEBUG_STATS false
#define LOG_MEM LOG_IF(WARNING, DEBUG_MEMORY_ALLOC)
#define LOG_STATS LOG_IF(WARNING, DEBUG_STATS)

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

class CLMLThreadEntry;

/*!
 * \brief CLML workspace.
 */
class CLMLWorkspace {
 public:
  /* Constructor */
  CLMLWorkspace();
  /*!
   * \brief Get the thread local ThreadEntry
   */
  virtual CLMLThreadEntry* GetThreadEntry();

  /* CLML Context */
  GET_ML_API_INTERFACE* h_ClmlIntf = nullptr;
  cl::OpenCLWorkspace* workspace = nullptr;
  cl::OpenCLThreadEntry* tentry = nullptr;
  cl_device_id device_id;
  cl_platform_id platform_id;

  /* Tuning Support */
  bool is_tuning_run;
  char* tuning_file;

  /* Recordable Queues */
  bool is_recordable_queue = false;

  /* On chip memory support */
  bool is_on_chip_memory = false;

  /* On chip memory size */
  size_t onchip_mem_size = 0;

  /* get the global workspace */
  static CLMLWorkspace* Global();

  bool ExtensionStringPresent(std::string extn);

  /* DDR memory management */
  std::map<cl_mem, std::pair<int, int>> ddr_global_pool;  // buf, size and ref count
};

/*! \brief Thread local workspace */
class CLMLThreadEntry {
 public:
  /* get the global workspace */
  static CLMLThreadEntry* ThreadLocal();
};

/*!
 * \brief CLML objects we cache in order to avoid needing to construct
 * a new layer each time.
 */
struct CachedLayer {
  /* List of all created CLML operation handles in graph */
  std::vector<cl_ml_op_qcom> function;
  /* The input tensor map  */
  std::map<int, std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> inputs;
  /* A place holder Tensor representing TVM NDArray as CLML Tensor */
  std::map<int, std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> in_placeholder;
  /* The Output tensor map */
  std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> outputs;
  /* A place holder Tensor representing TVM NDArray as CLML Tensor */
  std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> out_placeholder;
  /* Tensor shape exception list while returning from CLML Subgraph */
  std::map<int, std::vector<size_t>> out_shapes;
  /* Map of all tensors which need backing memory allocation */
  std::map<int, std::pair<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>, JSONGraphNode>>
      storage_map;
  /* Tensor memory descriptors list to set after backing memory allocation */
  std::vector<cl_ml_tensor_memory_desc_qcom> tensorMemDescs;
  cl_ml_tensor_mem_desc_set_qcom descriptorSet;
  /* List of layer names in subgraph */
  std::vector<std::string> layer_names;
  /* A dummy CLML tensor used across various ops */
  cl_ml_tensor_qcom unusedTensor = nullptr;

  /* Graph level tuning cache */
  cl_ml_tuningcache_qcom tuning_cache = nullptr;

  /* Memory management */
  std::map<int, int> storage_ref_map;  // NodeId & ref. count
  /* Activation node id & life span (the layer after which we can free) */
  std::map<int, int> life_span;
  std::map<size_t, size_t> on_chip_pool_size;                   // Mem start & size
  std::map<size_t, int> on_chip_pool_alloc_info;                // Mem start & node_id
  std::map<int, std::pair<size_t, size_t>> on_chip_alloc_plan;  // Final Alloc Plan
  std::map<int, size_t> on_chip_reject;                         // On-Chip reject info
  bool alloc_ping_pong;                                         // Allocation stratagy
  int in_chip_total_free;                                       // Total available
  int in_chip_total_alloc;                                      // Free memory
  int on_chip_alert_fail;                                       // Faliure due to fragmentation

  /* DDR memory planner */
  std::map<cl_mem, std::pair<int, bool>> ddr_storage_ref_map;  // local pool reference count
  std::map<int, cl_mem> ddr_alloc_plan;                        // allocation map <nid, cl_mem>

  cl_command_queue recordable_queue = nullptr;
  cl_recording_qcom recording = nullptr;
};

struct tensor_dims_t {
  uint32_t n, c, h, w;
};

#define CLML_INTF CLMLWorkspace::Global()->h_ClmlIntf
#define CLML_QUEUE \
  CLMLWorkspace::Global()->workspace->GetQueue(CLMLWorkspace::Global()->tentry->device)
#define CLML_CTX CLMLWorkspace::Global()->workspace->contexts[CLMLWorkspace::Global()->platform_id]

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_GRAPH_EXECUTOR_CLML
#endif  // TVM_RUNTIME_CONTRIB_CLML_CLML_RUNTIME_H_
