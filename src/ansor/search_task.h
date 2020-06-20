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
 * \file ansor/search_task.h
 * \brief Meta information and hardware parameters for a search task
 */

#ifndef TVM_ANSOR_SEARCH_TASK_H_
#define TVM_ANSOR_SEARCH_TASK_H_

#include <tvm/target/target.h>
#include <string>
#include "compute_dag.h"

namespace tvm {
namespace ansor {

class HardwareParams; class SearchTask;

/*! \brief Hardware related parameters */
class HardwareParamsNode : public Object {
 public:
  // The number of cores
  int num_cores;
  // The width of vector units in bytes
  int vector_unit_bytes;
  // The size of cache line in bytes
  int cache_line_bytes;
  // The max length of an axis to be unrolled or vectorized
  int max_unroll_vec;
  // The max split factor for the innermost tile
  int max_innermost_split_factor;

  // Limitation params for GPU
  int max_shared_memory_per_block{INT32_MAX};
  int max_registers_per_block{INT32_MAX};
  int max_threads_per_block{INT32_MAX};
  int max_vthread_extent{INT32_MAX};
  int warp_size{INT32_MAX};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_cores", &num_cores);
    v->Visit("vector_unit_bytes", &vector_unit_bytes);
    v->Visit("cache_line_bytes", &cache_line_bytes);
    v->Visit("max_unroll_vec", &max_unroll_vec);
    v->Visit("max_innermost_split_factor", &max_innermost_split_factor);

    v->Visit("max_shared_memory_per_block", &max_shared_memory_per_block);
    v->Visit("max_registers_per_block", &max_registers_per_block);
    v->Visit("max_threads_per_block", &max_threads_per_block);
    v->Visit("max_vthread_extent", &max_vthread_extent);
    v->Visit("warp_size", &warp_size);
  }

  static HardwareParams make(int num_cores, int vector_unit_bytes,
                             int cache_line_bytes, int max_unroll_vec,
                             int max_innermost_split_factor);

  static HardwareParams GetDefaultHardwareParams(const Target& target,
                                                 const Target& target_host);

  static constexpr const char* _type_key = "ansor.HardwareParams";
  TVM_DECLARE_FINAL_OBJECT_INFO(HardwareParamsNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(HardwareParams, ObjectRef, HardwareParamsNode);

/*! \brief Meta-info for a search task */
class SearchTaskNode : public Object {
 public:
  ComputeDAG compute_dag;
  std::string workload_key;
  Target target;
  Target target_host;
  HardwareParams hardware_params;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("compute_dag", &compute_dag);
    v->Visit("workload_key", &workload_key);
    v->Visit("target", &target);
    v->Visit("target_host", &target_host);
    v->Visit("hardware_params", &hardware_params);
  }

  static SearchTask make(ComputeDAG compute_dag, std::string workload_key,
                         Target target, Target target_host,
                         HardwareParams hardware_params);

  static constexpr const char* _type_key = "ansor.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(SearchTask, ObjectRef, SearchTaskNode);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_TASK_H_
