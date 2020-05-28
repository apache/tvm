/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/search_task.h
 * \brief Meta information for a search task
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
  int num_cores;
  int vector_unit_bytes;
  int cache_line_bytes;
  // The max length of the axis to be unrolled or vectorized
  int max_unroll_vec;
  // The max split factor for the innermost tile
  int max_innermost_split_factor;

  // Limit params for GPU schedule
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

  static constexpr const char *_type_key = "ansor.HardwareParams";
  TVM_DECLARE_FINAL_OBJECT_INFO(HardwareParamsNode, Object);
};
TVM_DEFINE_COW_NODE_REF(HardwareParams, ObjectRef, HardwareParamsNode);


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

  static constexpr const char *_type_key = "ansor.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};
TVM_DEFINE_COW_NODE_REF(SearchTask, ObjectRef, SearchTaskNode);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_TASK_H_
