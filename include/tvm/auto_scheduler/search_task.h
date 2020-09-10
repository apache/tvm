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
 * \file auto_scheduler/search_task.h
 * \brief Meta information and hardware parameters for a search task.
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_TASK_H_
#define TVM_AUTO_SCHEDULER_SEARCH_TASK_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/target/target.h>

namespace tvm {
namespace auto_scheduler {

class HardwareParams;

/*! \brief The parameters of target hardware used to guide the SearchPolicy. */
class HardwareParamsNode : public Object {
 public:
  /*! \brief The number of cores. */
  int num_cores;
  /*! \brief The width of vector units in bytes. */
  int vector_unit_bytes;
  /*! \brief The size of cache line in bytes. */
  int cache_line_bytes;

  // GPU related parameters got from device query API

  /*! \brief The max shared memory per block. */
  int max_shared_memory_per_block{INT32_MAX};
  /*! \brief The max register memory per block. */
  int max_registers_per_block{INT32_MAX};
  /*! \brief The max threads per block. */
  int max_threads_per_block{INT32_MAX};
  /*! \brief The max vthread extent. */
  int max_vthread_extent{INT32_MAX};
  /*! \brief The thread numbers of a warp. */
  int warp_size{INT32_MAX};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_cores", &num_cores);
    v->Visit("vector_unit_bytes", &vector_unit_bytes);
    v->Visit("cache_line_bytes", &cache_line_bytes);
    v->Visit("max_shared_memory_per_block", &max_shared_memory_per_block);
    v->Visit("max_registers_per_block", &max_registers_per_block);
    v->Visit("max_threads_per_block", &max_threads_per_block);
    v->Visit("max_vthread_extent", &max_vthread_extent);
    v->Visit("warp_size", &warp_size);
  }

  /*!
   * \brief Get the default hardware params.
   * \param target A `tvm.target`.
   * \param target_host A `tvm.target` for host device.
   * \return A HardwareParams object.
   */
  static HardwareParams GetDefaultHardwareParams(const Target& target, const Target& target_host);

  static constexpr const char* _type_key = "auto_scheduler.HardwareParams";
  TVM_DECLARE_FINAL_OBJECT_INFO(HardwareParamsNode, Object);
};

/*!
 * \brief Managed reference to HardwareParamsNode.
 * \sa HardwareParamsNode
 */
class HardwareParams : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param num_cores The number of cores.
   * \param vector_unit_bytes The width of vector units in bytes.
   * \param cache_line_bytes The size of cache line in bytes.
   */
  HardwareParams(int num_cores, int vector_unit_bytes, int cache_line_bytes);

  TVM_DEFINE_OBJECT_REF_METHODS(HardwareParams, ObjectRef, HardwareParamsNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareParamsNode);
};

/*!
 * \brief The computation information and hardware parameters for a specific schedule search task.
 */
class SearchTaskNode : public Object {
 public:
  /*! \brief The ComputeDAG for the compute declaration. */
  ComputeDAG compute_dag;
  /*! \brief The workload key for the compute declaration. */
  String workload_key;
  /*! \brief The target device of this search task. */
  Target target;
  /*! \brief The target host device of this search task. */
  Target target_host;
  /*! \brief Hardware parameters used in this search task. */
  HardwareParams hardware_params;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("compute_dag", &compute_dag);
    v->Visit("workload_key", &workload_key);
    v->Visit("target", &target);
    v->Visit("target_host", &target_host);
    v->Visit("hardware_params", &hardware_params);
  }

  static constexpr const char* _type_key = "auto_scheduler.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};

/*!
 * \brief Managed reference to SearchTaskNode.
 * \sa SearchTaskNode
 */
class SearchTask : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param compute_dag The ComputeDAG for the compute declaration.
   * \param workload_key The workload key for the compute declaration.
   * \param target The target device of this search task.
   * \param target_host The target host device of this search task.
   * \param hardware_params Hardware parameters used in this search task.
   */
  SearchTask(ComputeDAG compute_dag, String workload_key, Target target, Target target_host,
             Optional<HardwareParams> hardware_params);

  TVM_DEFINE_OBJECT_REF_METHODS(SearchTask, ObjectRef, SearchTaskNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_TASK_H_
