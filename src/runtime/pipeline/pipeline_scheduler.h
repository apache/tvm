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
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_SCHEDULER_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_SCHEDULER_H_
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "pipeline_struct.h"
namespace tvm {
namespace runtime {
/*!
 * \brief The class that executes the pipeline logic,it is used to initialize the thread pool,
    execute and schedule pipeline tasks, allocate and manage memory, etc.
 */
class PipelineScheduler {
 public:
  /*!
   * \brief Initialize the pipeline.
   * \param modules The list of graph executor module.
   * \param pipeline_config The dependency information of each graph executor module.
   */
  std::shared_ptr<GlobalRuntime> PipelineInit(const std::vector<Module>& modules,
                                              const ConfigPipelineExecution& pipeline_config,
                                              const InputConnectionConfig& input_connection_config);
  /*!
   * \brief Running the pipeline logic.
   * \param runtimes A list of backend runtime modules.
   */
  void PipelineRun(const std::vector<std::shared_ptr<BackendRuntime>>& runtimes);
  /*!
   * \brief Get a list of outputs.
   */
  Array<NDArray> PipelineGetOutput();

 private:
  /*!\brief The list of graph executors.*/
  std::vector<Module> graph_modules_;
  /*!\brief A list of NDArray used to storage outputs.*/
  Array<NDArray> output_arrays_;
  /*!\brief The global runtime to represent the pipeline executor.*/
  std::shared_ptr<GlobalRuntime> global_runtime_;
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_SCHEDULER_H_
