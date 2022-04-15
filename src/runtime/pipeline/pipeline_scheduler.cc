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
#include "pipeline_scheduler.h"

#include <unordered_map>
#include <utility>
#include <vector>
namespace tvm {
namespace runtime {
/*!
 * \brief Initialize the pipeline.
 * \param modules The list of graph executor modules.
 * \param pipeline_conf The dependency information of each graph executor module.
 */
std::vector<std::shared_ptr<BackendRuntime>> PipelineScheduler::PipelineInit(
    const std::vector<Module>& modules, const ConfigPipelineExecution& pipeline_config) {
  std::vector<std::shared_ptr<BackendRuntime>> runtimes;
  graph_modules_ = modules;
  global_runtime_ = std::make_shared<GlobalRuntime>(GLOBAL_MODULE_INDEX);
  // Creating a list of runtimes.
  for (size_t i = 0; i < graph_modules_.size(); i++) {
    auto run_item = std::make_shared<BackendRuntime>(graph_modules_[i], i);
    runtimes.push_back(run_item);
  }
  // Creating a list of NDArray in order to storage the outputs data.
  auto global_output_map = pipeline_config.GetGlobalConfigOutputBindings();
  for (size_t i = 0; i < global_output_map.size(); i++) {
    if (global_output_map.find(i) == global_output_map.end()) {
      LOG(FATAL) << "Not find global output index " << i;
    }
    ModuleOutputPair& output_pair = global_output_map[i];
    NDArray output = runtimes[output_pair.first]->CreateFromOutput(output_pair.second);
    output_arrays_.push_back(output);
  }
  // Initializing and then running the worker thread.
  for (auto runtime : runtimes) {
    runtime->InitializePipeline(pipeline_config, &runtimes, global_runtime_);
  }
  return runtimes;
}
/*!
 * \brief Running pipeline logic.
 * \param runtimes A list of backend runtime modules.
 * \param pipeline_config The dependency configuration of each runtime module.
 */
void PipelineScheduler::PipelineRun(const std::vector<std::shared_ptr<BackendRuntime>>& runtimes,
                                    ConfigPipelineExecution pipeline_config) {
  runtimes.front()->RunPipeline();
}
/*!
 * \brief Get a list of output.
 */
Array<NDArray> PipelineScheduler::PipelineGetOutput() {
  bool ret = global_runtime_->GetOutput(&output_arrays_);
  return ret ? output_arrays_ : Array<NDArray>{};
}
}  // namespace runtime
}  // namespace tvm
