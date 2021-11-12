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
 * \return Return a list of backend runtime module.
 */
std::vector<std::shared_ptr<BackendRuntime>> PipelineScheduler::PipelineInit(
    const std::vector<Module>& modules, ConfigPipelineExecution pipeline_config) {
  std::vector<std::shared_ptr<BackendRuntime>> runtimes;
  graph_modules_ = modules;
  for (size_t i = 0; i < graph_modules_.size(); i++) {
    auto runItem = std::make_shared<BackendRuntime>(graph_modules_[i], i);
    runtimes.push_back(runItem);
  }
  // Initialize the outputs array.
  auto& global_output_map = pipeline_config.GetGlobalConfigOutputBindings();
  for (size_t i = 0; i < global_output_map.size(); i++) {
    if (global_output_map.find(i) == global_output_map.end()) {
      LOG(FATAL) << "Not find global output index " << i;
    }
    ModuleOutputPair& output_pair = global_output_map[i];
    NDArray output = runtimes[output_pair.first]->CreateFromOutput(output_pair.second);
    output_array.push_back(output);
  }
  return runtimes;
}

/*!
 * \brief Exeute in the sequential mode.
 * \param runtimes A list of backend runtimes module.
 * \param pipeline_config The dependency information of each graph executor module.
 */
void PipelineScheduler::PipelineRunSequential(
    const std::vector<std::shared_ptr<BackendRuntime>>& runtimes,
    ConfigPipelineExecution pipeline_config) {
  for (size_t i = 0; i < runtimes.size(); i++) {
    // The offset in vector is the runtime execution order, this offset value should
    // be same with the the value of "runtime_idx" in runtime.
    if (static_cast<int>(i) != runtimes[i]->GetModuleIndex()) {
      LOG(FATAL) << "runtime index " << runtimes[i]->GetModuleIndex()
                 << " is not same as vector offset value " << i;
    }

    if (!pipeline_config.FindModuleInConfig(i)) {
      LOG(FATAL) << "Not find the configuration for the module " << i;
    }

    runtimes[i]->Run();
    // Check if there is any output need to be forward to other graph module or to be as
    // global output.
    int outputs_num = runtimes[i]->NumOutputs();
    for (int j = 0; j < outputs_num; j++) {
      ConfigBindings& out_binding = pipeline_config[i][j];
      std::unordered_map<int, std::string>& input_connections = out_binding.Get();
      NDArray output = runtimes[i]->GetOutput(j);
      for (auto bind : input_connections) {
        // If the value of "bind.first" less then 0 then this is not a graph module binding.
        if (bind.first < 0) continue;
        // Set input data for the graph module.
        runtimes[bind.first]->SetInput(bind.second, const_cast<DLTensor*>(output.operator->()));
      }
      // Store the output.
      if (out_binding.IsGlobalOutput()) {
        int global_idx = out_binding.GetGlobalOutputIndex();
        TVMArrayCopyFromTo(const_cast<DLTensor*>(output.operator->()),
                           const_cast<DLTensor*>(output_array[global_idx].operator->()), nullptr);
      }
    }
  }
}
/*!
 * \brief Execute pipeline.
 * \param runtimes A list of backend runtimes module.
 * \param pipeline_config The dependency information of each graph executor module.
 * \param sequential_mode If the execution is in sequential mode.
 */
void PipelineScheduler::PipelineRun(const std::vector<std::shared_ptr<BackendRuntime>>& runtimes,
                                    ConfigPipelineExecution pipeline_config, bool sequential_mode) {
  if (!sequential_mode) {
    // TODO(huajsj) remove this check after all of pipeline features in.
    LOG(FATAL) << "Currently Only supports sequential mode.";
  } else {
    PipelineRunSequential(runtimes, pipeline_config);
  }
}
/*!
 * \brief Stop the pipeline exection.
 */
void PipelineScheduler::PipelineStop() {
  // TODO(huajsj) Add stop logic.
}
/*!
 * \brief Get a list of outputs of the pipeline execution.
 */
Array<NDArray> PipelineScheduler::PipelineGetOutput() { return output_array; }
}  // namespace runtime
}  // namespace tvm
