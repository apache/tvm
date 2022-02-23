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
    runtime->InitializePipeline(pipeline_config, &runtimes);
  }
  return runtimes;
}
/*!
 * \brief Running the pipeline logic in the sequential mode.
 * \param runtimes A list of backend runtime modules.
 * \param pipeline_config The dependent configuration of each runtime module.
 */
void PipelineScheduler::PipelineRunSequential(
    const std::vector<std::shared_ptr<BackendRuntime>>& runtimes,
    ConfigPipelineExecution pipeline_config) {
  for (size_t i = 0; i < runtimes.size(); i++) {
    // The "runtimes" is a list of runtime sorted by the runtime index which should be
    // contiguous ascend.
    if (static_cast<int>(i) != runtimes[i]->GetModuleIndex()) {
      LOG(FATAL) << "Runtime index " << runtimes[i]->GetModuleIndex()
                 << " is not as same as vector offset value " << i;
    }

    if (!pipeline_config.FindModuleInConfig(i)) {
      LOG(FATAL) << "Not find the configuration for the module " << i;
    }

    runtimes[i]->Run();
    // Getting the output then forwarding into other module once it is configured as input of
    // another module or storaging into the "output_array" when the output is a global one.
    int outputs_num = runtimes[i]->NumOutputs();
    for (int j = 0; j < outputs_num; j++) {
      ConfigBindings& out_binding = pipeline_config[i][j];
      std::unordered_map<int, std::string>& input_connections = out_binding.Get();
      NDArray output = runtimes[i]->GetOutput(j);
      for (auto bind : input_connections) {
        // "bind.first < 0" means the bind is a global bind, by pass the forwarding for
        // a global bind.
        if (bind.first < 0) continue;
        // Setting the output as an input data into the runtime module.
        runtimes[bind.first]->SetInput(bind.second, const_cast<DLTensor*>(output.operator->()));
      }
      // Store the output.
      if (out_binding.IsGlobalOutput()) {
        int global_idx = out_binding.GetGlobalOutputIndex();
        TVMArrayCopyFromTo(const_cast<DLTensor*>(output.operator->()),
                           const_cast<DLTensor*>(output_arrays_[global_idx].operator->()), nullptr);
      }
    }
  }
}
/*!
 * \brief Running pipeline logic.
 * \param runtimes A list of backend runtime modules.
 * \param pipeline_config The dependency configuration of each runtime module.
 * \param sequential_mode Whether the execution is in a sequential mode.
 */
void PipelineScheduler::PipelineRun(const std::vector<std::shared_ptr<BackendRuntime>>& runtimes,
                                    ConfigPipelineExecution pipeline_config, bool sequential_mode) {
  if (!sequential_mode) {
    runtimes.front()->RunPipeline();
  } else {
    PipelineRunSequential(runtimes, pipeline_config);
  }
}
/*!
 * \brief Get a list of output.
 */
Array<NDArray> PipelineScheduler::PipelineGetOutput() { return output_arrays_; }
}  // namespace runtime
}  // namespace tvm
