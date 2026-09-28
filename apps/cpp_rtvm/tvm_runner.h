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
 * \file tvm_runner.h
 * \brief TVM model runner.
 */
#ifndef TVM_APPS_CPP_RTVM_RUNNER_H_
#define TVM_APPS_CPP_RTVM_RUNNER_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/tensor.h>

#include <map>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief various meta information related to the compiled TVM model.
 */
typedef struct _TVMMetaInfo {
  int n_inputs{0};
  /*! \brief -1 when "main" returns a single tensor, else the tuple size. Set by Run(). */
  int n_outputs{-1};
  std::vector<std::string> param_names;
  std::map<std::string, std::pair<std::vector<int64_t>, std::string>> input_info;
  std::map<std::string, std::pair<std::vector<int64_t>, std::string>> output_info;
} TVMMetaInfo;

/*!
 * \brief encapsulates TVM graph runtime functionality with simplified API interface.
 */
class TVMRunner {
 public:
  /*! \brief Constructor */
  TVMRunner(std::string path, std::string device);

  /*! \brief Initiates graph runtime and with the compiled model */
  int Load(void);
  /*! \brief Specify if the run programs should be dumped to binary and reused in the next runs */
  void UsePreCompiledPrograms(std::string);
  /*! \brief Executes one inference cycle */
  int Run(void);
  /*! \brief To create NDArray inputs from given npz file */
  int CreateInputNDArrayFromFile(std::string);
  /*! \brief To set the input from NDArray */
  int SetInput();
  /*! \brief To set the input from binary data */
  int SetInput(int, char*);
  /*! \brief To get the output NDArray and save it npy files */
  int GetOutput(std::string);
  /*! \brief Get the model output in binary format */
  int GetOutput(int, char*);
  /*! \brief Get the model output in NDArray object */
  Tensor GetOutputNDArray(int);
  /*! \brief Populates various meta information from graph runtime */
  TVMMetaInfo GetMetaInfo(void);
  /*! \brief Print function to show all meta information */
  void PrintMetaInfo(void);

  /*! \brief Print function to show all stats information */
  void PrintStats(void);

  // Public profiling information
  /*! Module load time */
  double r_module_load_ms{0};
  /*! Graph runtime creatint time */
  int r_graph_load_ms{0};
  /*! Params read time */
  int r_param_read_ms{0};
  /*! Params load time */
  int r_param_load_ms{0};
  /*! Pre compiled programs load time */
  int r_pre_compiled_load_ms{0};
  std::vector<Tensor> inputs_;

 private:
  /*! \brief Module handle for the shared object */
  ffi::Optional<ffi::Module> r_mod_handle;
  /*! \brief Graph runtime module handle */
  ffi::Optional<ffi::Module> r_graph_handle;
  /*! \brief The local model path from where we load the model */
  std::string r_model_path;
  /*! \brief The target device */
  std::string r_device;
  /*! \brief Holds meta information queried from graph runtime */
  TVMMetaInfo mInfo;
  /*! \brief Mark if the run method was called */
  bool r_run_was_called;
};

DLDeviceType GetTVMDevice(std::string device);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RTVM_RUNNER_H_
