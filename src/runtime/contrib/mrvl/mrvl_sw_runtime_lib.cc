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
 * \file src/runtime/contrib/mrvl/mrvl_sw_runtime_lib.cc
 * \brief Runtime library for Marvell Software Simulator.
 */

#include "mrvl_sw_runtime_lib.h"

#include <assert.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <vector>

#include "mrvl_base64.h"

using namespace tvm::runtime;

template <typename T>
static void NDArrayToFile(const tvm::runtime::NDArray& arr, std::ostream& os) {
  int ndim = arr->ndim;
  int tot_dim = 1;
  for (int i = 0; i < ndim; i++) {
    tot_dim *= arr->shape[i];
  }
  T* data_ptr = reinterpret_cast<T*>(arr->data);
  os << "\t\t[";
  os << std::endl;
  for (int i = 0; i < tot_dim; i++) {
    os << "\t\t\t" << std::setprecision(10) << data_ptr[i] << (i != tot_dim - 1 ? "," : "");
    os << std::endl;
  }
  os << "\t\t]";
}

static void WriteBinToDisk(const std::string& bin_file, const std::string& bin_code) {
  auto length = tvm::runtime::contrib::mrvl::b64strlen(bin_code);
  std::vector<unsigned char> byte_array(length);
  tvm::runtime::contrib::mrvl::b64decode(bin_code, byte_array.data());
  std::ofstream file_out;
  file_out.open(bin_file, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
  for (auto byte : byte_array) file_out << byte;
}

static void ReadInputsAndGenerateInputBin(TVMArgs args, const std::string& input_json,
                                          const std::string& input_bin,
                                          const std::string& bin_directory, size_t num_inputs) {
  std::ofstream file_out;
  file_out.open(input_json, std::ios_base::out | std::ios_base::trunc);
  file_out << "{" << std::endl;
  file_out << R"(    "inputs": [)" << std::endl;
  for (size_t i = 0; i < num_inputs; ++i) {
    const DLTensor* tensor;
    if (args[i].IsObjectRef<NDArray>()) {
      NDArray arr = args[i];
      tensor = arr.operator->();
    } else {
      tensor = args[i].operator DLTensor*();
    }
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < tensor->ndim; i++) {
      shape.push_back(tensor->shape[i]);
    }
    NDArray arr = NDArray::Empty(shape, tensor->dtype, tensor->device);
    arr.CopyFrom(tensor);
    NDArrayToFile<float>(arr, file_out);
    if (i != num_inputs - 1) {
      file_out << std::endl << "\t," << std::endl;
    }
  }
  file_out << std::endl << "\t]" << std::endl;
  file_out << "}" << std::endl;

  const auto* json_to_bin = tvm::runtime::Registry::Get("tvm.mrvl.JsonToBin");
  (*json_to_bin)(input_json, input_bin);
}

static void RunInferenceOnMlModel(const std::string& symbol_name, const std::string& bin_directory,
                                  const std::string& bin_file, const std::string& input_bin,
                                  const std::string& out_bin_prefix) {
  auto command = bin_directory + "/mrvl-mlsim " + "-m " + bin_file + " -d " + input_bin + " -o " +
                 out_bin_prefix;
  std::string sim_directory = "mrvl_sw_sim_" + symbol_name;
  const auto* run_sim = tvm::runtime::Registry::Get("tvm.mrvl.RunSim");
  (*run_sim)(command, sim_directory);
}

static void ReadOutputsAndUpdateRuntime(TVMArgs args, size_t num_inputs,
                                        const std::string& out_bin_prefix) {
  for (int out = num_inputs; out < args.size(); out++) {
    const DLTensor* outTensor;
    if (args[out].IsObjectRef<NDArray>()) {
      NDArray arr = args[out];
      outTensor = arr.operator->();
    } else {
      outTensor = args[out].operator DLTensor*();
    }
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < outTensor->ndim; i++) {
      shape.push_back(outTensor->shape[i]);
    }
    NDArray arr = NDArray::Empty(shape, outTensor->dtype, outTensor->device);
    int ndim = arr->ndim;
    int tot_dim = 1;
    for (int i = 0; i < ndim; i++) {
      tot_dim *= arr->shape[i];
    }
    float f;
    float* data = new float[tot_dim]();
    String outbin = out_bin_prefix + "-" + std::to_string(out - num_inputs) + ".bin";
    std::ifstream fin(outbin, std::ios::binary);
    ICHECK(fin.is_open()) << "Cannot open file: " << outbin;
    int i = 0;
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
      data[i] = f;
      ICHECK(i < tot_dim) << "Output data size mismatch";
      i++;
    }
    arr.CopyFromBytes(data, tot_dim * sizeof(float));
    arr.CopyTo(const_cast<DLTensor*>(outTensor));
    delete[] data;
  }
}

static void CleanUp(TVMArgs args, const std::string& bin_file, const std::string& input_json,
                    const std::string& input_bin, const std::string& out_bin_prefix,
                    size_t num_outputs) {
  const auto* clean_up = tvm::runtime::Registry::Get("tvm.mrvl.CleanUpSim");
  (*clean_up)(bin_file, input_json, input_bin, out_bin_prefix, num_outputs);
}

void tvm::runtime::contrib::mrvl::RunMarvellSimulator(TVMArgs args, const std::string& symbol_name,
                                                      const std::string& bin_code,
                                                      size_t num_inputs, size_t num_outputs) {
  // check $PATH for the presence of MRVL dependent tools/scripts
  std::string file_name("mrvl-mlsim");
  const auto* search_path = tvm::runtime::Registry::Get("tvm.mrvl.SearchPath");
  std::string tools_directory = (*search_path)(file_name);
  if (tools_directory.empty()) {
    ICHECK(false) << "mrvl-mlsim simulator not found! Please specify the path to Marvell "
                     "tools by adding it to $PATH.";
  }

  const auto* temp_dir = tvm::runtime::Registry::Get("tvm.mrvl.TempDir");
  std::string working_directory = (*temp_dir)();
  auto bin_file = working_directory + "/" + symbol_name + ".bin";
  auto input_json = working_directory + "/indata.json";
  auto input_bin = working_directory + "/input.bin";
  auto out_bin_prefix = working_directory + "/mrvl_sim_out";

  WriteBinToDisk(bin_file, bin_code);
  ReadInputsAndGenerateInputBin(args, input_json, input_bin, tools_directory, num_inputs);
  RunInferenceOnMlModel(symbol_name, tools_directory, bin_file, input_bin, out_bin_prefix);
  ReadOutputsAndUpdateRuntime(args, num_inputs, out_bin_prefix);
  CleanUp(args, bin_file, input_json, input_bin, out_bin_prefix, num_outputs);
}
