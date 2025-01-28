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
 * \file tvm_runner.cc
 * \brief TVM model runner implementation.
 */

#include "tvm_runner.h"

#include <cnpy.h>

#include <chrono>
#include <fstream>
#include <iterator>
#include <streambuf>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Get the TVM device id corresponding to device string.
 * \param device the target device in string format.
 * \return dl_device corresponding to the device string.
 */
DLDeviceType GetTVMDevice(std::string device) {
  if (!device.compare("cpu")) {
    return kDLCPU;
  } else if (!device.compare("llvm")) {
    return kDLCPU;
  } else if (!device.compare("cuda")) {
    return kDLCUDA;
  } else if (!device.compare("opencl")) {
    return kDLOpenCL;
  } else if (!device.compare("vulkan")) {
    return kDLVulkan;
  } else if (!device.compare("metal")) {
    return kDLMetal;
  } else if (!device.compare("vpi")) {
    return kDLVPI;
  } else if (!device.compare("rocm")) {
    return kDLROCM;
  } else if (!device.compare("oneapi")) {
    return kDLOneAPI;
  } else {
    LOG(FATAL) << "TVMRunner : Unsupported device :" << device;
  }
}

/*!
 * \brief Constructor for TVMRunner.
 * \param path where the tfm compiler artifacts present.
 * \param device the target device where we need to load the compiled model.
 */
TVMRunner::TVMRunner(std::string path, std::string device)
    : r_model_path(path), r_device(device), r_run_was_called(false) {
  LOG(INFO) << "TVMRunner Constructor:" << r_model_path << " Devices:" << r_device;
}

/*!
 * \brief Load Setup TVM graph runtime for given model.
 * \param 0 on success else error code.
 */
int TVMRunner::Load(void) {
  LOG(INFO) << "TVMRunner Load:" << r_model_path;
  // Load the lib file
  auto tstart = std::chrono::high_resolution_clock::now();

  r_mod_handle = Module::LoadFromFile((r_model_path + "/mod.so").c_str(), "so");
  auto tend = std::chrono::high_resolution_clock::now();
  r_module_load_ms = static_cast<double>((tend - tstart).count()) / 1e6;

  tstart = std::chrono::high_resolution_clock::now();
  // Read model json file
  std::ifstream json_reader((r_model_path + "/mod.json").c_str());
  CHECK(!json_reader.fail()) << "Failed to open json file:" << (r_model_path + "/mod.json").c_str();
  json_reader.seekg(0, std::ios_base::end);
  std::size_t json_size = json_reader.tellg();
  json_reader.seekg(0, std::ios_base::beg);
  std::string json_data;
  json_data.reserve(json_size);
  json_reader.read((char*)json_data.c_str(), json_size);
  json_reader.close();

  // Get ref to graph exeutor
  auto f_handle = tvm::runtime::Registry::Get("tvm.graph_executor.create");

  // Greate graph runtime
  r_graph_handle =
      (*f_handle)(json_data, r_mod_handle, static_cast<int>(GetTVMDevice(r_device)), 0);

  tend = std::chrono::high_resolution_clock::now();
  r_graph_load_ms = static_cast<double>((tend - tstart).count()) / 1e6;

  // Read params binary file
  tstart = std::chrono::high_resolution_clock::now();
  std::ifstream params_reader((r_model_path + "/mod.params").c_str(), std::ios::binary);
  CHECK(!params_reader.fail()) << "Failed to open json file:"
                               << (r_model_path + "/mod.params").c_str();

  params_reader.seekg(0, std::ios_base::end);
  std::size_t param_size = params_reader.tellg();
  params_reader.seekg(0, std::ios_base::beg);
  std::vector<char> param_data(param_size / sizeof(char));
  params_reader.read((char*)&param_data[0], param_size);
  params_reader.close();

  TVMByteArray params_arr;
  params_arr.data = (char*)&param_data[0];
  params_arr.size = param_size;

  tend = std::chrono::high_resolution_clock::now();
  r_param_read_ms = static_cast<double>((tend - tstart).count()) / 1e6;

  // Load parameters
  tstart = std::chrono::high_resolution_clock::now();
  r_graph_handle.GetFunction("load_params")(params_arr);
  tend = std::chrono::high_resolution_clock::now();
  r_param_load_ms = static_cast<double>((tend - tstart).count()) / 1e6;

  return 0;
}

/*!
 * \brief Specify if the run programs should be dumped to binary and reused in the next runs.
 * \param file_name File name where pre-compiled programs should be stored.
 */
void TVMRunner::UsePreCompiledPrograms(std::string file_name) {
  auto tstart = std::chrono::high_resolution_clock::now();
  if (r_run_was_called) {
    LOG(INFO) << "TVMRunner UsePreCompiledPrograms: should be called before first run";
    return;
  }
  auto f_get = r_mod_handle->GetFunction("opencl.GetPreCompiledPrograms", true);
  auto f_set = r_mod_handle->GetFunction("opencl.SetPreCompiledPrograms", true);
  if (f_get != nullptr && f_set != nullptr) {
    std::ifstream ifs(file_name, std::ios::in | std::ios::binary);
    if (ifs.fail()) {
      std::string ss = f_get();
      auto bytes = tvm::String(ss);
      std::ofstream fs(file_name, std::ofstream::binary);
      fs.write(bytes.c_str(), bytes.size());
    } else {
      ifs.seekg(0, std::ios_base::end);
      std::size_t blob_size = ifs.tellg();
      ifs.seekg(0, std::ios_base::beg);
      std::string blob_data;
      blob_data.reserve(blob_size);
      blob_data.resize(blob_size);
      ifs.read((char*)blob_data.c_str(), blob_size);
      ifs.close();
      f_set(String(blob_data));
    }
  }
  auto tend = std::chrono::high_resolution_clock::now();
  r_pre_compiled_load_ms = static_cast<double>((tend - tstart).count()) / 1e6;
}

/*!
 * \brief Calculated the memory size for the NDArray.
 * \param NDArray object.
 * \return size of the memory.
 */
inline size_t GetMemSize(NDArray& narr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < narr->ndim; ++i) {
    size *= static_cast<size_t>(narr->shape[i]);
  }
  size *= (narr->dtype.bits * narr->dtype.lanes + 7) / 8;
  return size;
}

/*!
 * \brief Get the input alloc mem size.
 * \param input_id The input id to query the mem size.
 * \return The memory size.
 */
size_t TVMRunner::GetInputMemSize(std::string input_id) {
  NDArray in_arr = r_graph_handle.GetFunction("get_input")(input_id);
  auto ssize = GetMemSize(in_arr);

  return ssize;
}

/*!
 * \brief Get the output alloc mem size.
 * \param output_id The output id to query the mem size.
 * \return The memory size.
 */
size_t TVMRunner::GetOutputMemSize(std::string output_id) {
  NDArray out_arr = r_graph_handle.GetFunction("get_output")(output_id);
  auto ssize = GetMemSize(out_arr);

  return ssize;
}

/*!
 * \brief Set the model inputs from npz file.
 * \param inputfile the npz file from where we read input tensor data.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput(std::string inputfile) {
  LOG(INFO) << "TVMRunner::SetInput (Numpy):" << inputfile;
  cnpy::npz_t npz_input = cnpy::npz_load(inputfile);

  for (auto& elem : mInfo.input_info) {
    LOG(INFO) << "Set Numpy Input for :" << elem.first;
    NDArray in_arr = r_graph_handle.GetFunction("get_input")(elem.first);
    auto ssize = GetMemSize(in_arr);

    if (npz_input.find(elem.first) != npz_input.end()) {
      in_arr.CopyFromBytes(npz_input[elem.first].data<char>(), ssize);
    } else {
      LOG(WARNING) << "Couldn't find input " << elem.first << " in npy input file";
    }
  }

  return 0;
}

/*!
 * \brief Set the model input from the given binary buffer.
 * \param input_id input node name.
 * \param raw_input binary input buffer to copy over input NDArray.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput(std::string input_id, char* raw_input) {
  NDArray in_arr = r_graph_handle.GetFunction("get_input")(input_id);
  auto ssize = GetMemSize(in_arr);
  in_arr.CopyFromBytes(raw_input, ssize);
  return 0;
}

/*!
 * \brief Set the model input from given NDArray with zero copy.
 * \param input_id input node name.
 * \param ndarr NDArray.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput(std::string input_id, NDArray& ndarr) {
  r_graph_handle.GetFunction("set_input_zero_copy")(input_id, ndarr);
  return 0;
}

/*!
 * \brief Get the model outputs and dump them to npz file.
 * \param outputfile the npz file to where we dump the output data.
 * \param 0 on success else error code.
 */
int TVMRunner::GetOutput(std::string outputfile) {
  LOG(INFO) << "TVMRunner::GetOutput (Numpy):" << outputfile;

  for (auto& elem : mInfo.output_info) {
    LOG(INFO) << "Get Output for :" << elem.first;
    NDArray out_arr = r_graph_handle.GetFunction("get_output")(elem.first);
    auto ssize = GetMemSize(out_arr);
    LOG(INFO) << "Output Size:" << ssize << "  bytes";

    void* data = (void*)malloc(ssize * (out_arr->dtype.bits * out_arr->dtype.lanes + 7) / 8);
    out_arr.CopyToBytes(data, ssize);
    std::vector<size_t> shape;

    for (int j = 0; j < out_arr->ndim; ++j) shape.push_back(out_arr->shape[j]);
    if (!elem.second.second.compare("float32")) {
      cnpy::npz_save<float>(outputfile, elem.first, (float*)data, shape, "a");
    } else if (!elem.second.second.compare("int8")) {
      cnpy::npz_save<int8_t>(outputfile, elem.first, (int8_t*)data, shape, "a");
    } else {
      LOG(WARNING) << "DType:" << elem.second.second << " is not supported for npy_save";
    }
    free(data);
  }

  return 0;
}

/*!
 * \brief Get output of the model as a binary buffer.
 * \param output_id output node name to read the data.
 * \param raw_output the buffer to copy the data to.
 * \param 0 on success else error code.
 */
int TVMRunner::GetOutput(std::string output_id, char* raw_output) {
  NDArray out_arr = r_graph_handle.GetFunction("get_output")(output_id);
  auto ssize = GetMemSize(out_arr);
  out_arr.CopyToBytes(raw_output, ssize);
  return 0;
}

/*!
 * \brief Set the model output from given NDArray with zero copy.
 * \param output_id output node name.
 * \param ndarr NDArray.
 * \param 0 on success else error code.
 */
int TVMRunner::SetOutput(std::string output_id, NDArray& ndarr) {
  r_graph_handle.GetFunction("set_output_zero_copy")(output_id, ndarr);
  return 0;
}

/*!
 * \brief Call one cycle of execution for the model.
 * \param 0 on success else error code.
 */
int TVMRunner::Run(void) {
  r_run_was_called = true;
  r_graph_handle.GetFunction("run")();
  return 0;
}

/*!
 * \brief Query various metadata from the grsph runtime.
 * \param 0 on success else error code.
 */
TVMMetaInfo TVMRunner::GetMetaInfo(void) {
  LOG(INFO) << "TVMRunner::GetMetaInfo";

  mInfo.n_inputs = r_graph_handle.GetFunction("get_num_inputs")();
  mInfo.n_outputs = r_graph_handle.GetFunction("get_num_outputs")();

  Map<String, ObjectRef> tvm_input_info = r_graph_handle.GetFunction("get_input_info")();
  auto shape_info = GetRef<Map<String, ObjectRef>>(tvm_input_info["shape"].as<MapNode>());
  auto dtype_info = GetRef<Map<String, ObjectRef>>(tvm_input_info["dtype"].as<MapNode>());
  for (const auto& kv : shape_info) {
    auto stuple = GetRef<ShapeTuple>(kv.second.as<ShapeTupleObj>());
    std::vector<int64_t> vshape;
    vshape.assign(stuple.begin(), stuple.end());
    auto dtype = GetRef<String>(dtype_info[kv.first].as<StringObj>());
    std::pair<std::vector<int64_t>, std::string> value = std::make_pair(vshape, dtype);
    mInfo.input_info.insert({kv.first, value});
  }

  tvm_input_info = r_graph_handle.GetFunction("get_output_info")();
  shape_info = GetRef<Map<String, ObjectRef>>(tvm_input_info["shape"].as<MapNode>());
  dtype_info = GetRef<Map<String, ObjectRef>>(tvm_input_info["dtype"].as<MapNode>());
  for (const auto& kv : shape_info) {
    auto stuple = GetRef<ShapeTuple>(kv.second.as<ShapeTupleObj>());
    std::vector<int64_t> vshape;
    vshape.assign(stuple.begin(), stuple.end());
    auto dtype = GetRef<String>(dtype_info[kv.first].as<StringObj>());
    std::pair<std::vector<int64_t>, std::string> value = std::make_pair(vshape, dtype);
    mInfo.output_info.insert({kv.first, value});
  }

  return mInfo;
}

/*!
 * \brief Print the meta information.
 * \param 0 on success else error code.
 */
void TVMRunner::PrintMetaInfo(void) {
  LOG(INFO) << "Meta Information:" << r_model_path;
  LOG(INFO) << "    Number of Inputs:" << mInfo.n_inputs;
  LOG(INFO) << "    Number of Outputs:" << mInfo.n_outputs;
  LOG(INFO) << "    Input MetaInfo:";
  for (auto& elem : mInfo.input_info) {
    std::ostringstream stream;
    stream << "[";
    copy(elem.second.first.begin(), elem.second.first.end() - 1,
         std::ostream_iterator<int>(stream, ", "));
    stream << elem.second.first.back() << "]";
    LOG(INFO) << "        Input:" << elem.first;
    LOG(INFO) << "            DType:" << elem.second.second;
    LOG(INFO) << "            Shape:" << stream.str();
  }
  LOG(INFO) << "    Output MetaInfo:";
  for (auto& elem : mInfo.output_info) {
    std::ostringstream stream;
    stream << "[";
    copy(elem.second.first.begin(), elem.second.first.end() - 1,
         std::ostream_iterator<int>(stream, ", "));
    stream << elem.second.first.back() << "]";
    LOG(INFO) << "        Output:" << elem.first;
    LOG(INFO) << "            DType:" << elem.second.second;
    LOG(INFO) << "            Shape:" << stream.str();
  }
}

/*!
 * \brief Print stats information.
 */
void TVMRunner::PrintStats(void) {
  LOG(INFO) << "Performance Stats:" << r_model_path;
  LOG(INFO) << "    Module Load              :" << r_module_load_ms << " ms";
  LOG(INFO) << "    Graph Runtime Create     :" << r_graph_load_ms << " ms";
  LOG(INFO) << "    Params Read              :" << r_param_read_ms << " ms";
  LOG(INFO) << "    Params Set               :" << r_param_load_ms << " ms";
  LOG(INFO) << "    Pre Compiled Progs Load  :" << r_pre_compiled_load_ms << " ms";
  LOG(INFO) << "Total Load Time     :"
            << r_module_load_ms + r_graph_load_ms + r_param_read_ms + r_param_load_ms +
                   r_pre_compiled_load_ms
            << " ms";
}

}  // namespace runtime
}  // namespace tvm
