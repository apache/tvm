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

#include <fstream>
#include <iterator>
#include <streambuf>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Get the TVM device id corresponding to device string.
 * \param device the target device in string format.
 * \return dl_device corresponding to the device string.
 */
int GetTVMDevice(std::string device) {
  if (!device.compare("cpu")) {
    return static_cast<int>(kDLCPU);
  } else if (!device.compare("llvm")) {
    return static_cast<int>(kDLCPU);
  } else if (!device.compare("cuda")) {
    return static_cast<int>(kDLCUDA);
  } else if (!device.compare("opencl")) {
    return static_cast<int>(kDLOpenCL);
  } else if (!device.compare("vulkan")) {
    return static_cast<int>(kDLVulkan);
  } else if (!device.compare("metal")) {
    return static_cast<int>(kDLMetal);
  } else if (!device.compare("vpi")) {
    return static_cast<int>(kDLVPI);
  } else if (!device.compare("rocm")) {
    return static_cast<int>(kDLROCM);
  } else if (!device.compare("oneapi")) {
    return static_cast<int>(kDLOneAPI);
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
  r_mod_handle = Module::LoadFromFile((r_model_path + "/mod.so").c_str(), "so");

  // Read model json file
  std::ifstream json_reader((r_model_path + "/mod.json").c_str());
  CHECK(!json_reader.fail()) << "Failed to open json file:" << (r_model_path + "/mod.json").c_str();
  std::string json_str((std::istreambuf_iterator<char>(json_reader)),
                       std::istreambuf_iterator<char>());
  json_reader.close();

  // Get ref to graph exeutor
  auto f_handle = tvm::runtime::Registry::Get("tvm.graph_executor.create");

  // Greate graph runtime
  r_graph_handle = (*f_handle)(json_str, r_mod_handle, GetTVMDevice(r_device), 0);

  // Read params binary file
  std::ifstream params_reader((r_model_path + "/mod.params").c_str(), std::ios::binary);
  CHECK(!params_reader.fail()) << "Failed to open json file:"
                               << (r_model_path + "/mod.params").c_str();
  const std::string params_str((std::istreambuf_iterator<char>(params_reader)),
                               std::istreambuf_iterator<char>());
  params_reader.close();
  TVMByteArray params_arr;
  params_arr.data = params_str.c_str();
  params_arr.size = params_str.length();

  // Load parameters
  r_graph_handle.GetFunction("load_params")(params_arr);

  return 0;
}

/*!
 * \brief Specify if the run programs should be dumped to binary and reused in the next runs.
 * \param file_name File name where pre-compiled programs should be stored.
 */
void TVMRunner::UsePreCompiledPrograms(std::string file_name) {
  if (r_run_was_called) {
    LOG(INFO) << "TVMRunner UsePreCompiledPrograms: should be called before first run";
    return;
  }
  auto f_get = r_mod_handle->GetFunction("opencl.GetPreCompiledPrograms", true);
  auto f_set = r_mod_handle->GetFunction("opencl.SetPreCompiledPrograms", true);
  if (f_get != nullptr && f_set != nullptr) {
    std::ifstream ifs(file_name, std::ios::in | std::ios::binary);
    if (ifs.fail()) {
      auto bytes = String(f_get());
      std::ofstream fs(file_name, std::ofstream::binary);
      fs.write(bytes.c_str(), bytes.size());
    } else {
      std::string bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
      f_set(String(bytes));
    }
  }
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
  LOG(INFO) << "TVMRunner::GetInputMemSize:" << input_id;

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
  LOG(INFO) << "TVMRunner::GetOutputMemSize:" << output_id;

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
  LOG(INFO) << "TVMRunner::SetInput (Raw)";
  NDArray in_arr = r_graph_handle.GetFunction("get_input")(input_id);
  auto ssize = GetMemSize(in_arr);
  in_arr.CopyFromBytes(raw_input, ssize);
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
  LOG(INFO) << "TVMRunner::GetOutput (Raw)";
  NDArray out_arr = r_graph_handle.GetFunction("get_output")(output_id);
  auto ssize = GetMemSize(out_arr);
  out_arr.CopyToBytes(raw_output, ssize);
  return 0;
}

/*!
 * \brief Call one cycle of execution for the model.
 * \param 0 on success else error code.
 */
int TVMRunner::Run(void) {
  LOG(INFO) << "TVMRunner::Run";
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
    std::vector<int> vshape;
    vshape.assign(stuple.begin(), stuple.end());
    auto dtype = GetRef<String>(dtype_info[kv.first].as<StringObj>());
    std::pair<std::vector<int>, std::string> value = std::make_pair(vshape, dtype);
    mInfo.input_info.insert({kv.first, value});
  }

  tvm_input_info = r_graph_handle.GetFunction("get_output_info")();
  shape_info = GetRef<Map<String, ObjectRef>>(tvm_input_info["shape"].as<MapNode>());
  dtype_info = GetRef<Map<String, ObjectRef>>(tvm_input_info["dtype"].as<MapNode>());
  for (const auto& kv : shape_info) {
    auto stuple = GetRef<ShapeTuple>(kv.second.as<ShapeTupleObj>());
    std::vector<int> vshape;
    vshape.assign(stuple.begin(), stuple.end());
    auto dtype = GetRef<String>(dtype_info[kv.first].as<StringObj>());
    std::pair<std::vector<int>, std::string> value = std::make_pair(vshape, dtype);
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

}  // namespace runtime
}  // namespace tvm
