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

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_map>
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
  TVM_FFI_UNREACHABLE();
}

/*! \brief Contents of a numpy .npy file. */
struct NpyArray {
  std::vector<int64_t> shape;
  DLDataType dtype;
  std::vector<char> data;
};

/*!
 * \brief Get the value text following a key in a npy header dict.
 * \param header The npy header dict, e.g. "{'descr': '<f4', 'shape': (1, 3), }".
 * \param key The dict key.
 * \return The header text starting right after "key:".
 */
static std::string NpyHeaderValue(const std::string& header, const std::string& key) {
  size_t pos = header.find("'" + key + "'");
  TVM_FFI_ICHECK(pos != std::string::npos) << "npy header has no '" << key << "': " << header;
  pos = header.find(':', pos);
  TVM_FFI_ICHECK(pos != std::string::npos) << "Malformed npy header: " << header;
  return header.substr(pos + 1);
}

/*!
 * \brief Load a numpy .npy file (format version 1.x - 3.x, little endian, C order).
 * \param fname Numpy file name.
 * \return The loaded array.
 */
NpyArray LoadNpy(const std::string& fname) {
  std::ifstream fs(fname, std::ios::binary);
  TVM_FFI_ICHECK(fs) << "Unable to open npy file " << fname;
  char magic[8];
  fs.read(magic, sizeof(magic));
  TVM_FFI_ICHECK(fs && std::memcmp(magic, "\x93NUMPY", 6) == 0) << "Not a npy file: " << fname;
  unsigned char len_bytes[4] = {0, 0, 0, 0};
  fs.read(reinterpret_cast<char*>(len_bytes), magic[6] == 1 ? 2 : 4);
  uint32_t header_len = len_bytes[0] | (len_bytes[1] << 8) | (len_bytes[2] << 16) |
                        (static_cast<uint32_t>(len_bytes[3]) << 24);
  std::string header(header_len, ' ');
  fs.read(header.data(), header_len);
  TVM_FFI_ICHECK(fs) << "Truncated npy header: " << fname;

  // descr, e.g. '<f4', '|u1', '|b1'
  std::string descr = NpyHeaderValue(header, "descr");
  size_t q0 = descr.find('\'');
  size_t q1 = descr.find('\'', q0 + 1);
  descr = descr.substr(q0 + 1, q1 - q0 - 1);
  TVM_FFI_ICHECK(descr.size() >= 3 && descr[0] != '>')
      << "Unsupported npy dtype '" << descr << "' in " << fname << " (big endian?)";
  int nbytes = std::stoi(descr.substr(2));
  std::string dtype;
  switch (descr[1]) {
    case 'f':
      dtype = "float" + std::to_string(nbytes * 8);
      break;
    case 'i':
      dtype = "int" + std::to_string(nbytes * 8);
      break;
    case 'u':
      dtype = "uint" + std::to_string(nbytes * 8);
      break;
    case 'b':
      dtype = "bool";
      break;
    default:
      LOG(FATAL) << "Unsupported npy dtype '" << descr << "' in " << fname;
  }

  std::string fortran_order = NpyHeaderValue(header, "fortran_order");
  TVM_FFI_ICHECK(fortran_order.find("False") < fortran_order.find(','))
      << "Fortran ordered npy files are not supported: " << fname;

  std::string shape_str = NpyHeaderValue(header, "shape");
  shape_str = shape_str.substr(shape_str.find('(') + 1);
  shape_str = shape_str.substr(0, shape_str.find(')'));
  std::replace(shape_str.begin(), shape_str.end(), ',', ' ');

  NpyArray arr;
  arr.dtype = ffi::StringToDLDataType(dtype);
  std::istringstream shape_ss(shape_str);
  int64_t dim;
  size_t numel = 1;
  while (shape_ss >> dim) {
    arr.shape.push_back(dim);
    numel *= static_cast<size_t>(dim);
  }
  arr.data.resize(numel * nbytes);
  fs.read(arr.data.data(), arr.data.size());
  TVM_FFI_ICHECK(fs) << "Truncated npy data: " << fname;
  return arr;
}

/*!
 * \brief Save raw tensor data as a numpy .npy file (format version 1.0).
 * \param fname Numpy file name.
 * \param dtype Data type of the tensor.
 * \param shape Shape of the tensor.
 * \param data Raw tensor bytes.
 * \return false if dtype can not be represented in npy.
 */
bool SaveNpy(const std::string& fname, DLDataType dtype, const std::vector<int64_t>& shape,
             const std::vector<char>& data) {
  char kind;
  switch (dtype.code) {
    case kDLFloat:
      kind = 'f';
      break;
    case kDLInt:
      kind = 'i';
      break;
    case kDLUInt:
      kind = 'u';
      break;
    case kDLBool:
      kind = 'b';
      break;
    default:
      return false;
  }
  if (dtype.lanes != 1 || dtype.bits % 8 != 0) return false;
  int nbytes = dtype.bits / 8;

  std::ostringstream header;
  header << "{'descr': '" << (nbytes == 1 ? '|' : '<') << kind << nbytes
         << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) header << shape[i] << ", ";
  header << "), }";
  std::string header_str = header.str();
  // Pad with spaces so that magic(6) + version(2) + len(2) + header ends with '\n' on 64 bytes.
  size_t total = 10 + header_str.size() + 1;
  header_str.append((64 - total % 64) % 64, ' ');
  header_str.push_back('\n');

  std::ofstream fs(fname, std::ios::binary);
  TVM_FFI_ICHECK(fs) << "Unable to create npy file " << fname;
  uint16_t header_len = static_cast<uint16_t>(header_str.size());
  const char len_bytes[2] = {static_cast<char>(header_len & 0xff),
                             static_cast<char>(header_len >> 8)};
  fs.write("\x93NUMPY\x01\x00", 8);
  fs.write(len_bytes, 2);
  fs.write(header_str.data(), header_str.size());
  fs.write(data.data(), data.size());
  TVM_FFI_ICHECK(fs) << "Failed writing npy file " << fname;
  return true;
}

// Function to trim whitespace from a string
std::string trim(const std::string& str) {
  size_t first = str.find_first_not_of(' ');
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, last - first + 1);
}

// Function to parse a JSON object
std::unordered_map<std::string, std::string> parse_json(const std::string& json_str) {
  std::unordered_map<std::string, std::string> json_map;
  std::istringstream ss(json_str);
  std::string line;

  while (std::getline(ss, line, ',')) {
    size_t colon_pos = line.find(':');
    if (colon_pos != std::string::npos) {
      std::string key = trim(line.substr(0, colon_pos));
      std::string value = trim(line.substr(colon_pos + 1));
      key.erase(remove(key.begin(), key.end(), '\"'), key.end());
      value.erase(remove(value.begin(), value.end(), '\"'), value.end());
      json_map[key] = value;
    }
  }

  return json_map;
}

/*!
 * \brief Calculated the memory size for the NDArray.
 * \param NDArray object.
 * \return size of the memory.
 */
inline size_t GetMemSize(Tensor& narr) {
  size_t size = 1;
  for (int64_t i = 0; i < narr->ndim; ++i) {
    size *= static_cast<size_t>(narr->shape[i]);
  }
  size *= (narr->dtype.bits * narr->dtype.lanes + 7) / 8;
  return size;
}

/*!
 * \brief Save Output Tensor to npy output file.
 * \param Tensor object.
 * \param index output index id.
 * \param fname output folder name,under that it will saving as index.npy.
 * \return 0 on success else error code.
 */
int SaveNDArrayToNpyFile(Tensor& nd_arr, int index, std::string fname) {
  auto ssize = GetMemSize(nd_arr);
  LOG(INFO) << "Output Size:" << ssize << "  bytes";

  std::vector<char> data(ssize);
  nd_arr.CopyToBytes(data.data(), ssize);
  std::vector<int64_t> shape(nd_arr->shape, nd_arr->shape + nd_arr->ndim);
  if (!SaveNpy(fname + "/" + std::to_string(index) + ".npy", nd_arr->dtype, shape, data)) {
    LOG(WARNING) << "DType:" << ffi::DLDataTypeToString(nd_arr->dtype)
                 << " is not supported for npy save";
  }
  return 0;
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
  LOG(INFO) << "TVMRunner Load:" << (r_model_path).c_str();
  // Load the lib file
  auto tstart = std::chrono::high_resolution_clock::now();

  ffi::Module executable = ffi::Module::LoadFromFile((r_model_path).c_str());
  auto fload_exec = executable->GetFunction("vm_load_executable");
  TVM_FFI_ICHECK(fload_exec.has_value()) << "TVM runtime cannot find vm_load_executable";
  r_graph_handle = (*fload_exec)().cast<ffi::Module>();
  // Get ref to graph executor
  (*r_graph_handle)
      ->GetFunction("vm_initialization")
      .value()(static_cast<int>(GetTVMDevice(r_device)), 0,
               static_cast<int>(memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
               static_cast<int>(memory::AllocatorType::kPooled));
  auto tend = std::chrono::high_resolution_clock::now();
  r_module_load_ms = std::chrono::duration<double, std::milli>(tend - tstart).count();

  return 0;
}

/*!
 * \brief Create model inputs NDarray from npy file.
 * \param inputfile the npy file from where we read input tensor data.
 * \param 0 on success else error code.
 */
int TVMRunner::CreateInputNDArrayFromFile(std::string inputfile) {
  LOG(INFO) << "TVMRunner::SetInput (Numpy):" << inputfile;
  for (int i = 0; i < mInfo.n_inputs; i++) {
    std::string param_name = (*r_graph_handle)
                                 ->GetFunction("get_function_param_name")
                                 .value()("main", i)
                                 .cast<std::string>();
    NpyArray npy_arry = LoadNpy(inputfile + "/" + param_name + ".npy");
    if (inputs_.size() <= i) inputs_.resize(i + 1);
    inputs_[i] = Tensor::Empty(ffi::Shape(npy_arry.shape.begin(), npy_arry.shape.end()),
                               npy_arry.dtype, DLDevice{GetTVMDevice(r_device), 0});
    auto ssize = GetMemSize(inputs_[i]);
    TVM_FFI_ICHECK_EQ(ssize, npy_arry.data.size());
    inputs_[i].CopyFromBytes(npy_arry.data.data(), ssize);
  }
  return 0;
}

/*!
 * \brief Set the model input from the given binary buffer.
 * \param input_id input node name.
 * \param raw_input binary input buffer to copy over input NDArray.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput(int index, char* raw_input) {
  if (inputs_.size() > index) {
    auto ssize = GetMemSize(inputs_[index]);
    inputs_[index].CopyFromBytes(raw_input, ssize);
  } else {
    LOG(FATAL) << "Input NDArray not created";
  }
  return 0;
}

/*!
 * \brief Set the model input from given NDArray with zero copy.
 * \param 0 on success else error code.
 */
int TVMRunner::SetInput() {
  // The Relax VM "set_input" function takes all the inputs for the function
  // in a single call: (func_name, input0, input1, ...).
  std::vector<ffi::AnyView> args;
  args.reserve(inputs_.size() + 1);
  args.emplace_back("main");
  for (int i = 0; i < inputs_.size(); i++) {
    args.emplace_back(inputs_[i]);
  }
  ffi::Any rv;
  (*r_graph_handle)
      ->GetFunction("set_input")
      .value()
      .CallPacked(ffi::PackedArgs(args.data(), static_cast<int32_t>(args.size())), &rv);
  return 0;
}

/*!
 * \brief Get the model outputs and dump them to npz file.
 * \param outputfile the npz file to where we dump the output data.
 * \param 0 on success else error code.
 */
int TVMRunner::GetOutput(std::string outputfile) {
  LOG(INFO) << "TVMRunner::GetOutput (Numpy):" << outputfile;

  // Check if the directory already exists otherwise create
  if (!std::filesystem::exists(outputfile)) std::filesystem::create_directory(outputfile);
  if (mInfo.n_outputs == -1) {
    Tensor out_arr = (*r_graph_handle)->GetFunction("get_output").value()("main").cast<Tensor>();
    SaveNDArrayToNpyFile(out_arr, 0, outputfile);
  } else {
    for (int i = 0; i < mInfo.n_outputs; i++) {
      Tensor out_arr =
          (*r_graph_handle)->GetFunction("get_output").value()("main", i).cast<Tensor>();
      SaveNDArrayToNpyFile(out_arr, i, outputfile);
    }
  }
  return 0;
}

/*!
 * \brief Get output of the model as a binary buffer.
 * \param output_id output node name to read the data.
 * \param raw_output the buffer to copy the data to.
 * \param 0 on success else error code.
 */
int TVMRunner::GetOutput(int index, char* raw_output) {
  if (mInfo.n_outputs == -1) {
    Tensor out_arr = (*r_graph_handle)->GetFunction("get_output").value()("main").cast<Tensor>();
    auto ssize = GetMemSize(out_arr);
    out_arr.CopyToBytes(raw_output, ssize);
  } else {
    Tensor out_arr =
        (*r_graph_handle)->GetFunction("get_output").value()("main", index).cast<Tensor>();
    auto ssize = GetMemSize(out_arr);
    out_arr.CopyToBytes(raw_output, ssize);
  }
  return 0;
}

/*!
 * \brief Get output of the model as a binary buffer.
 * \param index output node id to read the data.
 * \return output Tensor.
 */
Tensor TVMRunner::GetOutputNDArray(int index) {
  if (mInfo.n_outputs == -1) {
    return (*r_graph_handle)->GetFunction("get_output").value()("main").cast<Tensor>();
  } else {
    return (*r_graph_handle)->GetFunction("get_output").value()("main", index).cast<Tensor>();
  }
}

/*!
 * \brief Call one cycle of execution for the model.
 * \param 0 on success else error code.
 */
int TVMRunner::Run(void) {
  (*r_graph_handle)->GetFunction("invoke_stateful").value()("main");
  if (!r_run_was_called) {
    mInfo.n_outputs =
        (*r_graph_handle)->GetFunction("get_output_arity").value()("main").cast<int>();
    r_run_was_called = true;
  }
  return 0;
}

/*!
 * \brief Query various metadata from the graph runtime.
 * \param 0 on success else error code.
 */
TVMMetaInfo TVMRunner::GetMetaInfo(void) {
  LOG(INFO) << "TVMRunner::GetMetaInfo";
  mInfo.n_inputs = (*r_graph_handle)->GetFunction("get_function_arity").value()("main").cast<int>();
  inputs_.resize(mInfo.n_inputs);
  for (int i = 0; i < mInfo.n_inputs; i++) {
    mInfo.param_names.push_back((*r_graph_handle)
                                    ->GetFunction("get_function_param_name")
                                    .value()("main", i)
                                    .cast<std::string>());
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
  LOG(INFO) << "    Input MetaInfo:";
  for (int i = 0; i < mInfo.param_names.size(); i++) {
    LOG(INFO) << "param_names - " << mInfo.param_names[i];
  }
}

/*!
 * \brief Print stats information.
 */
void TVMRunner::PrintStats(void) {
  LOG(INFO) << "Performance Stats:" << r_model_path;
  LOG(INFO) << "Total Module Load Time     :" << r_module_load_ms << " ms";
}

}  // namespace runtime
}  // namespace tvm
