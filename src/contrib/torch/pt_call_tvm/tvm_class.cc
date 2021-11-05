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
#include <dlpack/dlpack.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>

#include <map>
#include <string>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace contrib {
namespace pytorch {

/*! \brief Class holding necessary components to call TVM graph runtime */
class TvmGraphModulePack {
 public:
  /*!
   * \brief Constructor.
   *
   * \param path Encoded path of graph runtime assets.
   * \param device_type int64_t, kDLCPU or kDLCUDA.
   * \param device_id int64_t.
   */
  explicit TvmGraphModulePack(std::string path, int64_t device_type, int64_t device_id)
      : path_(std::move(path)) {
    LOG(INFO) << "[TvmGraphModule] loading module at path: [" << path_ << "] on device ["
              << (device_type == kDLCUDA ? "cuda:" : "cpu:") << device_id << "]...";
    std::string lib_path, graph_path, params_path;
    DecodePaths(path_, &lib_path, &graph_path, &params_path);

    // load graph
    std::ifstream graph_in(graph_path);
    std::string graph_data((std::istreambuf_iterator<char>(graph_in)),
                           std::istreambuf_iterator<char>());
    graph_in.close();

    // load mod syslib
    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(lib_path);

    const auto runtime_create = *tvm::runtime::Registry::Get("tvm.graph_executor.create");

    // read params data
    std::ifstream params_in(params_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)),
                            std::istreambuf_iterator<char>());
    params_in.close();
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    // set devices
    module_ = runtime_create(graph_data, lib, device_type, device_id);
    const tvm::runtime::PackedFunc load_params = module_.GetFunction("load_params");
    load_params(params_arr);

    set_input = module_.GetFunction("set_input_zero_copy");
    run = module_.GetFunction("run");
    get_output = module_.GetFunction("get_output");
    set_output = module_.GetFunction("set_output_zero_copy");
    num_outputs_ = module_.GetFunction("get_num_outputs")();
  }

  static constexpr char kPathDelimiter = '|';

  /*!
   * \brief Decode lib_path, graph_path, params_path from encoded path.
   *
   * \param path The encoded path, concated with `kPathDelimiter`.
   * \param lib_path The path of .so lib file.
   * \param graph_path The path of graph.json.
   * \param params_path The path of params data.
   */
  static void DecodePaths(const std::string& path, std::string* lib_path, std::string* graph_path,
                          std::string* params_path) {
    std::vector<std::string> paths;
    for (size_t i = 0, pre = 0, lim = path.size(); i <= lim; ++i) {
      if (i == lim || path.at(i) == kPathDelimiter) {
        paths.push_back(path.substr(pre, i - pre));
        pre = i + 1;
      }
    }
    CHECK_EQ(paths.size(), 3u);
    *lib_path = paths.at(0);
    *graph_path = paths.at(1);
    *params_path = paths.at(2);
  }

  /*!
   * \brief Encode lib_path, graph_path, params_path by concat then with `kPathDelimiter`.
   *
   * \param lib_path The path of .so lib file.
   * \param graph_path The path of graph.json.
   * \param params_path The path of params data.
   *
   * \return The encoded path, concated with `kPathDelimiter`.
   */
  static std::string EncodePaths(const std::string& lib_path, const std::string& graph_path,
                                 const std::string& params_path) {
    return lib_path + kPathDelimiter + graph_path + kPathDelimiter + params_path;
  }

  const std::string& path() const { return path_; }

  const int64_t num_outputs() const { return num_outputs_; }

  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc run;
  tvm::runtime::PackedFunc get_output;
  tvm::runtime::PackedFunc set_output;

 private:
  tvm::runtime::Module module_;
  int64_t num_outputs_;
  std::string path_;
};

/*! \brief Class holding necessary components to call TVM VM runtime */
class TvmVMModulePack {
 public:
  /*!
   * \brief Constructor.
   *
   * \param path Encoded path of vm runtime assets.
   * \param device_type int64_t, kDLCPU or kDLCUDA.
   * \param device_id int64_t.
   */
  explicit TvmVMModulePack(std::string path, int64_t device_type, int64_t device_id)
      : path_(std::move(path)) {
    LOG(INFO) << "[TvmVMModule] loading module at path: [" << path_ << "] on device ["
              << (device_type == kDLCUDA ? "cuda:" : "cpu:") << device_id << "]...";
    // build tvm graph runtime
    std::string lib_path, code_path;
    DecodePaths(path_, &lib_path, &code_path);
    // load lib
    auto loaded_lib = tvm::runtime::Module::LoadFromFile(lib_path, "so");
    // load code
    std::ifstream code_in(code_path);
    std::string loaded_code((std::istreambuf_iterator<char>(code_in)),
                            std::istreambuf_iterator<char>());
    code_in.close();
    exe_ = tvm::runtime::vm::Executable::Load(loaded_code, loaded_lib);
    const auto runtime_create = *tvm::runtime::Registry::Get("runtime._VirtualMachine");
    vm_ = runtime_create(exe_);
    auto init_func = vm_.GetFunction("init", false);
    auto alloc_type = static_cast<int>(tvm::runtime::vm::AllocatorType::kPooled);
    if (device_type != kDLCPU) {
      // CPU is required for executing shape functions
      init_func(static_cast<int>(kDLCPU), 0, alloc_type, device_type, device_id, alloc_type);
    } else {
      init_func(device_type, device_id, alloc_type);
    }
    set_input = vm_.GetFunction("set_input", false);
    invoke = vm_.GetFunction("invoke", false);
  }

  static constexpr char kPathDelimiter = '|';

  /*!
   * \brief Decode lib_path, code_path from encoded path.
   *
   * \param path The encoded path, concated with `kPathDelimiter`.
   * \param lib_path The path of lib file.
   * \param code_path The path of code file.
   */
  static void DecodePaths(const std::string& path, std::string* lib_path, std::string* code_path) {
    std::vector<std::string> paths;
    for (size_t i = 0, pre = 0, lim = path.size(); i <= lim; ++i) {
      if (i == lim || path.at(i) == kPathDelimiter) {
        paths.push_back(path.substr(pre, i - pre));
        pre = i + 1;
      }
    }
    CHECK_EQ(paths.size(), 2u);
    *lib_path = paths.at(0);
    *code_path = paths.at(1);
  }

  /*!
   * \brief Encode lib_path, code_path by concat then with `kPathDelimiter`.
   *
   * \param lib_path The path of vm lib file.
   * \param code_path The path of code.
   *
   * \return The encoded path, concated with `kPathDelimiter`.
   */
  static std::string EncodePaths(const std::string& lib_path, const std::string& code_path) {
    return lib_path + kPathDelimiter + code_path;
  }

  const std::string& path() const { return path_; }

  tvm::runtime::PackedFunc set_input;
  tvm::runtime::PackedFunc invoke;

 private:
  tvm::runtime::Module exe_;
  tvm::runtime::Module vm_;
  std::string path_;
};

/*! \brief Pytorch custom class to call TVM */
class BaseTvmClass : public torch::jit::CustomClassHolder {
 public:
  /*!
   * \brief Constructor.
   *
   * \param num_inputs Number of inputs.
   * \param num_outputs Number of outputs.
   * \param device std::string, use the pytorch device str format, e.g. `cuda:0`, 'cpu'
   */
  BaseTvmClass(const int64_t num_inputs, const int64_t num_outputs, const std::string& device)
      : num_inputs_(num_inputs), num_outputs_(num_outputs) {
    auto torch_device = torch::Device(device);
    device_type_ = torch_device.is_cuda() ? kDLCUDA : kDLCPU;
    device_id_ = torch_device.index();
  }

  /*! \brief Virtual destructor. */
  virtual ~BaseTvmClass() {}

  /*!
   * \brief Get repr string of pytorch input shapes.
   *
   * \param shapes Pytorch shapes of type List[List[int]].
   *
   * \return std::string, the representation of inputs shapes.
   */
  static std::string TvmShapeRepr(const c10::List<c10::List<int64_t>>& shapes) {
    std::stringstream ss;
    for (const auto& shape : shapes) {
      for (const auto& sz : static_cast<c10::List<int64_t>>(shape)) {
        ss << sz << "_";
      }
      ss << "__";
    }
    return ss.str();
  }

  /*!
   * \brief Get input shapes.
   *
   * \param inputs Inputs with type List[Tensor].
   *
   * \return outputs with type List[List[int]].
   */
  static c10::List<c10::List<int64_t>> GetShapes(const c10::List<at::Tensor>& inputs) {
    c10::List<c10::List<int64_t>> shapes;
    for (const auto& input : inputs) {
      c10::List<int64_t> shape;
      for (const auto sz : static_cast<at::Tensor>(input).sizes()) {
        shape.push_back(sz);
      }
      shapes.push_back(shape);
    }
    return shapes;
  }

  /*!
   * \brief Move the TVM modules to given device.
   *
   * \param device String repr of the device to be moved to.
   */
  virtual void to(const std::string& device) = 0;

  // getters
  int64_t num_inputs() const { return num_inputs_; }

  int64_t num_outputs() const { return num_outputs_; }

  int64_t device_type() const { return device_type_; }

  int64_t device_id() const { return device_id_; }

  c10::DeviceType torch_device_type() const {
    return device_type() == kDLCUDA ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
  }

  bool is_on_same_device(const torch::Tensor& tensor) const {
    auto tensor_device_type = tensor.device().type();
    if (tensor_device_type == torch::DeviceType::CUDA) {
      return tensor_device_type == torch_device_type() && device_id() == tensor.device().index();
    }
    CHECK_EQ(tensor_device_type, torch::DeviceType::CPU);
    return tensor_device_type == torch_device_type();
  }

  std::string device() const { return torch::Device(torch_device_type(), device_id()).str(); }

  /*!
   * \brief Module forward.
   *
   * \param inputs Inputs with type List[Tensor].
   *
   * \return outputs with type List[Tensor].
   */
  virtual c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) = 0;

  /*!
   * \brief Serialize TVM Modules to Dict<string, string>
   */
  virtual c10::Dict<std::string, std::string> SerializeTvmModules() const = 0;

  /*!
   * \brief deserialize TVM Modules from Dict<string, string>
   */
  virtual void DeserializeTvmModules(const c10::Dict<std::string, std::string>& shape_path_map) = 0;

 protected:
  const int64_t num_inputs_;
  const int64_t num_outputs_;
  int64_t device_type_;
  int64_t device_id_;
};

/*! \brief Pytorch custom class to call TVM graph runtime */
class TvmGraphRuntimeClass : public BaseTvmClass {
 public:
  TvmGraphRuntimeClass(const int64_t num_inputs, const int64_t num_outputs,
                       const std::string& device)
      : BaseTvmClass(num_inputs, num_outputs, device) {}

  /*!
   * \brief Module forward.
   *
   * \param inputs Inputs with type List[Tensor].
   *
   * \return outputs with type List[Tensor].
   */
  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) override {
    CHECK_EQ(inputs.size(), num_inputs_);
    auto shape_repr = TvmShapeRepr(GetShapes(inputs));
    std::vector<DLTensor> args(num_inputs_ + num_outputs_);
    auto iter = tvm_modules_.find(shape_repr);
    CHECK(iter != tvm_modules_.end());
    const auto& tvm_pack = iter->second;
    std::vector<TensorAsBuf> buf_infos;
    buf_infos.reserve(num_inputs_ + num_outputs_);

    for (int i = 0; i < num_inputs_; ++i) {
      at::Tensor inp = inputs[i];
      CHECK(is_on_same_device(inp))
          << "input #" << i
          << " of forward is not on the same device with TvmGraphRuntime, expected " << device()
          << " but got " << inp.device().str();
      inp = inp.contiguous();
      buf_infos.emplace_back(inp);
      auto& input_buf = buf_infos[i];
      input_buf.CopyFromOrigin();
      input_buf.MakeDLTensor(&args[i]);
      tvm_pack.set_input(i, &args[i]);
    }
    // prepare output buffers
    c10::List<at::Tensor> outputs;
    outputs.reserve(num_outputs_);

    for (int i = 0; i < num_outputs_; ++i) {
      tvm::runtime::NDArray output_arr = tvm_pack.get_output(i);
      std::vector<int64_t> output_shape(output_arr->shape, output_arr->shape + output_arr->ndim);

      torch::ScalarType output_dtype = torch::ScalarType::Undefined;
      CHECK(GetTorchDtype(output_arr.DataType(), &output_dtype));

      CHECK(device_type_ == kDLCPU || device_type_ == kDLCUDA);
      const c10::DeviceType pt_device_type = (device_type_ == kDLCUDA ? torch::kCUDA : torch::kCPU);
      const auto options =
          torch::TensorOptions().dtype(output_dtype).device(pt_device_type, device_id_);

      outputs.emplace_back(torch::empty(output_shape, options));
      buf_infos.emplace_back(outputs[i]);
      auto& output_buf = buf_infos[num_inputs_ + i];
      output_buf.MakeDLTensor(&args[num_inputs_ + i]);
      tvm_pack.set_output(i, &args[num_inputs_ + i]);
    }
    tvm_pack.run();
    for (int i = 0; i < num_outputs_; ++i) {
      auto& output_buf = buf_infos[num_inputs_ + i];
      output_buf.CopyToOrigin();
    }
    return outputs;
  }

  /*!
   * \brief Load TVM graph runtime module.
   *
   * \param shapes Input shapes. List[List[int]].
   * \param lib_path Path of .so lib file.
   * \param graph_path Path of graph.json file.
   * \param params_path Path of params data file.
   */
  void LoadTvmModule(const c10::List<c10::List<int64_t>>& shapes, const std::string& lib_path,
                     const std::string& graph_path, const std::string& params_path) {
    std::string path = TvmGraphModulePack::EncodePaths(lib_path, graph_path, params_path);
    auto shape_repr = TvmShapeRepr(shapes);
    auto it_find = tvm_modules_.find(shape_repr);
    if (it_find != tvm_modules_.end()) {
      tvm_modules_.erase(it_find);
    }
    const auto it =
        tvm_modules_.emplace(shape_repr, TvmGraphModulePack(path, device_type_, device_id_)).first;
    if (it->second.num_outputs() != num_outputs_) {
      LOG(FATAL) << "tvm class num outputs mismatch, expected " << num_outputs_ << ", got "
                 << it->second.num_outputs();
    }
  }

  const std::map<std::string, TvmGraphModulePack>& tvm_modules() const { return tvm_modules_; }

  /*!
   * \brief Serialize TVM modules to shape map.
   *
   * \return shape_path_map Dict of shape_repr to path.
   */
  c10::Dict<std::string, std::string> SerializeTvmModules() const override {
    c10::Dict<std::string, std::string> shape_path_map;
    for (const auto& entry : tvm_modules()) {
      shape_path_map.insert(entry.first, entry.second.path());
    }
    return shape_path_map;
  }

  /*!
   * \brief Deserialize TVM modules from shape map.
   *
   * \param shape_path_map Dict of shape_repr to path.
   */
  void DeserializeTvmModules(const c10::Dict<std::string, std::string>& shape_path_map) override {
    tvm_modules_.clear();
    for (const auto& entry : shape_path_map) {
      const auto& shape_repr = entry.key();
      const auto& path = entry.value();
      tvm_modules_.emplace(shape_repr, TvmGraphModulePack(path, device_type_, device_id_));
    }
  }

  /*!
   * \brief Move the TVM modules to given device.
   *
   * \param device String repr of the device to be moved to.
   */
  void to(const std::string& device) override {
    if (device != this->device()) {
      auto torch_device = torch::Device(device);
      device_type_ = torch_device.is_cuda() ? kDLCUDA : kDLCPU;
      device_id_ = torch_device.index();
      DeserializeTvmModules(SerializeTvmModules());
    }
  }

 private:
  std::map<std::string, TvmGraphModulePack> tvm_modules_;
};

/*! \brief Pytorch custom class to call TVM graph runtime */
class TvmVMRuntimeClass : public BaseTvmClass {
 public:
  TvmVMRuntimeClass(const int64_t num_inputs, const int64_t num_outputs, const std::string& device)
      : BaseTvmClass(num_inputs, num_outputs, device) {}

  /*!
   * \brief Module forward.
   *
   * \param inputs Inputs with type List[Tensor].
   *
   * \return outputs with type List[Tensor].
   */
  c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) override {
    // get inputs repr str
    auto shape_repr = TvmShapeRepr(GetShapes(inputs));
    // get tvm pack
    auto iter = tvm_modules_.find(shape_repr);
    CHECK(iter != tvm_modules_.end()) << "tvm module pack not found for shape_repr " << shape_repr;
    const auto& tvm_pack = iter->second;

    // input tensors
    CHECK_EQ(inputs.size(), num_inputs_);
    std::vector<DLTensor> args(num_inputs_);
    std::vector<tvm::runtime::NDArray> args_arr(num_inputs_);

    for (int i = 0; i < num_inputs_; ++i) {
      TensorAsBuf input_buf(inputs[i]);
      input_buf.CopyFromOrigin();
      input_buf.MakeDLTensor(&args[i]);
      args_arr[i] =
          tvm::runtime::NDArray::FromDLPack(new DLManagedTensor({args[i], nullptr, nullptr}));
    }
    // set input
    std::vector<TVMValue> tvm_values(num_inputs_ + 1);
    std::vector<int> tvm_type_codes(num_inputs_ + 1);
    tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    setter(0, "main");
    for (int k = 0; k < num_inputs_; ++k) {
      setter(k + 1, args_arr[k]);
    }
    tvm_pack.set_input.CallPacked(
        tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), num_inputs_ + 1), nullptr);

    // run tvm
    tvm::runtime::TVMRetValue ret = tvm_pack.invoke("main");

    // get outputs
    std::vector<tvm::runtime::NDArray> output_arrs(num_outputs_);
    auto output_mismatch_msg = [](int actual, int expected) {
      std::stringstream ss;
      ss << "num_outputs not equal, actual:[" << actual << "] != expected:[" << expected << "]";
      return ss.str();
    };
    if (ret.type_code() == kTVMNDArrayHandle) {
      CHECK_EQ(num_outputs_, 1) << output_mismatch_msg(1, num_outputs_);
      output_arrs.at(0) = ret.AsObjectRef<tvm::runtime::NDArray>();
    } else if (ret.type_code() == kTVMObjectHandle) {
      const auto& adt = ret.AsObjectRef<tvm::runtime::ADT>();
      CHECK_EQ(adt.size(), num_outputs_) << output_mismatch_msg(adt.size(), num_outputs_);
      for (size_t i = 0; i < adt.size(); ++i) {
        CHECK(adt[i]->IsInstance<tvm::runtime::NDArray::ContainerType>())
            << "adt elements not tvm::runtime::NDArray";
        output_arrs.at(i) = tvm::runtime::Downcast<tvm::runtime::NDArray>(adt[i]);
      }
    } else {
      LOG(FATAL) << "unsupported return type with type_code = " << ret.type_code();
    }

    std::vector<DLTensor> output_args(num_outputs_);
    c10::List<at::Tensor> outputs;
    outputs.reserve(num_outputs_);

    for (int i = 0; i < num_outputs_; ++i) {
      const auto& output_arr = output_arrs[i];
      std::vector<int64_t> output_shape(output_arr->shape, output_arr->shape + output_arr->ndim);

      torch::ScalarType output_dtype = torch::ScalarType::Undefined;
      CHECK(GetTorchDtype(output_arr.DataType(), &output_dtype));

      CHECK(device_type_ == kDLCPU || device_type_ == kDLCUDA);
      const c10::DeviceType pt_device_type = (device_type_ == kDLCUDA ? torch::kCUDA : torch::kCPU);
      const auto options =
          torch::TensorOptions().dtype(output_dtype).device(pt_device_type, device_id_);

      outputs.emplace_back(torch::empty(output_shape, options));
      TensorAsBuf output_buf(outputs[i]);
      output_buf.MakeDLTensor(&output_args[i]);
      output_arr.CopyTo(&output_args[i]);
      output_buf.CopyToOrigin();
    }
    return outputs;
  }

  /*!
   * \brief Load TVM vm runtime module.
   *
   * \param shapes Input shapes. List[List[int]].
   * \param lib_path Path of .so lib file.
   * \param code_path Path of code file. Typically named code.ro
   */
  void LoadTvmModule(const c10::List<c10::List<int64_t>>& shapes, const std::string& lib_path,
                     const std::string& code_path) {
    std::string path = TvmVMModulePack::EncodePaths(lib_path, code_path);
    auto shape_repr = TvmShapeRepr(shapes);
    auto it_find = tvm_modules_.find(shape_repr);
    if (it_find != tvm_modules_.end()) {
      tvm_modules_.erase(it_find);
    }
    tvm_modules_.emplace(shape_repr, TvmVMModulePack(path, device_type_, device_id_));
  }

  const std::map<std::string, TvmVMModulePack>& tvm_modules() const { return tvm_modules_; }

  /*!
   * \brief Serialize TVM modules to shape map.
   *
   * \return shape_path_map Dict of shape_repr to path.
   */
  c10::Dict<std::string, std::string> SerializeTvmModules() const override {
    c10::Dict<std::string, std::string> shape_path_map;
    for (const auto& entry : tvm_modules()) {
      shape_path_map.insert(entry.first, entry.second.path());
    }
    return shape_path_map;
  }

  /*!
   * \brief Deserialize TVM modules from shape map.
   *
   * \param shape_path_map Dict of shape_repr to path.
   */
  void DeserializeTvmModules(const c10::Dict<std::string, std::string>& shape_path_map) override {
    tvm_modules_.clear();
    for (const auto& entry : shape_path_map) {
      const auto& shape_repr = entry.key();
      const auto& path = entry.value();
      tvm_modules_.emplace(shape_repr, TvmVMModulePack(path, device_type_, device_id_));
    }
  }

  /*!
   * \brief Move the TVM modules to given device.
   *
   * \param device String repr of the device to be moved to.
   */
  void to(const std::string& device) override {
    if (device != this->device()) {
      auto torch_device = torch::Device(device);
      device_type_ = torch_device.is_cuda() ? kDLCUDA : kDLCPU;
      device_id_ = torch_device.index();
      DeserializeTvmModules(SerializeTvmModules());
    }
  }

 private:
  std::map<std::string, TvmVMModulePack> tvm_modules_;
};

// <num_inputs, num_outputs, device, shape_path_map>
using SerializeTuple =
    std::tuple<int64_t, int64_t, std::string, c10::Dict<std::string, std::string>>;

/***** registries *****/
static auto __tvm_dsoop_graph_runtime_registry =
    torch::jit::class_<TvmGraphRuntimeClass>("tvm_dsoop", "TvmGraphModule")
        .def(torch::init<const int64_t, const int64_t, const std::string&>())
        .def("load_tvm_module", &TvmGraphRuntimeClass::LoadTvmModule)
        .def("forward", &TvmGraphRuntimeClass::forward)
        .def("to", &TvmGraphRuntimeClass::to)
        .def_pickle(
            [](const c10::intrusive_ptr<TvmGraphRuntimeClass>& self) -> SerializeTuple {
              return std::make_tuple(self->num_inputs(), self->num_outputs(), self->device(),
                                     self->SerializeTvmModules());
            },
            [](SerializeTuple tuple) -> c10::intrusive_ptr<TvmGraphRuntimeClass> {
              auto ptr = c10::make_intrusive<TvmGraphRuntimeClass>(
                  /*num_inputs=*/std::get<0>(tuple),
                  /*num_outputs=*/std::get<1>(tuple),
                  /*device=*/std::get<2>(tuple));
              ptr->DeserializeTvmModules(std::get<3>(tuple));
              return ptr;
            });

static auto __tvm_dsoop_vm_runtime_registry =
    torch::jit::class_<TvmVMRuntimeClass>("tvm_dsoop", "TvmVMModule")
        .def(torch::init<const int64_t, const int64_t, const std::string&>())
        .def("load_tvm_module", &TvmVMRuntimeClass::LoadTvmModule)
        .def("forward", &TvmVMRuntimeClass::forward)
        .def("to", &TvmVMRuntimeClass::to)
        .def_pickle(
            [](const c10::intrusive_ptr<TvmVMRuntimeClass>& self) -> SerializeTuple {
              return std::make_tuple(self->num_inputs(), self->num_outputs(), self->device(),
                                     self->SerializeTvmModules());
            },
            [](SerializeTuple tuple) -> c10::intrusive_ptr<TvmVMRuntimeClass> {
              auto ptr = c10::make_intrusive<TvmVMRuntimeClass>(
                  /*num_inputs=*/std::get<0>(tuple),
                  /*num_outputs=*/std::get<1>(tuple),
                  /*device=*/std::get<2>(tuple));
              ptr->DeserializeTvmModules(std::get<3>(tuple));
              return ptr;
            });

static auto __tvm_shape_repr_fn_registry =
    torch::RegisterOperators("tvm_dsoop::tvm_shape_repr", &BaseTvmClass::TvmShapeRepr);
}  // namespace pytorch
}  // namespace contrib
}  // namespace tvm
