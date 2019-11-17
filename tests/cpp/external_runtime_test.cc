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
 * \file external_runtime_test.cc
 * \brief Test an example runtime module to interpreting a json string.
 */
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cmath>
#include <sstream>
#include <string>

using tvm::runtime::Module;
using tvm::runtime::ModuleNode;
using tvm::runtime::NDArray;
using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::PackedFunc;
using tvm::runtime::TVMArgsSetter;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

void Add_(float* a, int len_a, float* b, int len_b, float* c) {
  for (int i = 0; i < len_a * len_b; i++) {
    c[i] = a[i] + b[i];
  }
}

int Add(TVMValue* value, int* type_code, int nargs) {
  CHECK_EQ(nargs, 3U) << "Expect 3 args, but get " << nargs << "\n";
  DLTensor* arg0 = static_cast<DLTensor*>(value[0].v_handle);
  DLTensor* arg1 = static_cast<DLTensor*>(value[1].v_handle);
  DLTensor* out = static_cast<DLTensor*>(value[2].v_handle);
  Add_(static_cast<float*>(arg0->data), arg0->shape[0],
       static_cast<float*>(arg1->data), arg1->shape[0],
       static_cast<float*>(out->data));
  return 0;
}

void Sub_(float* a, int len_a, float* b, int len_b, float* c) {
  for (int i = 0; i < len_a * len_b; i++) {
    c[i] = a[i] - b[i];
  }
}

int Sub(TVMValue* value, int* type_code, int nargs) {
  CHECK_EQ(nargs, 3U) << "Expect 3 args, but get " << nargs << "\n";
  DLTensor* arg0 = static_cast<DLTensor*>(value[0].v_handle);
  DLTensor* arg1 = static_cast<DLTensor*>(value[1].v_handle);
  DLTensor* out = static_cast<DLTensor*>(value[2].v_handle);
  Sub_(static_cast<float*>(arg0->data), arg0->shape[0],
       static_cast<float*>(arg1->data), arg1->shape[0],
       static_cast<float*>(out->data));
  return 0;
}

class ExampleJSonModule : public ModuleNode {
 public:
  ExampleJSonModule() {}
  ~ExampleJSonModule() {}

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "example_json_rt") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 3U);
        NDArray arg0 = args[0];
        NDArray arg1 = args[1];
        NDArray arg2 = args[2];
        this->data_entry_[0].CopyFrom(arg0);
        this->data_entry_[1].CopyFrom(arg1);
        this->data_entry_[2].CopyFrom(arg2);
        for (const auto& it : this->graph_) {
          this->run(it.first, it.second);
        }
        *rv = data_entry_.back();
      });
    } else {
      LOG(FATAL) << "Unkown runtime type: " << name << "\n";
      return PackedFunc();
    }
  }

  void run(int id, const std::vector<int>& inputs) {
    std::vector<TVMValue> values(inputs.size());
    std::vector<int> type_codes(inputs.size());
    TVMArgsSetter setter(values.data(), type_codes.data());

    if (op_id_[id] == "add" || op_id_[id] == "sub") {
      for (size_t i = 0; i < inputs.size(); i++) {
        setter(i, data_entry_[inputs[i]]);
      }
    }

    if (op_id_[id] == "add") {
      Add(values.data(), type_codes.data(), inputs.size());
    } else if (op_id_[id] == "sub") {
      Sub(values.data(), type_codes.data(), inputs.size());
    }
  }

  const char* type_key() const { return "examplejson"; }

  void SaveToBinary(dmlc::Stream* stream) final {
    // Write to a json string.
  }

  // Note this is a very simple json that only serves for demostration purpose.
  // Users usually have their own format and they can serialize it using the
  // SaveToBinary method and deserialize it using LoadFromFile.
  void ParseJson(const std::string& json) {
    std::string line;
    std::stringstream ss(json);

    while (std::getline(ss, line, '\n')) {
      std::stringstream ss2(line);
      std::string token;
      int id = 0;

      ss2 >> token;
      ss2 >> id;
      if (op_id_.size() <= static_cast<size_t>(id)) {
        op_id_.resize(id + 1);
        data_entry_.resize(id + 1);
      }

      int64_t total_elements = 1;
      std::vector<int64_t> shape;
      if (token == "input") {
        int64_t size = 0;
        while (ss2 >> size) {
          total_elements *= size;
          shape.push_back(size);
        }
      } else {
        op_id_[id] = token;
        bool shape_data = false;
        while (ss2 >> token) {
          if (token == "shape:") {
            shape_data = true;
          } else if (shape_data) {
            total_elements *= std::stoll(token);
            shape.push_back(std::stoll(token));
          } else if (token != "inputs:") {
            graph_[id].push_back(std::stoi(token));
          }
        }
        graph_[id].push_back(id);
      }
      DLContext ctx;
      ctx.device_type = static_cast<DLDeviceType>(1);
      ctx.device_id = 0;
      data_entry_[id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx);
    }
  }

  static Module LoadFromFile(const std::string& json, const std::string& format) {
    auto n = tvm::runtime::make_object<ExampleJSonModule>();
    n->ParseJson(json);
    return Module(n);
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {}
  std::string GetSource(const std::string& format = "") final { return ""; }

 private:
  // op -> inputs
  std::map<int, std::vector<int> > graph_;
  std::vector<NDArray> data_entry_;
  // id -> op
  std::vector<std::string> op_id_;
};

TEST(ExampleModule, Basic) {
  // This is a simple json format used for testing. Users/vendors can define
  // their own format.
  std::string json =
      "input 0 10 10\n"
      "input 1 10 10\n"
      "input 2 10 10\n"
      "add 3 inputs: 0 1 shape: 10 10\n"
      "sub 4 inputs: 3 2 shape: 10 10";

  Module mod = ExampleJSonModule::LoadFromFile(json, "");
  PackedFunc f = mod.GetFunction("example_json_rt", false);

  auto a_val = NDArray::Empty({10, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto b_val = NDArray::Empty({10, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto c_val = NDArray::Empty({10, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  float* pa = (float*)a_val.ToDLPack()->dl_tensor.data;
  float* pb = (float*)b_val.ToDLPack()->dl_tensor.data;
  float* pc = (float*)c_val.ToDLPack()->dl_tensor.data;

  // Assign values.
  for (int i = 0; i < 10 * 10; i++) {
    pa[i] = i;
    pb[i] = i + 1.0;
    pc[i] = i + 2.0;
  }

  NDArray out = f(a_val, b_val, c_val);
  float* p_out = (float*)out.ToDLPack()->dl_tensor.data;

  // Check correctness of result
  for (int i = 0; i < 10; i++) {
    CHECK_LT(std::fabs(p_out[i] - ((i + (i + 1.0) - (i + 2.0)))), 1e-5);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
