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
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/contrib/gcc.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <sstream>
#include <string>

namespace tvm {
namespace runtime {

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
  Add_(static_cast<float*>(arg0->data), arg0->shape[0], static_cast<float*>(arg1->data),
       arg1->shape[0], static_cast<float*>(out->data));
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
  Sub_(static_cast<float*>(arg0->data), arg0->shape[0], static_cast<float*>(arg1->data),
       arg1->shape[0], static_cast<float*>(out->data));
  return 0;
}

PackedFunc ExampleJSonModule::GetFunction(const std::string& name,
                                          const ObjectPtr<Object>& sptr_to_self) {
  if (this->graph_.find(name) != this->graph_.end()) {
    this->curr_subgraph_ = name;
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      for (uint32_t i = 0; i < args.size(); ++i) {
        NDArray arg = args[i];
        this->data_entry_[i].CopyFrom(arg);
      }
      for (const auto& it : this->graph_[this->curr_subgraph_]) {
        this->run(it.first, it.second);
      }
      *rv = data_entry_.back();
    });
  }
  else {
    LOG(FATAL) << "Unkown runtime type: " << name << "\n";
    return PackedFunc();
  }
}

void ExampleJSonModule::run(int id, const std::vector<int>& inputs) {
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

// Note this is a very simple json that only serves for demostration purpose.
// Users usually have their own format and they can serialize it using the
// SaveToBinary method and deserialize it using LoadFromFile.
void ExampleJSonModule::ParseJson(const std::string& json) {
  std::string line;
  std::string curr_subgraph;
  std::stringstream ss(json);

  while (std::getline(ss, line, '\n')) {
    std::stringstream ss2(line);
    std::string token;
    int id = 0;

    ss2 >> token;
    if (token.find("gcc_") != std::string::npos) {
      curr_subgraph = token;
      graph_[curr_subgraph];
      continue;
    }

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
          graph_[curr_subgraph][id].push_back(std::stoi(token));
        }
      }
      graph_[curr_subgraph][id].push_back(id);
    }
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(1);
    ctx.device_id = 0;
    data_entry_[id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx);
  }
}

TVM_REGISTER_GLOBAL("module.loadfile_gcc").set_body_typed(ExampleJSonModule::LoadFromFile);

}
}