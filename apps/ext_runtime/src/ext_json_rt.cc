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
 *
 * This is an exmaple runtime employed to show how we can interprete and execute
 * a json string that represents a simple computational (sub)graph. Users will
 * mainly need to implement four functions as follows:
 *  - GetFunction. It is used to get the packed function from the json runtime
 * module using a provided function name. This function returns a PackedFunc
 * that can be directly invoked by feeding it with parameters.
 *  - SaveToBinary. This function is used to achieve the serialization purpose.
 * The emitted binary stream can be directly saved to disk so that users can
 * load then back when needed.
 *  - LoadFromFile. This is a static function that acts as a helper to create
 * a json runtime module using a given json string. The json string could be
 * conveniently loaded from the front-end and passed to this interface through
 * PackedFunc.
 *  - LoadFromBinary. This function uses binary stream to load the json that
 * saved by SaveToBinary which essentially performs deserialization.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

// A simple JSON node that contains multiple inputs and a single output.
struct NodeEntry {
  int id;
  int output;
  std::vector<int> inputs;
};

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

void Mul_(float* a, int len_a, float* b, int len_b, float* c) {
  for (int i = 0; i < len_a * len_b; i++) {
    c[i] = a[i] * b[i];
  }
}

int Mul(TVMValue* value, int* type_code, int nargs) {
  CHECK_EQ(nargs, 3U) << "Expect 3 args, but get " << nargs << "\n";
  DLTensor* arg0 = static_cast<DLTensor*>(value[0].v_handle);
  DLTensor* arg1 = static_cast<DLTensor*>(value[1].v_handle);
  DLTensor* out = static_cast<DLTensor*>(value[2].v_handle);
  Mul_(static_cast<float*>(arg0->data), arg0->shape[0],
       static_cast<float*>(arg1->data), arg1->shape[0],
       static_cast<float*>(out->data));
  return 0;
}

class ExampleJsonModule : public ModuleNode {
 public:
  ExampleJsonModule(std::string graph_json) {
    this->graph_json_ = graph_json;
    ParseJson(graph_json);
  }

  /*!
   * \brief Get a PackedFunc from the example json module.
   *
   * \param name the name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   *
   * \return The function pointer when it is found, otherwise, PackedFunc(nullptr).
   */
  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    if (this->graph_.find(name) != this->graph_.end()) {
      this->curr_subgraph_ = name;
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        for (auto i = 0; i < args.size(); ++i) {
          CHECK(args[i].type_code() == kNDArrayContainer || args[i].type_code() == kArrayHandle)
              << "Expect NDArray or DLTensor as inputs"
              << "\n";
          if (args[i].type_code() == kArrayHandle) {
            DLTensor* arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
          } else {
            NDArray arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
          }
        }
        for (const auto& it : this->graph_[this->curr_subgraph_]) {
          this->Run(it.id, it.inputs, it.output);
        }
        CHECK_GT(graph_.count(this->curr_subgraph_), 0U);
        auto out_idx = graph_[this->curr_subgraph_].back().output;
        if (args[args.size() - 1].type_code() == kArrayHandle) {
          DLTensor* arg = args[args.size() - 1];
          this->data_entry_[out_idx].CopyTo(arg);
        } else {
          NDArray arg = args[args.size() - 1];
          this->data_entry_[out_idx].CopyTo(arg);
        }
        *rv = data_entry_.back();
      });
    } else {
      LOG(FATAL) << "Unkown runtime type: " << name << "\n";
      return PackedFunc();
    }
  }

  /*!
   * \brief Execute a function with provided arguments. The output will be
   * packed to the last argument according to TVM's calling convention.
   *
   * \param id The id of the function.
   * \param inputs The input indices that indicate where the data should be
   * fetched in the data entry pool.
   * \param output The output index.
   */
  void Run(int id, const std::vector<int>& inputs, int output) {
    std::vector<int> args(inputs.begin(), inputs.end());
    args.push_back(output);
    std::vector<TVMValue> values(args.size());
    std::vector<int> type_codes(args.size());
    TVMArgsSetter setter(values.data(), type_codes.data());

    if (op_id_[id] == "add" || op_id_[id] == "sub" || op_id_[id] == "mul") {
      for (size_t i = 0; i < args.size(); i++) {
        setter(i, data_entry_[args[i]]);
      }
    }

    if (op_id_[id] == "add") {
      Add(values.data(), type_codes.data(), args.size());
    } else if (op_id_[id] == "sub") {
      Sub(values.data(), type_codes.data(), args.size());
    } else if (op_id_[id] == "mul") {
      Mul(values.data(), type_codes.data(), args.size());
    } else {
      LOG(FATAL) << "Unknown op: " << op_id_[id] << "\n";
    }
  }

  const char* type_key() const { return "examplejson"; }

  void SaveToBinary(dmlc::Stream* stream) final {
      stream->Write(this->graph_json_);
  }

  /*!
   * \brief Parse the example json string.
   *
   * \param json. The json string that represents a simple computational graph.
   *
   * \Note this is a very simple json that only serves for demostration purpose.
   * Users usually have their own format and they can serialize it using the
   * SaveToBinary method and deserialize it using LoadFromFile.
   */
  void ParseJson(const std::string& json) {
    std::string line;
    std::string curr_subgraph;
    std::stringstream ss(json);

    while (std::getline(ss, line, '\n')) {
      std::stringstream ss2(line);
      std::string token;
      int id = 0;

      ss2 >> token;
      if (token.find("json_rt_") != std::string::npos) {
        curr_subgraph = token;
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
        NodeEntry entry;
        while (ss2 >> token) {
          if (token == "shape:") {
            shape_data = true;
          } else if (shape_data) {
            total_elements *= std::stoll(token);
            shape.push_back(std::stoll(token));
          } else if (token != "inputs:") {
            entry.inputs.push_back(std::stoi(token));
          }
        }
        entry.id = id;
        entry.output = id;
        graph_[curr_subgraph].push_back(entry);
      }
      DLContext ctx;
      ctx.device_type = static_cast<DLDeviceType>(1);
      ctx.device_id = 0;
      data_entry_[id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx);
    }
  }

  /*!
   * \brief Load a json module from a json string.
   *
   * \param json The json string that represents a computational graph.
   * \param format The format of the file which is not used here.
   *
   * \return The created json module.
   */
  static Module LoadFromFile(const std::string& json,
                             const std::string& format) {
    auto n = tvm::runtime::make_object<ExampleJsonModule>(json);
    return Module(n);
  }

  /*!
   * \brief Load a json module from stream.
   *
   * \param strm The binary stream to load json.
   *
   * \return The created json module.
   */
  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string graph_json;
    stream->Read(&graph_json);
    auto n = tvm::runtime::make_object<ExampleJsonModule>(graph_json);
    return Module(n);
  }

 private:
  std::string graph_json_;
  std::string curr_subgraph_;
  // A simple graph.
  std::map<std::string, std::vector<NodeEntry> > graph_;
  std::vector<NDArray> data_entry_;
  // id -> op
  std::vector<std::string> op_id_;
};

TVM_REGISTER_GLOBAL("ext_json_rt.create_json_rt")
.set_body_typed(ExampleJsonModule::LoadFromFile);

TVM_REGISTER_GLOBAL("module.loadbinary_examplejson")
.set_body_typed(ExampleJsonModule::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm

