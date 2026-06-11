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
 * \file src/runtime/json/json_node.h
 * \brief The graph nodes used by JSON runtime.
 */

#ifndef TVM_RUNTIME_CONTRIB_JSON_JSON_NODE_H_
#define TVM_RUNTIME_CONTRIB_JSON_JSON_NODE_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/data_type.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace json {

using JSONGraphAttrs = ffi::Map<ffi::String, ffi::Any>;

/*!
 * \brief The node entry in the serialized json graph.
 */
class JSONGraphNodeEntry {
 public:
  // Constructors.
  JSONGraphNodeEntry() = default;
  JSONGraphNodeEntry(int id, int index, int version = 0)
      : id_(id), index_(index), version_(version) {}

  /*!
   * \brief Serialize a node entry.
   */
  ffi::json::Value SaveToJSON() const {
    ffi::json::Array arr;
    arr.push_back(static_cast<int64_t>(id_));
    arr.push_back(static_cast<int64_t>(index_));
    arr.push_back(static_cast<int64_t>(version_));
    return arr;
  }

  /*!
   * \brief Deserialize the json array into a node entry.
   */
  void Load(ffi::json::Array arr) {
    TVM_FFI_ICHECK_GE(arr.size(), 2) << "invalid json format";
    id_ = static_cast<uint32_t>(arr[0].cast<int64_t>());
    index_ = static_cast<uint32_t>(arr[1].cast<int64_t>());
    if (arr.size() > 2) {
      version_ = static_cast<uint32_t>(arr[2].cast<int64_t>());
    } else {
      version_ = 0;
    }
  }

  /*! \brief The json graph node ID. */
  uint32_t id_;
  /*! \brief The entry index. */
  uint32_t index_;
  uint32_t version_;
};

/*!
 * \brief The node of the serialized json graph. It includes an array of
 * entries.
 */
class JSONGraphNode {
 public:
  // Constructors.
  JSONGraphNode() = default;
  JSONGraphNode(const std::string& name, const std::string& op_type,
                const std::vector<JSONGraphNodeEntry>& inputs = {}, size_t num_outputs = 1) {
    name_ = name;
    op_type_ = op_type;
    num_inputs_ = inputs.size();
    inputs_ = inputs;
    num_outputs_ = num_outputs;
  }

  /*!
   * \brief Serialize a node so that it can be saved to disk.
   */
  ffi::json::Value SaveToJSON() {
    if (!inputs_.empty()) {
      SetAttr("num_inputs", static_cast<int64_t>(inputs_.size()));
      SetAttr("num_outputs", static_cast<int64_t>(num_outputs_));
    }
    ffi::json::Object obj;
    obj.Set(ffi::String("op"), ffi::String(op_type_));
    obj.Set(ffi::String("name"), ffi::String(name_));
    if (!inputs_.empty()) {
      ffi::json::Array inputs_arr;
      for (const auto& e : inputs_) {
        inputs_arr.push_back(e.SaveToJSON());
      }
      obj.Set(ffi::String("inputs"), std::move(inputs_arr));
    }
    if (attrs_.size() > 0) {
      ffi::json::Object attrs_obj;
      for (const auto& kv : attrs_) {
        std::string key = kv.first;
        const ffi::Any& v = kv.second;
        attrs_obj.Set(ffi::String(key), v);
      }
      obj.Set(ffi::String("attrs"), std::move(attrs_obj));
    }
    return obj;
  }

  /*!
   * \brief Load the attribute of a node from a json object.
   */
  void LoadAttrs(ffi::json::Object obj) {
    for (const auto& kv : obj) {
      std::string key = std::string(kv.first.cast<ffi::String>());
      if (key == "num_inputs") {
        num_inputs_ = static_cast<uint32_t>(kv.second.cast<int64_t>());
      } else if (key == "num_outputs") {
        num_outputs_ = static_cast<uint32_t>(kv.second.cast<int64_t>());
      } else {
        attrs_.Set(key, kv.second);
      }
    }
    // Populate cached shape/dtype from attrs for fast access
    if (HasAttr("shape")) {
      shape_ = GetAttr<ffi::Array<ffi::Array<int64_t>>>("shape");
    }
    if (HasAttr("dtype")) {
      dtype_ = GetAttr<ffi::Array<DLDataType>>("dtype");
    }
    if (shape_.defined() && dtype_.defined()) {
      TVM_FFI_ICHECK_EQ(shape_.size(), dtype_.size());
    }
  }

  /*!
   * \brief Load a node from a json object.
   */
  void Load(ffi::json::Object obj) {
    for (const auto& kv : obj) {
      std::string key = std::string(kv.first.cast<ffi::String>());
      if (key == "op") {
        op_type_ = std::string(kv.second.cast<ffi::String>());
      } else if (key == "name") {
        name_ = std::string(kv.second.cast<ffi::String>());
      } else if (key == "inputs") {
        auto arr = kv.second.cast<ffi::json::Array>();
        inputs_.clear();
        for (const auto& e : arr) {
          JSONGraphNodeEntry entry;
          entry.Load(e.cast<ffi::json::Array>());
          inputs_.push_back(entry);
        }
      } else if (key == "attr" || key == "attrs") {
        this->LoadAttrs(kv.second.cast<ffi::json::Object>());
      } else {
        TVM_FFI_THROW(InternalError) << "Unknown key: " << key;
      }
    }
  }

  /*!
   * \brief Check if a node is a leaf node, i.e. input to the graph.
   *
   * \return True if the node has no input, otherwise, false.
   */
  bool IsLeaf() const { return inputs_.empty(); }

  /*!
   * \brief Return the number of outputs of the node.
   *
   * \return The number of the output.
   */
  uint32_t GetNumOutput() const { return num_outputs_; }

  /*!
   * \brief Return the input entries.
   *
   * \return The input entries.
   */
  std::vector<JSONGraphNodeEntry> GetInputs() const { return inputs_; }

  /*!
   * \brief Return the op type.
   *
   * \return The op type.
   */
  std::string GetOpType() const { return op_type_; }

  /*!
   * \brief Return the op name.
   *
   * \return The op name.
   */
  std::string GetOpName() const { return name_; }

  /*!
   * \brief Return the op output shapes.
   *
   * \return The shapes.
   */
  ffi::Array<ffi::Array<int64_t>> GetOpShape() const { return shape_; }

  /*!
   * \brief Return the op types.
   *
   * \return The types.
   */
  ffi::Array<DLDataType> GetOpDataType() const { return dtype_; }

  /*!
   * \brief Set the number of outputs of the node.
   *
   * \param num_outputs The number of output.
   */
  void SetNumOutput(uint32_t num_outputs) { num_outputs_ = num_outputs; }

  /*!
   * \brief Get the value of an attribute in the node.
   *
   * \tparam T The return type.
   * \param key The key for lookup.
   *
   * \return The value.
   */
  template <typename T>
  T GetAttr(const std::string& key) const {
    TVM_FFI_ICHECK(attrs_.count(key) > 0) << "Key: " << key << " is not found";
    return attrs_[key].cast<T>();
  }

  /*!
   * \brief Set an attribute for the node.
   *
   * \param key The key of the attribute.
   * \param value The attribute value (native type: int64_t, double, ffi::String, ffi::Array, etc.)
   */
  void SetAttr(const std::string& key, ffi::Any value) { attrs_.Set(key, std::move(value)); }

  /*!
   * \brief Set shape for the node.
   */
  void SetShape(const std::vector<std::vector<int64_t>>& shape) {
    ffi::Array<ffi::Array<int64_t>> arr;
    for (const auto& s : shape) {
      ffi::Array<int64_t> row;
      for (auto x : s) row.push_back(x);
      arr.push_back(std::move(row));
    }
    shape_ = arr;
    SetAttr("shape", std::move(arr));
  }

  /*!
   * \brief Set dtype for the node.
   */
  void SetDType(const std::vector<DLDataType>& dtype) {
    ffi::Array<ffi::String> str_arr;
    ffi::Array<DLDataType> dt_arr;
    for (const auto& d : dtype) {
      str_arr.push_back(ffi::String(ffi::DLDataTypeToString(d)));
      dt_arr.push_back(d);
    }
    dtype_ = std::move(dt_arr);
    SetAttr("dtype", std::move(str_arr));
  }

  /*!
   * \brief Check if node has attribute.
   *
   * \param key The key of the attribute.
   *
   * \return True if attribute exists, false otherwise.
   */
  bool HasAttr(const std::string& key) const { return attrs_.count(key) > 0; }

  void CaptureAttrs(const JSONGraphNode& that) {
    for (const auto& kv : that.attrs_) {
      attrs_.Set(kv.first, kv.second);
    }
  }

  virtual ~JSONGraphNode() {}

 private:
  /*! \brief The number of input. */
  uint32_t num_inputs_{0};
  /*! \brief The number of output. */
  uint32_t num_outputs_{1};
  /*! \brief The name of the op. It is the symbol that used for runtime lookup. */
  std::string name_;
  /*! \brief The operator type, i.e. input is "null". */
  std::string op_type_;
  /*! \brief The inputs of the node. */
  std::vector<JSONGraphNodeEntry> inputs_;
  /*! \brief Attribute of the node, including shape and dtype. */
  JSONGraphAttrs attrs_;
  /*! \brief Cached shape for fast access (also stored in attrs_). */
  ffi::Array<ffi::Array<int64_t>> shape_;
  /*! \brief Cached dtype for fast access (also stored in attrs_ as strings). */
  ffi::Array<DLDataType> dtype_;

  friend class JSONRuntimeBase;
};

}  // namespace json
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_JSON_JSON_NODE_H_
