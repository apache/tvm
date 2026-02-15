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
#include <tvm/ffi/extra/json.h>
#include <tvm/runtime/data_type.h>

#include <any>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace json {

using JSONGraphAttrs = std::unordered_map<std::string, std::any>;

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
    ICHECK_GE(arr.size(), 2) << "invalid json format";
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

namespace {

template <typename T>
bool SameType(const std::any& data) {
  return data.type() == typeid(T);
}

ffi::json::Value AnyVecToJson(const std::vector<std::any>& data) {
  ffi::json::Array arr;
  for (const auto& v : data) {
    if (SameType<std::string>(v)) {
      arr.push_back(ffi::String(std::any_cast<std::string>(v)));
    } else if (SameType<int>(v)) {
      arr.push_back(static_cast<int64_t>(std::any_cast<int>(v)));
    } else if (SameType<std::vector<size_t>>(v)) {
      const auto& sv = std::any_cast<const std::vector<size_t>&>(v);
      ffi::json::Array inner;
      for (auto x : sv) inner.push_back(static_cast<int64_t>(x));
      arr.push_back(std::move(inner));
    } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
      const auto& vv = std::any_cast<const std::vector<std::vector<int64_t>>&>(v);
      ffi::json::Array inner;
      for (const auto& row : vv) {
        ffi::json::Array r;
        for (auto x : row) r.push_back(x);
        inner.push_back(std::move(r));
      }
      arr.push_back(std::move(inner));
    } else if (SameType<std::vector<std::string>>(v)) {
      const auto& sv = std::any_cast<const std::vector<std::string>&>(v);
      ffi::json::Array inner;
      for (const auto& s : sv) inner.push_back(ffi::String(s));
      arr.push_back(std::move(inner));
    } else {
      LOG(FATAL) << "Not supported type in std::any vector";
    }
  }
  return arr;
}

ffi::json::Value AttrsToJson(const std::unordered_map<std::string, std::any>& data) {
  ffi::json::Object obj;
  for (const auto& kv : data) {
    const auto& k = kv.first;
    const auto& v = kv.second;
    if (SameType<std::string>(v)) {
      obj.Set(ffi::String(k), ffi::String(std::any_cast<std::string>(v)));
    } else if (SameType<int>(v)) {
      obj.Set(ffi::String(k), static_cast<int64_t>(std::any_cast<int>(v)));
    } else if (SameType<std::vector<size_t>>(v)) {
      const auto& sv = std::any_cast<const std::vector<size_t>&>(v);
      ffi::json::Array arr;
      for (auto x : sv) arr.push_back(static_cast<int64_t>(x));
      obj.Set(ffi::String(k), std::move(arr));
    } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
      const auto& vv = std::any_cast<const std::vector<std::vector<int64_t>>&>(v);
      ffi::json::Array arr;
      for (const auto& row : vv) {
        ffi::json::Array r;
        for (auto x : row) r.push_back(x);
        arr.push_back(std::move(r));
      }
      obj.Set(ffi::String(k), std::move(arr));
    } else if (SameType<std::vector<std::string>>(v)) {
      const auto& sv = std::any_cast<const std::vector<std::string>&>(v);
      ffi::json::Array arr;
      for (const auto& s : sv) arr.push_back(ffi::String(s));
      obj.Set(ffi::String(k), std::move(arr));
    } else if (SameType<std::vector<std::any>>(v)) {
      obj.Set(ffi::String(k), AnyVecToJson(std::any_cast<const std::vector<std::any>&>(v)));
    } else {
      LOG(FATAL) << "Not supported type in attrs";
    }
  }
  return obj;
}

}  // namespace

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
      SetAttr("num_inputs", std::to_string(inputs_.size()));
      SetAttr("num_outputs", std::to_string(num_outputs_));
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
    if (!attrs_.empty()) {
      obj.Set(ffi::String("attrs"), AttrsToJson(attrs_));
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
        std::string value = std::string(kv.second.cast<ffi::String>());
        num_inputs_ = strtoul(value.c_str(), nullptr, 10);
      } else if (key == "num_outputs") {
        std::string value = std::string(kv.second.cast<ffi::String>());
        num_outputs_ = strtoul(value.c_str(), nullptr, 10);
      } else if (key == "dtype") {
        auto outer = kv.second.cast<ffi::json::Array>();
        ICHECK_EQ(outer.size(), 1);
        auto inner = outer[0].cast<ffi::json::Array>();
        for (const auto& it : inner) {
          dtype_.push_back(ffi::StringToDLDataType(std::string(it.cast<ffi::String>())));
        }
      } else if (key == "shape") {
        auto outer = kv.second.cast<ffi::json::Array>();
        ICHECK_EQ(outer.size(), 1);
        auto shapes_arr = outer[0].cast<ffi::json::Array>();
        shape_.clear();
        for (const auto& s : shapes_arr) {
          auto row = s.cast<ffi::json::Array>();
          std::vector<int64_t> shape_vec;
          for (const auto& v : row) shape_vec.push_back(v.cast<int64_t>());
          shape_.push_back(shape_vec);
        }
      } else {
        auto outer = kv.second.cast<ffi::json::Array>();
        ICHECK_EQ(outer.size(), 1);
        auto inner = outer[0].cast<ffi::json::Array>();
        std::vector<std::string> tmp;
        for (const auto& v : inner) tmp.push_back(std::string(v.cast<ffi::String>()));
        attrs_[key] = tmp;
      }
    }
    ICHECK_EQ(shape_.size(), dtype_.size());
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
        LOG(FATAL) << "Unknown key: " << key;
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
  std::vector<std::vector<int64_t>> GetOpShape() const { return shape_; }

  /*!
   * \brief Return the op types.
   *
   * \return The types.
   */
  std::vector<DLDataType> GetOpDataType() const { return dtype_; }

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
    ICHECK_GT(attrs_.count(key), 0U) << "Key: " << key << " is not found";
    return std::any_cast<T>(attrs_.at(key));
  }

  /*!
   * \brief Set an attribute for the node.
   *
   * \tparam ValueT The type of the value being stored.
   * \param key The key of the attribute.
   * \param value The value of the attribute.
   */
  template <typename ValueT>
  void SetAttr(const std::string& key, const ValueT& value) {
    attrs_[key] = value;
  }

  /*!
   * \brief Check if node has attribute.
   *
   * \param key The key of the attribute.
   *
   * \return True if attribute exists, false otherwise.
   */
  bool HasAttr(const std::string& key) const { return attrs_.find(key) != attrs_.end(); }

  void CaptureAttrs(const JSONGraphNode& that) {
    for (const auto& kv : that.attrs_) {
      attrs_[kv.first] = kv.second;
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
  /*! \brief The shape of the node. */
  std::vector<std::vector<int64_t>> shape_;
  /*! \brief The type of the node. */
  std::vector<DLDataType> dtype_;
  /*! \brief The inputs of the node. */
  std::vector<JSONGraphNodeEntry> inputs_;
  /*!
   * \brief Attribute of the node. For simplicity, we store all attribute as
   * a list of std::string. It's the developer's resposibility to check the
   * required attribute of a certain op and convert it into the needed type.
   *
   * For example, for conv2d, this map could contain:
   *  attrs_["strides"] = ["1", "1"]
   *  attrs_["padding"] = ["0", "0", "0", "0"]
   *  attrs_["data_layout"] = ["NCHW"]
   *
   * when creating an execution engine, developers may need to use these
   * attributes and they can convert it into the needed type, i.e. padding to
   * int
   */
  JSONGraphAttrs attrs_;

  friend class JSONRuntimeBase;
};

}  // namespace json
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_JSON_JSON_NODE_H_
