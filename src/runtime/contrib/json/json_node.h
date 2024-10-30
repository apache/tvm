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
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/data_type.h>

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

using JSONGraphAttrs = std::unordered_map<std::string, dmlc::any>;

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
   * \param writer The json writer.
   */
  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginArray();
    writer->WriteArrayItem(id_);
    writer->WriteArrayItem(index_);
    writer->WriteArrayItem(version_);
    writer->EndArray();
  }

  /*!
   * \brief Deserialize the json string into a node entry.
   * \param reader The json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    ICHECK(reader->NextArrayItem()) << "invalid json format";
    reader->Read(&id_);
    ICHECK(reader->NextArrayItem()) << "invalid json format";
    reader->Read(&index_);
    if (reader->NextArrayItem()) {
      reader->Read(&version_);
      ICHECK(!reader->NextArrayItem()) << "invalid json format";
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
   * \param writer The json writer.
   */
  void Save(dmlc::JSONWriter* writer) {
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_type_);
    writer->WriteObjectKeyValue("name", name_);
    if (!inputs_.empty()) {
      SetAttr("num_inputs", std::to_string(inputs_.size()));
      SetAttr("num_outputs", std::to_string(num_outputs_));
      writer->WriteObjectKeyValue("inputs", this->inputs_);
    }
    if (!attrs_.empty()) {
      writer->WriteObjectKeyValue("attrs", attrs_);
    }
    writer->EndObject();
  }

  /*!
   * \brief Load the attribute of a node in the json string.
   * \param reader The json reader.
   */
  void LoadAttrs(dmlc::JSONReader* reader) {
    std::string key, value;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "num_inputs") {
        reader->Read(&value);
        num_inputs_ = strtoul(value.c_str(), nullptr, 10);
      } else if (key == "num_outputs") {
        reader->Read(&value);
        num_outputs_ = strtoul(value.c_str(), nullptr, 10);
      } else if (key == "dtype") {
        std::vector<std::string> tmp;
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&tmp);
        ICHECK(!reader->NextArrayItem());
        for (const auto& it : tmp) {
          dtype_.push_back(tvm::runtime::String2DLDataType(it));
        }
      } else if (key == "shape") {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&shape_);
        ICHECK(!reader->NextArrayItem());
      } else {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        std::vector<std::string> tmp;
        reader->Read(&tmp);
        attrs_[key] = tmp;
        ICHECK(!reader->NextArrayItem());
      }
    }
    ICHECK_EQ(shape_.size(), dtype_.size());
  }

  /*!
   * \brief Load a node in the json string.
   * \param reader The json reader.
   */
  void Load(dmlc::JSONReader* reader) {
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "op") {
        reader->Read(&op_type_);
      } else if (key == "name") {
        reader->Read(&name_);
      } else if (key == "inputs") {
        reader->Read(&inputs_);
      } else if (key == "attr" || key == "attrs") {
        this->LoadAttrs(reader);
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
    return dmlc::get<T>(attrs_.at(key));
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

namespace dmlc {
namespace json {
template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::unordered_map<std::string, dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::unordered_map<std::string, dmlc::any>& data) {
    for (const auto& kv : data) {
      auto k = kv.first;
      const dmlc::any& v = kv.second;
      if (SameType<std::vector<dmlc::any>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<dmlc::any>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndObject();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::unordered_map<std::string, dmlc::any>* data) {
    LOG(FATAL) << "Not implemented";
  }
};

template <>
struct Handler<std::shared_ptr<tvm::runtime::json::JSONGraphNode>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::runtime::json::JSONGraphNode>& data) {
    data->Save(writer);
  }

  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::runtime::json::JSONGraphNode>* data) {
    (*data)->Load(reader);
  }
};
}  // namespace json
}  // namespace dmlc

#endif  // TVM_RUNTIME_CONTRIB_JSON_JSON_NODE_H_
