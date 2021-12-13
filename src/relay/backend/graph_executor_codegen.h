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
 * \file tvm/src/relay/backend/graph_executor_codegen.h
 * \brief Graph executor codegen
 */
#ifndef TVM_RELAY_BACKEND_GRAPH_EXECUTOR_CODEGEN_H_
#define TVM_RELAY_BACKEND_GRAPH_EXECUTOR_CODEGEN_H_

#include <../../src/relay/backend/utils.h>

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace backend {

class GraphExecutorCodegen;
class GraphInputNode;
class GraphOpNode;
class GraphNode;

using GraphAttrs = std::unordered_map<std::string, dmlc::any>;
using GraphObjectPtr = std::shared_ptr<GraphNode>;

/*! \brief Node types */
enum GraphNodeType {
  kGraphNop,
  kGraphInputNode,
  kGraphOpNode,
  kGraphInputNodeExt,
  kGraphOpNodeExt,
};

/*! \brief Base Node class */
class GraphNode {
 public:
  GraphNode() {}
  virtual void Save(dmlc::JSONWriter* writer) const {}
  virtual void Load(dmlc::JSONReader* reader) {}
  virtual GraphNodeType Type() const { return kGraphNop; }
  virtual ~GraphNode() {}

 public:
  int num_outputs_{1};
  std::string name_;
  GraphAttrs attrs_;
};

class GraphNodeRef {
 public:
  GraphNodeRef() {}
  GraphNodeRef(int ident, int index, int version = 0)
      : ident_(ident), index_(index), version_(version) {}

  inline void Save(dmlc::JSONWriter* writer) const {
    writer->BeginArray();
    writer->WriteArrayItem(ident_);
    writer->WriteArrayItem(index_);
    writer->WriteArrayItem(version_);
    writer->EndArray();
  }

  inline void Load(dmlc::JSONReader* reader) { LOG(FATAL) << "Not implemented."; }

 protected:
  int ident_;
  int index_{0};
  int version_{0};
};

/*! \brief Input Node */
class GraphInputNode : public GraphNode {
 public:
  GraphInputNode() {}
  GraphInputNode(const std::string& name, const GraphAttrs& attrs) {
    name_ = name;
    attrs_ = attrs;
  }

  GraphNodeType Type() const override { return kGraphInputNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& attrs) {
    auto ptr = std::make_shared<GraphInputNode>(name, attrs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }

  inline void Load(dmlc::JSONReader* reader) override { LOG(FATAL) << "Not implemented."; }
};

/*! \brief Op Node */
class GraphOpNode : public GraphNode {
 public:
  GraphOpNode& operator=(const GraphOpNode& t) { return *this; }
  GraphOpNode() {}
  GraphOpNode(const std::string& name, const GraphAttrs& nd_attrs, const std::string& op_name,
              const std::vector<GraphNodeRef>& inputs, const GraphAttrs& attrs,
              size_t num_outputs = 1);

  GraphNodeType Type() const override;
  void Save(dmlc::JSONWriter* writer) const override;

  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& nd_attrs,
                                                  const std::string& op_name,
                                                  const std::vector<GraphNodeRef>& inputs,
                                                  const GraphAttrs& attrs, size_t num_outputs = 1) {
    auto ptr = std::make_shared<GraphOpNode>(name, nd_attrs, op_name, inputs, attrs, num_outputs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }

 public:
  std::string op_name_;
  std::vector<GraphNodeRef> inputs_;
  GraphAttrs op_attrs_;

 private:
  const std::string op_type_name_{"tvm_op"};
};

class ExternalJsonWriterCB {
 public:
  template <class T>
  void RegisterCB(T* const object,
                  void (T::*const mf)(dmlc::JSONWriter*, Array<tvm::runtime::Module>,
                                      std::vector<GraphObjectPtr>, std::vector<GraphNodeRef>)) {
    using namespace std::placeholders;
    callback_ = std::bind(mf, object, _1, _2, _3, _4);
    hasCallback_ = true;
  }
  void RegisterCB(void (*const fun)(dmlc::JSONWriter*, Array<tvm::runtime::Module>,
                                    std::vector<GraphObjectPtr>, std::vector<GraphNodeRef>)) {
    callback_ = fun;
    hasCallback_ = true;
  }
  void Exe(dmlc::JSONWriter* external_writer, Array<tvm::runtime::Module> mod,
           std::vector<GraphObjectPtr> nodes, std::vector<GraphNodeRef> heads) {
    ICHECK(hasCallback_) << "ERROR: no registered callback";
    callback_(external_writer, mod, nodes, heads);
  }
  inline bool HasCallback() { return hasCallback_; }

 private:
  std::function<void(dmlc::JSONWriter*, Array<tvm::runtime::Module>, std::vector<GraphObjectPtr>,
                     std::vector<GraphNodeRef>)>
      callback_;
  bool hasCallback_{false};
};

}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_GRAPH_EXECUTOR_CODEGEN_H_
