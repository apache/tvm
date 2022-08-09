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
#ifndef TVM_SCRIPT_PRINTER_FRAME_H_
#define TVM_SCRIPT_PRINTER_FRAME_H_

#include <tvm/node/node.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

class FrameNode : public Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  virtual ~FrameNode() = default;

  template <typename TCallback>
  void AddCallback(TCallback&& cb) {
    callbacks_.emplace_back(std::forward<TCallback>(cb));
  }

  virtual void EnterWithScope() {}

  virtual void ExitWithScope() {
    for (const std::function<void()>& callback : callbacks_) {
      callback();
    }
  }

  static constexpr const char* _type_key = "script.printer.Frame";
  TVM_DECLARE_BASE_OBJECT_INFO(FrameNode, Object);

 private:
  std::vector<std::function<void()>> callbacks_;
};

class Frame : public ObjectRef {
 protected:
  Frame() = default;

 public:
  virtual ~Frame() = default;
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Frame, ObjectRef, FrameNode);
};

class MetadataFrameNode : public FrameNode {
 public:
  Array<ObjectRef> metadata;

  void VisitAttrs(tvm::AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("metadata", &metadata);
  }

  static constexpr const char* _type_key = "script.printer.MetadataFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(MetadataFrameNode, FrameNode);
};

class MetadataFrame : public Frame {
 public:
  MetadataFrame();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(MetadataFrame, Frame, MetadataFrameNode);
};

class VarDefFrameNode : public FrameNode {
 public:
  Array<StmtDoc> stmts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("stmts", &stmts);
  }

  static constexpr const char* _type_key = "script.printer.VarDefFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarDefFrameNode, FrameNode);
};

class VarDefFrame : public Frame {
 public:
  explicit VarDefFrame();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(VarDefFrame, Frame, VarDefFrameNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_FRAME_H_
