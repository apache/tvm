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
#ifndef TVM_SCRIPT_PRINTER_IR_UTILS_H_
#define TVM_SCRIPT_PRINTER_IR_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/support/with.h>

#include <string>
#include <utility>

#include "../utils.h"

namespace tvm {
namespace script {
namespace printer {

class IRFrameNode : public FrameNode {
 public:
  Map<String, Array<GlobalInfo>>* global_infos = nullptr;
  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    // `global_infos` is not visited
  }

  static constexpr const char* _type_key = "script.printer.IRFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRFrameNode, FrameNode);
};

class IRFrame : public Frame {
 public:
  explicit IRFrame(const IRDocsifier& d) {
    ObjectPtr<IRFrameNode> n = make_object<IRFrameNode>();
    n->stmts.clear();
    n->d = d.get();
    n->global_infos = nullptr;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRFrame, Frame, IRFrameNode);
};

/*! \brief Redirected method for the ReprPrinter */
inline std::string ReprPrintIR(const ObjectRef& obj, const PrinterConfig& cfg) {
  IRDocsifier d(cfg);
  With<IRFrame> f(d);
  (*f)->AddDispatchToken(d, "ir");
  return Docsify(obj, d, *f, cfg);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_IR_UTILS_H_
