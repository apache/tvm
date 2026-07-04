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
#ifndef TVM_SCRIPT_PRINTER_DOC_PRINTER_EXPR_STRING_DOC_H_
#define TVM_SCRIPT_PRINTER_DOC_PRINTER_EXPR_STRING_DOC_H_

#include <tvm/script/printer/doc.h>

#include <utility>

namespace tvm {
namespace script {
namespace printer {

/*! \brief Internal Doc that renders an expression as the contents of a Python string literal. */
class ExprStringDocNode : public ExprDocNode {
 public:
  ExprDoc value{ffi::UnsafeInit()};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ExprStringDoc", ExprStringDocNode, ExprDocNode);
};

/*! \brief Managed reference to an internal ExprStringDocNode. */
class ExprStringDoc : public ExprDoc {
 public:
  explicit ExprStringDoc(ExprDoc value, const ffi::Optional<AccessPath>& object_path)
      : ExprDoc(MakeNode(std::move(value), object_path)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprStringDoc, ExprDoc, ExprStringDocNode);

 private:
  static ffi::ObjectPtr<ExprStringDocNode> MakeNode(ExprDoc value,
                                                    const ffi::Optional<AccessPath>& object_path) {
    ffi::ObjectPtr<ExprStringDocNode> node = ffi::make_object<ExprStringDocNode>();
    node->value = std::move(value);
    if (object_path.defined()) {
      node->source_paths.push_back(object_path.value());
    }
    return node;
  }
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_PRINTER_EXPR_STRING_DOC_H_
