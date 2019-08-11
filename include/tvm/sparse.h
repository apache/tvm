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
 * \file tvm/tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_SPARSE_H_
#define TVM_SPARSE_H_

#include "base.h"
#include "expr.h"

namespace tvm {

class SparseFormatNode;

enum SFormatType : int {
  kDense = 0,
  kSparse = 1,
};

/*!
 * \brief Tensor structure representing a possible input,
 *  or intermediate computation result.
 */
class SparseFormat : public NodeRef {
 public:
  SparseFormat() {}

  explicit SparseFormat(NodePtr<Node> n) : NodeRef(n) {}

  inline const SparseFormatNode* operator->() const;

  using ContainerType = SparseFormatNode;
};


class SparseFormatNode : public Node {
 public:
  Array<Expr> types;

  SparseFormatNode() {}
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("types", &types);
  };

  TVM_DLL static SparseFormat make(Array<Expr> types);

  static constexpr const char* _type_key = "SparseFormat";
  TVM_DECLARE_NODE_TYPE_INFO(SparseFormatNode, Node);
};

inline const SparseFormatNode* SparseFormat::operator->() const {
  return static_cast<const SparseFormatNode*>(node_.get());
}

SparseFormat DeclDenseFormat(int ndim);

}  // namespace tvm
#endif  // TVM_SPARSE_H_
