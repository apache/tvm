/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF liceenses this file
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
 *  Copyright (c) 2016 by Contributors
 * \file tensor.cc
 */
#include <tvm/sparse.h>

namespace tvm {

SparseFormat SparseFormatNode::make(Array<Expr> types) {
  NodePtr<SparseFormatNode> n = make_node<SparseFormatNode>();
  n->types = std::move(types);
  // for (auto type : types) {
  //   n->types.push_back(type.as<IntImm>()->value);
  // }
  return SparseFormat(n);
}

SparseFormat DeclDenseFormat(int ndim) {
  NodePtr<SparseFormatNode> n = make_node<SparseFormatNode>();
  for (int i = 0; i < ndim; ++i) {
    n->types.push_back(kDense);
  }
  return SparseFormat(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<SparseFormatNode>([](const SparseFormatNode *n, IRPrinter *p) {
    p->stream << "sparse_format([";
    for (size_t i = 0; i < n->types.size(); ++i) {
      auto ptype = n->types[i].as<IntImm>();
      CHECK(ptype);
      if (ptype->value == kDense) {
        p->stream << "dense";
      } else if (ptype->value == kSparse) {
        p->stream << "sparse";
      }

      if (i != n->types.size() - 1) {
        p->stream << ", ";
      }
    }
    p->stream << "])";
});

// SparseTensor SparseTensorNode::make(Array<Expr> shape,
//                                     Type dtype,
//                                     SparseFormat sformat,
//                                     Operation op,
//                                     int value_index) {
//   auto n = make_node<SparseTensorNode>();
//   n->shape = std::move(shape);
//   n->dtype = dtype;
//   n->sformat = std::move(sformat);
//   n->op = op;
//   n->value_index = value_index;
//   return SparseTensor(n);
// }

}// namespace tvm
