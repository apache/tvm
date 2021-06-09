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
 * \file tvm/relay/attrs/mlas_op.h
 * \brief Auxiliary attributes for mlas operators.
 */
#ifndef TVM_RELAY_ATTRS_MLAS_OP_H_
#define TVM_RELAY_ATTRS_MLAS_OP_H_

#include <tvm/ir/attrs.h>

namespace tvm {
namespace relay {

/*! \brief Attributes for MlasMatmul operator */
struct MlasMatmulAttrs : public tvm::AttrsNode<MlasMatmulAttrs> {
  bool packb;
  int K;
  int N;

  TVM_DECLARE_ATTRS(MlasMatmulAttrs, "relay.attrs.MlasMatmulAttrs") {
    TVM_ATTR_FIELD(packb).set_default(false).describe("packb");
    TVM_ATTR_FIELD(K).set_default(-1).describe("K");
    TVM_ATTR_FIELD(N).set_default(-1).describe("N");
  }
};

/*! \brief Attributes for MlasPackb operator */
struct MlasPackbAttrs : public tvm::AttrsNode<MlasPackbAttrs> {
  int K;
  int N;
  int size;
  bool transb;

  TVM_DECLARE_ATTRS(MlasPackbAttrs, "relay.attrs.MlasPackbAttrs") {
    TVM_ATTR_FIELD(K).describe("K");
    TVM_ATTR_FIELD(N).describe("N");
    TVM_ATTR_FIELD(size).describe("size");
    TVM_ATTR_FIELD(transb).describe("transb").set_default(true);
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_MLAS_OP_H_
