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
 * \file tvm/relay/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAY_ATTRS_ALGORITHM_H_
#define TVM_RELAY_ATTRS_ALGORITHM_H_

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in argsort operators */
struct ArgsortAttrs : public tvm::AttrsNode<ArgsortAttrs> {
  int axis;
  bool is_ascend;
  DataType dtype;

  TVM_DECLARE_ATTRS(ArgsortAttrs, "relay.attrs.ArgsortAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1)
      .describe("Axis along which to sort the input tensor."
                "If not given, the flattened array is used.");
    TVM_ATTR_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order."
                "By default, sort in ascending order");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
      .describe("DType of the output indices.");
  }
};

struct TopKAttrs : public tvm::AttrsNode<TopKAttrs> {
  int k;
  int axis;
  bool is_ascend;
  std::string ret_type;
  DataType dtype;

  TVM_DECLARE_ATTRS(TopKAttrs, "relay.attrs.TopkAttrs") {
    TVM_ATTR_FIELD(k).set_default(1)
      .describe("Number of top elements to select");
    TVM_ATTR_FIELD(axis).set_default(-1)
      .describe("Axis along which to sort the input tensor.");
    TVM_ATTR_FIELD(ret_type).set_default("both")
      .describe("The return type [both, values, indices]."
                "both - return both top k data and indices."
                "values - return top k data only."
                "indices - return top k indices only.");
    TVM_ATTR_FIELD(is_ascend).set_default(false)
      .describe("Whether to sort in ascending or descending order."
                "By default, sort in descending order");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
      .describe("Data type of the output indices.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_ALGORITHM_H_
