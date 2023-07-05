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
 * \brief Auxiliary attributes for random operators.
 */
#ifndef TVM_RELAY_ATTRS_RANDOM_H_
#define TVM_RELAY_ATTRS_RANDOM_H_

#include <tvm/ir/attrs.h>

namespace tvm {
namespace relay {

struct ThreefryGenerateAttrs : public tvm::AttrsNode<ThreefryGenerateAttrs> {
  Array<Integer> out_shape;

  TVM_DECLARE_ATTRS(ThreefryGenerateAttrs, "relay.attrs.ThreefryGenerateAttrs") {
    TVM_ATTR_FIELD(out_shape).describe("Shape of random numbers to generate");
  }
};

struct UniformAttrs : public tvm::AttrsNode<UniformAttrs> {
  Array<Integer> out_shape;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(UniformAttrs, "relay.attrs.UniformAttrs") {
    TVM_ATTR_FIELD(out_shape).describe("Shape of random numbers to generate");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Data type of the generated numbers");
  }
};

struct NormalAttrs : public tvm::AttrsNode<NormalAttrs> {
  Array<Integer> out_shape;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(NormalAttrs, "relay.attrs.NormalAttrs") {
    TVM_ATTR_FIELD(out_shape).describe("Shape of random numbers to generate");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Data type of the generated numbers");
  }
};

struct MultinomialAttrs : public tvm::AttrsNode<MultinomialAttrs> {
  Integer num_samples;

  TVM_DECLARE_ATTRS(MultinomialAttrs, "relay.attrs.MultinomialAttrs") {
    TVM_ATTR_FIELD(num_samples)
        .set_default(1)
        .describe("Number of samples to draw from the distribution.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_RANDOM_H_
