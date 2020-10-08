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
 *  Copyright (c) 2018 by Contributors.
 *
 * \file tvm/hago/quantize.h
 * \brief Header of definitions for quantization
 */
#ifndef TVM_HAGO_QUANTIZE_H_
#define TVM_HAGO_QUANTIZE_H_

#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <string>

namespace tvm {
namespace hago {

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  DataType in_dtype;
  DataType out_dtype;
  bool sign;
  std::string rounding;
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "hago.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(in_dtype)
      .describe("input data type");
    TVM_ATTR_FIELD(out_dtype)
      .describe("output data type");
    TVM_ATTR_FIELD(sign).set_default(true)
        .describe("whether to use signed data type.");
    TVM_ATTR_FIELD(rounding).set_default("round")
        .describe("rounding mode. Can be 'floor', 'ceil', 'round'");
    TVM_ATTR_FIELD(axis)
      .describe("specify axis for per-channel quantization.");
  }
};
}  // namespace hago
}  // namespace tvm
#endif  // TVM_HAGO_QUANTIZE_H_
