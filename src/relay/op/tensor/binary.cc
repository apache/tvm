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
 * \file binary.cc
 * \brief binary broadcast operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <topi/broadcast.h>
#include "../type_relations.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

#define RELAY_BINARY_COMPUTE(FTOPI)                        \
  [] (const Attrs& attrs,                                  \
      const Array<te::Tensor>& inputs,                     \
      const Type& out_type) -> Array<te::Tensor> {         \
    CHECK_EQ(inputs.size(), 2U);                           \
    return {FTOPI(inputs[0], inputs[1])};                  \
  }                                                        \

// Addition
RELAY_REGISTER_BINARY_OP("add")
.describe("Elementwise add with with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::add));

// Subtraction
RELAY_REGISTER_BINARY_OP("subtract")
.describe("Elementwise substract with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::subtract));

// Right shift
RELAY_REGISTER_BINARY_OP("right_shift")
.describe("Elementwise right shift with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::right_shift));


RELAY_REGISTER_BINARY_OP("left_shift")
.describe("Elementwise left shift with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::left_shift));


RELAY_REGISTER_BINARY_OP("maximum")
.describe("Elementwise maximum of two tensors with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::maximum));


RELAY_REGISTER_BINARY_OP("minimum")
.describe("Elementwise minimum of two tensors with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::minimum));


RELAY_REGISTER_BINARY_OP("divide")
.describe("Elementwise divide with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::divide));


RELAY_REGISTER_BINARY_OP("floor_divide")
.describe("Elementwise floor divide with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::floor_divide));


RELAY_REGISTER_BINARY_OP("multiply")
.describe("Elementwise multiply with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::multiply));


RELAY_REGISTER_BINARY_OP("power")
.describe("Elementwise power with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::power));


RELAY_REGISTER_BINARY_OP("mod")
.describe("Elementwise mod with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::mod));


RELAY_REGISTER_BINARY_OP("floor_mod")
  .describe("Elementwise floor mod with broadcasting")
  .set_support_level(1)
  .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::floor_mod));


RELAY_REGISTER_BINARY_OP("logical_and")
.describe("Elementwise logical AND with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::logical_and));


RELAY_REGISTER_BINARY_OP("logical_or")
.describe("Elementwise logical OR with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::logical_or));


RELAY_REGISTER_BINARY_OP("logical_xor")
.describe("Elementwise logical XOR with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::logical_xor));


RELAY_REGISTER_BINARY_OP("bitwise_and")
.describe("Elementwise bitwise AND with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::bitwise_and));


RELAY_REGISTER_BINARY_OP("bitwise_or")
.describe("Elementwise bitwise OR with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::bitwise_or));


RELAY_REGISTER_BINARY_OP("bitwise_xor")
.describe("Elementwise bitwise XOR with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::bitwise_xor));


RELAY_REGISTER_CMP_OP("equal")
.describe("Elementwise equal compare with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::equal));


RELAY_REGISTER_CMP_OP("not_equal")
.describe("Elementwise not equal with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::not_equal));


RELAY_REGISTER_CMP_OP("less")
.describe("Elementwise less than with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::less));


RELAY_REGISTER_CMP_OP("less_equal")
.describe("Elementwise less than or equal compare with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::less_equal));


RELAY_REGISTER_CMP_OP("greater")
.describe("Elementwise greater than compare with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::greater));


RELAY_REGISTER_CMP_OP("greater_equal")
.describe("Elementwise greater than or equal compare with broadcasting")
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::greater_equal));

}  // namespace relay
}  // namespace tvm
