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
* \brief Registration of broadcast operators
* \file broadcast.cc
*/
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <topi/broadcast.h>
#include <topi/util.h>

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

#define TOPI_REGISTER_BCAST_OP(OpName, Op)                              \
  TVM_REGISTER_GLOBAL(OpName)                                           \
  .set_body([](TVMArgs args, TVMRetValue *rv) {                         \
      bool lhs_is_tensor = args[0].IsObjectRef<tvm::te::Tensor>();      \
      bool rhs_is_tensor = args[1].IsObjectRef<tvm::te::Tensor>();      \
      if (lhs_is_tensor && rhs_is_tensor) {                             \
        *rv = Op(args[0].operator tvm::te::Tensor(),                    \
                 args[1].operator tvm::te::Tensor());                   \
      } else if (!lhs_is_tensor && rhs_is_tensor) {                     \
        *rv = Op(args[0].operator tvm::PrimExpr(),                      \
                 args[1].operator tvm::te::Tensor());                   \
      } else if (lhs_is_tensor && !rhs_is_tensor) {                     \
        *rv = Op(args[0].operator tvm::te::Tensor(),                    \
                 args[1].operator tvm::PrimExpr());                     \
      } else if (!lhs_is_tensor && !rhs_is_tensor) {                    \
        *rv = Op(args[0].operator tvm::PrimExpr(),                      \
                 args[1].operator tvm::PrimExpr());                     \
      }                                                                 \
    });                                                                 \

TOPI_REGISTER_BCAST_OP("topi.add", topi::add);
TOPI_REGISTER_BCAST_OP("topi.subtract", topi::subtract);
TOPI_REGISTER_BCAST_OP("topi.multiply", topi::multiply);
TOPI_REGISTER_BCAST_OP("topi.divide", topi::divide);
TOPI_REGISTER_BCAST_OP("topi.floor_divide", topi::floor_divide);
TOPI_REGISTER_BCAST_OP("topi.mod", topi::mod);
TOPI_REGISTER_BCAST_OP("topi.floor_mod", topi::floor_mod);
TOPI_REGISTER_BCAST_OP("topi.maximum", topi::maximum);
TOPI_REGISTER_BCAST_OP("topi.minimum", topi::minimum);
TOPI_REGISTER_BCAST_OP("topi.power", topi::power);
TOPI_REGISTER_BCAST_OP("topi.left_shift", topi::left_shift);
TOPI_REGISTER_BCAST_OP("topi.logical_and", topi::logical_and);
TOPI_REGISTER_BCAST_OP("topi.logical_or", topi::logical_or);
TOPI_REGISTER_BCAST_OP("topi.logical_xor", topi::logical_xor);
TOPI_REGISTER_BCAST_OP("topi.bitwise_and", topi::bitwise_and);
TOPI_REGISTER_BCAST_OP("topi.bitwise_or", topi::bitwise_or);
TOPI_REGISTER_BCAST_OP("topi.bitwise_xor", topi::bitwise_xor);
TOPI_REGISTER_BCAST_OP("topi.right_shift", topi::right_shift);
TOPI_REGISTER_BCAST_OP("topi.greater", topi::greater);
TOPI_REGISTER_BCAST_OP("topi.less", topi::less);
TOPI_REGISTER_BCAST_OP("topi.equal", topi::equal);
TOPI_REGISTER_BCAST_OP("topi.not_equal", topi::not_equal);
TOPI_REGISTER_BCAST_OP("topi.greater_equal", topi::greater_equal);
TOPI_REGISTER_BCAST_OP("topi.less_equal", topi::less_equal);

TVM_REGISTER_GLOBAL("topi.broadcast_to")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_to(args[0], args[1]);
  });

}  // namespace topi
