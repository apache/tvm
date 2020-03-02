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

#ifndef TOPI_CONTRIB_RANDOM_H_
#define TOPI_CONTRIB_RANDOM_H_

#include <string>
#include <tvm/te/operation.h>
#include "topi/detail/extern.h"

namespace topi {
namespace contrib {
using namespace tvm;
using namespace topi::detail;
/*!
* \brief Create an op that generates pseudo random number from a uniform P.D.(Prob distrib)
*
* \param shape The output shape
* \param minval The lower bound of Uniform Distribution(inclusive)
* \param maxval The upper bound of Uniform Distribution(exclusive)
* \param dtype  The datatype of the generated values in Uniform Distribution
* \param seed   Seed value for the Distribution
* \param name   Optional name parameter for the operation
*
* \return The output uniform P.D. tensor with filled psuedo random numbers
*/
inline tvm::Array<Tensor>
random_uniform(const Array<PrimExpr>& shape, const PrimExpr& minval,
               const PrimExpr& maxval, DataType dtype, Integer seed,
               std::string name = "random.uniform") {
  CHECK(dtype.is_float() || dtype.is_int());

  if (dtype.code() == kDLFloat) {  // float values
    if (dtype.bits() == 32 && dtype.lanes() == 1) {
      return make_extern({{shape}}, {dtype}, {},
                         [&](Array<Buffer> ins, Array<Buffer> outs) {
                           return call_packed({
                               PrimExpr("tvm.contrib.random.uniform"),
                               minval,
                               maxval,
                               pack_buffer(outs[0]),
                               seed,
                           });
                         },
                         name, "", {});
    } else {  // double values
      return make_extern({{shape}}, {dtype}, {},
                         [&](Array<Buffer> ins, Array<Buffer> outs) {
                           return call_packed({
                               PrimExpr("tvm.contrib.random.uniform.real"),
                               minval,
                               maxval,
                               pack_buffer(outs[0]),
                               seed,
                           });
                         },
                         name, "", {});
    }
  } else {  // integer values
  return make_extern({{shape}}, {dtype}, {},
                       [&](Array<Buffer> ins, Array<Buffer> outs) {
                         return call_packed({
                             PrimExpr("tvm.contrib.random.uniform.int"),
                             minval,
                             maxval,
                             pack_buffer(outs[0]),
                             seed,
                         });
                       },
                       name, "", {});
  }
}

}  // namespace contrib
}  // namespace topi

#endif  // TOPI_CONTRIB_RANDOM_H_
