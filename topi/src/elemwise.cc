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
* \brief Registration of elemwise operators
* \file elemwise.cc
*/
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <topi/elemwise.h>

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("topi.exp")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = exp(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.fast_exp")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = fast_exp(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.erf")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = erf(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.fast_erf")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = fast_erf(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.tan")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = tan(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.cos")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = cos(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.cosh")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = cosh(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sin")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sin(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sinh")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sinh(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.tanh")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = tanh(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.fast_tanh")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = fast_tanh(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.atan")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = atan(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sigmoid")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sigmoid(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sqrt")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sqrt(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.rsqrt")
.set_body([](TVMArgs args, TVMRetValue *rv) {
*rv = rsqrt(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.log")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = log(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.log2")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = log2(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.log10")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = log10(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.identity")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = identity(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.negative")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = negative(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.clip")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = clip(args[0], args[1], args[2]);
  });

TVM_REGISTER_GLOBAL("topi.cast")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = cast(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.reinterpret")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = reinterpret(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.elemwise_sum")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = elemwise_sum(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sign")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sign(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.full")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = full(args[0], args[1], args[2]);
  });

TVM_REGISTER_GLOBAL("topi.full_like")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = full_like(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.logical_not")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = logical_not(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.bitwise_not")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = bitwise_not(args[0]);
  });

}  // namespace topi
