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
 * \brief Registration of utils operators
 * \file utils.cc
 */

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/topi/detail/tensor_utils.h>

namespace tvm {
namespace topi {
TVM_REGISTER_GLOBAL("topi.utils.is_empty_shape").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::detail::is_empty_shape(args[0]);
});

TVM_REGISTER_GLOBAL("topi.utils.bilinear_sample_nchw").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = detail::bilinear_sample_nchw(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.utils.bilinear_sample_nhwc").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = detail::bilinear_sample_nhwc(args[0], args[1], args[2], args[3]);
});

}  // namespace topi
}  // namespace tvm
