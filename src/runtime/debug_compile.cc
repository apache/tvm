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
 * \file src/runtime/debug_compile.cc
 * \brief File used for debug migration
 */
// #include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/container/variant.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_hash.h>

#include <tvm/node/structural_hash.h>
// #include <tvm/runtime/profiling.h>
// #include <tvm/runtime/registry.h>
#include <tvm/ir/expr.h>

// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/tir/expr.h>

namespace tvm {
namespace debug {

using namespace tvm::runtime;

String Test() {
  PackedFunc pf([](TVMArgs args, TVMRetValue* rv) {
    String value = args[0];
    *rv = value;
  });
  String res = pf("xyz");
  LOG(INFO) << res;
  return res;
}

}  // namespace debug
}  // namespace tvm
