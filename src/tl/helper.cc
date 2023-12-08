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
 * \file tl/helper.cc
 * \brief helper functions for tile library.
 */

#include "helper.h"

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

Array<IterVar> ToIterVars(const Map<Var, Range>& vmap) {
  Array<IterVar> result;
  for (const auto& [var, range] : vmap) {
    result.push_back(IterVar(range, var, IterVarType::kDataPar));
  }
  return result;
}

Map<Var, Range> ToVMap(const Array<IterVar>& ivs) {
  Map<Var, Range> result;
  for (const auto& iv : ivs) {
    result.Set(iv->var, iv->dom);
  }
  return result;
}

}  // namespace tl
}  // namespace tvm
