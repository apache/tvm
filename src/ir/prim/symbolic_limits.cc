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
 * \file symbolic_limits.cc
 * \brief Shared symbolic infinity atoms for primitive operations and integer analysis.
 */
#include "const_fold.h"

namespace tvm {
namespace prim {
namespace detail {

PrimExpr SymbolicLimits::pos_inf_ = PrimVar("pos_inf", PrimType::Int(64));
PrimExpr SymbolicLimits::neg_inf_ = PrimVar("neg_inf", PrimType::Int(64));

}  // namespace detail
}  // namespace prim
}  // namespace tvm
