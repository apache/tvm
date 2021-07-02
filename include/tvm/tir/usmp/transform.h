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
 * \file tvm/tir/analysis.h
 * \brief Analysis utilities and passes for TIR Unified Static Memory Planner.
 */
#ifndef TVM_TIR_USMP_TRANSFORM_H_
#define TVM_TIR_USMP_TRANSFORM_H_

#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {
namespace usmp {

TVM_DLL Stmt ConvertForLoopsToSerial(const PrimFunc& func);

}
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_USMP_TRANSFORM_H_
