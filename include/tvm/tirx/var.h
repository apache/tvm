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
 * \file tvm/tirx/var.h
 * \brief Variables in the TIR.
 */
#ifndef TVM_TIR_VAR_H_
#define TVM_TIR_VAR_H_

#include <tvm/ffi/dtype.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>

#include <utility>

namespace tvm {
namespace tirx {

using VarNode = tvm::VarNode;
using Var = tvm::Var;

using Region = ffi::Array<Range>;

}  // namespace tirx

}  // namespace tvm

#endif  // TVM_TIR_VAR_H_
