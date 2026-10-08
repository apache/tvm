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
 * \file tvm/tirx/op/debug.h
 * \brief Debug operations for TIRx.
 */
#ifndef TVM_TIRX_OP_DEBUG_H_
#define TVM_TIRX_OP_DEBUG_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm::tirx {

/*!
 * \brief Print the content of a buffer during runtime.
 *
 * Arguments, in order:
 * - args[0]: data, The input data.
 * - args[1]: dtype, The data type.
 * - args[2]: is_string, Whether to print as a string.
 * - args[3]: is_scalar, Whether to print as a scalar.
 * - args[4]: ndim, The number of dimensions.
 * - args[5...]: args, trailing Expr operands.
 */
TVM_DLL const Op& print_buffer_op();

}  // namespace tvm::tirx

#endif  // TVM_TIRX_OP_DEBUG_H_
