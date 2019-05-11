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
*  Copyright (c) 2016 by Contributors
* \file nnvm/compiler/util.h
* \brief Utility functions for nnvm compiler
*/
#ifndef NNVM_COMPILER_UTIL_H_
#define NNVM_COMPILER_UTIL_H_

#include <tvm/expr.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace compiler {

/*
 * \brief Helper function to convert TShape to TVM array. Useful for
 * passing data from NNVM param structures to TOPI ops.
 *
 * \param shape The shape to convert
 *
 * \return An Array of Expr, where each element is a constant int32
 */
inline tvm::Array<tvm::Expr> ShapeToArray(TShape shape) {
  tvm::Array<tvm::Expr> result;
  for (auto i : shape) {
    result.push_back(tvm::make_const(tvm::Int(32), i));
  }
  return result;
}

/*
 * \brief Helper function to convert TShape to TVM array. Useful for
 * passing data from NNVM param structures to TOPI ops.
 *
 * \param shape The shape to convert
 *
 * \return An Array of Expr, where each element is a constant int32
 */
inline tvm::Array<tvm::Integer> ShapeToIntArray(TShape shape) {
  return tvm::Array<tvm::Integer>(ShapeToArray(shape).node_);
}
}  // namespace compiler
}  // namespace nnvm
#endif  // NNVM_COMPILER_UTIL_H_
