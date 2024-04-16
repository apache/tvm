/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file loop_vectorize.h
 * \brief A tool to automatically vectorize a for loop
 */

#ifndef TVM_TL_LOOP_VECTORIZE_H_
#define TVM_TL_LOOP_VECTORIZE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

int GetVectorizeSize(const For& loop);
For VectorizeLoop(const For& loop, int vectorize_hint = -1);

bool IndiceCanVectorize(PrimExpr expr, Var var, PrimExpr iter_var_size, int target_vectorized_size,
                        arith::Analyzer* analyzer);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LOOP_VECTORIZE_H_
