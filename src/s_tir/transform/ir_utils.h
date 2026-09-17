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

#ifndef TVM_S_TIR_TRANSFORM_IR_UTILS_H_
#define TVM_S_TIR_TRANSFORM_IR_UTILS_H_

#include <tvm/ffi/container/tuple.h>
#include <tvm/s_tir/stmt.h>

#include <unordered_map>

#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {

/*! \brief Convert a schedulable statement or module to SSA form. */
tirx::Stmt ConvertSSA(tirx::Stmt stmt);
IRModule ConvertSSA(IRModule mod);

/*!
 * \brief Convert match buffer target buffer access indices to original one.
 * \param indices The indices of the target buffer
 * \return The indices of source buffer.
 */
ffi::Array<PrimExpr> ConvertIndices(const MatchBufferRegion& match_buffer,
                                    const ffi::Array<PrimExpr>& indices);

/*!
 * \brief Convert match buffer target buffer region to original one.
 * \param region The sub-region of the target buffer
 * \return The region of source buffer.
 */
tirx::Region ConvertRegion(const MatchBufferRegion& match_buffer, const tirx::Region& region);

/*! \brief The quad used by StorageAlign for (buffer_idx, axis, factor, offset) */
using StorageAlignTuple = ffi::Tuple<int32_t, int32_t, int32_t, int32_t>;
/*! \brief A list of StorageAlignTuple, used by StorageAlign */
using StorageAlignAnnotation = ffi::Array<StorageAlignTuple>;
/*!
 * \brief Collect storage alignment annotations for all buffer vars within body.
 * \param body The stmt to collect.
 * \return The result dict from buffer var to storage align annotations.
 */
std::unordered_map<tirx::Var, StorageAlignAnnotation> CollectStorageAlignAnnotation(
    const tirx::Stmt& body);

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_TRANSFORM_IR_UTILS_H_
