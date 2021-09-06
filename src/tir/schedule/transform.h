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
#ifndef TVM_TIR_SCHEDULE_TRANSFORM_H_
#define TVM_TIR_SCHEDULE_TRANSFORM_H_

#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/******** Annotation ********/

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value);

/******** Buffer Related ********/

/*!
 * \brief Create a new buffer by changing the storage scope.
 * \param buffer The given buffer.
 * \param scope The target storage scope.
 * \return The new buffer with target storage scope.
 */
Buffer WithScope(const Buffer& buffer, const String& scope);

/*!
 * \brief Replaces the buffer within the specific sequence of regions
 * \param regions The regions whose buffers are to be replaced
 * \param source The buffer to be replaced
 * \param target The buffer to be replaced to
 * \return The new sequence of regions after replacement
 */
Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions, const Buffer& source,
                                  const Buffer& target);

/*!
 * \brief Replaces the buffer within the specific sequence of match_buffers
 * \param match_buffers The match_buffers whose buffers are to be replaced
 * \param source The buffer to be replaced
 * \param target The buffer to be replaced to
 * \return The new sequence of match_buffers after replacement
 */
Array<MatchBufferRegion> ReplaceBuffer(Array<MatchBufferRegion> match_buffers, const Buffer& source,
                                       const Buffer& target);
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRANSFORM_H_
