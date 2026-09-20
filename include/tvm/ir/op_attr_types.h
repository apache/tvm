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

/*! \file tvm/ir/op_attr_types.h
 *  \brief Shared expression operator effects.
 */
#ifndef TVM_IR_OP_ATTR_TYPES_H_
#define TVM_IR_OP_ATTR_TYPES_H_

#include <tvm/ffi/error.h>

#include <cstdint>
#include <ostream>

namespace tvm {

/*!
 * \brief The effect type of the call.
 */
enum class CallEffectKind : int {
  /*! \brief Function corresponds to an annotation(e.g. likely) and can translate to identity. */
  kExprAnnotation = 0,
  /*!
   * \brief Pure function that do not interacts
   *        with any external state.
   */
  kPure = 1,
  /*!
   * \brief Function's that may read from states(e.g. RAM)
   */
  kReadState = 2,
  /*!
   * \brief Function that may read/write from states(e.g. RAM).
   */
  kUpdateState = 3,
  /*!
   * \brief Opaque function, cannot make any assumption
   */
  kOpaque = kUpdateState,
  /*!
   * \brief Special intrinsic to annotate call arguments info
   *        only valid as a direct argument to a call.
   */
  kSpecialCallArg = 4,
  /*!
   * \brief Embed opaque information in the Expr, cannot be codegen.
   */
  kEmbedInfo = 5,
  /*!
   * \brief Function that changes control flow
   */
  kControlJump = 6,
};

inline std::ostream& operator<<(std::ostream& os, CallEffectKind side_effect) {
  switch (side_effect) {
    case CallEffectKind::kExprAnnotation:
      return os << "kExprAnnotation";

    case CallEffectKind::kPure:
      return os << "kPure";

    case CallEffectKind::kReadState:
      return os << "kReadState";

    case CallEffectKind::kUpdateState:
      return os << "kUpdateState";

    case CallEffectKind::kSpecialCallArg:
      return os << "kSpecialCallArg";

    case CallEffectKind::kEmbedInfo:
      return os << "kEmbedInfo";

    case CallEffectKind::kControlJump:
      return os << "kControlJump";

    default:
      TVM_FFI_THROW(InternalError) << "Unknown CallEffectKind: " << static_cast<int>(side_effect);
  }
}

/*! \brief Use integer to record the kind. */
using TCallEffectKind = int64_t;

}  // namespace tvm
#endif  // TVM_IR_OP_ATTR_TYPES_H_
