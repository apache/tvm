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
 * \file tvm/tir/op_attr_types.h
 * \brief Attribute types in the Op registry for TIR ops.
 *
 * These attributes can be set via OpRegEntry::set_attr
 *
 * \sa tvm/ir/op.h
 */
#ifndef TVM_TIR_OP_ATTR_TYPES_H_
#define TVM_TIR_OP_ATTR_TYPES_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/packed_func.h>

#include <ostream>

namespace tvm {
namespace tir {
/*!
 * \brief Global symbol of the op after lowering.
 */
using TGlobalSymbol = String;

/*!
 * \brief Whether the op is overloaded for vector form.
 */
using TVectorizable = bool;

/*!
 * \brief The intrinsic lowering function for given op.
 */
using FLowerIntrinsic = runtime::TypedPackedFunc<PrimExpr(PrimExpr)>;

/*!
 * \brief The legalization function for given tir op.
 */
using FLegalize = runtime::TypedPackedFunc<PrimExpr(PrimExpr)>;

/*!
 * \brief The operator's name in TVMScript printer
 */
using TScriptPrinterName = String;

/*!
 * \brief Specifies that TVMScript printer prints the dtype as the first/last argument.
          If not specified, dtype will not be printed.
 */
enum class ScriptDtypePrintLocation : int {
  /*!
   * \brief Do not print dtype as an argument.
   */
  kNone = 0,
  /*!
   * \brief Print dtype as the first argument.
   */
  kFirst = 1,
  /*!
   * \brief FPrint dtype as the last argument.
   */
  kLast = 2,
};

using TScriptDtypePrintLocation = Integer;

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
      LOG(FATAL) << "Unknown CallEffectKind: " << static_cast<int>(side_effect);
  }
}

/*! \brief Use integer to record the kind. */
using TCallEffectKind = Integer;

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_OP_ATTR_TYPES_H_
