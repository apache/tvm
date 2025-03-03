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
 * \file tvm/relax/attrs/index.h
 * \brief Attributes for indexing operators.
 */
#ifndef TVM_RELAX_ATTRS_INDEX_H_
#define TVM_RELAX_ATTRS_INDEX_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in take operator */
struct TakeAttrs : public tvm::AttrsNode<TakeAttrs> {
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(TakeAttrs, "relax.attrs.TakeAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis over which to select values.");
  }
};  // struct TakeAttrs

/*! \brief Attributes used in strided_slice operator */
struct StridedSliceAttrs : public tvm::AttrsNode<StridedSliceAttrs> {
  bool assume_inbound;

  TVM_DECLARE_ATTRS(StridedSliceAttrs, "relax.attrs.StridedSliceAttrs") {
    TVM_ATTR_FIELD(assume_inbound)
        .set_default(true)
        .describe(
            "Whether to assume the indices are in bound. If it is set to false, "
            "out of bound indices will be clipped to the bound.");
  }
};  // struct StridedSliceAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_INDEX_H_
