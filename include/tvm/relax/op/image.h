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
#ifndef TVM_RELAX_OP_IMAGE_H_
#define TVM_RELAX_OP_IMAGE_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {
namespace image {

// (TVM-TOOL) cc_op begin decl/image/*
/*!
 * TBD
 * \param x TODO(tvm-unity-team): add doc
 * \param size TODO(tvm-unity-team): add doc
 * \param roi TODO(tvm-unity-team): add doc
 * \param layout TODO(tvm-unity-team): add doc
 * \param method TODO(tvm-unity-team): add doc
 * \param coordinate_transformation_mode TODO(tvm-unity-team): add doc
 * \param rounding_method TODO(tvm-unity-team): add doc
 * \param bicubic_alpha TODO(tvm-unity-team): add doc
 * \param bicubic_exclude TODO(tvm-unity-team): add doc
 * \param extrapolation_value TODO(tvm-unity-team): add doc
 * \param out_dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call resize2d(relax::Expr x, relax::Expr size, Array<FloatImm> roi, String layout,
                     String method, String coordinate_transformation_mode, String rounding_method,
                     double bicubic_alpha, double bicubic_exclude, double extrapolation_value,
                     runtime::DataType out_dtype);
// (TVM-TOOL) cc_op end decl/image/*

}  // namespace image
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_IMAGE_H_
