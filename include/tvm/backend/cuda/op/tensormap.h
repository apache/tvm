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
 * \file tvm/backend/cuda/op/tensormap.h
 * \brief CUDA tensor-map encoding operations.
 */
#ifndef TVM_BACKEND_CUDA_OP_TENSORMAP_H_
#define TVM_BACKEND_CUDA_OP_TENSORMAP_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/op.h>

namespace tvm {
namespace backend {
namespace cuda {

/*! \brief Fixed encoding options for tensormap_encode_tiled. */
struct TensorMapEncodeTiledAttr : public AttrsNode {
  DLDataType descriptor_dtype;
  int64_t rank;
  int64_t interleave;
  int64_t swizzle;
  int64_t l2_promotion;
  int64_t oob_fill;
  int64_t force_cu_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.cuda.TensorMapEncodeTiledAttr", TensorMapEncodeTiledAttr,
                                    AttrsNode);
};

/*!
 * \brief Encode a tiled tensor map at invocation time.
 *
 * TensorMapEncodeTiledAttr stores the descriptor dtype, rank and fixed options.
 * Arguments are descriptor and data pointers, global dimensions (rank), byte
 * strides (rank - 1), box dimensions (rank), then element strides (rank).
 */
TVM_DLL const Op& tensormap_encode_tiled_op();

}  // namespace cuda
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_CUDA_OP_TENSORMAP_H_
