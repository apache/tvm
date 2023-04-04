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
 * \file create.h
 * \brief The functions to make Relax tensor-creation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_CREATE_H_
#define TVM_RELAX_OP_TENSOR_CREATE_H_

#include <tvm/relax/attrs/create.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Fill array with scalar value.
 * \param shape The shape of the created tensor.
 * \param fill_value The value to fill. Must be a scalar tensor.
 * \param dtype The data type of the created tensor.
 * If dtype is not given, it will by default use the dtype of fill_value.
 * \return The result tensor.
 */
Expr full(ObjectRef shape, Expr fill_value, DataType dtype);

/*!
 * \brief Construct a tensor such that
 * - its shape is the same as the input data tensor's shape,
 * - its value is filled with the input scalar fill value.
 * \param x The input tensor, which provides the shape, and dtype
 * when the input dtype is void.
 * \param fill_value The value to fill. Must be a scalar tensor.
 * \param dtype The data type of the created tensor. If it is
 * void, the input tensor's dtype will be used.
 * \return The result tensor.
 */
Expr full_like(Expr x, Expr fill_value, DataType dtype);

/*!
 * \brief Construct a tensor of all ones, with the input shape and dtype.
 * \param shape The shape of the created tensor.
 * \param dtype The data type of the created tensor.
 * \return The result tensor.
 */
Expr ones(Expr shape, DataType dtype);

/*!
 * \brief Construct a tensor with all ones, with shape of the input tensor shape.
 * \param x The input tensor, which provides the shape, and dtype
 * when the input dtype is void.
 * \param dtype The data type of the created tensor. If it is
 * void, the input tensor's dtype will be used.
 * \return The result tensor.
 */
Expr ones_like(Expr x, DataType dtype);

/*! \brief Construct a tensor of all zeros, with the input shape and dtype. */
Expr zeros(Expr shape, DataType dtype);

/*! \brief Construct a tensor with all zeros, with shape of the input tensor shape. */
Expr zeros_like(Expr x, DataType dtype);

/*! \brief Construct a tensor with evenly spaced elements. */
Expr arange(PrimValue start, PrimValue stop, PrimValue step, DataType dtype);

/*! \brief Return the lower triangular part of a matrix or a batch of matrices. */
Expr tril(Expr x, int k);

/*! \brief Return the upper triangular part of a matrix or a batch of matrices. */
Expr triu(Expr x, int k);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_CREATE_H_
