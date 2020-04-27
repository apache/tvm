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
 * \file topi/transform.h
 * \brief Transform op constructors
 */
#ifndef TOPI_TRANSFORM_H_
#define TOPI_TRANSFORM_H_

#include <tvm/tir/data_layout.h>
#include <tvm/te/operation.h>
#include <topi/tags.h>
#include <topi/detail/ravel_unravel.h>
#include <topi/detail/constant_utils.h>
#include <topi/detail/tensor_utils.h>

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace topi {
using namespace tvm;
using namespace tvm::te;
using namespace topi::detail;

/*!
* \brief Creates an operation to insert new dimensions of length 1
*
* \param x The input tensor
* \param axis The index of the first new dimension (allows negative
* indices as offsets from the last dimension)
* \param num_newaxis The number of new dimensions to insert
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the dim expansion operation
*/
inline Tensor expand_dims(const Tensor& x,
                          int axis,
                          int num_newaxis = 1,
                          std::string name = "T_expand_dims",
                          std::string tag = kBroadcast) {
  int ndim = static_cast<int>(x->shape.size());
  CHECK(-ndim - 1 <= axis && axis <= ndim)
    << "expand_dims only accepts `axis` in [-data.ndim - 1, data.ndim]"
    << ", but got axis = " << axis
    << ", and data.ndim = " << ndim;
  CHECK(num_newaxis >= 0)
    << "expand_dims only accepts `num_newaxis >= 0`"
    << ", but got num_newaxis = " << num_newaxis;
  if (axis < 0) {
    // Calculate offset from last dimension
    axis = ndim + axis + 1;
  }
  Array<PrimExpr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(x->shape[i]);
  }
  for (size_t i = 0; i < static_cast<size_t>(num_newaxis); ++i) {
    new_shape.push_back(1);
  }
  for (size_t i = axis; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(
    new_shape, [&](const Array<Var>& indices) {
      Array<PrimExpr> idx;
      for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        idx.push_back(indices[i]);
      }
      for (size_t i = axis + num_newaxis; i < indices.size(); ++i) {
        idx.push_back(indices[i]);
      }
      return x(idx);
    }, name, tag);
}

/*!
* \brief Permute the dimensions of an array
*
* \param x The input tensor
* \param axes The indices of the permutation. If this is empty,
* the dimensions will be reversed.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the transpose operation
*/
inline Tensor transpose(const Tensor& x,
                        Array<Integer> axes,
                        std::string name = "T_transpose",
                        std::string tag = kInjective) {
  if (!axes.defined() || axes.size() == 0) {
    axes = Array<Integer>();
    for (int i = static_cast<int>(x->shape.size()) - 1; i >= 0; --i) {
      axes.push_back(i);
    }
  }

  Array<PrimExpr> new_shape;
  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = static_cast<int>(axes[i]->value);
    int new_axis = axis;
    if (axis < 0) {
      new_axis = static_cast<int>(x->shape.size()) + axis;
      axes.Set(i, new_axis);
    }
    CHECK((new_axis >= 0) && (new_axis < static_cast<int>(x->shape.size())))
      << "axis=" << axis << " is invalid for the "
      << static_cast<int>(x->shape.size()) << "-dimensional input tensor";

    for (size_t j = 0; j < axes.size(); ++j) {
      if (i !=j) {
        CHECK(new_axis != static_cast<int>(axes[j]->value)) << "repeated axis in transpose";
      }
    }
    new_shape.push_back(x->shape[new_axis]);
  }

  return compute(
    new_shape, [&](const Array<Var>& indices) {
      std::vector<PrimExpr> idx;
      for (size_t i = 0; i < axes.size(); ++i) {
        idx.push_back(1);
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        int axis = static_cast<int>(axes[i]->value);
        idx[axis] = indices[i];
      }
      return x(idx);
    }, name, tag);
}

/*!
* \brief flip/reverse elements of an array in a particular axis
*
* \param x The input tensor
* \param axis The axis along which the tensors will be reveresed
* (allows negative indices)
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the reverse operation
*/
inline Tensor flip(const Tensor& x,
                   int axis = 0,
                   std::string name = "T_flip",
                   std::string tag = kInjective) {
  size_t src_tensor_dim = x->shape.size();
  int axis_inp = axis;

  if (axis < 0) {
    axis = static_cast<int>(x->shape.size()) + axis;
  }

  CHECK((0 <= axis) && (axis < static_cast<int>(x->shape.size())))
    << "axis=" << axis_inp << " is invalid for the "
    << static_cast<int>(x->shape.size()) << "-dimensional input tensor";

  // Reverse the Input Tensor in the axis specified
  return compute(
    x->shape, [&](const Array<Var>& indices) {
      Array<PrimExpr> real_indices;
      for (size_t i = 0; i < src_tensor_dim; ++i) {
        if (i == static_cast<size_t>(axis)) {
          real_indices.push_back(x->shape[i] - indices[i] - 1);
        } else {
          real_indices.push_back(indices[i]);
        }
      }
      return x(real_indices);
    }, name, tag);
}

/*!
* \brief Reshape a tensor
*
* \param x The input tensor
* \param newshape The new shape
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the reshape operation
*/
inline Tensor reshape(const Tensor& x,
                      Array<PrimExpr> newshape,
                      std::string name = "T_reshape",
                      std::string tag = kInjective) {
  auto x_shape = x->shape;
  Array<PrimExpr> target_shape;

  for (const auto &ele : newshape) {
    if (ele.as<IntImmNode>()) {
      target_shape.push_back(cast(DataType::Int(32), ele));
    } else {
      target_shape.push_back(ele);
    }
  }

  if (is_empty_shape(target_shape)) {
    return compute(target_shape,
                   [&](const Array<Var> &indices) { return tvm::cast(x->dtype, 0); },
                   name, tag);
  } else {
    return compute(
      target_shape, [&](const Array<Var>& indices) {
        return x(UnravelIndex(
          RavelIndex(Array<PrimExpr>{indices.begin(), indices.end()}, target_shape),
          x_shape));
      }, name, tag);
  }
}

/*!
 * \brief Converts a flat index or array of flat indices into a tuple of coordinate arrays
 *
 * \param x The input tensor having indices.
 * \param shape The shape tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor of coordinate arrays.
 */

inline Tensor unravel_index(const Tensor& x,
                            const Tensor& shape,
                            std::string name = "T_unravel",
                            std::string tag = kInjective) {
  auto x_shape = x->shape;
  auto shape_shape = shape->shape;

  Array<PrimExpr> oshape;
  oshape.push_back(shape_shape[0]);
  if (x_shape.size() != 0) {
    oshape.push_back(x_shape[0]);
  }

  auto func = [&](const Array<Var>& indices) {
    auto i = indices[0];
    std::vector<PrimExpr> indices_divs;
    PrimExpr ret = 0;
    PrimExpr cur_val = 0;
    PrimExpr index_val = 0;

    if (x_shape.size() != 0) {
      index_val = x[indices[1]];
    } else {
      index_val = x();
    }
    indices_divs.push_back(index_val);
    for (int v = GetConstInt(shape_shape[0]) - 1; v >= 0; --v) {
      ret = tvm::if_then_else(i == v, indexmod(indices_divs.back(), shape[v]), ret);
      cur_val = indexdiv(indices_divs.back(), shape[v]);
      indices_divs.push_back(cur_val);
    }
    return ret;
  };

  return compute(oshape, func, name, tag);
}

/*!
* \brief Remove size 1 dimensions from the shape of a tensor.
* The removed dimensions must have a constant size of 1.
*
* \param x The input tensor
* \param axis Indices of the dimensions to remove. If this is empty,
* all entries with a constant size of 1 will be removed.
 * \param atleast1d Whether the output need to be atleast1d.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the squeeze operation
*/
inline Tensor squeeze(const Tensor& x,
                      Array<Integer> axis,
                      bool atleast1d = false,
                      std::string name = "T_squeeze",
                      std::string tag = kInjective) {
  auto ndim = x->shape.size();
  std::vector<int> axis_val;
  if (!axis.defined() || axis.size() == 0) {
    for (size_t i = 0; i < ndim; ++i) {
      if (IsConstInt(x->shape[i]) && GetConstInt(x->shape[i]) == 1) {
        axis_val.push_back(static_cast<int>(i));
      }
    }
  } else {
    for (size_t i = 0; i < axis.size(); ++i) {
      int64_t val = axis[i]->value;
      if (val < 0) {
        val += static_cast<int>(x->shape.size());
      }
      CHECK_EQ(GetConstInt(x->shape[val]), 1) <<
          "Dimension " << val << " must have size 1";
      axis_val.push_back(val);
    }
  }

  std::unordered_set<int> axis_set(axis_val.begin(), axis_val.end());

  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < ndim; ++i) {
    if (axis_set.count(static_cast<int>(i)) == 0) {
      out_shape.push_back(x->shape[i]);
    }
  }
  if (out_shape.size() == 0 && atleast1d) {
    out_shape.push_back(1);
  }

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      Array<PrimExpr> real_indices;
      int flag = 0;
      for (size_t i = 0; i < ndim; ++i) {
        if (axis_set.count(static_cast<int>(i)) == 0) {
          real_indices.push_back(indices[i - flag]);
        } else {
          real_indices.push_back(0);
          flag += 1;
        }
      }
      return x(real_indices);
    }, name, tag);
}

/*!
* \brief Join a sequence of tensors along an existing axis
*
* \param inputs The input tensors
* \param axis The axis along which the tensors will be joined
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the concatenate operation
*/
inline Tensor concatenate(const Array<Tensor>& inputs,
                          int axis = 0,
                          std::string name = "T_concat",
                          std::string tag = kInjective) {
  int ndim = static_cast<int>(inputs[0]->shape.size());
  CHECK(-ndim <= axis && axis < ndim)
    << "concatenate only accepts `axis` in [-ndim, ndim)"
    << ", but got axis = " << axis
    << ", and ndim = " << ndim;
  if (axis < 0) {
    axis += ndim;
  }
  CHECK_LT(axis, inputs[0]->shape.size()) <<
    "axis out of bounds";

  Array<PrimExpr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }
  arith::Analyzer analyzer;
  PrimExpr join_size = axis_sizes[0];
  for (size_t i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  join_size = analyzer.Simplify(join_size);
  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(i == static_cast<size_t>(axis) ? join_size : inputs[0]->shape[i]);
  }

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      auto ret = inputs[0](indices);
      auto ind = indices[axis];
      for (size_t i = 0; i < inputs.size() - 1; ++i) {
        ind -= axis_sizes[i];

        Array<PrimExpr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        idx.push_back(ind);
        for (size_t i = axis + 1; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }

        ret = tvm::if_then_else(ind >= 0,
                                inputs[i + 1](idx),
                                ret);
      }
      return ret;
    }, name, tag);
}

/*!
* \brief Join a sequence of tensors along a new axis.
*
* \param inputs The input tensors
* \param axis The axis along which the tensors will be stacked
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the stack operation
*/
inline Tensor stack(const Array<Tensor>& inputs,
                    int axis = 0,
                    std::string name = "T_stack",
                    std::string tag = kInjective) {
  int ndim = static_cast<int>(inputs[0]->shape.size());
  CHECK(-ndim - 1 <= axis && axis <= ndim)
    << "stack only accepts `axis` in [-ndim, ndim)"
    << ", but got axis = " << axis
    << ", and ndim = " << ndim;
  if (axis < 0) {
    axis += ndim + 1;
  }
  CHECK_LT(axis, inputs[0]->shape.size() + 1) <<
    "axis out of bounds";

  const int stack_size = static_cast<int>(inputs.size());
  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
    out_shape.push_back(inputs[0]->shape[i]);
  out_shape.push_back(stack_size);
  for (size_t i = static_cast<size_t>(axis); i < static_cast<size_t>(ndim); ++i)
    out_shape.push_back(inputs[0]->shape[i]);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      Array<PrimExpr> idx;
      for (size_t i = 0; i < indices.size(); ++i)
        if (i != static_cast<size_t>(axis))
          idx.push_back(indices[i]);
      auto ind = indices[axis];
      auto ret = inputs[0](idx);
      for (int i = 0; i < static_cast<int>(inputs.size() - 1); ++i) {
        ret = tvm::if_then_else(ind == i + 1,
                                inputs[i + 1](idx),
                                ret);
      }
      return ret;
    }, name, tag);
}

/*!
* \brief Split a tensor into multiple sub-tensors
*
* \param x The input tensor
* \param split_indices The indices to split the input at. This must be in ascending
* order.
* \param axis The axis to split along.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the split operation
*/
inline Array<Tensor> split(const Tensor& x,
                           Array<Integer> split_indices,
                           int axis,
                           std::string name = "T_split",
                           std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(x->shape.size());
  }
  CHECK_LT(axis, x->shape.size()) << "axis out of bounds";

  auto src_axis_size = static_cast<int>(GetConstInt(x->shape[axis]));
  std::vector<int> begin_ids;
  begin_ids.push_back(0);

  for (Integer idx : split_indices) {
    int val = static_cast<int>(idx->value);
    CHECK_GT(val, begin_ids.back())
        << "split_indices must be sorted";
    begin_ids.push_back(val);
  }

  Array< Array<PrimExpr> > out_shapes;
  for (size_t i = 0; i < begin_ids.size(); ++i) {
    int out_axis_size;
    if (i == begin_ids.size() - 1) {
      out_axis_size = src_axis_size - begin_ids[i];
    } else {
      out_axis_size = begin_ids[i + 1] - begin_ids[i];
    }

    Array<PrimExpr> shape;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
      shape.push_back(x->shape[i]);
    }
    shape.push_back(out_axis_size);
    for (size_t i = axis + 1; i < x->shape.size(); ++i) {
      shape.push_back(x->shape[i]);
    }

    out_shapes.push_back(shape);
  }

  Array<Tensor> result;
  for (size_t i = 0; i < begin_ids.size(); ++i) {
    result.push_back(
      compute(
        out_shapes[i], [&](const Array<Var>& indices) {
          auto begin = begin_ids[i];
          Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(indices[j]);
          }
          real_indices.push_back(indices[axis] + begin);
          for (size_t j = axis + 1; j < indices.size(); ++j) {
            real_indices.push_back(indices[j]);
          }

          return x(real_indices);
        }, name, tag));
  }

  return result;
}

/*!
* \brief strided_slice of a tensor
*
* \param x The input tensor
* \param begin The indices to begin with in the slicing
* \param end Indicies indicating end of the slice
* \param strides Specifies the stride values, it can be negative
* in that case, the input tensor will be reversed in that particular axis
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the split operation
*/
inline Tensor strided_slice(const Tensor& x,
                            const Array<Integer>& begin,
                            const Array<Integer>& end,
                            const Array<Integer>& strides,
                            std::string name = "T_strided_slice",
                            std::string tag = kInjective) {
  size_t src_tensor_dim = static_cast<size_t>(x->shape.size());
  // Setup the ranges.
  // NOTE: this code duplicates the shape inference logic relay.op
  // Consider to refactor in the future.
  std::vector<int64_t> stride_vec;
  for (Integer i : strides) {
    CHECK(i.defined());
    stride_vec.push_back(i->value);
  }
  for (size_t i = stride_vec.size(); i < src_tensor_dim; ++i) {
    stride_vec.push_back(1);
  }
  const int64_t max_range = std::numeric_limits<int64_t>::max();

  std::vector<int64_t> begin_vec;
  for (size_t i = 0; i < begin.size(); ++i) {
    if (!begin[i].defined()) {
      // value=None
      begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
    } else {
      begin_vec.push_back(begin[i]->value);
    }
  }
  for (size_t i = begin_vec.size(); i < src_tensor_dim; ++i) {
    begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
  }

  std::vector<int64_t> end_vec;
  for (size_t i = 0; i < end.size(); ++i) {
    // allow end to be None
    if (!end[i].defined()) {
      end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
    } else {
      end_vec.push_back(end[i]->value);
    }
  }
  for (size_t i = end_vec.size(); i < src_tensor_dim; ++i) {
    end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
  }
  // Compute
  Array<PrimExpr> out_shape;
  Array<PrimExpr> begin_expr;
  Array<PrimExpr> strides_expr;

  for (size_t i = 0; i < src_tensor_dim; ++i) {
    int64_t begin_range = stride_vec[i] < 0 ? -1 : 0;
    int64_t dim_i = GetConstInt(x->shape[i]);
    int64_t end_range = stride_vec[i] < 0 ? dim_i - 1 : dim_i;
    // transform negative indices to positive value, clips on the correct range
    auto index_canonicalization = [dim_i, begin_range, end_range](int64_t index) {
      if (index < 0) {
        index += dim_i;
      }
      return std::min(std::max(index, begin_range), end_range);
    };

    int64_t begin_i = index_canonicalization(begin_vec[i]);
    int64_t end_i = index_canonicalization(end_vec[i]);

    int interval = std::abs(end_i - begin_i);
    int slice_size = static_cast<int>((interval
                                     + std::abs(stride_vec[i]) - 1) / std::abs(stride_vec[i]));
    CHECK(stride_vec[i] < 0 ? (end_i <= begin_i) : (begin_i <= end_i))
      << ": Input [Begin=" << begin_vec[i] << ", End=" << end_vec[i]
      << "] is invalid for axis=" << i;

    begin_expr.push_back(make_const(begin[0].dtype(), begin_i));
    strides_expr.push_back(make_const((strides.size() != 0 ? strides[0].dtype() : begin[0].dtype()),
                                     stride_vec[i]));
    out_shape.push_back(slice_size);
  }

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      Array<PrimExpr> real_indices;
      for (size_t i = 0; i < src_tensor_dim; ++i) {
        real_indices.push_back(indices[i] * strides_expr[i] + begin_expr[i]);
      }
      return x(real_indices);
    }, name, tag);
}

/*!
* \brief Split a tensor into a number of sub-tensors
*
* \param x The input tensor
* \param num_sections The number of sections to split the tensor into.
* this must be an integer factor of the size of the axis being split.
* \param axis The axis to split along.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the split operation
*/
inline Array<Tensor> split_sections(const Tensor& x,
                                    int num_sections,
                                    int axis,
                                    std::string name = "T_split_sections",
                                    std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(x->shape.size());
  }
  CHECK_LT(axis, x->shape.size()) << "axis out of bounds";

  auto src_axis_size = static_cast<int>(GetConstInt(x->shape[axis]));

  CHECK_GT(num_sections, 0) << "Slice count must be > 0";
  CHECK_EQ(src_axis_size % num_sections, 0)
    << "num_sections must be an integer factor of the size of axis " << axis
    << " (" << src_axis_size << ")";

  Array<Integer> split_indices;
  auto seg_size = src_axis_size / num_sections;
  for (int i = 0; i < num_sections; ++i) {
    // region at index 0 is added by split()
    if (i != 0) {
      split_indices.push_back(seg_size * i);
    }
  }

  return split(x, split_indices, axis, name, tag);
}

/*!
* \brief Take elements from an flattened input array when axis is None.
*
* \param a The source array.
* \param indices The indices of the values to extract.
* \param mode The mode of the operation.
* \param name The name of the operation.
* \param mode The mode of to handle out of bound indices.
* \param tag The tag to mark the operation.
*
* \return A Tensor whose op member is the take operation
*/
inline Tensor take(const Tensor& a,
                   const Tensor& indices,
                   std::string mode = "clip",
                   std::string name = "T_take",
                   std::string tag = kInjective) {
  Array<PrimExpr> a_shape = a->shape;
  Array<PrimExpr> out_shape = indices->shape;
  PrimExpr a_size = 1;
  for (size_t i = 0; i < a_shape.size(); ++i) {
    a_size = a_size * a_shape[i];
  }

  if (mode == "clip") {
    return compute(
        out_shape, [&](const Array<Var>& out_index) {
          auto idx = tvm::min(tvm::max(0, indices(out_index)), a_size - 1);
          return a(UnravelIndex(idx, a_shape));
        }, name, tag);
  } else if (mode == "fast") {
    LOG(WARNING) << "Fast mode segfaults when there are out-of-bounds indices. "
                    "Make sure input indices are in bound";
    return compute(
        out_shape, [&](const Array<Var>& out_index) {
          return a(UnravelIndex(indices(out_index), a_shape));
        }, name, tag);
  } else {  // mode == "wrap"
    return compute(
        out_shape, [&](const Array<Var>& out_index) {
          auto idx = truncmod(truncmod(indices(out_index), a_size) + a_size, a_size);
          return a(UnravelIndex(idx, a_shape));
        }, name, tag);
  }
}


/*!
* \brief Mask the out-of-boundary elements of each sequence.
*
* \param data The source array.
* \param valid_length The real length of each sequence.
* \param mask_value The masking value.
* \param axis The axis of the temporal dimension of the sequence
* \param name The name of the operation.
* \param tag The tag to mark the operation.
*
* \return A Tensor whose op member is the sequence_mask operation
*/
inline Tensor sequence_mask(const Tensor& data,
                            const Tensor& valid_length,
                            double mask_value,
                            int axis,
                            std::string name = "T_sequence_mask",
                            std::string tag = kInjective) {
  CHECK(axis == 0 || axis == 1) << "axis must be either 0 or 1";
  CHECK_EQ(valid_length->shape.size(), 1) << "valid_length must have ndim=1, i.e., (batch_size,).";
  auto length_dim = data->shape[axis];
  auto batch_dim = data->shape[1 - axis];
  Array<PrimExpr> out_shape = data->shape;
  Tensor out = compute(
      out_shape, [&](const Array<Var>& out_index) {
        Array<PrimExpr> len_index;
        auto tid = out_index[axis];
        auto bid = out_index[1 - axis];
        len_index.push_back(bid);
        PrimExpr ret = tvm::if_then_else(
            tvm::cast(valid_length->dtype, tid) >= valid_length(len_index),
            tvm::tir::make_const(data->dtype, mask_value), data(out_index));
        return ret;
      }, name, tag);
  return out;
}

/*!
* \brief Take elements from an array along an axis.
*
* \param a The source array.
* \param indices The indices of the values to extract.
* \param axis The axis over which to select values. By default,
* the flattened input array is used.
* \param mode The mode for handling out of bound indices.
* \param name The name of the operation.
* \param tag The tag to mark the operation.
*
* \return A Tensor whose op member is the take operation
*/
inline Tensor take(const Tensor& a,
                   const Tensor& indices,
                   int axis,
                   std::string mode = "clip",
                   std::string name = "T_take",
                   std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(a->shape.size());
  }
  CHECK_GE(axis, 0) << "axis out of bounds";
  CHECK_LT(axis, a->shape.size()) << "axis out of bounds";
  auto axis_dim = a->shape[axis];

  int indices_len = static_cast<int>(indices->shape.size());
  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < a->shape.size(); ++i) {
    if (axis == static_cast<int>(i)) {
      for (size_t j = 0; j < indices->shape.size(); ++j) {
        out_shape.push_back(indices->shape[j]);
      }
    } else {
      out_shape.push_back(a->shape[i]);
    }
  }
  if (mode == "clip") {
    return compute(
        out_shape, [&](const Array<Var>& out_index) {
          Array<PrimExpr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis+indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          auto idx = tvm::min(tvm::max(0, indices(indices_position)),
                              axis_dim - 1);
          real_indices.push_back(idx);
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          return a(real_indices);
        }, name, tag);
  } else if (mode == "fast") {
    LOG(WARNING) << "Fast mode segfaults when there are out-of-bounds indices. "
                    "Make sure input indices are in bound";
    return compute(
        out_shape, [&](const Array<Var>& out_index) {
          Array<PrimExpr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis+indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          real_indices.push_back(indices(indices_position));
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          return a(real_indices);
        }, name, tag);
  } else {  // mode == "wrap"
    return compute(
        out_shape, [&](const Array<Var>& out_index) {
          Array<PrimExpr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis+indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          auto idx = truncmod(truncmod(indices(indices_position), axis_dim) + axis_dim, axis_dim);
          real_indices.push_back(idx);
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          return a(real_indices);
        }, name, tag);
  }
}

/*!
* \brief Return the elements, either from x or y, depending on the condition.
*
* \param condition The condition array.
* \param x First array to be selected.
* \param y Second array to be selected.
* \param name The name of the operation.
* \param tag The tag to mark the operation.
*
* \return A Tensor selected from x or y depending on condition.
*/
inline Tensor where(const Tensor& condition,
                    const Tensor& x,
                    const Tensor& y,
                    std::string name = "T_where",
                    std::string tag = kBroadcast) {
  CHECK_EQ(x->shape.size(), y->shape.size())
    << "x and y must have the same shape.Got different number of dimension: "
    << x->shape.size() << " vs " << y->shape.size();
  CHECK_EQ(x->dtype, y->dtype) << "x and y must have the same dtype: "
                               << x->dtype << " vs " << y->dtype;
  Array<PrimExpr> oshape = x->shape;
  Tensor out;

  if (condition->shape.size() != 1) {
    CHECK_EQ(condition->shape.size(), x->shape.size())
      << "condition array must be either have the same shape as x or to be a "
         "1-D array.Got different number of dimension: "
      << condition->shape.size() << " vs " << x->shape.size();
    out = compute(
      oshape, [&](const Array<Var>& indices) {
        return tvm::tir::SelectNode::make(condition(indices) != 0, x(indices), y(indices));
      }, name, tag);
  } else {
    CHECK_EQ(topi::GetConstInt(condition->shape[0]), topi::GetConstInt(x->shape[0]))
      << "If condition is 1-D, the first dimension must be the same as x: "
      << condition->shape[0] << " vs " << x->shape[0];
    out = compute(
      oshape, [&](const Array<Var>& indices) {
        Array<PrimExpr> condition_idx{indices[0]};
        return tvm::tir::SelectNode::make(condition(condition_idx) != 0,
                                     x(indices), y(indices));
      }, name, tag);
  }
  return out;
}

/*!
* \brief Creates an operation to repeat elements of an array
*
* \param x The input tensor
* \param repeats The number of repetitions for each element
* \param axis The axis along which to repeat values (allows
* negative indices as offsets from the last dimension)
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the repeat operation
*/
inline Tensor repeat(const Tensor& x,
                     int repeats,
                     int axis,
                     std::string name = "T_repeat",
                     std::string tag = kBroadcast) {
  int ndim = static_cast<int>(x->shape.size());
  CHECK(-ndim - 1 <= axis && axis <= ndim)
    << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
    << ", but got axis = " << axis
    << ", and data.ndim = " << ndim;
  CHECK(repeats >= 1)
    << "repeat only accepts `repeats >= 1`"
    << ", but got repeats = " << repeats;
  if (axis < 0) {
    // Calculate offset from last dimension
    axis += ndim;
  }
  Array<PrimExpr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(x->shape[i]);
  }
  new_shape.push_back(repeats * x->shape[axis]);
  for (size_t i = axis + 1; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(
    new_shape, [&](const Array<Var>& indices) {
      Array<PrimExpr> idx;
      for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        idx.push_back(indices[i]);
      }
      idx.push_back(indexdiv(indices[axis], repeats));
      for (size_t i = axis + 1; i < indices.size(); ++i) {
        idx.push_back(indices[i]);
      }
      return x(idx);
    }, name, tag);
}

/*!
* \brief Creates an operation to tile elements of an array
*
* \param x The input tensor
* \param reps The number of times for repeating the tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the tile operation
*/
inline Tensor tile(const Tensor& x,
                   Array<Integer> reps,
                   std::string name = "T_tile",
                   std::string tag = kBroadcast) {
  size_t ndim = x->shape.size();
  size_t rdim = reps.size();
  size_t tdim = (ndim > rdim) ? ndim : rdim;
  Array<PrimExpr> data_shape;
  Array<PrimExpr> reps_shape;
  Array<PrimExpr> new_shape;
  if (ndim == rdim) {
    for (size_t i = 0; i < ndim; ++i) {
      data_shape.push_back(x->shape[i]);
      reps_shape.push_back(reps[i]);
    }
  } else if (ndim > rdim) {
    for (size_t i = 0; i < ndim; ++i)
      data_shape.push_back(x->shape[i]);
    for (size_t i = 0; i < (ndim - rdim); ++i)
      reps_shape.push_back(1);
    for (size_t i = 0; i < rdim; ++i)
      reps_shape.push_back(reps[i]);
  } else {
    for (size_t i = 0; i < (rdim - ndim); ++i)
      data_shape.push_back(1);
    for (size_t i = 0; i < ndim; ++i)
      data_shape.push_back(x->shape[i]);
    for (size_t i = 0; i < rdim; ++i)
      reps_shape.push_back(reps[i]);
  }
  for (size_t i = 0; i < tdim; ++i)
    new_shape.push_back(data_shape[i] * reps_shape[i]);

  if (is_empty_shape(new_shape)) {
    return compute(new_shape,
                   [&](const Array<Var>& indices) { return tvm::cast(x->dtype, 0);},
                   name, tag);
  } else {
    return compute(
      new_shape, [&](const Array<Var>& indices) {
        Array<PrimExpr> idx;
        if (ndim >= rdim) {
          for (size_t i = 0; i < ndim; ++i)
            idx.push_back(indexmod(indices[i], x->shape[i]));
        } else {
          for (size_t i = 0; i < ndim; ++i)
            idx.push_back(indexmod(indices[rdim - ndim + i], x->shape[i]));
        }
        return x(idx);
      }, name, tag);
  }
}

/*!
* \brief Gather elements from a n-dimension array.
*
* \param data The source array.
* \param indices The indices of the values to extract.
* \param name The name of the operation.
* \param tag The tag to mark the operation.
*
* \return A Tensor whose op member is the gather_nd operation
*/
inline Tensor gather_nd(const Tensor& data,
                        const Tensor& indices,
                        std::string name = "T_gather_nd",
                        std::string tag = kInjective) {
  size_t ndim_d = data->shape.size();
  size_t ndim_i = indices->shape.size();
  CHECK_GT(ndim_i, 1) << "indices tensor must have at least 2 dimensions";
  size_t indices_dim0 = static_cast<size_t>(GetConstInt(indices->shape[0]));
  CHECK_LE(indices_dim0, ndim_d) << "dim 0 of indices tensor must be no more "
                                 << "than dimensions of data tensor";
  Array<PrimExpr> out_shape;
  for (size_t i = 1; i < ndim_i; ++i) {
    out_shape.push_back(indices->shape[i]);
  }
  for (size_t i = indices_dim0; i < ndim_d; ++i) {
    out_shape.push_back(data->shape[i]);
  }
  if (out_shape.size() == 0) {
    out_shape.push_back(make_const(DataType::Int(32), 1));
  }
  return compute(
        out_shape, [&](const Array<Var>& out_index) {
          Array<PrimExpr> indices_position;
          indices_position.push_back(0);
          for (size_t i = 0; i < ndim_i - 1; ++i) {
            indices_position.push_back(out_index[i]);
          }
          Array<PrimExpr> real_indices;
          for (size_t i = 0; i < indices_dim0; ++i) {
            indices_position.Set(0, make_const(DataType::Int(32), i));
            if (indices->dtype.is_int()) {
              real_indices.push_back(indices(indices_position));
            } else {
              real_indices.push_back(
                  tvm::cast(tvm::DataType::Int(32), indices(indices_position)));
            }
          }
          for (size_t i = ndim_i - 1; i < out_index.size(); ++i) {
            real_indices.push_back(out_index[i]);
          }
          return data(real_indices);
        }, name, tag);
}

/*!
 * \brief Creates an operation that calculates a matrix multiplication
 *  (row-major notation):
 *      A(i, k) * B(k, j), if trans_a == trans_b
 *          the usual transposed combinations, otherwise
 *
 * \param A The matrix A
 * \param B The matrix B
 * \param trans_a Is A's layout transposed?
 * \param trans_b Is B's layout transposed?
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the matmul operation
 */
inline tvm::te::Tensor matmul(const tvm::te::Tensor& A,
                           const tvm::te::Tensor& B,
                           bool trans_a = false,
                           bool trans_b = false,
                           std::string name = "T_matmul",
                           std::string tag = kMatMul) {
  tvm::Array<tvm::PrimExpr> output_shape{A->shape[trans_a ? 1 : 0],
                                     B->shape[trans_b ? 0 : 1]};
  auto k = tvm::te::reduce_axis(tvm::Range{0, A->shape[trans_a ? 0 : 1]}, "k");
  auto l = [&](tvm::tir::Var i, tvm::tir::Var j) {
    return tvm::sum((trans_a ? A[k][i] : A[i][k]) * (trans_b ? B[j][k] : B[k][j]),
                    {k});
  };
  return tvm::te::compute(output_shape, l, name, tag);
}

/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param axes The number of the dimensions to reduce over
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor computing the result
 */
inline Tensor tensordot(const Tensor& A,
                        const tvm::te::Tensor& B,
                        int axes = 2,
                        std::string name = "T_tensordot",
                        std::string tag = kMatMul) {
  CHECK_GE(A->shape.size(), axes);
  CHECK_GE(B->shape.size(), axes);

  Array<PrimExpr> output_shape(A->shape.begin(), A->shape.end() + (-axes));
  for (auto it = B->shape.begin() + axes; it != B->shape.end(); ++it)
    output_shape.push_back(*it);

  Array<IterVar> iter_vars;
  for (int i = 0; i < axes; ++i)
    iter_vars.push_back(reduce_axis(Range(0, B->shape[i]), "k" + std::to_string(i)));

  auto func =
    [&A, &B, &iter_vars, axes]
    (const Array<Var>& input_indices) {
      Array<PrimExpr> A_indices(
          input_indices.begin(),
          input_indices.begin() + (A->shape.size() - axes));
      for (auto& v : iter_vars)
        A_indices.push_back(v);

      Array<PrimExpr> B_indices;
      for (auto& v : iter_vars)
        B_indices.push_back(v);

      auto it = input_indices.begin() + (A->shape.size() - axes);
      for (; it != input_indices.end(); ++it)
        B_indices.push_back(*it);

      // Some passes don't like reductions with empty axis, so avoid it here
      if (iter_vars.empty())
        return A(A_indices) * B(B_indices);
      else
        return sum(A(A_indices) * B(B_indices), iter_vars);
    };

  return compute(output_shape, func, name, tag);
}

/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param A_axes The indices of the dimensions of tensor A to reduce over
 * \param B_axes The indices of the dimensions of tensor B to reduce over
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor computing the result
 */
inline Tensor tensordot(const Tensor& A,
                        const tvm::te::Tensor& B,
                        Array<PrimExpr> A_axes,
                        Array<PrimExpr> B_axes,
                        std::string name = "T_tensordot",
                        std::string tag = kMatMul) {
  CHECK_EQ(A_axes.size(), B_axes.size());

  auto A_axes_val = GetConstIntValues(A_axes, "A_axes");
  auto B_axes_val = GetConstIntValues(B_axes, "B_axes");

  Array<PrimExpr> output_shape;
  for (unsigned i = 0; i < A->shape.size(); ++i)
    if (std::find(A_axes_val.begin(), A_axes_val.end(), i) == A_axes_val.end())
      output_shape.push_back(A->shape[i]);
  for (unsigned i = 0; i < B->shape.size(); ++i)
    if (std::find(B_axes_val.begin(), B_axes_val.end(), i) == B_axes_val.end())
      output_shape.push_back(B->shape[i]);

  Array<IterVar> iter_vars;
    for (unsigned i = 0; i < B_axes_val.size(); ++i)
      iter_vars.push_back(reduce_axis(Range(0, B->shape[B_axes_val[i]]), "k" + std::to_string(i)));

  auto func =
    [&A, &B, &iter_vars, A_axes_val, B_axes_val]
    (const Array<Var>& input_indices) {
      int idx_input = 0;
      Array<PrimExpr> A_indices;
      for (unsigned i = 0; i < A->shape.size(); ++i) {
        auto axes_pos = std::find(A_axes_val.begin(), A_axes_val.end(), i);
        if (axes_pos == A_axes_val.end())
          A_indices.push_back(input_indices[idx_input++]);
        else
          A_indices.push_back(iter_vars[axes_pos - A_axes_val.begin()]);
      }

      Array<PrimExpr> B_indices;
      for (unsigned i = 0; i < B->shape.size(); ++i) {
        auto axes_pos = std::find(B_axes_val.begin(), B_axes_val.end(), i);
        if (axes_pos == B_axes_val.end())
          B_indices.push_back(input_indices[idx_input++]);
        else
          B_indices.push_back(iter_vars[axes_pos - B_axes_val.begin()]);
      }
      return sum(A(A_indices) * B(B_indices), iter_vars);
    };
  return compute(output_shape, func, name, tag);
}

inline Tensor arange(const PrimExpr& start,
                     const PrimExpr& stop,
                     const PrimExpr& step,
                     DataType dtype,
                     std::string name = "T_arange",
                     std::string tag = kInjective) {
  PrimExpr num_elem = tvm::cast(tvm::DataType::Int(32), tvm::ceil(
      tvm::cast(tvm::DataType::Float(32), stop - start) / step));
  Array<PrimExpr> shape;
  return compute({num_elem}, [&](const Array<Var>& indices) {
    return tvm::cast(dtype, start + step * indices[0]);
  }, name, tag);
}

/*!
 * \brief Transform the layout according to \p src_layout and \p dst_layout
 * \param src the source input.
 * \param src_layout the source layout.
 * \param dst_layout the destination layout.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return A tensor with shape in \p dst_layout
 */
inline Tensor layout_transform(const Tensor& src,
                               const std::string& src_layout,
                               const std::string& dst_layout,
                               const std::string name = "T_layout_trans",
                               const std::string tag = kInjective) {
  Layout src_layout_struct = LayoutNode::make(src_layout);
  Layout dst_layout_struct = LayoutNode::make(dst_layout);

  if (src_layout_struct.Equals(dst_layout_struct)) {
    return src;
  }

  CHECK(src_layout_struct.defined() && dst_layout_struct.defined())
    << "cannot convert from/to undefined layout";

  auto layout_converter = tir::BijectiveLayout(src_layout_struct, dst_layout_struct);
  CHECK(layout_converter.defined())
    << "cannot convert from " << src_layout << " to " << dst_layout;

  Array<PrimExpr> dst_shape = layout_converter.ForwardShape(src->shape);

  return compute(
    dst_shape, [&](const Array<Var>& dst_indices) {
      Array<PrimExpr> dst_indices_expr(dst_indices.begin(), dst_indices.end());
      Array<PrimExpr> src_indices = layout_converter.BackwardIndex(dst_indices_expr);
      return src(src_indices);
  }, name, tag);
}

/*!
 * \brief Get the shape of input tensor.
 * \param src the input tensor.
 * \param dtype the type of the elements in the tensor.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return Tensor of input shape.
 */
inline Tensor shape(const Tensor& src,
                    DataType dtype,
                    const std::string name = "T_shape",
                    const std::string tag = kInjective) {
  int ndim = static_cast<int>(src->shape.size());
  Array<PrimExpr> out_shape{ndim};
  return compute(out_shape, [&](const Array<Var>& indices) {
    auto idx = indices[0];
    PrimExpr ret = 0;
    for (int i = 0; i < ndim; ++i) {
      ret = tvm::if_then_else(idx == i, src->shape[i], ret);
    }
    return tvm::cast(dtype, ret);
  }, name, tag);
}

/*!
 * \brief Get the size of input tensor.
 * \param src the input tensor.
 * \param dtype the type of the elements in the tensor.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return Tensor of input shape.
 */
inline Tensor ndarray_size(const Tensor& src,
                           const DataType& dtype,
                           const std::string& name = "ndarray_size",
                           const std::string& tag = kInjective) {
  int ndim = static_cast<int>(src->shape.size());
  Array<PrimExpr> out_ndarray_size = {1};
  return compute(out_ndarray_size, [&](const Array<Var>& indices) {
    PrimExpr ret = 1;
    for (int i = 0; i < ndim; ++i) {
      ret *= src->shape[i];
    }
    return tvm::cast(dtype, ret);
  }, name, tag);
}

/*!
 * \brief Returns a one-hot tensor where the locations repsented by indices take value on_value,
    other locations take value off_value.
 * \param indices locations to set to on_value.
 * \param on_value value that locations represented by indices take on.
 * \param off_value value that other locations take on.
 * \param depth depth of the one-hot dimension.
 * \param axis axis to fill.
 * \param dtype data type of the output tensor.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return one-hot tensor.
 */
inline Tensor one_hot(const Tensor& indices,
                      const PrimExpr on_value,
                      const PrimExpr off_value,
                      int depth,
                      int axis,
                      const DataType& dtype,
                      const std::string name = "T_one_hot",
                      const std::string tag = kInjective) {
  Array<PrimExpr> oshape;
  int ndim = indices->shape.size() + 1;
  int indices_index = 0;
  int true_axis = (axis == -1) ? indices->shape.size() : axis;
  for (int i = 0; i < ndim; i++) {
    if (i == true_axis) {
      oshape.push_back(Integer(depth));
    } else {
      oshape.push_back(indices->shape[indices_index++]);
    }
  }

  PrimExpr on_value_cast = cast(dtype, on_value);
  PrimExpr off_value_cast = cast(dtype, off_value);
  return compute(oshape, [&](const Array<Var>& iter_vars) {
    Array<Var> indices_indices;
    for (size_t i = 0; i < iter_vars.size(); i++) {
      if (static_cast<int>(i) == true_axis) {
        continue;
      }

      indices_indices.push_back(iter_vars[i]);
    }

    auto idx = iter_vars[true_axis];
    return tir::SelectNode::make(indices(indices_indices) == idx, on_value_cast, off_value_cast);
  }, name, tag);
}

}  // namespace topi
#endif  // TOPI_TRANSFORM_H_
