/*!
 *  Copyright (c) 2017 by Contributors
 * \file topi/transform.h
 * \brief Transform op constructors
 */
#ifndef TOPI_TRANSFORM_H_
#define TOPI_TRANSFORM_H_

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include "topi/tags.h"
#include "topi/detail/ravel_unravel.h"
#include "topi/detail/constant_utils.h"
#include "tvm/tvm.h"

namespace topi {
using namespace tvm;
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
                          std::string name = "tensor",
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
  Array<Expr> new_shape;
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
      Array<Expr> idx;
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
                        Array<Expr> axes,
                        std::string name = "tensor",
                        std::string tag = kInjective) {
  if (axes.size() == 0) {
    axes = Array<Expr>();
    for (int i = static_cast<int>(x->shape.size()) - 1; i >= 0; --i) {
      axes.push_back(i);
    }
  }

  auto axes_val = GetConstIntValues(axes, "axes");
  for (size_t i = 0; i < axes_val.size(); ++i) {
    int axis = axes_val[i];
    if (axes_val[i] < 0) {
      axes_val[i] = static_cast<int>(x->shape.size()) + axes_val[i];
    }
    CHECK((0 <= axes_val[i]) && (axes_val[i] < static_cast<int>(x->shape.size())))
      << "axis=" << axis << " is invalid for the "
      << static_cast<int>(x->shape.size()) << "-dimensional input tensor";

    CHECK(1 == std::count(std::begin(axes_val), std::end(axes_val), axes_val[i]))
      << "repeated axis in transpose";
  }

  Array<Expr> new_shape;
  for (size_t i = 0; i < axes_val.size(); ++i) {
    new_shape.push_back(x->shape[axes_val[i]]);
  }
  return compute(
    new_shape, [&](const Array<Var>& indices) {
      std::vector<Expr> idx;
      for (size_t i = 0; i < axes_val.size(); ++i) {
        idx.push_back(1);
      }
      for (size_t i = 0; i < axes_val.size(); ++i) {
        idx[axes_val[i]] = indices[i];
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
                   std::string name = "tensor",
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
      Array<Expr> real_indices;
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
                      Array<Expr> newshape,
                      std::string name = "tensor",
                      std::string tag = kInjective) {
  auto x_shape = x->shape;
  return compute(
    newshape, [&](const Array<Var>& indices) {
      return x(UnavelIndex(RavelIndex(indices, newshape), x_shape));
    }, name, tag);
}

/*!
* \brief Remove size 1 dimensions from the shape of a tensor.
* The removed dimensions must have a constant size of 1.
*
* \param x The input tensor
* \param axis Indices of the dimensions to remove. If this is empty,
* all entries with a constant size of 1 will be removed.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the squeeze operation
*/
inline Tensor squeeze(const Tensor& x,
                      Array<Expr> axis,
                      std::string name = "tensor",
                      std::string tag = kInjective) {
  auto axis_val = GetConstIntValues(axis, "axis");
  auto ndim = x->shape.size();
  if (axis_val.size() == 0) {
    for (size_t i = 0; i < ndim; ++i) {
      if (IsConstInt(x->shape[i]) && GetConstInt(x->shape[i]) == 1) {
        axis_val.push_back(static_cast<int>(i));
      }
    }
  } else {
    for (size_t i = 0; i < axis_val.size(); ++i) {
      if (axis_val[i] < 0) {
        axis_val[i] += static_cast<int>(x->shape.size());
      }
      CHECK_EQ(GetConstInt(x->shape[axis_val[i]]), 1) <<
        "Dimension " << axis[i] << " must have size 1";
    }
  }

  std::unordered_set<int> axis_set(axis_val.begin(), axis_val.end());

  Array<Expr> out_shape;
  for (size_t i = 0; i < ndim; ++i) {
    if (axis_set.count(static_cast<int>(i)) == 0) {
      out_shape.push_back(x->shape[i]);
    }
  }
  if (out_shape.size() == 0) {
    out_shape.push_back(1);
  }

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      Array<Expr> real_indices;
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
                          std::string name = "tensor",
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

  Array<Expr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }

  Expr join_size = axis_sizes[0];
  for (size_t i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  join_size = tvm::ir::Simplify(join_size);
  Array<Expr> out_shape;
  for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(i == static_cast<size_t>(axis) ? join_size : inputs[0]->shape[i]);
  }

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      auto ret = inputs[0](indices);
      auto ind = indices[axis];
      for (size_t i = 0; i < inputs.size() - 1; ++i) {
        ind -= axis_sizes[i];

        Array<Expr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        idx.push_back(ind);
        for (size_t i = axis + 1; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }

        ret = tvm::select(ind >= 0,
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
                           Array<Expr> split_indices,
                           int axis,
                           std::string name = "tensor",
                           std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(x->shape.size());
  }
  CHECK_LT(axis, x->shape.size()) << "axis out of bounds";

  auto src_axis_size = static_cast<int>(GetConstInt(x->shape[axis]));

  auto split_indices_val = GetConstIntValues(split_indices, "split_indices");
  CHECK(std::is_sorted(split_indices_val.begin(), split_indices_val.end())) <<
    "split_indices must be sorted";

  std::vector<int> begin_ids;
  begin_ids.push_back(0);
  std::copy(split_indices_val.begin(), split_indices_val.end(), std::back_inserter(begin_ids));

  Array< Array<Expr> > out_shapes;
  for (size_t i = 0; i < begin_ids.size(); ++i) {
    int out_axis_size;
    if (i == begin_ids.size() - 1) {
      out_axis_size = src_axis_size - begin_ids[i];
    } else {
      out_axis_size = begin_ids[i + 1] - begin_ids[i];
    }

    Array<Expr> shape;
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
          Array<Expr> real_indices;
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
                            const Array<Expr>& begin,
                            const Array<Expr>& end,
                            const Array<Expr>& strides,
                            std::string name = "tensor",
                            std::string tag = kInjective) {
  size_t src_tensor_dim = static_cast<size_t>(x->shape.size());
  std::vector<int64_t> begin_vec = GetConstInt64Values(begin, "begin");
  std::vector<int64_t> end_vec = GetConstInt64Values(end, "end");
  std::vector<int64_t> stride_vec = GetConstInt64Values(strides, "strides");
  // in case user has not provided begin indices for all the axes,
  // then inflate it with default value = 0
  for (size_t i = begin_vec.size(); i < src_tensor_dim; ++i) {
    begin_vec.push_back(0);
  }
  // in case user has not provided end indices for all the axes,
  // then inflate it with default value = input_tensor.shape[axis]
  for (size_t i = end_vec.size(); i < src_tensor_dim; ++i) {
    end_vec.push_back(GetConstInt(x->shape[i]));
  }
  // in case user has not provided stride values,
  // then inflate it with default value = 1
  for (size_t i = stride_vec.size(); i < src_tensor_dim; ++i) {
    stride_vec.push_back(1);
  }

  Array<Expr> out_shape;
  Array<Expr> begin_expr;
  Array<Expr> strides_expr;

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
    CHECK(stride_vec[i] < 0 ? (end_i < begin_i) : (begin_i < end_i))
      << ": Input [Begin=" << begin_vec[i] << ", End=" << end_vec[i]
      << "] is invalid for axis=" << i;

    begin_expr.push_back(make_const(begin[0].type(), begin_i));
    strides_expr.push_back(make_const((strides.size() != 0 ? strides[0].type() : begin[0].type()),
                                     stride_vec[i]));
    out_shape.push_back(slice_size);
  }

  return compute(
    out_shape, [&](const Array<Var>& indices) {
      Array<Expr> real_indices;
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
                           std::string name = "tensor",
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

  Array<Expr> split_indices;
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
* \param name The name of the operation.
* \param tag The tag to mark the operation.
*
* \return A Tensor whose op member is the take operation
*/
inline Tensor take(const Tensor& a,
                   const Tensor& indices,
                   std::string name = "tensor",
                   std::string tag = kInjective) {
  Array<Expr> a_shape = a->shape;
  Array<Expr> out_shape;
  for (size_t j = 0; j < indices->shape.size(); ++j) {
    out_shape.push_back(indices->shape[j]);
  }

  return compute(
        out_shape, [&](const Array<Var>& out_index) {
          Array<Expr> indices_position;
          for (size_t j = 0; j < indices->shape.size(); ++j) {
            indices_position.push_back(out_index[j]);
          }
          return a(UnavelIndex(indices(indices_position), a_shape));
        }, name, tag);
}

/*!
* \brief Take elements from an array along an axis.
*
* \param a The source array.
* \param indices The indices of the values to extract.
* \param axis The axis over which to select values. By default,
* the flattened input array is used.
* \param name The name of the operation.
* \param tag The tag to mark the operation.
*
* \return A Tensor whose op member is the take operation
*/
inline Tensor take(const Tensor& a,
                   const Tensor& indices,
                   int axis,
                   std::string name = "tensor",
                   std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(a->shape.size());
  }
  CHECK_LT(axis, a->shape.size()) << "axis out of bounds";

  int indices_len = static_cast<int>(indices->shape.size());
  Array<Expr> out_shape;
  for (size_t i = 0; i < a->shape.size(); ++i) {
    if (axis == static_cast<int>(i)) {
      for (size_t j = 0; j < indices->shape.size(); ++j) {
        out_shape.push_back(indices->shape[j]);
      }
    } else {
      out_shape.push_back(a->shape[i]);
    }
  }
  return compute(
        out_shape, [&](const Array<Var>& out_index) {
          Array<Expr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis+indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          Array<Expr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          real_indices.push_back(indices(indices_position));
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          return a(real_indices);
        }, name, tag);
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
                    std::string name = "tensor",
                    std::string tag = kInjective) {
  CHECK_EQ(x->shape.size(), y->shape.size())
    << "x and y must have the same shape.Got different number of dimension: "
    << x->shape.size() << " vs " << y->shape.size();
  CHECK_EQ(x->dtype, y->dtype) << "x and y must have the same dtype: "
                               << x->dtype << " vs " << y->dtype;
  Array<Expr> oshape = x->shape;
  Tensor out;

  if (condition->shape.size() != 1) {
    CHECK_EQ(condition->shape.size(), x->shape.size())
      << "condition array must be either have the same shape as x or to be a "
         "1-D array.Got different number of dimension: "
      << condition->shape.size() << " vs " << x->shape.size();
    out = compute(
      oshape, [&](const Array<Var>& indices) {
        return tvm::select(condition(indices) != 0, x(indices), y(indices));
      }, name, tag);
  } else {
    CHECK_EQ(topi::GetConstInt(condition->shape[0]), topi::GetConstInt(x->shape[0]))
      << "If condition is 1-D, the first dimension must be the same as x: "
      << condition->shape[0] << " vs " << x->shape[0];
    out = compute(
      oshape, [&](const Array<Var>& indices) {
        Array<Expr> condition_idx{indices[0]};
        return tvm::select(condition(condition_idx) != 0,
                           x(indices), y(indices));
      }, name, tag);
  }
  return out;
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
inline tvm::Tensor matmul(const tvm::Tensor& A,
                           const tvm::Tensor& B,
                           bool trans_a = false,
                           bool trans_b = false,
                           std::string name = "tensor",
                           std::string tag = kMatMul) {
  tvm::Array<tvm::Expr> output_shape{A->shape[trans_a ? 1 : 0],
                                     B->shape[trans_b ? 0 : 1]};
  auto k = tvm::reduce_axis(tvm::Range{0, A->shape[trans_a ? 0 : 1]}, "k");
  auto l = [&](tvm::Var i, tvm::Var j) {
    return tvm::sum((trans_a ? A[k][i] : A[i][k]) * (trans_b ? B[j][k] : B[k][j]),
                    {k});
  };
  return tvm::compute(output_shape, l, name, tag);
}


}  // namespace topi
#endif  // TOPI_TRANSFORM_H_
