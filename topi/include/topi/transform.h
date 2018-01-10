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

#include "topi/tags.h"
#include "topi/detail/ravel_unravel.h"
#include "topi/detail/constant_utils.h"
#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

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
  if (axis < 0) {
    // Calculate offset from last dimension
    axis = static_cast<int>(x->shape.size()) + axis + 1;
  }

  Array<Expr> new_shape;
  for (int i = 0; i < axis; ++i) {
    new_shape.push_back(x->shape[i]);
  }
  for (int i = 0; i < num_newaxis; ++i) {
    new_shape.push_back(1);
  }
  for (int i = axis; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(new_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    for (int i = 0; i < axis; ++i) {
      idx.push_back(indices[i]);
    }
    for (int i = axis + num_newaxis; i < indices.size(); ++i) {
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
  std::vector<int> axes,
  std::string name = "tensor",
  std::string tag = kInjective) {
  if (axes.size() == 0) {
    for (int i = static_cast<int>(x->shape.size()) - 1; i >= 0; --i) {
      axes.push_back(i);
    }
  }

  Array<Expr> new_shape;
  for (int i = 0; i < axes.size(); ++i) {
    new_shape.push_back(x->shape[axes[i]]);
  }
  return compute(new_shape, [&](const Array<Var>& indices) {
    std::vector<Expr> idx;
    for (int i = 0; i < axes.size(); ++i) {
      idx.push_back(1);
    }
    for (int i = 0; i < axes.size(); ++i) {
      idx[axes[i]] = indices[i];
    }
    return x(idx);
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
  return compute(newshape, [&](const Array<Var>& indices) {
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
  std::vector<int> axis,
  std::string name = "tensor",
  std::string tag = kInjective) {
  auto ndim = x->shape.size();
  if (axis.size() == 0) {
    for (int i = 0; i < ndim; ++i) {
      if (IsConstInt(x->shape[i]) && GetConstInt(x->shape[i]) == 1) {
        axis.push_back(i);
      }
    }
  } else {
    for (int i = 0; i < ndim; ++i) {
      if (axis[i] < 0) {
        axis[i] += static_cast<int>(x->shape.size());
      }
      CHECK_EQ(GetConstInt(x->shape[axis[i]]), 1) <<
        "Dimension " << axis[i] << " must have size 1";
    }
  }

  std::unordered_set<int> axis_set(axis.begin(), axis.end());

  Array<Expr> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (axis_set.count(i) == 0) {
      out_shape.push_back(x->shape[i]);
    }
  }
  if (out_shape.size() == 0) {
    out_shape.push_back(1);
  }

  return compute(out_shape, [&](const Array<Var>& indices) {
    Array<Expr> real_indices;
    int flag = 0;
    for (int i = 0; i < ndim; ++i) {
      if (axis_set.count(i) == 0) {
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
  if (axis < 0) {
    axis += static_cast<int>(inputs[0]->shape.size());
  }
  CHECK_LT(axis, inputs[0]->shape.size()) <<
    "axis out of bounds";

  std::vector<Expr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }

  Expr join_size = axis_sizes[0];
  for (int i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  std::vector<Expr> out_shape;
  for (int i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(i == axis ? join_size : inputs[0]->shape[i]);
  }

  return compute(out_shape, [&](const Array<Var>& indices) {
    auto ret = inputs[0](indices);
    auto ind = indices[axis];
    for (int i = 0; i < inputs.size() - 1; ++i) {
      ind -= axis_sizes[i];

      Array<Expr> idx;
      for (int i = 0; i < axis; ++i) {
        idx.push_back(indices[i]);
      }
      idx.push_back(ind);
      for (int i = axis + 1; i < indices.size(); ++i) {
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
  std::vector<int> split_indices,
  int axis,
  std::string name = "tensor",
  std::string tag = kInjective) {
  auto src_axis_size = static_cast<int>(GetConstInt(x->shape[axis]));

  CHECK(std::is_sorted(split_indices.begin(), split_indices.end())) <<
    "split_indices must be sorted";

  std::vector<int> begin_ids;
  begin_ids.push_back(0);
  std::copy(split_indices.begin(), split_indices.end(), std::back_inserter(begin_ids));

  std::vector<Array<Expr>> out_shapes;
  for (int i = 0; i < begin_ids.size(); ++i) {
    int out_axis_size;
    if (i == begin_ids.size() - 1) {
      out_axis_size = src_axis_size - begin_ids[i];
    } else {
      out_axis_size = begin_ids[i + 1] - begin_ids[i];
    }

    std::vector<Expr> shape;
    std::copy(x->shape.begin(), x->shape.begin() + axis, std::back_inserter(shape));
    shape.push_back(out_axis_size);
    std::copy(x->shape.begin() + axis + 1, x->shape.end(), std::back_inserter(shape));

    out_shapes.push_back(shape);
  }

  Array<Tensor> result;
  for (int i = 0; i < begin_ids.size(); ++i) {
    result.push_back(
      compute(out_shapes[i], [&](const Array<Var>& indices) {
      auto begin = begin_ids[i];
      Array<Expr> real_indices;
      for (int j = 0; j < axis; ++j) {
        real_indices.push_back(indices[j]);
      }
      real_indices.push_back(indices[axis] + begin);
      for (int j = axis + 1; j < indices.size(); ++j) {
        real_indices.push_back(indices[j]);
      }

      return x(real_indices);
    }, name, tag));
  }

  return result;
}

}  // namespace topi
#endif  // TOPI_TRANSFORM_H_
