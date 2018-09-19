/*!
 *  Copyright (c) 2017 by Contributors
 * \file topi/reduction.h
 * \brief Reduction op constructors
 */
#ifndef TOPI_REDUCTION_H_
#define TOPI_REDUCTION_H_

#include <algorithm>
#include <string>
#include <set>
#include <vector>
#include <iterator>

#include "topi/broadcast.h"
#include "topi/elemwise.h"
#include "topi/tags.h"
#include "topi/transform.h"
#include "topi/detail/ravel_unravel.h"
#include "topi/detail/constant_utils.h"
#include "tvm/tvm.h"

/*!
 * \brief macro flag to enable some legacy behavior which requires
 * reduction result to be at least 1d.
 */
#ifndef TOPI_REDUCE_ATLEAST1D
#define TOPI_REDUCE_ATLEAST1D 0
#endif

namespace topi {
using namespace tvm;

/*! \brief The operation to use for CommReduce */
using FReduce = std::function<Expr(Expr source, const Array<IterVar>& axis)>;

/*! \brief The operation to use for CommReduceIdx */
using FCommReduce = std::function<
  Array<Expr>(Array<Expr> exprs, const Array<IterVar>& axis, Expr* condition)>;

/*!
* \brief Convert a reduction axis which could be empty or have negative
* elements into a real axis with valid dimension indices.
*
* \return A non-empty sorted array of valid dimension indices, with no duplicates.
* If the input axis is empty, the result will be an axis including all dimensions.
* If any input element is negative, it will be treated as an offset from the
* last dimension (same as python indexing rules).
*/
inline std::vector<int> GetRealAxis(int ndim, const std::vector<int>& axis) {
  std::vector<int> real_axis;
  if (axis.size() == 0) {
    for (int i = 0; i < ndim; ++i) {
      real_axis.push_back(i);
    }
  } else {
    // Use a set so duplicates are removed and the dims are sorted
    std::set<int> dims;
    for (auto ele : axis) {
      if (ele < 0) {
        ele += ndim;
      }
      if (ele >= ndim) {
        LOG(ERROR) << ele << " exceeds the maximum dimension " << ndim;
      }
      dims.emplace(ele);
    }
    std::copy(dims.begin(), dims.end(), std::back_inserter(real_axis));
  }
  return real_axis;
}

/*! \brief Enumerate the axes for a reduce op */
inline Array<IterVar> MakeReduceAxes(const std::vector<int>& real_axis, const Tensor& data) {
  Array<IterVar> reduce_axes;
  for (auto i : real_axis) {
    std::string name = "k" + std::to_string(i);
    reduce_axes.push_back(
      tvm::reduce_axis(Range(0, data->shape[i]), name));
  }
  return reduce_axes;
}

/*! \brief Calculate the target shape for a reduce op */
inline Array<Expr> MakeReduceTargetShape(const std::vector<int>& real_axis,
                                         const Tensor& data,
                                         bool keepdims) {
  auto ndim = data->shape.size();
  Array<Expr> target_shape;
  if (keepdims) {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        target_shape.push_back(1);
      } else {
        target_shape.push_back(data->shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) == real_axis.end()) {
        // real_axis does not contain i
        target_shape.push_back(data->shape[i]);
      }
    }
  }
  if (target_shape.size() == 0 && TOPI_REDUCE_ATLEAST1D) {
    target_shape.push_back(1);
  }
  return target_shape;
}

/*!
 * \brief Create a reduction operation.
 *
 * \param data The input tensor.
 * \param func The reduction function eg. tvm::sum
 * \param target_shape The output Tensor shape.
 * \param reduce_axes The real axes along which the reduction is performed.
 * \param squeeze_axes The real axes to squeeze. Unsqueezed, reduced axes will
 *                     have shape 1 in the output tensor.
 *
 * \return The result tensor.
 */
inline Tensor DoCommReduce(const Tensor& data,
                           FReduce func,
                           const Array<Expr>& target_shape,
                           const std::vector<int>& reduce_axes,
                           const std::vector<int>& squeeze_axes) {
  auto r_axes = MakeReduceAxes(reduce_axes, data);
  auto compute = [&](const Array<Var>& indices) {
    Array<Expr> eval_range;
    Array<Var> eval_indices;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < data->shape.size(); ++i) {
      bool squeeze_i = std::find(squeeze_axes.begin(), squeeze_axes.end(), i) != squeeze_axes.end();
      if (std::find(reduce_axes.begin(), reduce_axes.end(), i) != reduce_axes.end()) {
        // real_axis contains i
        eval_range.push_back(r_axes[red_counter]);
        eval_indices.push_back(r_axes[red_counter]->var);
        red_counter++;
        arg_counter += !squeeze_i;
        continue;
      }
      eval_range.push_back(indices[arg_counter]);
      arg_counter++;
    }

    return func(data(eval_range), r_axes);
  };

  return tvm::compute(target_shape, compute, data->op->name + "_red", kCommReduce);
}

/*!
 * \brief Create a reduction operation.
 *
 * \param data The input tensor.
 * \param axis The axes along which the reduction is performed.
 * \param func The reduction function eg. tvm::sum
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 *
 * \return The result tensor.
 */
inline Tensor CommReduce(const Tensor& data,
                         const Array<Expr>& axis,
                         FReduce func,
                         bool keepdims = false) {
  auto ndim = data->shape.size();
  CHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto axis_val = detail::GetConstIntValues(axis, "axis");
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis_val);
  auto target_shape = MakeReduceTargetShape(real_axis, data, keepdims);
  return DoCommReduce(data, func, target_shape, real_axis,
      keepdims ? std::vector<int>() : real_axis);
}

/*!
* \brief Create an index reduction operation.
*
* \param data The input tensor.
* \param axis The axes along which the reduction is performed.
* \param func The reduction function
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return The result tensor.
*/
inline Tensor CommReduceIdx(const Tensor& data,
                            const Array<Expr>& axis,
                            FCommReduce func,
                            bool keepdims = false) {
  auto ndim = data->shape.size();
  CHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto axis_val = detail::GetConstIntValues(axis, "axis");
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis_val);
  auto reduce_axes = MakeReduceAxes(real_axis, data);
  auto target_shape = MakeReduceTargetShape(real_axis, data, keepdims);

  auto compute = [ndim, keepdims, &real_axis, &reduce_axes, &func, &data]
  (const Array<Var>& indices) {
    Array<Expr> eval_range;
    Array<Var> eval_indices;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        eval_range.push_back(reduce_axes[red_counter]);
        eval_indices.push_back(reduce_axes[red_counter]->var);
        red_counter++;
      } else {
        if (!keepdims) {
          eval_range.push_back(indices[arg_counter]);
          arg_counter++;
        } else {
          eval_range.push_back(indices[i]);
        }
      }
    }

    Array<Expr> ravel_shape;
    for (auto i : real_axis) {
      ravel_shape.push_back(data->shape[i]);
    }
    auto idx = detail::RavelIndex(eval_indices, ravel_shape);
    return func({ idx, data(eval_range) }, reduce_axes, nullptr);
  };

  auto temp_idx_val = tvm::compute(target_shape, compute,
                                   data->op->name + "_red_temp", kCommReduceIdx);
  auto temp_idx = temp_idx_val[0];
  auto temp_val = temp_idx_val[1];
  return tvm::compute(
    target_shape,
    [&temp_idx](const Array<Var>& indices) { return temp_idx(indices); },
    data->op->name + "_red",
    kCommReduceIdx);
}

/*! \brief A combiner function for a reduction */
using FCombine = std::function<Array<Expr>(Array<Var> lhs, Array<Var> rhs)>;

/*! \brief An initializer function for a reduction */
using FIdentity = std::function<Array<Expr>(std::vector<Type> types)>;

/*!
 * \brief Create a commutative reducer for a reduction
 *
 * \param fcombine A function to combine exprs
 * \param fidentity A function to initialize elements
 * \param name The name of the operation
 *
 * \return A reducer function which creates a reduce expression over an axis.
 */
inline FCommReduce MakeCommReducer(FCombine fcombine,
                                   FIdentity fidentity,
                                   std::string name = "reduce") {
  return [fcombine, fidentity, &name]
  (Array<Expr> exprs, const Array<IterVar>& axis, Expr* condition) {
    Array<Var> lhs, rhs;
    std::vector<Type> dtypes;

    for (size_t i = 0; i < exprs.size(); ++i) {
      auto dtype = exprs[i].type();
      dtypes.push_back(dtype);
      lhs.push_back(var(name + "_lhs_" + std::to_string(i), dtype));
      rhs.push_back(var(name + "_rhs_" + std::to_string(i), dtype));
    }

    auto result = fcombine(lhs, rhs);
    auto id_elem = fidentity(dtypes);
    auto cond = condition != nullptr ? *condition : tvm::const_true();

    auto combiner = tvm::ir::CommReducerNode::make(lhs, rhs, result, id_elem);
    Array<Expr> outputs;
    for (size_t i = 0; i < exprs.size(); ++i) {
      outputs.push_back(tvm::ir::Reduce::make(combiner, exprs, axis, cond, static_cast<int>(i)));
    }
    return outputs;
  };
}

/*! \brief Wrap tvm::min to ensure we get the correct overload */
inline Expr MinOp(Expr source, Array<IterVar> axis) {
  return tvm::min(source, axis);
}

/*! \brief Wrap tvm::max to ensure we get the correct overload */
inline Expr MaxOp(Expr source, Array<IterVar> axis) {
  return tvm::max(source, axis);  // NOLINT(*)
}

/*! \brief Wrap tvm::prod to ensure we get the correct overload */
inline Expr ProdOp(Expr source, Array<IterVar> axis) {
  return tvm::prod(source, axis);  // NOLINT(*)
}

/*!
* \brief Creates an operation that sums array elements over a given axis
*
* \param data The input tensor
* \param axis The axis to sum over. If axis is empty, the operation will
* sum over all elements of the array.
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return A Tensor whose op member is the sum operation
*/
inline Tensor sum(const Tensor& data, Array<Expr> axis, bool keepdims = false) {
  return CommReduce(data, axis, tvm::sum, keepdims);
}

inline Tensor collapse_sum(const Tensor& data, Array<Expr> target_shape) {
  CHECK_GE(data->shape.size(), target_shape.size());
  auto ishape = detail::GetConstIntValues(data->shape, "ishape");
  auto oshape = detail::GetConstIntValues(target_shape, "oshape");

  std::vector<int> reduce_axes;
  std::vector<int> squeeze_axes;
  for (int i_ax = ishape.size() - 1,
      o_ax = oshape.size() - 1; i_ax >= 0; --i_ax) {
    if (o_ax >= 0 && ishape[i_ax] == oshape[o_ax]) {
      --o_ax;
      continue;
    }
    reduce_axes.push_back(i_ax);
    if (o_ax < 0) {  // squeeze o_ax if was added during expansion
      squeeze_axes.push_back(i_ax);
    } else if (oshape[o_ax] == 1) {
      --o_ax;
    }
  }

  if (reduce_axes.size() == 0) return topi::identity(data, "tensor", kCommReduce);

  std::reverse(reduce_axes.begin(), reduce_axes.end());
  std::reverse(squeeze_axes.begin(), squeeze_axes.end());
  return DoCommReduce(data, tvm::sum, target_shape, reduce_axes, squeeze_axes);
}

/*!
* \brief Creates an operation that finds the minimum of elements over
* a given axis.
*
* \param data The input tensor
* \param axis The axis to find the minimum over. If axis is empty, the
* operation will find the minimum over all elements of the array.
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return A Tensor whose op member is the min operation
*/
inline Tensor min(const Tensor& data, Array<Expr> axis, bool keepdims = false) {
  return CommReduce(data, axis, MinOp, keepdims);
}

/*!
* \brief Creates an operation that finds the maximum of elements over
* a given axis.
*
* \param data The input tensor
* \param axis The axis to find the maximum over. If axis is empty, the
* operation will find the maximum over all elements of the array.
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return A Tensor whose op member is the max operation
*/
inline Tensor max(const Tensor& data, Array<Expr> axis, bool keepdims = false) {  // NOLINT(*)
  return CommReduce(data, axis, MaxOp, keepdims);
}

/*!
* \brief Creates an operation that finds the indices of the minimum
* values over a given axis.
*
* \param data The input tensor
* \param axis The axis along which the argmin is performed. If axis is empty,
* the operation will find the minimum index over all elements of the array.
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return A Tensor whose op member is the argmin operation
*/
inline Tensor argmin(const Tensor& data, Array<Expr> axis, bool keepdims = false) {
  auto fcombine = [](Array<Var> lhs, Array<Var> rhs) {
    Array<Expr> result;
    result.push_back(tvm::select(lhs[1] <= rhs[1], lhs[0], rhs[0]));  // idx
    result.push_back(tvm::select(lhs[1] <= rhs[1], lhs[1], rhs[1]));  // val
    return result;
  };
  auto fidentity = [](std::vector<Type> types) {
    Array<Expr> result;
    result.push_back(tvm::make_const(types[0], -1));  // idx
    result.push_back(types[1].max());  // val
    return result;
  };
  auto func = MakeCommReducer(fcombine, fidentity, "argmin");
  return CommReduceIdx(data, axis, func, keepdims);
}

/*!
* \brief Creates an operation that finds the indices of the maximum
* values over a given axis.
*
* \param data The input tensor
* \param axis The axis along which the argmax is performed. If axis is empty,
* the operation will find the maximum index over all elements of the array.
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return A Tensor whose op member is the argmax operation
*/
inline Tensor argmax(const Tensor& data, Array<Expr> axis, bool keepdims = false) {
  auto fcombine = [](Array<Var> lhs, Array<Var> rhs) {
    Array<Expr> result;
    result.push_back(tvm::select(lhs[1] >= rhs[1], lhs[0], rhs[0]));  // idx
    result.push_back(tvm::select(lhs[1] >= rhs[1], lhs[1], rhs[1]));  // val
    return result;
  };
  auto fidentity = [](std::vector<Type> types) {
    Array<Expr> result;
    result.push_back(tvm::make_const(types[0], -1));  // idx
    result.push_back(types[1].min());  // val
    return result;
  };
  auto func = MakeCommReducer(fcombine, fidentity, "argmax");
  return CommReduceIdx(data, axis, func, keepdims);
}

/*!
* \brief Creates product operation over given axis.
*
* \param data The input tensor
* \param axis The axis to do product over. If axis is empty, the
* operation will do the product over all elements of the array.
* \param keepdims If this is set to true, the axes which are reduced are
* left in the result as dimensions with size one. This enables the result
* to broadcast correctly against the input array.
*
* \return A Tensor whose op member is the prod operation
*/
inline Tensor prod(const Tensor& data, Array<Expr> axis, bool keepdims = false) {  // NOLINT(*)
  return CommReduce(data, axis, ProdOp, keepdims);
}

}  // namespace topi
#endif  // TOPI_REDUCTION_H_
