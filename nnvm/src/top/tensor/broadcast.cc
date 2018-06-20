/*!
 *  Copyright (c) 2017 by Contributors
 * \file broadcast.cc
 * \brief broadcast operator.
 */
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
#include <nnvm/top/nn.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/broadcast.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

// broadcast_to
DMLC_REGISTER_PARAMETER(BroadcastToParam);

inline bool BroadcastToInferShape(const NodeAttrs& attrs,
                                  std::vector<TShape>* in_attrs,
                                  std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;

  const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
  CHECK_EQ(ishape.ndim(), param.shape.ndim())
      << "Operand of shape " << ishape
      << " cannot be broadcasted to " << param.shape;
  TShape oshape = param.shape;
  for (dim_t i = 0; i < ishape.ndim(); ++i) {
    if (oshape[i] != 0) {
      CHECK(ishape[i] == oshape[i] || ishape[i] == 1)
        << "Array cannot be broadcasted from " <<
          ishape << " to " << param.shape;
    } else {
      oshape[i] = ishape[i];
    }
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(broadcast_to)
.describe(R"code(Broadcasts the input array to a new shape.

Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                           [ 1.,  2.,  3.]])

The dimension which you do not want to change can also be kept as `0` which means copy the original value.
So with `shape=(2,0)`, we will obtain the same result as in the above example.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr_parser(ParamParser<BroadcastToParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BroadcastToParam>)
.set_attr<FInferShape>("FInferShape", BroadcastToInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
    const Array<Tensor>& inputs,
    const Array<Tensor>& out_info) {
      const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
      auto shape = ShapeToArray(param.shape);
      return Array<Tensor>{ topi::broadcast_to(inputs[0], shape) };
  })
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4);

// binary broadcast op
inline bool BinaryBroadcastShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_attrs,
                                 std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& lhs = (*in_attrs)[0];
  const TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (lhs.ndim() == 0 || rhs.ndim() == 0) return false;

  if (lhs == rhs) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *out_attrs, 0, lhs);
    return true;
  }
  TShape out(std::max(lhs.ndim(), rhs.ndim()));
  dim_t bl = out.ndim() - lhs.ndim();
  dim_t br = out.ndim() - rhs.ndim();
  for (dim_t i = 0; i < out.ndim(); ++i) {
    dim_t l = 1, r = 1;
    if (i >= bl) l = lhs[i - bl];
    if (i >= br) r = rhs[i - br];
    if (l != r) {
      if (l == 0 || r == 0) {
        out[i] = 0;
      } else {
        CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes "
          << lhs << " " << rhs << ", l=" << l << ", r=" << r;
        out[i] = std::max(l, r);
      }
    } else {
      out[i] = l;
    }
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out);
  return true;
}

inline bool BinaryBroadcastCorrectLayout(const NodeAttrs& attrs,
                                         std::vector<Layout> *ilayouts,
                                         const std::vector<Layout> *last_ilayouts,
                                         std::vector<Layout> *olayouts) {
  CHECK_EQ(ilayouts->size(), 2U);
  CHECK_EQ(olayouts->size(), 1U);
  Layout lhs = (*ilayouts)[0];
  Layout rhs = (*ilayouts)[1];
  Layout out(Layout::Undef());

  if (lhs.defined() && rhs.defined()) {
    if (lhs == rhs) {
      NNVM_ASSIGN_LAYOUT(*olayouts, 0, lhs);
      return true;
    }
    // For example, NCHW <-> CHW, N16nCH16cW <-> HCW16c, etc, are broadcast-convertible
    // because as the definition, CHW can broadcast with NCHW.
    // For the second case, we can convert HCW16c to CH16cW then it can broadcast with N16nCH16cW.
    // But CNHW <-> CHW, NCHW16n <-> CHW are not,
    // because not matter how we adjust the layout of 'CHW',
    // we can never have an 'N' between 'C' and "HW".
    size_t l_start = 0, r_start = 0;
    size_t l = 0, r = 0;
    bool find_first_match = false;
    while (l < lhs.ndim() && r < rhs.ndim()) {
      if (!rhs.contains(Layout::to_superdim(lhs[l]))) {
        CHECK(!find_first_match) << lhs << " and " << rhs << " are not broadcast-convertible";
        l_start = ++l;
      } else if (!lhs.contains(Layout::to_superdim(rhs[r]))) {
        CHECK(!find_first_match) << lhs << " and " << rhs << " are not broadcast-convertible";
        r_start = ++r;
      } else {
        find_first_match = true;
        ++l; ++r;
      }
    }
    if (l_start > 0 && r_start > 0) {
      LOG(FATAL) << lhs << " and " << rhs << " are not broadcast-convertible";
    } else if (l_start > 0) {
      rhs = lhs.sublayout(l_start, lhs.ndim()-l_start);
      out = lhs;
    } else if (r_start > 0) {
      lhs = rhs.sublayout(r_start, rhs.ndim()-r_start);
      out = rhs;
    } else {
      // prior to keep left layout
      rhs = lhs;
      out = lhs;
    }
  } else if (lhs.defined()) {
    const Layout& last_lhs = last_ilayouts->at(0);
    if (last_lhs.defined()) {
      CHECK(lhs.convertible(last_lhs)) << "current lhs layout " << lhs
                                       << " cannot be converted to the original one " << last_lhs;
      lhs = last_lhs;
      // cannot decide output layout
    }
  } else if (rhs.defined()) {
    const Layout& last_rhs = last_ilayouts->at(1);
    if (last_rhs.defined()) {
      CHECK(rhs.convertible(last_rhs)) << "current rhs layout " << rhs
                                       << " cannot be converted to the original one " << last_rhs;
      rhs = last_rhs;
      // cannot decide output layout
    }
  }
  NNVM_ASSIGN_LAYOUT(*ilayouts, 0, lhs);
  NNVM_ASSIGN_LAYOUT(*ilayouts, 1, rhs);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, out);
  return true;
}

#define NNVM_REGISTER_BINARY_BROADCAST_OP(name)                     \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<FInferShape>("FInferShape", BinaryBroadcastShape)       \
  .set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)           \
  .set_attr<FCorrectLayout>("FCorrectLayout",                       \
    BinaryBroadcastCorrectLayout)                                   \
  .set_attr<FInplaceOption>("FInplaceOption",                       \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .set_attr<FTVMCompute>(                                           \
    "FTVMCompute", [](const NodeAttrs& attrs,                       \
      const Array<Tensor>& inputs,                                  \
      const Array<Tensor>& out_info) {                              \
        return Array<Tensor>{                                       \
          topi::name(inputs[0], inputs[1]) };                       \
    })                                                              \
  .add_argument("lhs", "Tensor", "first input")                     \
  .add_argument("rhs", "Tensor", "second input")


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_add)
.add_alias("__add_symbol__")
.describe(R"code(Returns element-wise sum of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_add(x, y) = [[ 1.,  1.,  1.],
                          [ 2.,  2.,  2.]]

)code" NNVM_ADD_FILELINE);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_sub)
.add_alias("__sub_symbol__")
.describe(R"code(Returns element-wise difference of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                          [ 0.,  0.,  0.]]

)code" NNVM_ADD_FILELINE);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_mul)
.add_alias("__mul_symbol__")
.describe(R"code(Returns element-wise product of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                          [ 1.,  1.,  1.]]
)code" NNVM_ADD_FILELINE);


NNVM_REGISTER_BINARY_BROADCAST_OP(broadcast_div)
.add_alias("__div_symbol__")
.describe(R"code(Returns element-wise division of the input arrays with broadcasting.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_div(x, y) = [[ 3.,  3.,  3.],
                          [ 2.,  2.,  2.]]

)code" NNVM_ADD_FILELINE);

}  // namespace top
}  // namespace nnvm
