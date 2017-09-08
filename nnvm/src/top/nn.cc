/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./op_common.h"
#include "./elemwise_op_common.h"

namespace nnvm {
namespace top {

// dense
DMLC_REGISTER_PARAMETER(DenseParam);

inline std::vector<std::string> DenseListInputNames(const NodeAttrs& attrs) {
  const DenseParam& param = nnvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

inline bool DenseInferShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape) {
  const DenseParam& param = nnvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[DenseParam::kData];
  TShape oshape = (*out_shape)[0];
  // require data to be known
  if (dshape.ndim() ==  0) return false;
  dim_t num_input;
  num_input = dshape.ProdShape(1, dshape.ndim());
  SHAPE_ASSIGN_CHECK(*in_shape, DenseParam::kWeight, TShape({param.units, num_input}));
  if (param.use_bias) {
    SHAPE_ASSIGN_CHECK(*in_shape, DenseParam::kBias, TShape({param.units}));
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0, TShape({dshape[0], param.units}));
  if (oshape.ndim() != 0) {
    dshape[0] = oshape[0];
    SHAPE_ASSIGN_CHECK(*in_shape, DenseParam::kData, dshape);
  }
  return true;
}

NNVM_REGISTER_OP(dense)
.NNVM_DESCRIBE(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **bias**: `(units,)`
- **out**: `(x1, x2, ..., xn, num_hidden)`

The learnable parameters include both ``weight`` and ``bias``.

If ``use_bias`` is set to be false, then the ``bias`` term is ignored.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(DenseParam::__FIELDS__())
.set_attr_parser(ParamParser<DenseParam>)
.set_num_outputs(1)
.set_num_inputs([](const NodeAttrs& attrs) {
    const DenseParam& param = nnvm::get<DenseParam>(attrs.parsed);
    return param.use_bias ? 3 : 2;
  })
.set_attr<FListInputNames>("FListInputNames", DenseListInputNames)
.set_attr<FInferShape>("FInferShape", DenseInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_support_level(1);

// relu
NNVM_REGISTER_ELEMWISE_UNARY_OP(relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(input, 0)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);
}  // namespace top
}  // namespace nnvm
