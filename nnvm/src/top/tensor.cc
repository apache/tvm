/*!
 *  Copyright (c) 2017 by Contributors
 * \file tensor.cc
 * \brief Property def of tensor operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "./op_common.h"
#include "./elemwise_op_common.h"

namespace nnvm {
namespace top {
// sigmoid
NNVM_REGISTER_ELEMWISE_UNARY_OP(sigmoid)
.describe(R"code(Computes sigmoid.

.. math::
   y = 1 / (1 + exp(-x))

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// tanh
NNVM_REGISTER_ELEMWISE_UNARY_OP(tanh)
.describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.

.. math::
   tanh(x) = sinh(x) / cosh(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// exp
NNVM_REGISTER_ELEMWISE_UNARY_OP(exp)
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   exp(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// log
NNVM_REGISTER_ELEMWISE_UNARY_OP(log)
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// flatten
inline bool FlattenInferShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  uint32_t target_dim = 1;
  for (uint32_t i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({dshape[0], target_dim}));
  return true;
}

NNVM_REGISTER_OP(flatten)
.describe(R"code(Flattens the input array into a 2-D array by collapsing the higher dimensions.

For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
the input array into an output array of shape ``(d1, d2*...*dk)``.

Example::

    x = [[
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [    [1,2,3],
        [4,5,6],
        [7,8,9]
    ]],

    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]

)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", FlattenInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.add_argument("data", "Tensor", "Input data.")
.set_support_level(1);

// concatenate
DMLC_REGISTER_PARAMETER(ConcatenateParam);

inline bool ConcatenateInferShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_shape,
                                  std::vector<TShape> *out_shape) {
  const ConcatenateParam& param = nnvm::get<ConcatenateParam>(attrs.parsed);
  TShape dshape;
  dim_t size = 0;
  bool has_zero = false;
  for (size_t i = 0; i < in_shape->size(); ++i) {
    TShape tmp = (*in_shape)[i];
    if (tmp.ndim()) {
      CHECK_LT(static_cast<dim_t>(param.axis), tmp.ndim())
          << "concat dim " << param.axis << " out of range of input shape " << tmp;
      has_zero = tmp[param.axis] == 0 || has_zero;
      size += tmp[param.axis];
      tmp[param.axis] = 0;
      shape_assign(&dshape, tmp);
    }
  }

  TShape tmp = (*out_shape)[0];
  if (tmp.ndim()) {
    CHECK_LT(static_cast<dim_t>(param.axis), tmp.ndim())
        << "concat dim " << param.axis << " out of range of input shape " << tmp;
    tmp[param.axis] = 0;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == 0) return false;

  for (size_t i = 0; i < in_shape->size(); ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, dshape);
  }

  if (!has_zero) dshape[param.axis] = size;
  SHAPE_ASSIGN_CHECK(*out_shape, 0, dshape);
  return dshape.Size() != 0;
}

NNVM_REGISTER_OP(concatenate)
.describe(R"code(Joins input arrays along a given axis.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.

Example::

   x = [[1,1],[2,2]]
   y = [[3,3],[4,4],[5,5]]
   z = [[6,6], [7,7],[8,8]]

   concatenate(x,y,z,dim=0) = [[ 1.,  1.],
                               [ 2.,  2.],
                               [ 3.,  3.],
                               [ 4.,  4.],
                               [ 5.,  5.],
                               [ 6.,  6.],
                               [ 7.,  7.],
                               [ 8.,  8.]]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concatenate(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                             [ 4.,  4.,  7.,  7.],
                             [ 5.,  5.,  8.,  8.]]

)code" NNVM_ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(nnvm::kVarg)
.set_attr_parser(ParamParser<ConcatenateParam>)
.add_argument("data", "Tensor-or-Tensor[]", "List of arrays to concatenate")
.set_attr<FInferShape>("FInferShape", ConcatenateInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.add_arguments(ConcatenateParam::__FIELDS__())
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_add)
.describe(R"code(Element-wise add

)code")
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.describe(R"code(Element-wise substraction

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mul)
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_div)
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

// cast
DMLC_REGISTER_PARAMETER(CastParam);

inline bool CastInferType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return true;
}

NNVM_REGISTER_OP(cast)
.describe(R"code(Cast the content of input to dtype.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data array")
.set_attr_parser(ParamParser<CastParam>)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", CastInferType)
.add_arguments(CastParam::__FIELDS__())
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
