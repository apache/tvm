/*!
 *  Copyright (c) 2017 by Contributors
 * \file reduce.cc
 * \brief reduce operator.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

// reduce
DMLC_REGISTER_PARAMETER(ReduceParam);

inline TShape ReduceShapeImpl(const TShape& ishape,
                              const TShape& axis,
                              bool keepdims,
                              bool exclude) {
  if (axis.ndim() == 0) {
    if (keepdims) {
      return TShape(ishape.ndim());
    } else {
      return TShape(1);
    }
  }
  CHECK_LT(axis[axis.ndim() - 1], ishape.ndim())
    << "Reduction axis " << axis[axis.ndim() - 1]
    << " Exceeds input dimensions " << ishape;

  if (keepdims) {
    TShape oshape(ishape);
    if (exclude) {
      for (dim_t i = 0, j = 0; i < ishape.ndim(); ++i) {
        if (j < axis.ndim() && i == axis[j]) {
          ++j;
          continue;
        }
        oshape[i] = 1;
      }
      return oshape;
    }

    for (dim_t i = 0; i < axis.ndim(); ++i) {
      oshape[axis[i]] = 1;
    }
    return oshape;
  }

  if (exclude) {
    TShape oshape = TShape(axis.ndim());
    for (dim_t i = 0; i < axis.ndim(); ++i) {
      oshape[i] = ishape[axis[i]];
    }
    return oshape;
  }
  TShape oshape = TShape(std::max<dim_t>(1, ishape.ndim() - axis.ndim()));
  for (dim_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
    if (j < axis.ndim() && i == axis[j]) {
      ++j;
      continue;
    }
    oshape[k++] = ishape[i];
  }
  return oshape;
}

inline bool ReduceShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
  NNVM_ASSIGN_INPUT_SHAPE(
      attrs, *out_attrs, 0,
      ReduceShapeImpl((*in_attrs)[0], param.axis,
                      param.keepdims, param.exclude));
  return true;
}

template<typename PType>
inline void AxesParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  std::sort(&param.axis[0], &param.axis[param.axis.ndim()]);
  attrs->parsed = std::move(param);
}

#define NNVM_REGISTER_REDUCE_OP(op)                                     \
  NNVM_REGISTER_OP(op)                                                  \
  .add_argument("data", "Tensor", "The input")                          \
  .add_arguments(ReduceParam::__FIELDS__())                             \
  .set_attr_parser(AxesParamParser<ReduceParam>)                        \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReduceParam>) \
  .set_attr<FInferShape>("FInferShape", ReduceShape)                    \
  .set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)               \
  .set_num_inputs(1)                                                    \
  .set_num_outputs(1)                                                   \



NNVM_REGISTER_REDUCE_OP(sum)
.describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" NNVM_ADD_FILELINE);

NNVM_REGISTER_REDUCE_OP(max)
.describe(R"code(Computes the max of array elements over given axes.

)code" NNVM_ADD_FILELINE);

NNVM_REGISTER_REDUCE_OP(min)
.describe(R"code(Computes the min of array elements over given axes.

)code" NNVM_ADD_FILELINE);


}  // namespace top
}  // namespace nnvm
