/*!
 *  Copyright (c) 2017 by Contributors
 * \file tensor.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef NNVM_TOP_TENSOR_H_
#define NNVM_TOP_TENSOR_H_

namespace nnvm {
namespace top {

struct ConcatParam : public dmlc::Parameter<ConcatParam> {
  int dim;
  DMLC_DECLARE_PARAMETER(ConcatParam) {
    DMLC_DECLARE_FIELD(dim).set_range(0,  4).set_default(1)
    .describe("the axis to be concated.");
  }
};

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_TENSOR_H_
