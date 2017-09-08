/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef NNVM_TOP_NN_H_
#define NNVM_TOP_NN_H_

#include <dmlc/base.h>
#include <dmlc/parameter.h>

namespace nnvm {
namespace top {

struct DenseParam : public dmlc::Parameter<DenseParam> {
  int units;
  bool use_bias;
  DMLC_DECLARE_PARAMETER(DenseParam) {
    DMLC_DECLARE_FIELD(units).set_lower_bound(1)
    .describe("Number of hidden units of the dense transformation.");
    DMLC_DECLARE_FIELD(use_bias).set_default(true)
    .describe("Whether to use bias parameter");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_NN_H_
