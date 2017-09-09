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

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float rate;

  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(rate).set_default(0.5)
        .set_range(0, 1)
        .describe("Fraction of the input that gets dropped out during training time.");
  }
};

struct BatchNormParam : public dmlc::Parameter<BatchNormParam> {
  int axis;
  float epsilon;
  float momentum;
  bool center;
  bool scale;

  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(axis).set_default(1)
      .describe("Specify which shape axis the channel is specified.");
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-5f)
        .describe("Small float added to variance to avoid dividing by zero.");
    DMLC_DECLARE_FIELD(center).set_default(true)
        .describe("If True, add offset of `beta` to normalized tensor."
                  "If False, `beta` is ignored.");
    DMLC_DECLARE_FIELD(scale).set_default(true)
        .describe("If True, multiply by `gamma`. If False, `gamma` is not used."
                  "When the next layer is piecewise linear (also e.g. `nn.relu`),"
                  "this can be disabled since the scaling"
                  "will be done by the next layer.");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kGamma = 1;
  static const constexpr int kBeta = 2;
  static const constexpr int kMovingMean = 3;
  static const constexpr int kMovingVariance = 4;
};

struct SoftmaxParam : public dmlc::Parameter<SoftmaxParam> {
  int axis;

  DMLC_DECLARE_PARAMETER(SoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
      .describe("The axis to sum over when computing softmax.");
  }
};

struct LogSoftmaxParam : public dmlc::Parameter<LogSoftmaxParam> {
  int axis;

  DMLC_DECLARE_PARAMETER(LogSoftmaxParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
      .describe("The axis to sum over when computing softmax.");
  }
};

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_NN_H_
