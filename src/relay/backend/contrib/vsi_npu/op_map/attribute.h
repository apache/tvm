
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_ATTRIBUTE_H_
#define TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_ATTRIBUTE_H_

#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>

#include <algorithm>
#include <vector>

#include "tim/vx/tensor.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {
namespace op_map {

void attrs_vector_transform(tvm::relay::Shape tvm_attr, std::vector<uint32_t>& vx_attr) {
  std::transform(tvm_attr.begin(), tvm_attr.end(), std::back_inserter(vx_attr),
                 [](const PrimExpr& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });
}

void attrs_int_vector_transform(tvm::relay::Shape tvm_attr, std::vector<int32_t>& vx_attr) {
  std::transform(tvm_attr.begin(), tvm_attr.end(), std::back_inserter(vx_attr),
                 [](const PrimExpr& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });
}

void attrs_vector_transform2(Array<Integer> tvm_attr, std::vector<uint32_t>& vx_attr) {
  std::transform(tvm_attr.begin(), tvm_attr.end(), std::back_inserter(vx_attr),
                 [](const Integer& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });
}

void attrs_int_vector_transform2(Array<Integer> tvm_attr, std::vector<int32_t>& vx_attr) {
  std::transform(tvm_attr.begin(), tvm_attr.end(), std::back_inserter(vx_attr),
                 [](const Integer& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });
}

void attrs_uint_transform(uint32_t tvm_attr, uint32_t& vx_attr) { vx_attr = tvm_attr; }

void attrs_int_transform(int tvm_attr, int& vx_attr) { vx_attr = tvm_attr; }

void attrs_roundtype_transform(bool tvm_attr, tim::vx::RoundType& round_type) {
  round_type = tvm_attr ? tim::vx::RoundType::CEILING : tim::vx::RoundType::FLOOR;
}

void attrs_bool_transform(bool tvm_attr, bool& vx_attr) { vx_attr = tvm_attr; }

void attrs_string_transform(std::string tvm_attr, std::string& vx_attr) { vx_attr = tvm_attr; }

void attrs_padtype_transform(tvm::relay::Shape tvm_attr, tim::vx::PadType& vx_attr) {
  std::vector<uint32_t> padding;
  std::transform(tvm_attr.begin(), tvm_attr.end(), std::back_inserter(padding),
                 [](const PrimExpr& dim) { return static_cast<int>(dim.as<IntImmNode>()->value); });
  std::vector<uint32_t> valid_pads = {0, 0, 0, 0};
  if (padding == valid_pads) {
    vx_attr = tim::vx::PadType::VALID;
  } else {
    vx_attr = tim::vx::PadType::SAME;
  }
}

#define TRANS_ATTR(attr_name, creator) creator(tvm_attr_struct->attr_name, tim::vx_##attr_name)

struct TvxConv2dAttrs {
  std::vector<uint32_t> kernel_size;
  std::vector<uint32_t> strides;
  std::vector<uint32_t> dilation;
  std::vector<uint32_t> padding;
  tim::vx::PadType pad_type;
  int groups;
  TvxConv2dAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<Conv2DAttrs>();
    attrs_vector_transform(tvm_attr_struct->kernel_size, kernel_size);
    attrs_vector_transform(tvm_attr_struct->strides, strides);
    attrs_vector_transform(tvm_attr_struct->dilation, dilation);
    attrs_vector_transform(tvm_attr_struct->padding, padding);
    attrs_int_transform(tvm_attr_struct->groups, groups);
    attrs_padtype_transform(tvm_attr_struct->padding, pad_type);
  }
};

struct TvxDeConv2dAttrs {
  std::vector<uint32_t> kernel_size;
  std::vector<uint32_t> strides;
  int channels;
  tim::vx::PadType pad_type;
  std::vector<uint32_t> padding;
  int groups;
  TvxDeConv2dAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<Conv2DTransposeAttrs>();
    attrs_vector_transform(tvm_attr_struct->kernel_size, kernel_size);
    attrs_vector_transform(tvm_attr_struct->strides, strides);
    attrs_vector_transform(tvm_attr_struct->padding, padding);
    attrs_padtype_transform(tvm_attr_struct->padding, pad_type);
    attrs_int_transform(tvm_attr_struct->groups, groups);
    channels = static_cast<int>(tvm_attr_struct->channels.as<IntImmNode>()->value);
  }
};

struct TvxPool2DAttrs {
  std::vector<uint32_t> pool_size;
  std::vector<uint32_t> strides;
  tim::vx::RoundType ceil_mode;
  tim::vx::PadType pad_type;
  static const int kMaxPool = 0;
  static const int kAvgPool = 1;
  TvxPool2DAttrs(const Call call, int type) {
    if (type == kMaxPool) {
      auto tvm_attr_struct = call->attrs.as<MaxPool2DAttrs>();
      attrs_vector_transform(tvm_attr_struct->pool_size, pool_size);
      attrs_vector_transform(tvm_attr_struct->strides, strides);
      attrs_roundtype_transform(tvm_attr_struct->ceil_mode, ceil_mode);
      attrs_padtype_transform(tvm_attr_struct->padding, pad_type);
    } else {
      auto tvm_attr_struct = call->attrs.as<AvgPool2DAttrs>();
      attrs_vector_transform(tvm_attr_struct->pool_size, pool_size);
      attrs_vector_transform(tvm_attr_struct->strides, strides);
      attrs_roundtype_transform(tvm_attr_struct->ceil_mode, ceil_mode);
      attrs_padtype_transform(tvm_attr_struct->padding, pad_type);
    }
  }
};

struct TvxSoftmaxAttrs {
  uint32_t axis;
  TvxSoftmaxAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<SoftmaxAttrs>();
    attrs_uint_transform(tvm_attr_struct->axis, axis);
  }
};

struct TvxTransposeAttrs {
  std::vector<uint32_t> axes;
  TvxTransposeAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<TransposeAttrs>();
    attrs_vector_transform2(tvm_attr_struct->axes, axes);
  }
};

struct TvxReshapeAttrs {
  std::vector<uint32_t> newshape;
  TvxReshapeAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<ReshapeAttrs>();
    attrs_vector_transform2(tvm_attr_struct->newshape, newshape);
  }
};

struct TvxSqueezeAttrs {
  std::vector<uint32_t> axis;
  TvxSqueezeAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<SqueezeAttrs>();
    attrs_vector_transform2(tvm_attr_struct->axis, axis);
  }
};

struct TvxDepthtoSpaceAttrs {
  int block_size;
  TvxDepthtoSpaceAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<SubPixelAttrs>();
    attrs_int_transform(tvm_attr_struct->block_size, block_size);
  }
};

struct TvxReduceMeanAttrs {
  std::vector<int32_t> axis;
  bool keepdims;
  TvxReduceMeanAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<ReduceAttrs>();
    attrs_bool_transform(tvm_attr_struct->keepdims, keepdims);
    attrs_int_vector_transform2(tvm_attr_struct->axis, axis);
  }
};

struct TvxResizeAttrs {
  std::vector<int32_t> size;
  std::string method;
  std::string coordinate_transformation_mode;
  TvxResizeAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<Resize2DAttrs>();
    attrs_int_vector_transform(tvm_attr_struct->size, size);
    attrs_string_transform(tvm_attr_struct->method, method);
    attrs_string_transform(tvm_attr_struct->coordinate_transformation_mode,
                           coordinate_transformation_mode);
  }
};

struct TvxClipAttrs {
  float min;
  float max;
  TvxClipAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<ClipAttrs>();
    min = tvm_attr_struct->a_min;
    max = tvm_attr_struct->a_max;
  }
};

struct TvxLeakyReluAttrs {
  float alpha;
  TvxLeakyReluAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<LeakyReluAttrs>();
    alpha = tvm_attr_struct->alpha;
  }
};

struct TvxPadAttrs {
  std::vector<std::vector<uint32_t>> pad_width;
  TvxPadAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<PadAttrs>();
    for (uint32_t i = 0; i < tvm_attr_struct->pad_width.size(); i++) {
      std::vector<uint32_t> pad_width_tmp;
      attrs_vector_transform2(tvm_attr_struct->pad_width[i], pad_width_tmp);
      pad_width.push_back(pad_width_tmp);
    }
  }
};

struct TvxDeconvAttrs {
  tim::vx::PadType pad_type;
  std::vector<uint32_t> kernel_size;
  std::vector<uint32_t> strides;
  std::vector<uint32_t> padding;
  int groups;
  TvxDeconvAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<Conv2DTransposeAttrs>();
    attrs_vector_transform(tvm_attr_struct->kernel_size, kernel_size);
    attrs_vector_transform(tvm_attr_struct->strides, strides);
    attrs_vector_transform(tvm_attr_struct->padding, padding);
    attrs_padtype_transform(tvm_attr_struct->padding, pad_type);
    attrs_int_transform(tvm_attr_struct->groups, groups);
  }
};

struct TvxDilateAttrs {
  std::vector<uint32_t> strides;
  TvxDilateAttrs(const Call call) {
    auto tvm_attr_struct = call->attrs.as<DilateAttrs>();
    attrs_vector_transform(tvm_attr_struct->strides, strides);
  }
};

}  // namespace op_map
}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif