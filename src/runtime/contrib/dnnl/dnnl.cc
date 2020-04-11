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

/*!
 * \file src/runtime/contrib/dnnl/dnnl.cc
 * \brief TVM compatible wrappers for dnnl kernels.
 */

#include "dnnl_kernel.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace dnnl;

typedef struct {
  void** data;
} DnnlPackedArgs;

// Read from memory, write to handle
inline void read_from_dnnl_memory(void* handle, const memory& mem) {
  size_t bytes = mem.get_desc().get_size();

  uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
  std::copy(src, src + bytes, reinterpret_cast<uint8_t*>(handle));
}

void dnnl_conv2d_common(float* data, float* weights, float* bias, float* out, int p_N_, int p_C_,
                        int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph_, int p_Pw_, int p_Kh_,
                        int p_Kw_, int p_Sh_, int p_Sw_, primitive_attr attr) {
  using tag = memory::format_tag;
  using dt = memory::data_type;
  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims conv2d_src_tz = {p_N_, p_C_, p_H_, p_W_};
  memory::dims conv2d_weights_tz = {p_O_, p_C_, p_Kh_, p_Kw_};
  if (p_G_ > 1) conv2d_weights_tz = {p_G_, 1, p_C_ / p_G_, p_Kh_, p_Kw_};
  memory::dims conv2d_bias_tz = {p_O_};
  memory::dims conv2d_dst_tz = {p_N_, p_O_, (p_H_ - p_Kh_ + 2 * p_Ph_ + p_Sh_) / p_Sh_,
                                (p_W_ - p_Kw_ + 2 * p_Pw_ + p_Sw_) / p_Sw_};
  memory::dims conv2d_strides = {p_Sh_, p_Sw_};
  memory::dims conv2d_padding = {p_Ph_, p_Pw_};

  auto user_src_memory = memory({{conv2d_src_tz}, dt::f32, tag::nchw}, eng, data);
  auto user_weights_memory =
      memory({{conv2d_weights_tz}, dt::f32, (p_G_ > 1) ? tag::goihw : tag::oihw}, eng, weights);
  auto conv2d_user_bias_memory = memory({{conv2d_bias_tz}, dt::f32, tag::x}, eng, bias);

  auto conv2d_src_md = memory::desc({conv2d_src_tz}, dt::f32, tag::any);
  auto conv2d_bias_md = memory::desc({conv2d_bias_tz}, dt::f32, tag::any);
  auto conv2d_weights_md = memory::desc({conv2d_weights_tz}, dt::f32, tag::any);
  auto conv2d_dst_md = memory::desc({conv2d_dst_tz}, dt::f32, tag::nchw);

  auto conv2d_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, conv2d_src_md, conv2d_weights_md,
      conv2d_bias_md, conv2d_dst_md, conv2d_strides, conv2d_padding, conv2d_padding);
  auto conv2d_prim_desc = convolution_forward::primitive_desc(conv2d_desc, attr, eng);

  auto conv2d_src_memory = user_src_memory;
  auto conv2d_weights_memory = user_weights_memory;
  auto conv2d_dst_memory = memory(conv2d_prim_desc.dst_desc(), eng);

  auto conv = convolution_forward(conv2d_prim_desc);
  conv.execute(s, {{DNNL_ARG_SRC, conv2d_src_memory},
                   {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                   {DNNL_ARG_BIAS, conv2d_user_bias_memory},
                   {DNNL_ARG_DST, conv2d_dst_memory}});
  s.wait();
  read_from_dnnl_memory(out, conv2d_dst_memory);
}

extern "C" void dnnl_conv2d(float* data, float* weights, float* out, int p_N_, int p_C_, int p_H_,
                            int p_W_, int p_O_, int p_G_, int p_Ph_, int p_Pw_, int p_Kh_,
                            int p_Kw_, int p_Sh_, int p_Sw_) {
  primitive_attr attr;
  std::vector<float> bias(p_O_, 0);
  return dnnl_conv2d_common(data, weights, bias.data(), out, p_N_, p_C_, p_H_, p_W_, p_O_, p_G_,
                            p_Ph_, p_Pw_, p_Kh_, p_Kw_, p_Sh_, p_Sw_, attr);
}

primitive_attr create_attr_with_relu_post_op() {
  post_ops ops;
  ops.append_eltwise(1.f, algorithm::eltwise_relu, 0.f, 0.f);

  primitive_attr attr;
  attr.set_post_ops(ops);

  return attr;
}

extern "C" void dnnl_fused_conv2d_relu(float* data, float* weights, float* out, int p_N_, int p_C_,
                                       int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph_, int p_Pw_,
                                       int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_) {
  std::vector<float> bias(p_O_, 0);
  return dnnl_conv2d_common(data, weights, bias.data(), out, p_N_, p_C_, p_H_, p_W_, p_O_, p_G_,
                            p_Ph_, p_Pw_, p_Kh_, p_Kw_, p_Sh_, p_Sw_,
                            create_attr_with_relu_post_op());
}

extern "C" void dnnl_fused_conv2d_bias_relu(float* data, float* weights, float* bias, float* out,
                                            int p_N_, int p_C_, int p_H_, int p_W_, int p_O_,
                                            int p_G_, int p_Ph_, int p_Pw_, int p_Kh_, int p_Kw_,
                                            int p_Sh_, int p_Sw_) {
  return dnnl_conv2d_common(data, weights, bias, out, p_N_, p_C_, p_H_, p_W_, p_O_, p_G_, p_Ph_,
                            p_Pw_, p_Kh_, p_Kw_, p_Sh_, p_Sw_, create_attr_with_relu_post_op());
}

extern "C" void dnnl_dense(float* data, float* weight, float* out, int p_B_,
                           int p_I_, int p_O_) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims data_tz = {p_B_, p_I_};
  memory::dims weight_tz = {p_O_, p_I_};
  memory::dims bias_tz = {p_O_};
  memory::dims dst_tz = {p_B_, p_O_};

  auto data_md = memory::desc{{data_tz}, dt::f32, tag::nc};
  auto weight_md = memory::desc({{weight_tz}, dt::f32, tag::nc});
  auto bias_md = memory::desc({{bias_tz}, dt::f32, tag::x});
  auto dst_md = memory::desc({{dst_tz}, dt::f32, tag::nc});

  std::vector<float> bias(p_O_, 0);
  auto data_memory = memory(data_md, eng, data);
  auto weight_memory = memory(weight_md, eng, weight);
  auto bias_memory = memory(bias_md, eng, bias.data());
  auto dst_memory = memory(dst_md, eng);

  auto dense_desc = inner_product_forward::desc(
      prop_kind::forward_inference, data_md, weight_md, bias_md, dst_md);
  auto dense_prim_desc = inner_product_forward::primitive_desc(dense_desc, eng);
  assert(dst_md == dense_prim_desc.dst_desc());

  auto dense = inner_product_forward(dense_prim_desc);
  dense.execute(s, {{DNNL_ARG_SRC, data_memory},
                    {DNNL_ARG_WEIGHTS, weight_memory},
                    {DNNL_ARG_BIAS, bias_memory},
                    {DNNL_ARG_DST, dst_memory}});
  s.wait();
  read_from_dnnl_memory(out, dst_memory);
}

extern "C" void dnnl_relu(float* data, float* out, int p_N_, int p_C_, int p_H_,
                          int p_W_) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims data_tz = {p_N_, p_C_, p_H_, p_W_};

  auto data_md = memory::desc{{data_tz}, dt::f32, tag::nchw};

  auto data_memory = memory(data_md, eng, data);
  auto dst_memory = memory(data_md, eng);

  auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                         algorithm::eltwise_relu, data_md, 0);
  auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, eng);
  assert(data_md == relu_prim_desc.dst_desc());

  auto relu = eltwise_forward(relu_prim_desc);
  relu.execute(s, {{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, dst_memory}});
  s.wait();
  read_from_dnnl_memory(out, dst_memory);
}

extern "C" void dnnl_bn(float* data, float* gamma, float* beta, float* mean, float* variance,
                        float* out, float* new_mean, float* new_variance, int p_N_, int p_C_,
                        int p_H_, int p_W_, int p_E_) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims data_tz = {p_N_, p_C_, p_H_, p_W_};

  auto data_md = memory::desc{{data_tz}, dt::f32, tag::nchw};

  auto data_memory = memory(data_md, eng, data);
  auto dst_memory = memory(data_md, eng);

  auto bn_desc = batch_normalization_forward::desc(
      prop_kind::forward_inference, data_md, p_E_,
      normalization_flags::use_global_stats |
          normalization_flags::use_scale_shift);
  auto bn_prim_desc = batch_normalization_forward::primitive_desc(bn_desc, eng);
  assert(data_md == bn_prim_desc.dst_desc());

  float* weight = reinterpret_cast<float*>(malloc(sizeof(float) * 2 * p_C_));
  memcpy(weight, gamma, sizeof(float) * p_C_);
  memcpy(weight + p_C_, beta, sizeof(float) * p_C_);

  auto weight_memory = memory(bn_prim_desc.weights_desc(), eng, weight);
  auto mean_memory = memory(bn_prim_desc.mean_desc(), eng, mean);
  auto variance_memory = memory(bn_prim_desc.variance_desc(), eng, variance);

  auto bn = batch_normalization_forward(bn_prim_desc);
  bn.execute(s, {{DNNL_ARG_SRC, data_memory},
                 {DNNL_ARG_DST, dst_memory},
                 {DNNL_ARG_SCALE_SHIFT, weight_memory},
                 {DNNL_ARG_MEAN, mean_memory},
                 {DNNL_ARG_VARIANCE, variance_memory}});
  s.wait();
  read_from_dnnl_memory(out, dst_memory);
  free(weight);
}

extern "C" void dnnl_add(float* data, float* weight, float* out, int p_N_,
                         int p_C_, int p_H_, int p_W_) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims data_tz = {p_N_, p_C_, p_H_, p_W_};

  auto data_md = memory::desc{{data_tz}, dt::f32, tag::nchw};
  auto weight_md = memory::desc({{data_tz}, dt::f32, tag::nchw});
  auto dst_md = memory::desc({{data_tz}, dt::f32, tag::nchw});

  auto data_memory = memory(data_md, eng, data);
  auto weight_memory = memory(weight_md, eng, weight);
  auto dst_memory = memory(dst_md, eng);

  auto add_desc =
      binary::desc(algorithm::binary_add, data_md, weight_md, dst_md);
  auto add_prim_desc = binary::primitive_desc(add_desc, eng);
  assert(dst_md == add_prim_desc.dst_desc());

  auto add = binary(add_prim_desc);
  add.execute(s, {{DNNL_ARG_SRC_0, data_memory},
                  {DNNL_ARG_SRC_1, weight_memory},
                  {DNNL_ARG_DST, dst_memory}});
  s.wait();
  read_from_dnnl_memory(out, dst_memory);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
