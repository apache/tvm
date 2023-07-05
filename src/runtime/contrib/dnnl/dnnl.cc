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

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "dnnl_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace dnnl;

typedef struct {
  void** data;
} DnnlPackedArgs;

inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape,
                                                memory::data_type dtype) {
  using tag = memory::format_tag;

  dnnl::memory::desc data_md;

  switch (shape.size()) {
    case 2:
      data_md = dnnl::memory::desc({shape, dtype, tag::ab});
      break;
    case 3:
      data_md = dnnl::memory::desc({shape, dtype, tag::abc});
      break;
    case 4:
      data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
      break;
    case 5:
      data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
      break;
    default:
      LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
      break;
  }
  return data_md;
}

// Read from memory, write to handle
inline void read_from_dnnl_memory(void* handle, const memory& mem) {
  size_t bytes = mem.get_desc().get_size();

  uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
  std::copy(src, src + bytes, reinterpret_cast<uint8_t*>(handle));
}

void dnnl_conv2d_common(float* data, float* weights, float* bias, float* out, int p_N_, int p_C_,
                        int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph0_, int p_Pw0_, int p_Ph1_,
                        int p_Pw1_, int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_, primitive_attr attr,
                        bool channel_last, bool pre_cast, bool post_cast) {
  using tag = memory::format_tag;
  using dt = memory::data_type;
  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims conv2d_src_tz = {p_N_, p_C_, p_H_, p_W_};
  memory::dims conv2d_weights_tz = {p_O_, p_C_, p_Kh_, p_Kw_};
  if (p_G_ > 1) conv2d_weights_tz = {p_G_, 1, p_C_ / p_G_, p_Kh_, p_Kw_};
  memory::dims conv2d_bias_tz = {p_O_};
  memory::dims conv2d_dst_tz = {p_N_, p_O_, (p_H_ - p_Kh_ + p_Ph0_ + p_Ph1_ + p_Sh_) / p_Sh_,
                                (p_W_ - p_Kw_ + p_Pw0_ + p_Pw1_ + p_Sw_) / p_Sw_};
  memory::dims conv2d_strides = {p_Sh_, p_Sw_};
  memory::dims conv2d_padding0 = {p_Ph0_, p_Pw0_};
  memory::dims conv2d_padding1 = {p_Ph1_, p_Pw1_};

  auto user_src_memory =
      memory({{conv2d_src_tz}, pre_cast ? dt::f32 : dt::bf16, channel_last ? tag::nhwc : tag::nchw},
             eng, data);
  auto user_weights_memory = memory({{conv2d_weights_tz},
                                     (pre_cast && post_cast) ? dt::f32 : dt::bf16,
                                     channel_last ? tag::hwio : tag::oihw},
                                    eng, weights);
  if (p_G_ > 1)
    user_weights_memory = memory({{conv2d_weights_tz},
                                  (pre_cast && post_cast) ? dt::f32 : dt::bf16,
                                  channel_last ? tag::ghwio : tag::goihw},
                                 eng, weights);
  auto conv2d_user_bias_memory =
      memory({{conv2d_bias_tz}, (pre_cast && post_cast) ? dt::f32 : dt::bf16, tag::x}, eng, bias);
  auto user_dst_memory = memory(
      {{conv2d_dst_tz}, post_cast ? dt::f32 : dt::bf16, channel_last ? tag::nhwc : tag::nchw}, eng,
      out);

  auto conv2d_src_md =
      memory::desc({conv2d_src_tz}, (pre_cast && post_cast) ? dt::f32 : dt::bf16, tag::any);
  auto conv2d_bias_md =
      memory::desc({conv2d_bias_tz}, (pre_cast && post_cast) ? dt::f32 : dt::bf16, tag::any);
  auto conv2d_weights_md =
      memory::desc({conv2d_weights_tz}, (pre_cast && post_cast) ? dt::f32 : dt::bf16, tag::any);
  auto conv2d_dst_md =
      memory::desc({conv2d_dst_tz}, (pre_cast && post_cast) ? dt::f32 : dt::bf16, tag::any);

  auto conv2d_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, conv2d_src_md, conv2d_weights_md,
      conv2d_bias_md, conv2d_dst_md, conv2d_strides, conv2d_padding0, conv2d_padding1);
  auto conv2d_prim_desc = convolution_forward::primitive_desc(conv2d_desc, attr, eng);

  // reorder if src layout not DNNL chosen.
  auto conv2d_src_memory = user_src_memory;
  if (conv2d_prim_desc.src_desc() != user_src_memory.get_desc()) {
    conv2d_src_memory = memory(conv2d_prim_desc.src_desc(), eng);
    auto reorder_src = reorder(user_src_memory, conv2d_src_memory);
    reorder_src.execute(s, {{DNNL_ARG_FROM, user_src_memory}, {DNNL_ARG_TO, conv2d_src_memory}});
  }

  // reorder if weights layout not DNNL chosen.
  auto conv2d_weights_memory = user_weights_memory;
  if (conv2d_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    conv2d_weights_memory = memory(conv2d_prim_desc.weights_desc(), eng);
    auto reorder_weights = reorder(user_weights_memory, conv2d_weights_memory);
    reorder_weights.execute(
        s, {{DNNL_ARG_FROM, user_weights_memory}, {DNNL_ARG_TO, conv2d_weights_memory}});
  }

  auto conv2d_dst_memory = user_dst_memory;
  if (conv2d_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
    conv2d_dst_memory = memory(conv2d_prim_desc.dst_desc(), eng);
  }

  auto conv = convolution_forward(conv2d_prim_desc);
  conv.execute(s, {{DNNL_ARG_SRC, conv2d_src_memory},
                   {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                   {DNNL_ARG_BIAS, conv2d_user_bias_memory},
                   {DNNL_ARG_DST, conv2d_dst_memory}});

  // reorder if dst layout not DNNL chosen.
  if (conv2d_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
    reorder(conv2d_dst_memory, user_dst_memory)
        .execute(s, {{DNNL_ARG_FROM, conv2d_dst_memory}, {DNNL_ARG_TO, user_dst_memory}});
  }

  s.wait();
}

extern "C" void dnnl_conv2d(float* data, float* weights, float* out, int p_N_, int p_C_, int p_H_,
                            int p_W_, int p_O_, int p_G_, int p_Ph0_, int p_Pw0_, int p_Ph1_,
                            int p_Pw1_, int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_) {
  primitive_attr attr;
  std::vector<float> bias(p_O_, 0);
  return dnnl_conv2d_common(data, weights, bias.data(), out, p_N_, p_C_, p_H_, p_W_, p_O_, p_G_,
                            p_Ph0_, p_Pw0_, p_Ph1_, p_Pw1_, p_Kh_, p_Kw_, p_Sh_, p_Sw_, attr, false,
                            true, true);
}

primitive_attr create_attr_with_relu_post_op() {
  post_ops ops;
  ops.append_eltwise(1.f, algorithm::eltwise_relu, 0.f, 0.f);

  primitive_attr attr;
  attr.set_post_ops(ops);

  return attr;
}

extern "C" void dnnl_fused_conv2d_relu(float* data, float* weights, float* out, int p_N_, int p_C_,
                                       int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph0_,
                                       int p_Pw0_, int p_Ph1_, int p_Pw1_, int p_Kh_, int p_Kw_,
                                       int p_Sh_, int p_Sw_) {
  std::vector<float> bias(p_O_, 0);
  return dnnl_conv2d_common(data, weights, bias.data(), out, p_N_, p_C_, p_H_, p_W_, p_O_, p_G_,
                            p_Ph0_, p_Pw0_, p_Ph1_, p_Pw1_, p_Kh_, p_Kw_, p_Sh_, p_Sw_,
                            create_attr_with_relu_post_op(), false, true, true);
}

extern "C" void dnnl_fused_conv2d_bias_relu(float* data, float* weights, float* bias, float* out,
                                            int p_N_, int p_C_, int p_H_, int p_W_, int p_O_,
                                            int p_G_, int p_Ph0_, int p_Pw0_, int p_Ph1_,
                                            int p_Pw1_, int p_Kh_, int p_Kw_, int p_Sh_,
                                            int p_Sw_) {
  return dnnl_conv2d_common(data, weights, bias, out, p_N_, p_C_, p_H_, p_W_, p_O_, p_G_, p_Ph0_,
                            p_Pw0_, p_Ph1_, p_Pw1_, p_Kh_, p_Kw_, p_Sh_, p_Sw_,
                            create_attr_with_relu_post_op(), false, true, true);
}

extern "C" void dnnl_dense(float* data, float* weight, float* out, int p_B_, int p_I_, int p_O_) {
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

  auto dense_desc = inner_product_forward::desc(prop_kind::forward_inference, data_md, weight_md,
                                                bias_md, dst_md);
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

extern "C" void dnnl_relu(float* data, float* out, std::vector<int64_t> shape) {
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  auto data_md = GenDNNLMemDescByShape(shape, dt::f32);

  auto data_memory = memory(data_md, eng, data);
  auto dst_memory = memory(data_md, eng);

  auto relu_desc =
      eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_relu, data_md, 0);
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
      normalization_flags::use_global_stats | normalization_flags::use_scale_shift);
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

// should comply with src/relay/backend/contrib/dnnl/codegen.cc
#define DNNL_BINARY_ADD 0
#define DNNL_BINARY_MUL 1

extern "C" void dnnl_binary_op(float* data, float* weight, float* out, int algo_type,
                               std::vector<int64_t> shape) {
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  auto data_md = GenDNNLMemDescByShape(shape, dt::f32);

  auto data_memory = memory(data_md, eng, data);
  auto weight_memory = memory(data_md, eng, weight);
  auto dst_memory = memory(data_md, eng);

  algorithm algo = algorithm::undef;
  switch (algo_type) {
    case DNNL_BINARY_ADD:
      algo = algorithm::binary_add;
      break;
    case DNNL_BINARY_MUL:
      algo = algorithm::binary_mul;
      break;
    default:
      LOG(FATAL) << "Unsupported dnnl algorithm: " << algo_type;
      break;
  }

  auto add_desc = binary::desc(algo, data_md, data_md, data_md);
  auto add_prim_desc = binary::primitive_desc(add_desc, eng);
  assert(data_md == add_prim_desc.dst_desc());

  auto add = binary(add_prim_desc);
  add.execute(
      s,
      {{DNNL_ARG_SRC_0, data_memory}, {DNNL_ARG_SRC_1, weight_memory}, {DNNL_ARG_DST, dst_memory}});
  s.wait();
  read_from_dnnl_memory(out, dst_memory);
}

// DNNL Conv2d single OP
TVM_REGISTER_GLOBAL("tvm.contrib.dnnl.conv2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* input = args[0];
  DLTensor* weights = args[1];
  DLTensor* output = args[2];
  int p_Ph0_ = args[3], p_Pw0_ = args[4], p_Ph1_ = args[5], p_Pw1_ = args[6], p_Sh_ = args[7],
      p_Sw_ = args[8], p_G_ = args[9];
  bool channel_last = args[10];
  bool pre_cast = args[11];
  bool post_cast = args[12];

  int p_N_ = input->shape[0], p_C_ = input->shape[1], p_H_ = input->shape[2],
      p_W_ = input->shape[3], p_O_ = output->shape[1], p_Kh_ = weights->shape[2],
      p_Kw_ = weights->shape[3];

  if (channel_last) {
    p_N_ = input->shape[0];
    p_H_ = input->shape[1];
    p_W_ = input->shape[2];
    p_C_ = input->shape[3];
    p_O_ = output->shape[3];
    p_Kh_ = weights->shape[0];
    p_Kw_ = weights->shape[1];
  }

  std::vector<float> bias(p_O_, 0);
  primitive_attr attr;
  return dnnl_conv2d_common(static_cast<float*>(input->data), static_cast<float*>(weights->data),
                            bias.data(), static_cast<float*>(output->data), p_N_, p_C_, p_H_, p_W_,
                            p_O_, p_G_, p_Ph0_, p_Pw0_, p_Ph1_, p_Pw1_, p_Kh_, p_Kw_, p_Sh_, p_Sw_,
                            attr, channel_last, pre_cast, post_cast);
});

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
