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
 * \file src/relay/backend/contrib/dnnl/query_layout.cc
 * \brief layout auto-query func.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../../../runtime/contrib/dnnl/dnnl_utils.h"
#include "../../../../runtime/regex.h"
#include "../../utils.h"
#include "dnnl.hpp"
namespace tvm {
namespace relay {
namespace contrib {

using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using tvm::runtime::contrib::dtype_dl2dnnl;

template <typename T, typename U>
inline void array_set(T* arr, const U& val, size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = static_cast<T>(val);
}

template <typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
  for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}

template <typename T>
inline void swap(T& t1, T& t2) {
  T tmp(t1);
  t1 = t2;
  t2 = tmp;
}

template <typename T, typename U, typename F>
inline void simultaneous_sort(T* vals, T* vals_2nd_level, U* keys, size_t size, F comparator) {
  if (size == 0) return;

  for (size_t i = 0; i < size - 1; ++i) {
    bool swapped = false;

    for (size_t j = 0; j < size - i - 1; j++) {
      auto res = comparator(vals[j], vals[j + 1]);
      if (res == 0) res = comparator(vals_2nd_level[j], vals_2nd_level[j + 1]);

      if (res > 0) {
        swap(vals[j], vals[j + 1]);
        swap(vals_2nd_level[j], vals_2nd_level[j + 1]);
        swap(keys[j], keys[j + 1]);
        swapped = true;
      }
    }

    if (swapped == false) break;
  }
}

void compute_blocks(dims_t blocks, const dnnl::memory::desc* md) {
  using format_kind_t = dnnl_format_kind_t;
  const format_kind_t blocked = dnnl_blocked;
  if (!(md->data.format_kind == blocked)) {
    array_set(blocks, 0, md->data.ndims);
    return;
  }
  array_set(blocks, 1, md->data.ndims);
  const auto& bd = md->data.format_desc.blocking;
  for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
    blocks[bd.inner_idxs[iblk]] *= bd.inner_blks[iblk];
}

inline bool has_runtime_strides(const dnnl::memory::desc* md) {
  using format_kind_t = dnnl_format_kind_t;
  const format_kind_t blocked = dnnl_blocked;
  if (!(md->data.format_kind == blocked)) return false;
  for (int d = 0; d < md->data.ndims; ++d)
    if (md->data.format_desc.blocking.strides[d] == DNNL_RUNTIME_DIM_VAL) return true;
  return false;
}

std::string md2fmt_tag_str(const dnnl::memory::desc* md) {
  const auto& blk = md->data.format_desc.blocking;

  dims_t blocks = {0};
  compute_blocks(blocks, md);

  char dim_chars[DNNL_MAX_NDIMS + 1];

  dims_t ou_blocks = {0};
  array_copy(ou_blocks, md->data.padded_dims, md->data.ndims);

  bool plain = true;
  for (int d = 0; d < md->data.ndims; ++d) {
    dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + static_cast<char>(d);
    if (blocks[d] != 1) plain = false;
    ou_blocks[d] /= blocks[d];
  }

  // Can't report meaningful tag for runtime dimensions.
  if (has_runtime_strides(md)) return "*";

  dims_t strides;
  array_copy(strides, blk.strides, md->data.ndims);

  simultaneous_sort(strides, ou_blocks, dim_chars, md->data.ndims,
                    [](dim_t a, dim_t b) { return b - a; });

  dim_chars[md->data.ndims] = '\0';

  std::string s(dim_chars);

  if (!plain) {
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
      char c = ('a' + static_cast<char>(blk.inner_idxs[iblk]));
      s += (std::to_string(blk.inner_blks[iblk]) + c);
    }
  }
  return s;
}

dnnl::memory::dims str2dims(const std::string& str_shape, bool dilates = false,
                            std::string interval = ",") {
  // Split strings
  std::vector<std::string> str_dims;
  size_t pos = 0, start = 0;
  while ((pos = str_shape.find(interval, start)) != std::string::npos) {
    std::string str_dim = str_shape.substr(start, pos - start);
    if (pos > start) str_dims.push_back(str_dim);
    start = pos + interval.size();
  }
  if (str_shape.size() > start) {
    str_dims.push_back(str_shape.substr(start));
  }
  // transfer string to dims
  dnnl::memory::dims out_dims;
  if (dilates) {
    std::transform(str_dims.begin(), str_dims.end(), std::back_inserter(out_dims),
                   [](const std::string& str) { return std::stoi(str) - 1; });
  } else {
    std::transform(str_dims.begin(), str_dims.end(), std::back_inserter(out_dims),
                   [](const std::string& str) { return std::stoi(str); });
  }
  return out_dims;
}

void check_shapes(const std::vector<std::string> shapes) {
  std::string valid_pat("(\\d*)(,(\\d*))*");
  bool checked = tvm::runtime::regex_match(shapes[0], valid_pat);
  for (size_t i = 1; i < shapes.size() - 1; i++) {
    checked &= tvm::runtime::regex_match(shapes[i], valid_pat);
  }
  checked &= tvm::runtime::regex_match(shapes[shapes.size() - 1], "\\d*");
  if (!checked) {
    LOG(FATAL) << "Invalid input args for query dnnl optimal layout.";
  }
}

void check_layout(bool var, bool ref) {
  if (var != ref) {
    LOG(FATAL) << "Invalid input layout for query dnnl optimal layout.";
  }
}

std::string get_optimal_layout_for_conv(std::string data_layout, std::string kernel_layout,
                                        std::string weight_shape, std::string out_shape,
                                        std::string paddings, std::string strides,
                                        std::string dilates, std::string G, std::string dtype) {
  check_layout(tvm::runtime::regex_match(data_layout, "NC(D?)(H?)W"), true);
  check_layout(tvm::runtime::regex_match(kernel_layout, "(G?)OI(D?)(H?)W"), true);
  check_shapes({weight_shape, out_shape, paddings, strides, dilates, G});

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);
  using tag = dnnl::memory::format_tag;

  dnnl::memory::dim groups = std::stoi(G);
  dnnl::memory::dims weight_dims_ = str2dims(weight_shape);
  dnnl::memory::dims weight_dims = weight_dims_;

  if (groups > 1) {
    if (weight_dims_.size() == 5) {
      weight_dims = {groups * weight_dims_[1], groups * weight_dims_[2], weight_dims_[3],
                     weight_dims_[4]};
    } else {
      weight_dims[1] = weight_dims[1] * groups;
    }
  }

  dnnl::memory::dims out_dims = str2dims(out_shape);
  dnnl::memory::dims padding_dims = str2dims(paddings);
  dnnl::memory::dims padding_dims_l(padding_dims.begin(),
                                    padding_dims.begin() + padding_dims.size() / 2);
  dnnl::memory::dims padding_dims_r(padding_dims.end() - padding_dims.size() / 2,
                                    padding_dims.end());
  dnnl::memory::dims strides_dims = str2dims(strides);
  dnnl::memory::dims dilates_dims = str2dims(dilates, true);

  dnnl::memory::dims input_dims = out_dims;
  input_dims[1] = weight_dims[1];
  for (size_t i = 2; i < out_dims.size(); i++) {
    dnnl::memory::dim K = weight_dims[i];
    dnnl::memory::dim S = strides_dims[i - 2];
    dnnl::memory::dim D = dilates_dims[i - 2];
    dnnl::memory::dim PL = padding_dims_l[i - 2];
    dnnl::memory::dim PR = padding_dims_r[i - 2];
    dnnl::memory::dim DK = 1 + (K - 1) * (D + 1);
    input_dims[i] = out_dims[i] * S - PL - PR + DK - 1;
  }

  dnnl::memory::dims conv_src_dims = input_dims;
  dnnl::memory::dims conv_weights_dims = weight_dims;
  if (groups > 1) {
    conv_weights_dims = {groups, out_dims[1] / groups, input_dims[1] / groups};
    conv_weights_dims.insert(conv_weights_dims.end(), weight_dims.begin() + 2, weight_dims.end());
  }

  dnnl::memory::dims conv_dst_dims = out_dims;
  dnnl::memory::dims conv_strides = strides_dims;
  dnnl::memory::dims conv_dilates = dilates_dims;
  dnnl::memory::dims conv_padding_l = padding_dims_l;
  dnnl::memory::dims conv_padding_r = padding_dims_r;

  auto dnnl_dtype = dtype_dl2dnnl(tvm::runtime::String2DLDataType(dtype));
  auto conv_src_md = dnnl::memory::desc({conv_src_dims}, dnnl_dtype, tag::any);
  auto conv_weights_md = dnnl::memory::desc({conv_weights_dims}, dnnl_dtype, tag::any);
  auto conv_dst_md = dnnl::memory::desc({conv_dst_dims}, dnnl_dtype, tag::any);

  auto conv_desc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
      conv_weights_md, conv_dst_md, conv_strides, conv_dilates, conv_padding_l, conv_padding_r);

  auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, eng);

  auto src_format = conv_prim_desc.src_desc();
  auto weights_format = conv_prim_desc.weights_desc();
  auto dst_format = conv_prim_desc.dst_desc();
  std::string src_df, weight_df, dst_df;

  src_df = md2fmt_tag_str(&src_format);
  weight_df = md2fmt_tag_str(&weights_format);
  dst_df = md2fmt_tag_str(&dst_format);
  std::string res = src_df + "," + weight_df + "," + dst_df;
  return res;
}

std::string get_optimal_layout_for_conv_transpose(std::string data_layout,
                                                  std::string kernel_layout,
                                                  std::string weight_shape, std::string out_shape,
                                                  std::string paddings, std::string output_paddings,
                                                  std::string strides, std::string dilates,
                                                  std::string G, std::string dtype) {
  check_layout(tvm::runtime::regex_match(data_layout, "NC(D?)(H?)W"), true);
  check_layout(tvm::runtime::regex_match(kernel_layout, "(G?)((IO)|(OI))(D?)(H?)W"), true);
  check_shapes({weight_shape, out_shape, paddings, output_paddings, strides, dilates, G});

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);
  using tag = dnnl::memory::format_tag;

  dnnl::memory::dim groups = std::stoi(G);
  dnnl::memory::dims weight_dims_ = str2dims(weight_shape);
  dnnl::memory::dims weight_dims = weight_dims_;
  if (groups > 1) {
    if (weight_dims_.size() == 5) {
      weight_dims = {groups * weight_dims_[1], groups * weight_dims_[2], weight_dims_[3],
                     weight_dims_[4]};
    } else {
      weight_dims[1] = weight_dims[1] * groups;
    }
  }
  dnnl::memory::dims out_dims = str2dims(out_shape);
  dnnl::memory::dims padding_dims = str2dims(paddings);
  dnnl::memory::dims padding_dims_l(padding_dims.begin(),
                                    padding_dims.begin() + padding_dims.size() / 2);
  dnnl::memory::dims padding_dims_r(padding_dims.end() - padding_dims.size() / 2,
                                    padding_dims.end());
  dnnl::memory::dims output_padding_dims = str2dims(output_paddings);
  dnnl::memory::dims strides_dims = str2dims(strides);
  dnnl::memory::dims dilates_dims = str2dims(dilates, true);

  dnnl::memory::dims input_dims = out_dims;
  if (out_dims[1] == weight_dims[0]) {
    input_dims[1] = weight_dims[1];
  } else {
    input_dims[1] = weight_dims[0];
    std::swap(weight_dims[0], weight_dims[1]);
  }
  for (size_t i = 2; i < out_dims.size(); i++) {
    dnnl::memory::dim K = weight_dims[i];
    dnnl::memory::dim S = strides_dims[i - 2];
    dnnl::memory::dim D = dilates_dims[i - 2];
    dnnl::memory::dim PL = padding_dims_l[i - 2];
    dnnl::memory::dim PR = padding_dims_r[i - 2];
    dnnl::memory::dim OP = output_padding_dims[i - 2];
    dnnl::memory::dim DK = 1 + (K - 1) * (D + 1);
    input_dims[i] = (out_dims[i] - DK + PL + PR - OP) / S + 1;
  }

  dnnl::memory::dims deconv_src_dims = input_dims;
  dnnl::memory::dims deconv_weights_dims = weight_dims;
  if (groups > 1) {
    deconv_weights_dims = {groups, out_dims[1] / groups, input_dims[1] / groups};
    deconv_weights_dims.insert(deconv_weights_dims.end(), weight_dims.begin() + 2,
                               weight_dims.end());
  }
  dnnl::memory::dims deconv_dst_dims = out_dims;
  dnnl::memory::dims deconv_strides = strides_dims;
  dnnl::memory::dims deconv_dilates = dilates_dims;
  dnnl::memory::dims deconv_padding_l = padding_dims_l;
  dnnl::memory::dims deconv_padding_r = padding_dims_r;

  auto dnnl_dtype = dtype_dl2dnnl(tvm::runtime::String2DLDataType(dtype));
  auto deconv_src_md = dnnl::memory::desc({deconv_src_dims}, dnnl_dtype, tag::any);
  auto deconv_weights_md = dnnl::memory::desc({deconv_weights_dims}, dnnl_dtype, tag::any);
  auto deconv_dst_md = dnnl::memory::desc({deconv_dst_dims}, dnnl_dtype, tag::any);

  auto deconv_desc = dnnl::deconvolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct, deconv_src_md,
      deconv_weights_md, deconv_dst_md, deconv_strides, deconv_dilates, deconv_padding_l,
      deconv_padding_r);

  auto deconv_prim_desc = dnnl::deconvolution_forward::primitive_desc(deconv_desc, eng);

  auto src_format = deconv_prim_desc.src_desc();
  auto weights_format = deconv_prim_desc.weights_desc();
  auto dst_format = deconv_prim_desc.dst_desc();
  std::string src_df, weight_df, dst_df;

  src_df = md2fmt_tag_str(&src_format);
  weight_df = md2fmt_tag_str(&weights_format);
  dst_df = md2fmt_tag_str(&dst_format);
  std::string res = src_df + "," + weight_df + "," + dst_df;
  return res;
}

TVM_REGISTER_GLOBAL("relay.ir.get_optimal_layout_for_conv")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = get_optimal_layout_for_conv(args[0], args[1], args[2], args[3], args[4], args[5],
                                        args[6], args[7], args[8]);
    });

TVM_REGISTER_GLOBAL("relay.ir.get_optimal_layout_for_conv_transpose")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = get_optimal_layout_for_conv_transpose(args[0], args[1], args[2], args[3], args[4],
                                                  args[5], args[6], args[7], args[8], args[9]);
    });

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
