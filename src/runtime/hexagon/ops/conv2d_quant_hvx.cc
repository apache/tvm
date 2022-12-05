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

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <inttypes.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include "conv2d.h"

extern "C" int conv2d_packed_quant(TVMValue* args, int* type_codes, int num_args, TVMValue* out_val,
                                   int out_code, void* res_handle);

namespace tvm {
namespace runtime {
namespace hexagon {
inline uint8_t* getElementPtr_int8(int block_out_y, int block_out_x, int block_out_c, int yi,
                                   int xi, int ci, const DLTensor& block) {
  auto block_ptr =
      tvm::runtime::hexagon::conv_utils::nhwc_at(block, 0, block_out_y, block_out_x, block_out_c);
  const int width_stride = 32;
  const int height_stride = width_stride * 8;
  auto block_offset = yi * height_stride + xi * width_stride + ci;
  auto first_element_ptr = reinterpret_cast<uint8_t*>(block_ptr);
  return first_element_ptr + block_offset;
}

inline int8_t* getWgtPtr_int8(int out_i, int out_o, int h, int w, int i, int o,
                              const DLTensor& wgt_vtcm, int width) {
  auto data = static_cast<intptr_t*>(wgt_vtcm.data);
  auto chunk = data[out_i * wgt_vtcm.shape[3] + out_o];
  auto base_chunk_ptr = reinterpret_cast<int8_t*>(chunk);
  auto wgt_chunk_offset = tvm::runtime::hexagon::conv_utils::hwio_to_sm_8b(width, h, w, i, o);
  return base_chunk_ptr + wgt_chunk_offset;
}

int32_t saturate_uint8(int32_t val) { return std::max(std::min(val, 255), 0); }

int32_t saturate_int8(int32_t val) { return std::max(std::min(val, 127), -128); }

/**
 * @brief Compute the quantized convolution along with requantize with output quantization params to
 * get uint8 outputs
 *
 * The quantized convolution is represented by the below equation
 * out_scale(out_q - out_zp) = Σr,s,c(act_scale(act_q[n,h+r,w+s,c] - act_zp) *
 *                             wgt_scale(wgt_q[r,s,c,o] - wgt_zp))
 * => out_q = Σr,s,c((act_q[n,h+r,w+s,c] - act_zp) * (wgt_q[r,s,c,o] - wgt_zp))
 *            * (act_scale*wgt_scale/out_scale) + out_zp
 * out_q = Σr,s,c((act_q[n,h+r,w+s,c] - act_zp) * (wgt_zp_q[r,s,c,o])) *
 * (act_scale*wgt_scale/out_scale) + out_zp, where wgt_zp_q = (wgt_q[r,s,c,o] - wgt_zp)
 *
 *  Assumptions/Limitations:
 *  - Strided convolution is not yet supported so the stride variables are unused
 *
 * @param cr_out blockized output tensor with zeros already filled in
 * @param cr_act blockized activations
 * @param cr_filt Chunkified weights as returned from output of prepare_hwio
 * @param out_shape Original output shape of the tensor before blockization
 * @param act_shape Original input shape
 * @param filt_shape Original filter shape
 * @param act_scale Quantization scale for activation
 * @param act_zp Activations zero point
 * @param wgt_scale Quantization scale for weights
 * @param wgt_zp Weights zero point
 * @param out_scale Quantization scale for output
 * @param out_zp Output zero point
 * @param fixed_final_scale Fixed point value of final_scale= (act_scale*wgt_scale/out_scale)
 * @param scale_factor Scale factor for the fixed_final_scale
 */
void conv_layer_int8_hvx_whole(DLTensor& cr_out, const DLTensor& cr_act,  // NOLINT(*)
                               const DLTensor& cr_filt, const DLTensor& out_shape,
                               const DLTensor& act_shape, const DLTensor& filt_shape,
                               float act_scale, int act_zp, float wgt_scale, int wgt_zp,
                               float out_scale, int out_zp, int fixed_final_scale,
                               int scale_factor) {
  namespace conv_utils = tvm::runtime::hexagon::conv_utils;
  int filt_height = filt_shape.shape[0];
  int filt_width = filt_shape.shape[1];
  int filt_idepth = filt_shape.shape[2];

  int a_depth = cr_act.shape[3];

  int o_height = cr_out.shape[1];
  int o_width = cr_out.shape[2];
  int o_depth = cr_out.shape[3];

  int out_height = out_shape.shape[1];
  int out_width = out_shape.shape[2];

  uint8_t act_zp_u8 = static_cast<uint8_t>(act_zp);
  int8_t wgt_zp_i8 = static_cast<int8_t>(wgt_zp);

  HVX_Vector act_zp_vec = Q6_Vb_vsplat_R(act_zp_u8);
  HVX_Vector wgt_zp_vec = Q6_Vb_vsplat_R(wgt_zp_i8);
  HVX_VectorPair wgt_zp_vec_pair = Q6_Wh_vsxt_Vb(wgt_zp_vec);

  ICHECK_EQ(a_depth, cr_filt.shape[2]) << "input depth should match weights input channels";
  ICHECK_EQ(o_depth, cr_filt.shape[3]) << "output depth should match the weights output channel";

  uint32_t scale_u = static_cast<uint32_t>(fixed_final_scale);
  HVX_Vector scale_vec = Q6_V_vsplat_R(scale_u);
  uint32_t new_scale_factor = static_cast<uint32_t>(scale_factor - 16);
  HVX_Vector out_zp_vec = Q6_V_vsplat_R(out_zp);

  auto computeOutVec = [&cr_act, &cr_filt, &act_zp_vec, &wgt_zp_vec_pair, &out_zp_vec, &scale_vec,
                        new_scale_factor, filt_height, filt_width,
                        filt_idepth](int out_h, int out_w, int out_c, int h, int w) -> HVX_Vector {
    HVX_Vector out_vec = Q6_V_vzero();
    for (int fh = 0; fh < filt_height; ++fh) {
      for (int fw = 0; fw < filt_width; ++fw) {
        for (int c = 0; c < conv_utils::round_up(filt_idepth, 4); c += 4) {
          int act_h = out_h * 8 + h + fh;
          int act_ho = act_h / 8;
          int act_hi = act_h % 8;

          int act_w = out_w * 8 + w + fw;
          int act_wo = act_w / 8;
          int act_wi = act_w % 8;

          int act_co = c / 32;
          int act_ci = c % 32;

          uint8_t* act_ptr =
              getElementPtr_int8(act_ho, act_wo, act_co, act_hi, act_wi, act_ci, cr_act);

          uint32_t four_act_elems = *reinterpret_cast<uint32_t*>(act_ptr);
          HVX_Vector act_vec = Q6_V_vsplat_R(four_act_elems);
          int8_t* wgt_ptr = getWgtPtr_int8(act_co, out_c, fh, fw, act_ci, 0, cr_filt, filt_width);

          HVX_Vector* wgt_vec_ptr = reinterpret_cast<HVX_Vector*>(wgt_ptr);
          HVX_Vector wgt_vec = *wgt_vec_ptr;

          HVX_VectorPair act_vec_zp_diff = Q6_Wh_vsub_VubVub(act_vec, act_zp_vec);
          HVX_VectorPair wgt_i16_vec_nodiff = Q6_Wh_vsxt_Vb(wgt_vec);
          HVX_VectorPair wgt_i16_vec = Q6_Wh_vsub_WhWh_sat(wgt_i16_vec_nodiff, wgt_zp_vec_pair);

          out_vec = Q6_Vw_vdmpyacc_VwVhVh_sat(out_vec, Q6_V_lo_W(act_vec_zp_diff),
                                              Q6_V_lo_W(wgt_i16_vec));
          out_vec = Q6_Vw_vdmpyacc_VwVhVh_sat(out_vec, Q6_V_hi_W(act_vec_zp_diff),
                                              Q6_V_hi_W(wgt_i16_vec));
        }
      }
    }
    HVX_Vector mul_vec = Q6_Vw_vmpye_VwVuh(out_vec, scale_vec);
    HVX_Vector scaled_vec = Q6_Vw_vasr_VwR(mul_vec, new_scale_factor);
    HVX_Vector sum_vec = Q6_Vw_vadd_VwVw(scaled_vec, out_zp_vec);
    return sum_vec;
  };

  auto saturateAndStore = [&cr_out, &computeOutVec](int out_h, int out_w, int out_c, int h, int w) {
    uint8_t* out_ptr = getElementPtr_int8(out_h, out_w, out_c, h, w, 0, cr_out);
    HVX_Vector* out_vec_ptr = reinterpret_cast<HVX_Vector*>(out_ptr);
    HVX_Vector out_vec1, out_vec2, out_vec3, out_vec4, out_vec;
    out_vec1 = computeOutVec(out_h, out_w, out_c, h, w);
    out_vec2 = computeOutVec(out_h, out_w, out_c, h, w + 1);
    out_vec3 = computeOutVec(out_h, out_w, out_c, h, w + 2);
    out_vec4 = computeOutVec(out_h, out_w, out_c, h, w + 3);

    HVX_Vector half_vec1 = Q6_Vh_vpack_VwVw_sat(out_vec2, out_vec1);
    HVX_Vector half_vec2 = Q6_Vh_vpack_VwVw_sat(out_vec4, out_vec3);
    out_vec = Q6_Vub_vpack_VhVh_sat(half_vec2, half_vec1);
    *out_vec_ptr = out_vec;
  };

  for (int out_c = 0; out_c < o_depth; ++out_c) {
    for (int out_h = 0; out_h < o_height; ++out_h) {
      int max_y = std::min(8, out_height - out_h * 8);
      for (int out_w = 0; out_w < o_width; ++out_w) {
        int max_x = std::min(8, out_width - out_w * 8);
        for (int h = 0; h < max_y; ++h) {
          if (max_x == 8) {
            for (int w = 0; w < max_x; w += 4) {
              saturateAndStore(out_h, out_w, out_c, h, w);
            }
          } else {
            int w = 0;
            if (max_x >= 4) {
              saturateAndStore(out_h, out_w, out_c, h, w);
              w = 4;
            }
            uint8_t* out_ptr = getElementPtr_int8(out_h, out_w, out_c, h, w, 0, cr_out);
            HVX_Vector* out_vec_ptr = reinterpret_cast<HVX_Vector*>(out_ptr);
            HVX_Vector out_vec1, out_vec2, out_vec3, out_vec;
            if (max_x % 4 == 1) {
              out_vec1 = computeOutVec(out_h, out_w, out_c, h, w);
              HVX_Vector half_vec = Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(), out_vec1);
              out_vec = Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(), half_vec);
              *out_vec_ptr = out_vec;
            } else if (max_x % 4 == 2) {
              out_vec1 = computeOutVec(out_h, out_w, out_c, h, w);
              out_vec2 = computeOutVec(out_h, out_w, out_c, h, w + 1);
              HVX_Vector half_vec = Q6_Vh_vpack_VwVw_sat(out_vec2, out_vec1);
              out_vec = Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(), half_vec);
              *out_vec_ptr = out_vec;
            } else if (max_x % 4 == 3) {
              out_vec1 = computeOutVec(out_h, out_w, out_c, h, w);
              out_vec2 = computeOutVec(out_h, out_w, out_c, h, w + 1);
              out_vec3 = computeOutVec(out_h, out_w, out_c, h, w + 2);
              HVX_Vector half_vec1 = Q6_Vh_vpack_VwVw_sat(out_vec2, out_vec1);
              HVX_Vector half_vec2 = Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(), out_vec3);
              out_vec = Q6_Vub_vpack_VhVh_sat(half_vec2, half_vec1);
              *out_vec_ptr = out_vec;
            }
          }
        }
      }
    }
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

int conv2d_packed_quant(TVMValue* args, int* type_codes, int num_args, TVMValue* out_val,
                        int out_code, void* res_handle) {
  namespace conv_utils = tvm::runtime::hexagon::conv_utils;
  ICHECK_EQ(num_args, 13) << "Unexpected number of arguments";
  ICHECK_EQ(type_codes[0], kTVMDLTensorHandle)
      << "First argument is expected to be the input tensor";  // Input activations
  ICHECK_EQ(type_codes[1], kTVMDLTensorHandle)
      << "Second argument is expected to be the weights tensor";  // Weights
  ICHECK_EQ(type_codes[2], kDLFloat) << "Third argument is expected to be the activation scale";
  ICHECK_EQ(type_codes[3], kDLInt) << "Fourth argument is expected to be the activation zero point";
  ICHECK_EQ(type_codes[4], kDLFloat) << "Fifth argument is expected to be the weight scale";
  ICHECK_EQ(type_codes[5], kDLInt) << "Sixth argument is expected to be the weight zero point";
  ICHECK_EQ(type_codes[6], kDLFloat) << "Seventh argument is expected to be the output scale";
  ICHECK_EQ(type_codes[7], kDLInt) << "Eigth argument is expected to be the output zero point";
  ICHECK_EQ(type_codes[8], kDLInt) << "Nineth argument is expected to be the stride_h";  // stride_h
  ICHECK_EQ(type_codes[9], kDLInt) << "Tenth argument is expected to be the stride_w";   // stride_w
  ICHECK_EQ(type_codes[10], kDLInt) << "Eleventh argument is expected to be fixed final scale";
  ICHECK_EQ(type_codes[11], kDLInt) << "Twelfth argument is expected to be scale factor";
  ICHECK_EQ(type_codes[12], kTVMDLTensorHandle)
      << "Thirteenth argument is expected to be the output tensor";  // output

  auto* act_flat = static_cast<DLTensor*>(args[0].v_handle);
  auto* wgt_flat = static_cast<DLTensor*>(args[1].v_handle);
  auto* out_flat = static_cast<DLTensor*>(args[12].v_handle);

  // Temporary assertion until multiple batches are supported
  ICHECK_EQ(act_flat->shape[0], 1) << "Input batch size more than 1 is not supported yet";

  // Temporary assertion until multiple batches are supported
  ICHECK_EQ(out_flat->shape[0], 1) << "Output batch size more than 1 is not supported yet";

  float act_scale = args[2].v_float64;
  int act_zp = args[3].v_int64;
  LOG_INFO << "act_scale: " << act_scale << ", act_zp: " << act_zp;

  float wgt_scale = args[4].v_float64;
  int wgt_zp = args[5].v_int64;
  LOG_INFO << "wgt_scale: " << wgt_scale << ", wgt_zp: " << wgt_zp;

  float out_scale = args[6].v_float64;
  int out_zp = args[7].v_int64;
  LOG_INFO << "out_scale: " << out_scale << ", out_zp: " << out_zp;

  int stride_h = args[8].v_int64;
  int stride_w = args[9].v_int64;
  LOG_INFO << "stride_h: " << stride_h << ", stride_w: " << stride_w;

  int fixed_final_scale = args[10].v_int64;
  int scale_factor = args[11].v_int64;
  LOG_INFO << "fixed_final_scale: " << fixed_final_scale << ", scale_factor: " << scale_factor;

  auto* device_api = tvm::runtime::DeviceAPI::Get(conv_utils::hexagon_device, false);
  ICHECK(device_api != nullptr);
  tvm::runtime::String vtcm_scope = "global.vtcm";

  auto act_vtcm =
      conv_utils::prepare_nhwc<uint8_t, 8, 8, 32>(device_api, act_flat, /*copy_data=*/true);

  int num_wgt_chunks = conv_utils::calculate_num_weight_chunks(
      wgt_flat->shape, /* chunk_height */ wgt_flat->shape[0],
      /* chunk_width */ wgt_flat->shape[1], /* chunk_in_channel */ 32, /* chunk_out_channel */ 32);
  auto wgt_ptr_table =
      reinterpret_cast<void**>(__builtin_alloca(num_wgt_chunks * sizeof(uintptr_t)));

  auto wgt_vtcm =
      conv_utils::prepare_hwio_8b(device_api, wgt_flat, num_wgt_chunks, wgt_ptr_table, wgt_zp);

  auto out_vtcm =
      conv_utils::prepare_nhwc<uint8_t, 8, 8, 32>(device_api, out_flat, /*copy_data=*/false);

  auto act_shape = conv_utils::SDLTensor<4>(nullptr, act_flat->dtype, nullptr, act_flat->shape);
  auto filt_shape = conv_utils::SDLTensor<4>(nullptr, wgt_flat->dtype, nullptr, wgt_flat->shape);
  auto out_shape = conv_utils::SDLTensor<4>(nullptr, out_flat->dtype, nullptr, out_flat->shape);

  tvm::runtime::hexagon::conv_layer_int8_hvx_whole(
      out_vtcm, act_vtcm, wgt_vtcm, out_shape, act_shape, filt_shape, act_scale, act_zp, wgt_scale,
      wgt_zp, out_scale, out_zp, fixed_final_scale, scale_factor);

  conv_utils::deblockize_hwc<uint8_t, 8, 8, 32>(out_flat->data, out_vtcm.data, out_flat->shape[1],
                                                out_flat->shape[2], out_flat->shape[3]);

  conv_utils::release(device_api, out_vtcm);
  conv_utils::release(device_api, wgt_vtcm);
  conv_utils::release(device_api, act_vtcm);

  return 0;
}
