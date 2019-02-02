/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtilsAvx2.h>
#include <random>
#include "fbgemm_utils.h"
#include <cpuinfo.h>

namespace tvm {
namespace contrib {

using namespace runtime;
using namespace fbgemm;


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.fully_connected_int8")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* X = args[0]; //M*K quantized int8 input
    DLTensor* W = args[1]; //N*K quantized int8 weight
    DLTensor* B = args[2]; //N quantized int8 bias
    // ignore the axis and axis_w now for testing purpose
    DLTensor* Y = args[3];
    int threads = args[8];

    CHECK_EQ(X->ndim, 2);
    CHECK_EQ(W->ndim, 2);
    CHECK_EQ(B->ndim, 1);
    CHECK_EQ(X->shape[1], W->shape[1]);
    CHECK_EQ(W->shape[0], B->shape[0]);

    float  ReQuant_multiplier = (double)args[7];
    std::int32_t x_zero_point = args[4]; 
    std::int32_t w_zero_point = args[5]; 
    std::int32_t y_zero_point = args[6]; 

    int m = X->shape[0];
    int n = W->shape[0];
    int k = X->shape[1];

    std::vector<std::int32_t> row_offsets_(PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
    std::vector<std::int32_t> Y_int32_(n*m);
    std::vector<std::int32_t> column_offsets_;

    std::vector<TensorQuantizationParams> temp_qparams;
    temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

    PackBMatrix<std::int8_t, std::int32_t> packB(	
      matrix_op_t::Transpose,	
      k,	
      n,
      reinterpret_cast<const std::int8_t*>(W->data),
      k,
      nullptr, 
      1);

    PackAWithRowOffset<std::uint8_t> packA(
      matrix_op_t::NoTranspose,
      m,
      k,
      reinterpret_cast<const std::uint8_t*>(X->data),
      k,
      nullptr,
      1,
      row_offsets_.data());    

    ComputeColumnOffsets<std::int8_t>(
      k,
      n,
      reinterpret_cast<const std::int8_t*>(W->data),
      temp_qparams, 
      column_offsets_);

    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false> outputProcObj(
      doNothingObj,
      &ReQuant_multiplier,
      y_zero_point,
      x_zero_point,
      &w_zero_point,
      packA.getRowOffsetBuffer(),
      column_offsets_.data(),
      reinterpret_cast<const std::int32_t*>(B->data),
      n);

    fbgemmPacked(
      packA,
      packB,
      reinterpret_cast<std::uint8_t*>(Y->data),
      Y_int32_.data(),
      n,
      outputProcObj,
      0,
      threads); // num_threads

});


}  // namespace contrib
}  // namespace tvm
