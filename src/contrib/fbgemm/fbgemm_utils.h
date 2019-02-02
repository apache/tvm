/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#ifndef TVM_CONTRIB_FBGEMM_FBGEMM_UTILS_H_
#define TVM_CONTRIB_FBGEMM_FBGEMM_UTILS_H_
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#include <cmath>
#include <vector>
#include <random>
#include <fbgemm/QuantUtilsAvx2.h>
#include <fbgemm/Fbgemm.h>


namespace tvm {
namespace contrib {
using namespace fbgemm;


// Helper functions for precompute the column offset
template <typename T>
void ComputeColumnOffsets(
    int num_rows,
    int num_cols,
    const T* W,
    const std::vector<TensorQuantizationParams>& qparams,
    std::vector<std::int32_t>& col_offsets);

}
}
#endif
