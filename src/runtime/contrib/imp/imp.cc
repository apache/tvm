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

#include "imp.h"

namespace tvm {
namespace runtime {
namespace contrib {

#define IMP_BINARY_ADD 0
#define IMP_BINARY_MUL 1

extern "C" void imp_binary_op(uint32_t* data, uint32_t* weight, uint32_t* out, int algo_type,
                               std::vector<int64_t> shape) {
      imp_ops(data, weight, out, algo_type);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm