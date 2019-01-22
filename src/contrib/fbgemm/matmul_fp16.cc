/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/AlignedVec.h>
#include <random>


namespace tvm {
namespace contrib {

using namespace runtime;
using namespace fbgemm;
using namespace std;


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.matmul_fp16")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];

    CHECK_EQ(A->ndim, 2);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(A->shape[1], B->shape[0]);

    int m = A->shape[0];
    int n = B->shape[1];
    int k = A->shape[1];
    float alpha = 1.f, beta = 0.f;

    // execuse me for the copy_in and copy_out, not familiar with fbgemm api interface
    aligned_vector<int> Aint(m * k);
    aligned_vector<int> Bint(k * n);
    Aint.assign(static_cast<int*>(A->data), static_cast<int*>(A->data) + m*k);
    Bint.assign(static_cast<int*>(B->data), static_cast<int*>(B->data) + k*n);
    aligned_vector<float> A_in(Aint.begin(), Aint.end());
    aligned_vector<float> B_in(Bint.begin(), Bint.end());
		
    aligned_vector<float> C_in(m * n, NAN);
    // fbgemm fp16
    PackedGemmMatrixFP16 Bp(matrix_op_t::NoTranspose, k, n, alpha, B_in.data());
    cblas_gemm_compute(matrix_op_t::NoTranspose, m, A_in.data(), Bp, beta, C_in.data());
    for(int i = 0; i<m*n; i++)
        *(static_cast<int*>(C->data) + i) = C_in[i];

});


}  // namespace contrib
}  // namespace tvm
