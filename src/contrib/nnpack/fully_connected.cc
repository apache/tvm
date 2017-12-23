/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <nnpack.h>
#include "./nnpack_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.nnpack.fully_connected_inference")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    NNPackThreadLocalEntry *entry = NNPackThreadLocalEntry::ThreadLocal();
    nnp_initialize();
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    NNPackConfig(args[3]);

    CHECK_EQ(A->ndim, 1);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(C->ndim, 1);
    CHECK_EQ(B->shape[0], C->shape[0]);
    CHECK_EQ(B->shape[1], A->shape[0]);
    CHECK(C->strides == nullptr);
    CHECK(B->strides == nullptr);
    CHECK(A->strides == nullptr);
    CHECK(TypeMatch(A->dtype, kDLFloat, 32));
    CHECK(TypeMatch(B->dtype, kDLFloat, 32));
    CHECK(TypeMatch(C->dtype, kDLFloat, 32));

    nnp_fully_connected_inference(B->shape[1],
                                  B->shape[0],
                                  static_cast<float*>(A->data),
                                  static_cast<float*>(B->data),
                                  static_cast<float*>(C->data),
                                  entry->threadpool);
  });


TVM_REGISTER_GLOBAL("tvm.contrib.nnpack.fully_connected_output")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    NNPackThreadLocalEntry *entry = NNPackThreadLocalEntry::ThreadLocal();
    nnp_initialize();
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    NNPackConfig(args[3]);

    CHECK_EQ(A->ndim, 2);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(C->ndim, 2);
    CHECK_EQ(B->shape[0], C->shape[1]);
    CHECK_EQ(B->shape[1], A->shape[1]);
    CHECK_EQ(A->shape[0], C->shape[0]);
    CHECK(C->strides == nullptr);
    CHECK(B->strides == nullptr);
    CHECK(A->strides == nullptr);
    CHECK(TypeMatch(A->dtype, kDLFloat, 32));
    CHECK(TypeMatch(B->dtype, kDLFloat, 32));
    CHECK(TypeMatch(C->dtype, kDLFloat, 32));

    nnp_fully_connected_output(A->shape[0],
                               B->shape[1],
                               B->shape[0],
                               static_cast<float*>(A->data),
                               static_cast<float*>(B->data),
                               static_cast<float*>(C->data),
                               entry->threadpool,
                               NULL);
  });

}  // namespace contrib
}  // namespace tvm
