/*!
 *  Copyright (c) 2018 by Contributors
 * \file random/sgx_random_engine.h
 * \brief SGX trusted random engine
 */
#include <dmlc/logging.h>
#include <sgx_trts.h>
#include <algorithm>
#include "../../runtime/sgx/common.h"

namespace tvm {
namespace contrib {

/*!
 * \brief An interface for generating [tensors of] random numbers.
 */
class RandomEngine {
 public:
   /*!
    * \brief Creates a RandomEngine, suggesting the use of a provided seed.
    */
  explicit RandomEngine(unsigned seed) {
    LOG(WARNING) << "SGX RandomEngine does not support seeding.";
  }

   /*!
    * \brief Seeds the underlying RNG, if possible.
    */
  inline void Seed(unsigned seed) {
    LOG(WARNING) << "SGX RandomEngine does not support seeding.";
  }

   /*!
    * \return the seed associated with the underlying RNG.
    */
  inline unsigned GetSeed() const {
    LOG(WARNING) << "SGX RandomEngine does not support seeding.";
    return 0;
  }

   /*!
    * \return a random integer sampled from the RNG.
    */
  inline unsigned GetRandInt() {
    int rand_int;
    TVM_SGX_CHECKED_CALL(
        sgx_read_rand(reinterpret_cast<unsigned char*>(&rand_int), sizeof(int)));
    return rand_int;
  }

   /*!
    * \brief Fills a tensor with values drawn from Unif(low, high)
    */
  void SampleUniform(DLTensor* data, float low, float high) {
    CHECK_GT(high, low) << "high must be bigger than low";
    CHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    std::generate_n(static_cast<float*>(data->data), size, [&] () {
      float max_int = static_cast<float>(std::numeric_limits<unsigned>::max());
      float unif01 = GetRandInt() / max_int;
      return low + unif01 * (high - low);
    });
  }
};

}  // namespace contrib
}  // namespace tvm
