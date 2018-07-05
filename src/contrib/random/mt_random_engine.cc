/*!
 *  Copyright (c) 2018 by Contributors
 * \file random/mt_random_engine.cc
 * \brief mt19937 random engine
 */
#include <dmlc/logging.h>
#include <algorithm>
#include <ctime>
#include <random>

namespace tvm {
namespace contrib {

/*!
 * \brief An interface for generating [tensors of] random numbers.
 */
class RandomEngine {
 public:
   /*!
    * \brief Creates a RandomEngine using a default seed.
    */
  RandomEngine() {
    this->Seed(time(0));
  }

   /*!
    * \brief Creates a RandomEngine, suggesting the use of a provided seed.
    */
  explicit RandomEngine(unsigned seed) {
    this->Seed(seed);
  }

   /*!
    * \brief Seeds the underlying RNG, if possible.
    */
  inline void Seed(unsigned seed) {
    rnd_engine_.seed(seed);
    this->rseed_ = static_cast<unsigned>(seed);
  }

   /*!
    * \return the seed associated with the underlying RNG.
    */
  inline unsigned GetSeed() const {
    return rseed_;
  }

   /*!
    * \return a random integer sampled from the RNG.
    */
  inline unsigned GetRandInt() {
    return rnd_engine_();
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

    if (data->ctx.device_type == kDLCPU) {
      std::uniform_real_distribution<float> uniform_dist(low, high);
      std::generate_n(static_cast<float*>(data->data), size, [&] () {
        return uniform_dist(rnd_engine_);
      });
    } else {
      LOG(FATAL) << "Do not support random.uniform on this device yet";
    }
  }

   /*!
    * \brief Fills a tensor with values drawn from Normal(loc, scale**2)
    */
  void SampleNormal(DLTensor* data, float loc, float scale) {
    CHECK_GT(scale, 0) << "standard deviation must be positive";
    CHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    if (data->ctx.device_type == kDLCPU) {
      std::normal_distribution<float> normal_dist(loc, scale);
      std::generate_n(static_cast<float*>(data->data), size, [&] () {
        return normal_dist(rnd_engine_);
      });
    } else {
      LOG(FATAL) << "Do not support random.normal on this device yet";
    }
  }

 private:
  std::mt19937 rnd_engine_;
  unsigned rseed_;
};

}  // namespace contrib
}  // namespace tvm
