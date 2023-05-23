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
 * \file random_engine.h
 * \brief Random number generator. It provides a generic interface consistent with
 * `std::uniform_random_bit_generator`
 */
#ifndef TVM_SUPPORT_RANDOM_ENGINE_H_
#define TVM_SUPPORT_RANDOM_ENGINE_H_
#include <tvm/runtime/logging.h>

#include <cstdint>
#include <random>

namespace tvm {
namespace support {

/*!
 * \brief This linear congruential engine is a drop-in replacement for std::minstd_rand. It strictly
 *  corresponds to std::minstd_rand and is designed to be platform-independent.
 * \note Our linear congruential engine is a complete implementation of
 *  std::uniform_random_bit_generator so it can be used as generator for any STL random number
 *  distribution. However, parts of std::linear_congruential_engine's member functions are not
 *  included for simplification. For full member functions of std::minstd_rand, please check out the
 *  following link: https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine
 */

class LinearCongruentialEngine {
 public:
  using TRandState = int64_t;
  /*! \brief The result type. */
  using result_type = uint64_t;
  /*! \brief The multiplier */
  static constexpr TRandState multiplier = 48271;
  /*! \brief The increment */
  static constexpr TRandState increment = 0;
  /*! \brief The modulus */
  static constexpr TRandState modulus = 2147483647;
  /*! \brief The minimum possible value of random state here. */
  static constexpr result_type min() { return 0; }
  /*! \brief The maximum possible value of random state here. */
  static constexpr result_type max() { return modulus - 1; }

  /*!
   * \brief Get a device random state
   * \return The random state
   */
  static TRandState DeviceRandom() { return (std::random_device()()) % modulus; }

  /*!
   * \brief Operator to move the random state to the next and return the new random state. According
   *  to definition of linear congruential engine, the new random state value is computed as
   *  new_random_state = (current_random_state * multiplier + increment) % modulus.
   * \return The next current random state value in the type of result_type.
   * \note In order for better efficiency, the implementation here has a few assumptions:
   *  1. The multiplication and addition won't overflow.
   *  2. The given random state pointer `rand_state_ptr` is not nullptr.
   *  3. The given random state `*(rand_state_ptr)` is in the range of [0, modulus - 1].
   */
  result_type operator()() {
    (*rand_state_ptr_) = ((*rand_state_ptr_) * multiplier + increment) % modulus;
    return *rand_state_ptr_;
  }
  /*!
   * \brief Normalize the random seed to the range of [1, modulus - 1].
   * \param rand_state The random seed.
   * \return The normalized random seed.
   */
  static TRandState NormalizeSeed(TRandState rand_state) {
    if (rand_state == -1) {
      rand_state = DeviceRandom();
    } else {
      rand_state %= modulus;
    }
    if (rand_state == 0) {
      rand_state = 1;
    }
    if (rand_state < 0) {
      LOG(FATAL) << "ValueError: Random seed must be non-negative";
    }
    return rand_state;
  }
  /*!
   * \brief Change the start random state of RNG with the seed of a new random state value.
   * \param rand_state The random state given in result_type.
   */
  void Seed(TRandState rand_state) {
    ICHECK(rand_state_ptr_ != nullptr);
    *rand_state_ptr_ = NormalizeSeed(rand_state);
  }

  /*!
   * \brief Fork a new seed for another RNG from current random state.
   * \return The forked seed.
   */
  TRandState ForkSeed() {
    // In order for reproducibility, we compute the new seed using RNG's random state and a
    // different set of parameters. Note that both 32767 and 1999999973 are prime numbers.
    return ((*this)() * 32767) % 1999999973;
  }

  /*!
   * \brief Construct a random number generator with a random state pointer.
   * \param rand_state_ptr The random state pointer given in result_type*.
   * \note The random state is not checked for whether it's nullptr and whether it's in the range of
   *  [0, modulus-1]. We assume the given random state is valid or the Seed function would be
   *  called right after the constructor before any usage.
   */
  explicit LinearCongruentialEngine(TRandState* rand_state_ptr) {
    rand_state_ptr_ = rand_state_ptr;
  }

 private:
  TRandState* rand_state_ptr_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_RANDOM_ENGINE_H_
