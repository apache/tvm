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

#include <random>

#include "../utils.h"

namespace tvm {
namespace tir {

struct PrimeTable {
  /*! \brief The table contains prime numbers in [2, kMaxPrime) */
  static constexpr const int32_t kMaxPrime = 65536;
  /*! \brief The exact number of prime numbers in the table */
  static constexpr const int32_t kNumPrimes = 6542;
  /*!
   * \brief For each number in [2, kMaxPrime), the index of its min factor.
   * For example, if min_factor_idx[x] = i, then the min factor of x is primes[i].
   */
  int32_t min_factor_idx[kMaxPrime];
  /*! \brief The prime numbers in [2, kMaxPrime) */
  std::vector<int32_t> primes;
  /*!
   * \brief The power of each prime number.
   * pow_table[i, j] stores the result of pow(prime[i], j + 1)
   */
  std::vector<std::vector<int32_t>> pow_tab;

  /*! \brief Get a global instance of the prime table */
  static const PrimeTable* Global() {
    static const PrimeTable table;
    return &table;
  }

  /*! \brief Constructor, pre-computes all info in the prime table */
  PrimeTable() {
    constexpr const int64_t int_max = std::numeric_limits<int32_t>::max();
    // Euler's sieve: prime number in linear time
    for (int32_t i = 0; i < kMaxPrime; ++i) {
      min_factor_idx[i] = -1;
    }
    primes.reserve(kNumPrimes);
    for (int32_t x = 2; x < kMaxPrime; ++x) {
      if (min_factor_idx[x] == -1) {
        min_factor_idx[x] = primes.size();
        primes.push_back(x);
      }
      for (size_t i = 0; i < primes.size(); ++i) {
        int64_t factor = primes[i];
        int64_t y = x * factor;
        if (y >= kMaxPrime) {
          break;
        }
        min_factor_idx[y] = i;
        if (x % factor == 0) {
          break;
        }
      }
    }
    ICHECK_EQ(static_cast<int32_t>(primes.size()), static_cast<int32_t>(kNumPrimes));
    // Calculate the power table for each prime number
    pow_tab.reserve(primes.size());
    for (int32_t prime : primes) {
      std::vector<int32_t> tab;
      tab.reserve(32);
      for (int64_t pow = prime; pow <= int_max; pow *= prime) {
        tab.push_back(pow);
      }
      tab.shrink_to_fit();
      pow_tab.emplace_back(std::move(tab));
    }
  }
  /*!
   * \brief Factorize a number n, and return in a cryptic format
   * \param n The number to be factorized
   * \return A list of integer pairs [(i_1, j_1), (i_2, j_2), ..., (i_l, j_l)]
   * For each pair (i, j), we define
   *    (a, b) = (j, 1)             if i == -1 (in this case j must be a prime number)
   *             (primes[i], j)     if i != -1
   * Then the factorization is
   *    n = (a_1 ^ b_1) * (a_2 ^ b_2) ... (a_l ^ b_l)
   */
  std::vector<std::pair<int32_t, int32_t>> Factorize(int32_t n) const {
    std::vector<std::pair<int32_t, int32_t>> result;
    result.reserve(16);
    int32_t i = 0, n_primes = primes.size();
    // Phase 1: n >= kMaxPrime
    for (int32_t j; n >= kMaxPrime && i < n_primes && primes[i] * primes[i] <= n; ++i) {
      for (j = 0; n % primes[i] == 0; n /= primes[i], ++j) {
      }
      if (j != 0) {
        result.emplace_back(i, j);
      }
    }
    // if i >= n_primes or primes[i] > sqrt(n), then n must be a prime number
    if (n >= kMaxPrime) {
      result.emplace_back(-1, n);
      return result;
    }
    // Phase 2: n < kMaxPrime
    for (int32_t j; n > 1;) {
      int32_t i = min_factor_idx[n];
      for (j = 0; n % primes[i] == 0; n /= primes[i], ++j) {
      }
      result.emplace_back(i, j);
    }
    return result;
  }
};

int32_t SampleInt(support::LinearCongruentialEngine::TRandState* rand_state, int32_t min_inclusive,
                  int32_t max_exclusive) {
  CHECK(min_inclusive < max_exclusive)
      << "ValueError: max_exclusive must be greater than min_inclusive.";
  if (min_inclusive + 1 == max_exclusive) {
    return min_inclusive;
  }
  support::LinearCongruentialEngine rand_(rand_state);
  std::uniform_int_distribution<int32_t> dist(min_inclusive, max_exclusive - 1);
  return dist(rand_);
}

std::vector<int32_t> SampleWithoutReplacement(
    support::LinearCongruentialEngine::TRandState* rand_state, int32_t n, int32_t k) {
  if (k == 1) {
    return {SampleInt(rand_state, 0, n)};
  }
  if (k == 2) {
    int32_t result0 = SampleInt(rand_state, 0, n);
    int32_t result1 = SampleInt(rand_state, 0, n - 1);
    if (result1 >= result0) {
      result1 += 1;
    }
    return {result0, result1};
  }
  std::vector<int32_t> order(n);
  for (int32_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  for (int32_t i = 0; i < k; ++i) {
    int32_t j = SampleInt(rand_state, i, n);
    if (i != j) {
      std::swap(order[i], order[j]);
    }
  }
  return {order.begin(), order.begin() + k};
}

int64_t SampleCategorical(support::LinearCongruentialEngine::TRandState* rand_state,
                          const Array<Integer>& candidates, const Array<FloatImm>& probs,
                          Optional<Integer>* decision) {
  CHECK(candidates.size() == probs.size())
      << "ValueError: number of candidates does not match number of probabilities.";
  int32_t i = -1;
  int32_t n = candidates.size();
  if (decision->defined()) {
    const auto* int_imm = decision->as<IntImmNode>();
    i = int_imm->value;
    CHECK(0 <= i && i < n) << "ValueError: Wrong decision value, where n = " << n
                           << ", but decision is: " << i;
  } else {
    std::vector<double> weights = support::AsVector<FloatImm, double>(probs);
    std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
    support::LinearCongruentialEngine rand_(rand_state);
    i = dist(rand_);
    ICHECK(0 <= i && i < n) << "ValueError: Unexpected decision generated, where n = " << n
                            << ", but decision is: " << i;
  }

  *decision = Integer(i);  // decision is guaranteed not to be nullptr.
  return candidates[i].IntValue();
}

std::function<int32_t()> MakeMultinomialSampler(
    support::LinearCongruentialEngine::TRandState* rand_state, const std::vector<double>& weights) {
  ICHECK(!weights.empty());
  std::vector<double> sums;
  sums.reserve(weights.size());
  double sum = 0.0;
  for (double w : weights) {
    sums.push_back(sum += w);
  }
  return [rng = support::LinearCongruentialEngine(rand_state).ForkSeed(),
          dist = std::uniform_real_distribution<double>(0.0, sum),
          sums = std::move(sums)]() mutable -> int32_t {
    support::LinearCongruentialEngine rand_(&rng);
    double p = dist(rand_);
    int32_t idx = std::lower_bound(sums.begin(), sums.end(), p) - sums.begin();
    int32_t n = sums.size();
    CHECK_LE(0, idx);
    CHECK_LE(idx, n);
    return (idx == n) ? (n - 1) : idx;
  };
}

std::vector<int64_t> SamplePerfectTile(support::LinearCongruentialEngine::TRandState* rand_state,
                                       int32_t extent, int32_t n_splits) {
  CHECK_GE(extent, 1) << "ValueError: Cannot tile a loop with 0 or negative extent";
  CHECK_GE(n_splits, 1) << "ValueError: Cannot tile a loop to 0 or negative splits";
  // Handle special case that we can potentially accelerate
  if (n_splits == 1) {
    return {extent};
  }
  if (extent == 1) {
    return std::vector<int64_t>(n_splits, 1);
  }
  // Enumerate each pair (i, j), we define
  //    (a, p) = (j, 1)             if i == -1 (in this case j must be a prime number)
  //             (primes[i], j)     if i != -1
  // Then the factorization is
  //    extent = (a_1 ^ p_1) * (a_2 ^ p_2) ... (a_l ^ p_l)
  const PrimeTable* prime_tab = PrimeTable::Global();
  std::vector<std::pair<int32_t, int32_t>> factorized = prime_tab->Factorize(extent);
  if (n_splits == 2) {
    // n_splits = 2, this can be taken special care of,
    // because general reservoir sampling can be avoided to accelerate the sampling
    int32_t result0 = 1;
    int32_t result1 = 1;
    for (const std::pair<int32_t, int32_t>& ij : factorized) {
      // Case 1: (a, p) = (j, 1), where j is a prime number
      if (ij.first == -1) {
        (SampleInt(rand_state, 0, 2) ? result1 : result0) *= ij.second;
        continue;
      }
      // Case 2: (a = primes[i], p = 1)
      int32_t p = ij.second;
      const int32_t* pow = prime_tab->pow_tab[ij.first].data() - 1;
      int32_t x1 = SampleInt(rand_state, 0, p + 1);
      int32_t x2 = p - x1;
      if (x1 != 0) {
        result0 *= pow[x1];
      }
      if (x2 != 0) {
        result1 *= pow[x2];
      }
    }
    return {result0, result1};
  }
  // Data range:
  //    2 <= extent <= 2^31 - 1
  //    3 <= n_splits <= max tiling splits
  //    1 <= p <= 31
  std::vector<int64_t> result(n_splits, 1);
  for (const std::pair<int32_t, int32_t>& ij : factorized) {
    // Handle special cases to accelerate sampling
    // Case 1: (a, p) = (j, 1), where j is a prime number
    if (ij.first == -1) {
      result[SampleInt(rand_state, 0, n_splits)] *= ij.second;
      continue;
    }
    // Case 2: (a = primes[i], p = 1)
    int32_t p = ij.second;
    if (p == 1) {
      result[SampleInt(rand_state, 0, n_splits)] *= prime_tab->primes[ij.first];
      continue;
    }
    // The general case. We have to sample uniformly from the solution of:
    //    x_1 + x_2 + ... + x_{n_splits} = p
    // where x_i >= 0
    // Data range:
    //    2 <= p <= 31
    //    3 <= n_splits <= max tiling splits
    std::vector<int32_t> sampled =
        SampleWithoutReplacement(rand_state, p + n_splits - 1, n_splits - 1);
    std::sort(sampled.begin(), sampled.end());
    sampled.push_back(p + n_splits - 1);
    const int32_t* pow = prime_tab->pow_tab[ij.first].data() - 1;
    for (int32_t i = 0, last = -1; i < n_splits; ++i) {
      int32_t x = sampled[i] - last - 1;
      last = sampled[i];
      if (x != 0) {
        result[i] *= pow[x];
      }
    }
  }
  return result;
}

std::vector<int64_t> SamplePerfectTile(support::LinearCongruentialEngine::TRandState* rand_state,
                                       int32_t extent, int32_t n_splits,
                                       int32_t max_innermost_factor) {
  if (max_innermost_factor == -1) {
    return SamplePerfectTile(rand_state, extent, n_splits);
  }
  CHECK_GE(n_splits, 2) << "ValueError: Cannot tile a loop into " << n_splits << " splits";
  while (true) {
    std::vector<int64_t> result = SamplePerfectTile(rand_state, extent, n_splits);
    if (result.back() <= max_innermost_factor) {
      return result;
    }
  }
}

std::vector<int64_t> SamplePerfectTile(
    support::LinearCongruentialEngine::TRandState* rand_state,  //
    const tir::StmtSRef& loop_sref, int32_t n_splits, int32_t max_innermost_factor,
    Optional<Array<Integer>>* decision) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  const int64_t* extent = GetLoopIntExtent(loop);
  std::vector<int64_t> result;
  if (extent == nullptr) {
    // Case 1. Handle loops with non-constant length
    result = std::vector<int64_t>(n_splits, 1);
    result[0] = -1;
  } else if (decision->defined()) {
    // Case 2. Use previous decision
    result = support::AsVector<Integer, int64_t>(decision->value());
    int n = result.size();
    ICHECK_GE(n, 2);
    int64_t len = *extent;
    for (int i = n - 1; i > 0; --i) {
      int64_t& l = result[i];
      // A previous decision could become invalid because of the change of outer tiles
      // To handle this case properly, we check if the tiling strategy is still perfect.
      // If not, we use a trivial default solution (1, 1, ..., 1, L) for rest of the tiles
      if (len % l != 0) {
        l = len;
      }
      len /= l;
    }
    result[0] = len;
  } else {
    // Case 3. Use fresh new sampling result
    result = SamplePerfectTile(rand_state, *extent, n_splits, max_innermost_factor);
    if (max_innermost_factor != -1) {
      ICHECK_LE(result.back(), max_innermost_factor);
    }
  }
  *decision = support::AsArray<int64_t, Integer>(result);
  return result;
}

tir::StmtSRef SampleComputeLocation(tir::ScheduleState self,
                                    support::LinearCongruentialEngine::TRandState* rand_state,
                                    const StmtSRef& block_sref, Optional<Integer>* decision) {
  // Step 1. Collect all possible compute-at locations.
  auto [location_srefs, location_indices] = CollectComputeLocation(self, block_sref);
  ICHECK_EQ(location_srefs.size(), location_indices.size());

  // Step 2. If there was a previous decision, keep the decision unchanged if it exists in the
  // location candidates. Otherwise, pick the location before the previous decision.
  // Step 3. If there was not a previous decision, sample a decision from the collected locations.
  if (decision->defined()) {
    int64_t old_decision = Downcast<Integer>(*decision)->value;
    auto it = std::lower_bound(location_indices.begin(), location_indices.end(), old_decision);
    int idx = it - location_indices.begin();

    if (it != location_indices.end() && *it == old_decision) {
      *decision = Integer(old_decision);
      return location_srefs[idx];
    } else if (it != location_indices.begin()) {
      *decision = Integer(location_indices[idx - 1]);
      return location_srefs[idx - 1];
    } else {
      *decision = Integer(-1);
      return StmtSRef::RootMark();
    }
  } else {
    int sampled_idx = SampleInt(rand_state, 0, location_indices.size());
    *decision = Integer(location_indices[sampled_idx]);
    return location_srefs[sampled_idx];
  }
}

/******** InstructionKind Registration ********/

struct SampleCategoricalTraits : public UnpackedInstTraits<SampleCategoricalTraits> {
  static constexpr const char* kName = "SampleCategorical";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 1;

  static ExprRV UnpackedApplyToSchedule(Schedule sch,               //
                                        Array<Integer> candidates,  //
                                        Array<FloatImm> probs,      //
                                        Optional<Integer> decision) {
    return sch->SampleCategorical(candidates, probs, decision);
  }

  static String UnpackedAsPython(Array<String> outputs,      //
                                 Array<Integer> candidates,  //
                                 Array<FloatImm> probs,      //
                                 Optional<Integer> decision) {
    PythonAPICall py("sample_categorical");
    py.Input("candidates", candidates);
    py.Input("probs", probs);
    py.Decision(decision);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct SamplePerfectTileTraits : public UnpackedInstTraits<SamplePerfectTileTraits> {
  static constexpr const char* kName = "SamplePerfectTile";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 1;

  static Array<ExprRV> UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, Integer n,
                                               Integer max_innermost_factor,
                                               Optional<Array<Integer>> decision) {
    return sch->SamplePerfectTile(loop_rv, n->value, max_innermost_factor->value, decision);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Integer n,
                                 Integer max_innermost_factor, Optional<Array<Integer>> decision) {
    PythonAPICall py("sample_perfect_tile");
    py.Input("loop", loop_rv);
    py.Input("n", n->value);
    py.Input("max_innermost_factor", max_innermost_factor->value);
    py.Decision(decision);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct SampleComputeLocationTraits : public UnpackedInstTraits<SampleComputeLocationTraits> {
  static constexpr const char* kName = "SampleComputeLocation";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 1;

  static LoopRV UnpackedApplyToSchedule(Schedule sch,      //
                                        BlockRV block_rv,  //
                                        Optional<Integer> decision) {
    return sch->SampleComputeLocation(block_rv, decision);
  }

  static String UnpackedAsPython(Array<String> outputs,  //
                                 String block_rv,        //
                                 Optional<Integer> decision) {
    PythonAPICall py("sample_compute_location");
    py.Input("block", block_rv);
    py.Decision(decision);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SampleCategoricalTraits);
TVM_REGISTER_INST_KIND_TRAITS(SamplePerfectTileTraits);
TVM_REGISTER_INST_KIND_TRAITS(SampleComputeLocationTraits);

}  // namespace tir
}  // namespace tvm
