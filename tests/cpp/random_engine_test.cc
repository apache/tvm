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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/support/random_engine.h>

TEST(RandomEngine, Randomness) {
  int64_t rand_state = 0;

  tvm::support::LinearCongruentialEngine rng(&rand_state);
  rng.Seed(0x114514);

  bool covered[100];
  memset(covered, 0, sizeof(covered));
  for (int i = 0; i < 100000; i++) {
    covered[rng() % 100] = true;
  }
  for (int i = 0; i < 100; i++) {
    ICHECK(covered[i]);
  }
}

TEST(RandomEngine, Reproducibility) {
  int64_t rand_state_a = 0, rand_state_b = 0;
  tvm::support::LinearCongruentialEngine rng_a(&rand_state_a), rng_b(&rand_state_b);

  rng_a.Seed(0x23456789);
  rng_b.Seed(0x23456789);

  for (int i = 0; i < 100000; i++) {
    ICHECK_EQ(rng_a(), rng_b());
  }
}

TEST(RandomEngine, Serialization) {
  int64_t rand_state_a = 0, rand_state_b = 0;
  tvm::support::LinearCongruentialEngine rng_a(&rand_state_a), rng_b(&rand_state_b);

  rng_a.Seed(0x56728);

  rand_state_b = rand_state_a;
  for (int i = 0; i < 100000; i++) ICHECK_EQ(rng_a(), rng_b());

  for (int i = 0; i < 123456; i++) rng_a();

  rand_state_b = rand_state_a;
  for (int i = 0; i < 100000; i++) ICHECK_EQ(rng_a(), rng_b());
}
