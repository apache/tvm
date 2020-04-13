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
 * \file util.cc
 * \brief The utils for arithmetic analysis.
 */
#include <tvm/arith/util.h>
#include <dmlc/logging.h>

namespace tvm {
namespace arith {

std::tuple<int64_t, int64_t, int64_t> xgcd(int64_t a, int64_t b) {
  int64_t s = 0, old_s = 1;
  int64_t t = 1, old_t = 0;
  int64_t r = b, old_r = a;

  while (r != 0) {
    int64_t q = old_r / r;
    std::swap(r, old_r);
    r -= q * old_r;
    std::swap(s, old_s);
    s -= q * old_s;
    std::swap(t, old_t);
    t -= q * old_t;
  }

  CHECK_EQ(a % old_r, 0);
  CHECK_EQ(b % old_r, 0);
  CHECK(old_r == old_s*a + old_t*b);

  return std::make_tuple(old_r, old_s, old_t);
}

}  // namespace arith
}  // namespace tvm
