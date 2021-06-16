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
 * \file src/relay/backend/contrib/ethosn/ethosn_variant.h
 * \brief Ethos-N utility functions.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_VARIANT_H_
#define TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_VARIANT_H_

#include <algorithm>
#include <string>

/*!
 * \brief Create an Ethos-N variant string.
 * \param variant A string specifing the variant type.
 * \param tops A string specifing the number of tops.
 * \param ple_ratio A string specifing the PLE ratio.
 * \return Ethos-N variant string.
 */
std::string MakeVariant(std::string variant, std::string tops, std::string ple_ratio) {
  // Transform variant string to lowercase for comparison
  std::string variant_string = variant;
  std::transform(variant_string.begin(), variant_string.end(), variant_string.begin(), ::tolower);
  std::string variant_n78 = "ethos-n78";
  if (variant_string == variant_n78) {
    variant = "Ethos-N78_" + tops + "TOPS_" + ple_ratio + "PLE_RATIO";
  }
  return variant;
}

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_VARIANT_H_
