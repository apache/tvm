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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_HVX_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_HVX_H_

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonHvx {
 public:
  //! \brief Constructor.
  // TODO(HWE): Pass in a parameter for which HVX instance to bind
  HexagonHvx();

  //! \brief Destructor.
  ~HexagonHvx();

  //! \brief Prevent copy construction of HexagonHvx.
  HexagonHvx(const HexagonHvx&) = delete;

  //! \brief Prevent copy assignment with HexagonHvx.
  HexagonHvx& operator=(const HexagonHvx&) = delete;

  //! \brief Prevent move construction.
  HexagonHvx(HexagonHvx&&) = delete;

  //! \brief Prevent move assignment.
  HexagonHvx& operator=(HexagonHvx&&) = delete;

  //! \brief Lock one HVX to the calling thread.
  void Lock();

  //! \brief Unlock the HVX for the calling thread.
  void Unlock();

  //! \brief Number of HVX units reserved.
  int ReservedCount() { return reserved_count_; }

 private:
  int reserved_count_;

  void Acquire();
  void Release();
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_HVX_H_
