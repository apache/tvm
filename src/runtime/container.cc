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
/*
 * \file src/runtime/container.cc
 * \brief POD container type implementations.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/container.h>
#include <cstdint>
#include "object_internal.h"
#include "runtime_base.h"

namespace tvm {
namespace runtime {

template <typename Iterator>
ADT::ADT(uint32_t tag, Iterator begin, Iterator end) {
  size_t num_elems = std::distance(begin, end);
  auto ptr = make_array<ADTObj, ObjectRef>(num_elems);
  ptr->tag_ = tag;
  ptr->size_ = num_elems;
  ptr->Init(begin, end);
  data_ = std::move(ptr);
}

ADT ADT::Tuple(std::vector<ObjectRef> fields) { return ADT(0, fields); }

}  // namespace runtime
}  // namespace tvm
