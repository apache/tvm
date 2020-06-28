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
 * \file ansor/utils.cc
 * \brief Common utilities.
 */

#include "utils.h"

namespace tvm {
namespace ansor {

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}

ThreadPool& ThreadPool::Global() {
  static ThreadPool* pool = new ThreadPool();
  static int ct = 0;

  ct = (ct + 1) % ThreadPool::REFRESH_EVERY;

  if (ct == 0) {
    pool->Abort();
    delete pool;
    pool = new ThreadPool();
  }

  if (pool->NumWorkers() == 0) {
    pool->Launch(std::thread::hardware_concurrency());
  }

  return *pool;
}

}  // namespace ansor
}  // namespace tvm
