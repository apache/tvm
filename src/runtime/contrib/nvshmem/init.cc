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
#include <nvshmem.h>
#include <nvshmemx.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace runtime {

ShapeTuple InitNVSHMEMUID() {
  nvshmemx_uniqueid_t uid;
  nvshmemx_get_uniqueid(&uid);
  std::vector<int64_t> uid_64;
  uid_64.push_back(static_cast<int64_t>(uid.version));
  for (int i = 0; i < UNIQUEID_PADDING; ++i) {
    uid_64.push_back(static_cast<int64_t>(uid.internal[i]));
  }
  return ShapeTuple(uid_64);
}

void InitNVSHMEM(ShapeTuple uid_64, int num_workers) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker != nullptr);
  CHECK_EQ(uid_64.size(), UNIQUEID_PADDING + 1)
      << "ValueError: The length of unique_id must be " << UNIQUEID_PADDING << ", but got "
      << uid_64.size() << ".";

  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

  nvshmemx_uniqueid_t uid;
  uid.version = static_cast<int>(uid_64[0]);
  for (int i = 0; i < UNIQUEID_PADDING; ++i) {
    uid.internal[i] = static_cast<char>(uid_64[i + 1]);
  }
  nvshmemx_set_attr_uniqueid_args(worker->worker_id, num_workers, &uid, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CUDA_CALL(cudaSetDevice(mype_node));
  LOG_INFO << "NVSHMEM init finished: mype=" << nvshmem_my_pe() << " "
           << ", npes=" << nvshmem_n_pes();
}

TVM_REGISTER_GLOBAL("runtime.disco.nvshmem.init_nvshmem_uid").set_body_typed(InitNVSHMEMUID);

TVM_REGISTER_GLOBAL("runtime.disco.nvshmem.init_nvshmem").set_body_typed(InitNVSHMEM);

}  // namespace runtime
}  // namespace tvm
