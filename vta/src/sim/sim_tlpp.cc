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
 * \file sim_tlpp.cc
 * \brief simulate core level pipe line parallism logic.
 */
#include <vta/sim_tlpp.h>
TlppVerify::TlppVerify() {
  done_ = 0;
}

void TlppVerify::Clear() {
  fsim_handle_ = nullptr;
  run_fsim_function_ = nullptr;
  for (int i = 0; i < COREMAX; i++) {
    while (insnq_array_[i].size()) {
      insnq_array_[i].pop();
    }
  }
  done_ = 0;
}

uint64_t TlppVerify::GetOperationCode(const VTAGenericInsn *insn) {
  const VTAMemInsn* mem = reinterpret_cast<const VTAMemInsn*>(insn);
  return mem->opcode;
}

CORE_TYPE TlppVerify::GetCoreType(uint64_t operation_code,
                              const VTAGenericInsn *insn) {
  CORE_TYPE core_type = COREGEMM;
  const VTAMemInsn* mem = reinterpret_cast<const VTAMemInsn*>(insn);
  switch (operation_code) {
    case VTA_OPCODE_GEMM:
    case VTA_OPCODE_ALU:
      core_type = COREGEMM;
      break;
    case VTA_OPCODE_LOAD:
      if (mem->memory_type == VTA_MEM_ID_INP||
          mem->memory_type == VTA_MEM_ID_WGT) {
        core_type = CORELOAD;
      }
      break;
    case VTA_OPCODE_STORE:
      core_type = CORESTORE;
      break;
    default:
      break;
  }
  return core_type;
}

bool TlppVerify::DependencyProcess(bool before_run,
    bool pop_prev, bool pop_next,
    bool push_prev, bool push_next,
    Dep_q_t *pop_prev_q, Dep_q_t *pop_next_q,
    Dep_q_t *push_prev_q, Dep_q_t *push_next_q,
    CORE_TYPE push_to_prev_q_indx, CORE_TYPE push_to_next_q_indx) {

  int val = 1;
  if (before_run) {
    if (pop_prev && pop_prev_q->size() == 0) {
      return false;
    }
    if (pop_next && pop_next_q->size() == 0) {
      return false;
    }
    if (pop_next) pop_next_q->pop();
    if (pop_prev) pop_prev_q->pop();
  } else {
    if (push_prev) {
      push_prev_q->push(val);
      dep_push_event_.push(push_to_prev_q_indx);
    }
    if (push_next) {
      push_next_q->push(val);
      dep_push_event_.push(push_to_next_q_indx);
    }
  }
  return true;
}

bool TlppVerify::InsnDependencyCheck(const VTAGenericInsn *insn,
                                     bool before_run) {
  const VTAMemInsn* mem = reinterpret_cast<const VTAMemInsn*>(insn);
  bool pop_prev = mem->pop_prev_dep;
  bool pop_next = mem->pop_next_dep;
  bool push_prev = mem->push_prev_dep;
  bool push_next = mem->push_next_dep;
  CORE_TYPE core_type = GetCoreType(GetOperationCode(insn), insn);
  bool bcheck = false;
  switch (core_type) {
    case COREGEMM:
      bcheck = DependencyProcess(before_run, pop_prev,
          pop_next, push_prev, push_next,
          &l2g_q_, &s2g_q_, &g2l_q_, &g2s_q_, CORELOAD, CORESTORE);
      break;
    case CORELOAD:
      bcheck = DependencyProcess(before_run, pop_prev,
          pop_next, push_prev, push_next,
          nullptr, &g2l_q_, nullptr, &l2g_q_, COREMAX, COREGEMM);
      break;
    case CORESTORE:
      bcheck = DependencyProcess(before_run, pop_prev,
          pop_next, push_prev, push_next,
          &g2s_q_, nullptr, &s2g_q_, nullptr, COREGEMM, COREMAX);
      break;
    case COREMAX:
      assert(0);
      break;
  }

  return bcheck;
}

void TlppVerify::CoreRun(CORE_TYPE core_type) {
  const VTAGenericInsn *insn = PickFrontInsn(core_type);
  while (insn) {
    /*!
     * Check need to read any dependency queue for wait.
     */
    if (!InsnDependencyCheck(insn, true)) {
      break;
    }
    /*!
     * Execute the instruction.
     */
    run_fsim_function_(insn, fsim_handle_);
    /*!
     *check if need to write any dependency queue for notify.
     */
    InsnDependencyCheck(insn, false);
    /*!
     * If instruction is FINISH set done flag.
     * notification.
     */
    done_ = GetOperationCode(insn) == VTA_OPCODE_FINISH;

    if (debug_) {
      printf("this is thread for %s\n", GetCoreTypeName(core_type));
    }
    ConsumeFrontInsn(core_type);
    insn = PickFrontInsn(core_type);
  }
  return;
}

void TlppVerify::EventProcess(void) {
  while (dep_push_event_.size()) {
      CORE_TYPE core_type = dep_push_event_.front();
      dep_push_event_.pop();
      CoreRun(core_type);
  }
}

void TlppVerify::TlppSynchronization(Run_Function run_function,
                                         void *fsim_handle,
                                         bool debug) {
  fsim_handle_ = fsim_handle;
  run_fsim_function_ = run_function;
  debug_ = debug;
  done_ = 0;
  do {
    /*
     * Pick a random core to run first.
     */
    unsigned int seed = time(NULL);
    uint8_t core_start = rand_r(&seed)%COREMAX;
    for (int i = 0; i < COREMAX; i++) {
      CoreRun(static_cast<CORE_TYPE>((core_start + i) % COREMAX));
    }
    EventProcess();
  }while (!done_);
  Clear();
  return;
}

void TlppVerify::TlppPushInsn(const VTAGenericInsn *insn) {
  uint64_t operation_code = GetOperationCode(insn);
  CORE_TYPE core_type = GetCoreType(operation_code, insn);
  insnq_array_[core_type].push(static_cast<const void *>(insn));
  return;
}

const VTAGenericInsn *TlppVerify::PickFrontInsn(uint64_t core_type) {
  const void *return_value = nullptr;
  if (insnq_array_[core_type].size()) {
    return_value = insnq_array_[core_type].front();
  }
  return reinterpret_cast<const VTAGenericInsn *> (return_value);
}

void TlppVerify::ConsumeFrontInsn(uint64_t core_type) {
  if (insnq_array_[core_type].size()) {
    insnq_array_[core_type].pop();
  }
}
