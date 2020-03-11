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
 * \file sim_tlpp.h
 * \brief TVM VTA multiple thread simulator header file.
 */
#ifndef VTA_SIM_TLPP_H_
#define VTA_SIM_TLPP_H_
#include <vta/hw_spec.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <ctime>
#include <cassert>
#include <queue>

#define SCOREGEMM "gemm"
#define SCORELOAD "load"
#define SCORESTORE "store"
#define SCOREUNKNOWN "unknown"
typedef void (*Run_Function)(const VTAGenericInsn *, void *);
typedef enum {COREGEMM = 0, CORELOAD, CORESTORE, COREMAX} CORE_TYPE;
typedef std::queue<const void*> Insn_q_t;
typedef std::queue<int> Dep_q_t;
/*!
 * \brief simulate core level pipe line parallism logic.
 */
class TlppVerify {
 public:
    /*! Return TlppVefiy class instance.*/
    static TlppVerify *Global() { static TlppVerify Cls; return &Cls;}

    /*!
     *  \brief Loop to process instruction and verify tlpp logic.
     *  \param run_function function pointer to excute instruction .
     *  \param fsim_handle class pointer of function simulator class Device.
     *  \param debug to enable/disable debug
     */
    void TlppSynchronization(Run_Function run_function,
                             void *fsim_handle,
                             bool debug = false);
    /*!
     *  \brief Push instruction into queue for later excute.
     *  \param insn instructions.
     */
    void TlppPushInsn(const VTAGenericInsn *insn);
    /*! \ Event pump to handle dependency event. */
    void EventProcess(void);
    /*! \ Schedule a paticular core to run. */
    void CoreRun(CORE_TYPE core_type);

 private:
    /*! TlppVerify construction function.*/
    TlppVerify();
    /*!
     * \brief clear class variable.
     */
    void Clear();
    /*!
     * \ brief check if the insn dependency condition satisfy and do notify.
     * \ param insn instructions.
     * \ param before_run identify this check is happen before
     *   instruction excute or after instruction excute, for before
     *   scenario need to check if depency condition satisfy, for post
     *   case need to check if need to send notfication.
     */
    bool InsnDependencyCheck(const VTAGenericInsn *insn, bool before_run);
    /*!
     * \ brief get operation code from insn
     * \ param insn instructions
     */
    uint64_t GetOperationCode(const VTAGenericInsn *insn);
    /*!
     * \ brief find which core should run this instruction.
     * \ param operation_code operation type like load/gemm etc.
     * \ param insn instructions.
     */
    CORE_TYPE GetCoreType(uint64_t operation_code, const VTAGenericInsn *insn);
    /*!
     * \ brief , pick up first instruction for specify core.
     * \ param core_type core type
     */
    const VTAGenericInsn *PickFrontInsn(uint64_t core_type);
    /*!
     * \ brief consume one instruction after pass dependency condition.
     * \ param core_type core type
     */
    void ConsumeFrontInsn(uint64_t core_type);
    /*!
     * \ brief, process dependency logic
     * param before_run if this call happen before instruction run.
     * param pop_prev if instruction have previous core dependency.
     * param pop_next if instruction have depency for next core.
     * param pop_prev_q notification from previous core.
     * param pop_next_q notification from next core.
     * param push_prev_q notification queue need to send notification
     * for prevous core.
     * param push_next_q notification queue need to send notification
     * from next core.
     * push_to_prev_q_indx which core need wake up if have notification
     * fro previous core.
     * push_to_next_q_indx which core need wake up if have notification
     * fro next core.
     */
    bool DependencyProcess(bool before_run,
        bool pop_prev, bool pop_next,
        bool push_prev, bool push_next,
        Dep_q_t *pop_prev_q, Dep_q_t *pop_next_q,
        Dep_q_t *push_prev_q, Dep_q_t *push_next_q,
        CORE_TYPE push_to_prev_q_indx, CORE_TYPE push_to_next_q_indx);
    /*!
     * \ brief , return name based on core type.
     * \ param core_type core type
     */
    inline const char * GetCoreTypeName(CORE_TYPE core_type) {
      return (core_type == COREGEMM) ? SCOREGEMM :
        (core_type == CORELOAD) ? SCORELOAD :
        (core_type == CORESTORE) ? SCORESTORE :
        SCOREUNKNOWN;
    }
    /*! debug flag*/
    bool debug_;
    /*! function simulator device class pointer*/
    void *fsim_handle_;
    /*! function simulator instruction excute function pointer*/
    Run_Function run_fsim_function_;
    /*! instruction queue for each core*/
    Insn_q_t insnq_array_[COREMAX];
    /*! dependency queue from load to gemm*/
    Dep_q_t l2g_q_;
    /*! dependency queue from store to gemm*/
    Dep_q_t s2g_q_;
    /*! dependency queue from gemm to load*/
    Dep_q_t g2l_q_;
    /*! dependency queue from gemm to store*/
    Dep_q_t g2s_q_;
    /*! computation done*/
    int done_;
    /*! event queue for core wake up*/
    std::queue<CORE_TYPE> dep_push_event_;
};
#endif  // VTA_SIM_TLPP_H_
