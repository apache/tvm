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
 
 //soDLA lisence see https://github.com/soDLA-publishment/soDLA/blob/soDLA_beta/LICENSE.soDLA 
 
package somnia

import chisel3._
import chisel3.experimental._
import chisel3.util._


class cmacConfiguration extends project_spec
{
    val CMAC_BPE = SOMNIA_BPE //bits per element
    val CMAC_ATOMC = PE_MAC_ATOMIC_C_SIZE
    val CMAC_ATOMK = PE_MAC_ATOMIC_K_SIZE
    val CMAC_SLCG_NUM = 3+PE_MAC_ATOMIC_K_SIZE
    val CMAC_RESULT_WIDTH = PE_MAC_RESULT_WIDTH
    val CMAC_IN_RT_LATENCY = 2   //both for data&pd
    val CMAC_OUT_RT_LATENCY = 2   //both for data&pd
    val CMAC_OUT_RETIMING = 3   //only data
    val CMAC_ACTV_LATENCY = 2   //only data
    val CMAC_DATA_LATENCY = (CMAC_IN_RT_LATENCY+CMAC_OUT_RT_LATENCY+CMAC_OUT_RETIMING+CMAC_ACTV_LATENCY)
    val MAC_PD_LATENCY = (CMAC_OUT_RETIMING+CMAC_ACTV_LATENCY-3)     //pd must be 3T earlier than data
    val PKT_nvdla_stripe_info_stripe_st_FIELD = 5
    val PKT_nvdla_stripe_info_stripe_end_FIELD = 6
    val PKT_nvdla_stripe_info_layer_end_FIELD = 8
}


class cmac_core_actv(implicit val conf: somniaConfig) extends Bundle{
    val nz = Output(Bool())
    val data = Output(UInt(conf.SOMNIA_BPE.W))
}

class cmac_reg_dual_flop_outputs extends Bundle{
    val conv_mode = Output(Bool())
    val proc_precision = Output(UInt(2.W))
}