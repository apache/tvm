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
 * soDLA license is at https://github.com/soDLA-publishment/soDLA/blob/soDLA_beta/LICENSE.soDLA
 */
 

package somnia

import chisel3._
import chisel3.experimental._
import chisel3.util._
import scala.math._


class project_spec extends nv_somnia
{
    val PE_MAC_ATOMIC_C_SIZE = MAC_ATOMIC_C_SIZE/SPLIT_NUM
    val PE_MAC_ATOMIC_K_SIZE = MAC_ATOMIC_K_SIZE/SPLIT_NUM
    val PE_MAC_ATOMIC_C_SIZE_LOG2 = log2Ceil(PE_MAC_ATOMIC_C_SIZE)
    val PE_MAC_ATOMIC_K_SIZE_LOG2 = log2Ceil(PE_MAC_ATOMIC_K_SIZE)
    val PE_MAC_RESULT_WIDTH = 2*SOMNIA_BPE + log2Ceil(PE_MAC_ATOMIC_C_SIZE)
}