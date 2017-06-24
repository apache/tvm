/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.tvm.types

import ml.dmlc.tvm.APIInternal

// TVM context structure
object TVMContext {
  def apply(deviceType: String, deviceId: Int): TVMContext = {
    new TVMContext(deviceType, deviceId)
  }

  val MASK2STR = Map(
    1 -> "cpu",
    2 -> "gpu",
    4 -> "opencl",
    8 -> "metal",
    9 -> "vpi"
  )
  val STR2MASK = Map(
    "cpu" -> 1,
    "gpu" -> 2,
    "cuda" -> 2,
    "cl" -> 4,
    "opencl" -> 4,
    "metal" -> 8,
    "vpi" -> 9
  )
}

class TVMContext(private val deviceType: Int, private val deviceId: Int) {
  private val RPC_SESS_MASK = 128

  def this(deviceType: String, deviceId: Int) = {
    this(TVMContext.STR2MASK(deviceType), deviceId)
  }

  // Whether this device exist.
  def exist: Boolean = {
    val ret = APIInternal("_GetDeviceAttr")(deviceType, deviceId, 0)
    ret.asInstanceOf[TVMValueLong].value != 0
  }

  // Maximum number of threads on each block.
  def maxThreadsPerBlock: Long = {
    val ret = APIInternal("_GetDeviceAttr")(deviceType, deviceId, 1)
    ret.asInstanceOf[TVMValueLong].value
  }

  // Number of threads that executes in concurrent.
  def warpSize: Long = {
    val ret = APIInternal("_GetDeviceAttr")(deviceType, deviceId, 2)
    ret.asInstanceOf[TVMValueLong].value
  }

  // Synchronize until jobs finished at the context.
  def sync(): Unit = {
    // TODO: checkCall(_LIB.TVMSynchronize(self, None))
  }

  override def hashCode: Int = {
    (deviceType << 16) | deviceId
  }

  override def equals(other: Any): Boolean = {
    other match {
      case obj: TVMContext =>
        deviceId == obj.deviceId && deviceType == obj.deviceType
      case _ =>
        false
    }
  }

  override def toString: String = {
    if (deviceType >= RPC_SESS_MASK) {
      val tblId = deviceType / RPC_SESS_MASK - 1
      val devType = deviceType % RPC_SESS_MASK
      s"remote[$tblId]:${TVMContext.MASK2STR(devType)}($deviceId)"
    } else {
      s"${TVMContext.MASK2STR(deviceType)}($deviceId)"
    }
  }
}
