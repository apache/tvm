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

object TVMType {
  implicit def str2Type(dtype: String): TVMType = TVMType(dtype)

  val CODE2STR = Map(
    0 -> "int",
    1 -> "uint",
    2 -> "float",
    4 -> "handle"
  )

  val INT = 0
  val UINT = 1
  val FLOAT = 2
  val HANDLE = 4

  def apply(typeStr: String, lanes: Int = 1): TVMType = {
    new TVMType(typeStr, lanes)
  }
}

class TVMType(val typeStr: String, val lanes: Int = 1) {
  private val (typeCodeTemp, bitsTemp) =
    if (typeStr.startsWith("int")) {
      (0, typeStr.substring(3).toInt)
    } else if (typeStr.startsWith("uint")) {
      (1, typeStr.substring(4).toInt)
    } else if (typeStr.startsWith("float")) {
      (2, typeStr.substring(5).toInt)
    } else if (typeStr.startsWith("handle")) {
      (4, 64)
    } else {
      throw new IllegalArgumentException("Do not know how to handle type " + typeStr)
    }

  val typeCode = typeCodeTemp
  val bits = if (bitsTemp == 0) 32 else bitsTemp
  if ((bits & (bits - 1)) != 0 || bits < 8) {
    throw new IllegalArgumentException("Do not know how to handle type " + typeStr)
  }

  def numOfBytes: Int = {
    bits / 8
  }

  override def hashCode: Int = {
    (typeCode << 16) | (bits  << 8) | lanes
  }

  override def equals(other: Any): Boolean = {
    if (other != null && other.isInstanceOf[TVMType]) {
      val otherInst = other.asInstanceOf[TVMType]
      (bits == otherInst.bits) && (typeCode == otherInst.typeCode) && (lanes == otherInst.lanes)
    } else {
      false
    }
  }

  override def toString: String = {
    val str = s"${TVMType.CODE2STR(typeCode)}$bits"
    if (lanes != 1) {
      str + lanes
    } else {
      str
    }
  }
}
