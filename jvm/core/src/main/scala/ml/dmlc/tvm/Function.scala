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

package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.TVMValue

import scala.collection.mutable.ArrayBuffer

object Function {
  private[tvm] def initAPI(filter: String => Boolean, getName: String => String)
    : Map[String, Function] = {
    listGlobalFuncNames().filter(filter).map(fullName => {
      val funcName = getName(fullName)
      (funcName, getGlobalFunc(fullName, isResident = true).get)
    }).toMap
  }

  /**
   * Get list of global functions registered.
   * @return List of global functions names.
   */
  private[tvm] def listGlobalFuncNames(): IndexedSeq[String] = {
    val names = ArrayBuffer.empty[String]
    checkCall(_LIB.tvmFuncListGlobalNames(names))
    names.toIndexedSeq
  }

  /**
   * Get a global function by name.
   * @param name The name of the function.
   * @param isResident Whether it is a global function. FIXME: not a good name
   * @param allowMissing Whether allow missing function or raise an error.
   * @return The function to be returned, None if function is missing.
   */
  private def getGlobalFunc(name: String, isResident: Boolean = false,
                            allowMissing: Boolean = false): Option[Function] = {
    val handle = new RefFunctionHandle()
    checkCall(_LIB.tvmFuncGetGlobal(name, handle))
    if (handle.value != 0) {
      Some(new Function(handle.value, isResident))
    } else {
      if (allowMissing) {
        None
      } else {
        throw new IllegalArgumentException("Cannot find global function " + name)
      }
    }
  }
}

/**
 * Initialize the function with handle
 * @param handle the handle to the underlying function.
 * @param isResident Whether this is a resident function in jvm
 */
// scalastyle:off finalize
class Function(private val handle: FunctionHandle, private val isResident: Boolean) {
  override protected def finalize(): Unit = {
    if (!isResident) {
      checkCall(_LIB.tvmFuncFree(handle))
    }
  }

  def apply(args: TVMValue*): TVMValue = {
    val ret = new RefTVMValue()
    checkCall(_LIB.tvmFuncCall(handle, args.toArray, ret))
    ret.value
  }
}
// scalastyle:on finalize
