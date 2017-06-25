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

import ml.dmlc.tvm.types._
import ml.dmlc.tvm.types.TVMValue._

// scalastyle:off println
object HelloWorld {
  def main(args: Array[String]): Unit = {
    val filename = "myadd.so"
    val mod = Module.load(filename)
    println(mod.entryFunc)

    val ctx = TVMContext("cpu", 0)
    println("CPU exist: " + ctx.exist)

    val shape = Shape(2)

    val arr = NDArray.empty(shape)
    println(arr.shape)

    arr.set(Array(3.0f, 4.0f))
    println(arr.shape)

    val res = NDArray.empty(shape)
    mod(arr, arr, res)

    println("arr to Array: [" + arr.toArray.mkString(",") + "]")
    println("res to Array: [" + res.toArray.mkString(",") + "]")
  }
}
// scalastyle:on println
