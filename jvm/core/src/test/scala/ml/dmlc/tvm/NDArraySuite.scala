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

import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}

class NDArraySuite extends FunSuite with BeforeAndAfterAll with Matchers {
  test("from float32") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "float32")
    ndarray.set(Array(1, 2, 3, 4))
    assert(ndarray.toArray === Array(1.0, 2.0, 3.0, 4.0))
  }

  test("from float64") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "float64")
    ndarray.set(Array(1, 2, 3, 4))
    assert(ndarray.toArray === Array(1.0, 2.0, 3.0, 4.0))
  }

  test("from int8") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "int8")
    ndarray.set(Array(1, 2, 3, 4))
    assert(ndarray.toArray === Array(1.0, 2.0, 3.0, 4.0))
  }

  test("from int16") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "int16")
    ndarray.set(Array(1, 2, 3, 4))
    assert(ndarray.toArray === Array(1.0, 2.0, 3.0, 4.0))
  }

  test("from int32") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "int32")
    ndarray.set(Array(1, 2, 3, 4))
    assert(ndarray.toArray === Array(1.0, 2.0, 3.0, 4.0))
  }

  test("from int64") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "int64")
    ndarray.set(Array(1, 2, 3, 4))
    assert(ndarray.toArray === Array(1.0, 2.0, 3.0, 4.0))
  }

  test("from uint8") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "uint8")
    ndarray.set(Array(128, 2, 3, 4))
    assert(ndarray.toArray === Array(128.0, 2.0, 3.0, 4.0))
  }

  test("from uint16") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "uint16")
    ndarray.set(Array(65535, 2, 3, 4))
    assert(ndarray.toArray === Array(65535.0, 2.0, 3.0, 4.0))
  }

  test("from uint32") {
    val ndarray = NDArray.empty(Shape(2, 2), dtype = "uint32")
    ndarray.set(Array(4294967295.0, 2, 3, 4))
    assert(ndarray.toArray === Array(4294967295.0, 2.0, 3.0, 4.0))
  }
}
