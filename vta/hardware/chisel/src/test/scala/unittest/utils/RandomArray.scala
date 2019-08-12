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

package unittest.util

import scala.util.Random
import scala.math.pow

class RandomArray(val len: Int, val bits: Int) {
  val r = new Random
  if (bits < 1) throw new IllegalArgumentException ("bits should be greater than 1")

  def any : Array[Int] = {
    return Array.fill(len) { r.nextInt(pow(2, bits).toInt) - pow(2, bits-1).toInt }
  }

  def positive : Array[Int] = {
    return Array.fill(len) { r.nextInt(pow(2, bits-1).toInt) }
  }

  def negative : Array[Int] = {
    return Array.fill(len) { 0 - r.nextInt(pow(2, bits-1).toInt) }
  }
}
