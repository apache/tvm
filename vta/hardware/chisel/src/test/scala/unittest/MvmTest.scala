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
 
package unittest

import chisel3._
import chisel3.util._
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester}
import scala.math.pow
import unittest.util._
import vta.core._

class TestMatrixVectorMultiplication(c: MatrixVectorMultiplication) extends PeekPokeTester(c) {
    
  /* mvm_ref
   *
   * This is a software function that computes dot product with a programmable shift
   * This is used as a reference for the hardware
   */
  def mvmRef(inp: Array[Int], wgt: Array[Array[Int]], shift: Int) : Array[Int] = {
    val size = inp.length
    val res = Array.fill(size) {0}
    for (i <- 0 until size) {
        var dot = 0
        for (j <- 0 until size) {
          dot += wgt(i)(j) * inp(j)
        }
        res(i) = dot * pow(2, shift).toInt
    }
    return res
  }

  val cycles = 5
  for (i <- 0 until cycles) {
    // generate data based on bits
    val inpGen = new RandomArray(c.size, c.inpBits)
    val wgtGen = new RandomArray(c.size, c.wgtBits)
    val in_a = inpGen.any
    val in_b = Array.fill(c.size) { wgtGen.any }
    val res = mvmRef(in_a, in_b, 0)  
    val inpMask = helper.getMask(c.inpBits)
    val wgtMask = helper.getMask(c.wgtBits)
    val accMask = helper.getMask(c.accBits)
    
    for (i <- 0 until c.size) {
      poke(c.io.inp.data.bits(0)(i), in_a(i) & inpMask)
      poke(c.io.acc_i.data.bits(0)(i), 0)
      for (j <- 0 until c.size) {
        poke(c.io.wgt.data.bits(i)(j), in_b(i)(j) & wgtMask)
      }
    }
    
    poke(c.io.reset, 0)
    
    poke(c.io.inp.data.valid, 1)
    poke(c.io.wgt.data.valid, 1)
    poke(c.io.acc_i.data.valid, 1)
      
    step(1)

    poke(c.io.inp.data.valid, 0)
    poke(c.io.wgt.data.valid, 0)
    poke(c.io.acc_i.data.valid, 0)

    // wait for valid signal
    while (peek(c.io.acc_o.data.valid) == BigInt(0)) {
      step(1) // advance clock
    } 
    if (peek(c.io.acc_o.data.valid) == BigInt(1)) {
      for (i <- 0 until c.size) {
          expect(c.io.acc_o.data.bits(0)(i), res(i) & accMask)
      }
    }
  }
}
