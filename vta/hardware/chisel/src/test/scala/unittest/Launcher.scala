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
// taken from https://github.com/freechipsproject/chisel-testers

import chisel3._
import chisel3.iotesters.{Driver, TesterOptionsManager}
import unittest.util._
import vta.core._
import vta.util.config._
import vta.shell._

class TestConfig extends Config(new CoreConfig ++ new PynqConfig)

/* Launcher.
 *
 * The Launcher object includes a test list for the TestRunner to check.
 * Users can utilize this Launcher to run custom tests.
 *
 * How to Use:
 * When the user input: sbt 'test:runMain unittest.Launcher mvm'
 * the TestRunner will look for 'mvm' in the map and executes the 
 * test that 'mvm' is mapped to
 */
object Launcher {
  implicit val p: Parameters = new TestConfig
  val tests = Map(
    "mvm" -> { (manager: TesterOptionsManager) =>
      Driver.execute(() => new MatrixVectorMultiplication, manager) {
        (c) => new TestMatrixVectorMultiplication(c)
      }
    },
		"alu" -> { (manager: TesterOptionsManager) =>
      Driver.execute(() => new AluVector, manager) {
        (c) => new TestAluVector(c)
      }   
    } 
  )

  def main(args: Array[String]): Unit = {
    TestRunner(tests, args)
  }
}
