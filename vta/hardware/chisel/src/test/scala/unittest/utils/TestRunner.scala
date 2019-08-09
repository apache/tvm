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
// taken from https://github.com/freechipsproject/chisel-testers

import scala.collection.mutable.ArrayBuffer
import chisel3.iotesters._

object TestRunner {

  def apply(testMap: Map[String, TesterOptionsManager => Boolean], args: Array[String]): Unit = {
    var successful = 0
    val errors = new ArrayBuffer[String]

    val optionsManager = new TesterOptionsManager()
    optionsManager.doNotExitOnHelp()

    optionsManager.parse(args)

    val programArgs = optionsManager.commonOptions.programArgs

    if(programArgs.isEmpty) {
      println("Available tests")
      for(x <- testMap.keys) {
        println(x)
      }
      println("all")
      System.exit(0)
    }

    val testsToRun = if(programArgs.exists(x => x.toLowerCase() == "all")) {
      testMap.keys
    }
    else {
      programArgs
    }

    for(testName <- testsToRun) {
      testMap.get(testName) match {
        case Some(test) =>
          println(s"Starting $testName")
          try {
            optionsManager.setTopName(testName)
            optionsManager.setTargetDirName(s"test_run_dir/$testName")
            if(test(optionsManager)) {
              successful += 1
            }
            else {
              errors += s"$testName: test error occurred"
            }
          }
          catch {
            case exception: Exception =>
              exception.printStackTrace()
              errors += s"$testName: exception ${exception.getMessage}"
            case t : Throwable =>
              errors += s"$testName: throwable ${t.getMessage}"
          }
        case _ =>
          errors += s"Bad Test name: $testName"
      }

    }
    if(successful > 0) {
      println(s"Tests passing: $successful")
    }
    if(errors.nonEmpty) {
      println("=" * 80)
      println(s"Errors: ${errors.length}: in the following tests")
      println(errors.mkString("\n"))
      println("=" * 80)
    }
  }
}
