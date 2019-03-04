// See LICENSE.txt for license details.
package utils

import scala.collection.mutable.ArrayBuffer
import chisel3.iotesters._

object TutorialRunner {
  def apply(section: String, tutorialMap: Map[String, TesterOptionsManager => Boolean], args: Array[String]): Unit = {
    var successful = 0
    val errors = new ArrayBuffer[String]

    val optionsManager = new TesterOptionsManager()
    optionsManager.doNotExitOnHelp()

    optionsManager.parse(args)

    val programArgs = optionsManager.commonOptions.programArgs

    if(programArgs.isEmpty) {
      println("Available tutorials")
      for(x <- tutorialMap.keys) {
        println(x)
      }
      println("all")
      System.exit(0)
    }

    val problemsToRun = if(programArgs.exists(x => x.toLowerCase() == "all")) {
      tutorialMap.keys
    }
    else {
      programArgs
    }

    for(testName <- problemsToRun) {
      tutorialMap.get(testName) match {
        case Some(test) =>
          println(s"Starting tutorial $testName")
          try {
            optionsManager.setTopName(testName)
            optionsManager.setTargetDirName(s"ip/$section/${testName.toLowerCase()}")
            if(test(optionsManager)) {
              successful += 1
            }
            else {
              errors += s"Tutorial $testName: test error occurred"
            }
          }
          catch {
            case exception: Exception =>
              exception.printStackTrace()
              errors += s"Tutorial $testName: exception ${exception.getMessage}"
            case t : Throwable =>
              errors += s"Tutorial $testName: throwable ${t.getMessage}"
          }
        case _ =>
          errors += s"Bad tutorial name: $testName"
      }

    }
    if(successful > 0) {
      println(s"Tutorials passing: $successful")
    }
    if(errors.nonEmpty) {
      println("=" * 80)
      println(s"Errors: ${errors.length}: in the following tutorials")
      println(errors.mkString("\n"))
      println("=" * 80)
      System.exit(1)
    }
  }
}
