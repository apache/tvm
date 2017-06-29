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

import java.io.File

import ml.dmlc.tvm.types.TVMValue
import org.slf4j.{LoggerFactory, Logger}

private[tvm] object Base {
  private val logger: Logger = LoggerFactory.getLogger("tvm4j")

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)
  class RefTVMValue(val value: TVMValue = null)

  type CPtrAddress = Long

  type FunctionHandle = CPtrAddress
  type RefFunctionHandle = RefLong

  type ModuleHandle = CPtrAddress
  type RefModuleHandle = RefLong

  type TVMArrayHandle = CPtrAddress
  type RefTVMArrayHandle = RefLong

  try {
    try {
      tryLoadLibraryOS("tvm4j")
    } catch {
      case e: UnsatisfiedLinkError =>
        logger.warn("TVM native library not found in path. " +
          "Copying native library from the archive. " +
          "Consider installing the library somewhere in the path " +
          "(for Windows: PATH, for Linux: LD_LIBRARY_PATH), " +
          "or specifying by Java cmd option -Djava.library.path=[lib path].")
        NativeLibraryLoader.loadLibrary("tvm4j")
    }
  } catch {
    case e: UnsatisfiedLinkError =>
      logger.error("Couldn't find native library tvm4j")
      throw e
  }

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadLibraryOS(libname: String): Unit = {
    try {
      logger.info(s"Try loading $libname from native path.")
      System.loadLibrary(libname)
    } catch {
      case e: UnsatisfiedLinkError =>
        val os = System.getProperty("os.name")
        // ref: http://lopica.sourceforge.net/os.html
        if (os.startsWith("Linux")) {
          tryLoadLibraryXPU(libname, "linux-x86_64")
        } else if (os.startsWith("Mac")) {
          tryLoadLibraryXPU(libname, "osx-x86_64")
        } else {
          // TODO(yizhi) support windows later
          throw new UnsatisfiedLinkError()
        }
    }
  }

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadLibraryXPU(libname: String, arch: String): Unit = {
    try {
      // try gpu first
      logger.info(s"Try loading $libname-$arch-gpu from native path.")
      System.loadLibrary(s"$libname-$arch-gpu")
    } catch {
      case e: UnsatisfiedLinkError =>
        logger.info(s"Try loading $libname-$arch-cpu from native path.")
        System.loadLibrary(s"$libname-$arch-cpu")
    }
  }

  val _LIB = new LibInfo
  val tvmLibFilename = System.getProperty("libtvm.so.path")
  if (tvmLibFilename == null || !new File(tvmLibFilename).isFile
        || _LIB.nativeLibInit(tvmLibFilename) != 0) {
    NativeLibraryLoader.extractResourceFileToTempDir("libtvm_runtime.so", target => {
      logger.info("Loading tvm runtime from {}", target.getPath)
      checkCall(_LIB.nativeLibInit(target.getPath))
    })
  }

  Runtime.getRuntime.addShutdownHook(new Thread() {
    override def run(): Unit = {
      _LIB.shutdown()
    }
  })

  // helper function definitions
  /**
   * Check the return value of C API call
   *
   * This function will raise exception when error occurs.
   * Wrap every API call with this function
   * @param ret return value from API calls
   */
  def checkCall(ret: Int): Unit = {
    if (ret != 0) {
      throw new TVMError(_LIB.tvmGetLastError())
    }
  }

  // Convert ctypes returned doc string information into parameters docstring.
  def ctypes2docstring(
                        argNames: Seq[String],
                        argTypes: Seq[String],
                        argDescs: Seq[String]): String = {

    val params =
      (argNames zip argTypes zip argDescs) map { case ((argName, argType), argDesc) =>
        val desc = if (argDesc.isEmpty) "" else s"\n$argDesc"
        s"$argName : $argType$desc"
      }
    s"Parameters\n----------\n${params.mkString("\n")}\n"
  }
}

private[tvm] class TVMError(val err: String) extends Exception(err)
