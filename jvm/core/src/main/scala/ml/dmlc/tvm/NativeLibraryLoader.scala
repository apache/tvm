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

import java.io._
import org.slf4j.{LoggerFactory, Logger}

private[tvm] class NativeLibraryLoader

private[tvm] object NativeLibraryLoader {
  private val logger: Logger = LoggerFactory.getLogger(classOf[NativeLibraryLoader])

  private val libPathInJar = "/lib/native/"
  private val _tempDir: File =
    try {
      val tempDir = File.createTempFile("tvm", "")
      if (!tempDir.delete || !tempDir.mkdir) {
        throw new IOException(s"Couldn't create directory ${tempDir.getAbsolutePath}")
      }

      /*
       * Different cleanup strategies for Windows and Linux.
       * TODO: shutdown hook won't work on Windows
       */
      if (getUnifiedOSName != "Windows") {
        Runtime.getRuntime.addShutdownHook(new Thread() {
          override def run(): Unit = {
            for (f <- tempDir.listFiles()) {
              logger.info("Deleting {}", f.getAbsolutePath)
              if (!f.delete()) {
                logger.warn(s"Couldn't delete temporary file ${f.getAbsolutePath}")
              }
            }
            logger.info(s"Deleting ${tempDir.getAbsolutePath}")
            if (!tempDir.delete()) {
              logger.warn(s"Couldn't delete temporary directory ${tempDir.getAbsolutePath}")
            }
          }
        })
        tempDir
      } else {
        throw new RuntimeException("Windows not supported yet.")
      }
    } catch {
      case ex: IOException =>
        logger.error("Couldn't create temporary directory: {}", ex.getMessage)
        null
    }

  /**
   * Find the library as a resource in jar, copy it to a tempfile
   * and load it using System.load(). The name of the library has to be the
   * base name, it is mapped to the corresponding system name using
   * System.mapLibraryName(). e.g., the library "foo" is called "libfoo.so"
   * under Linux and "foo.dll" under Windows, but you just have to pass "foo" to
   * the loadLibrary().
   *
   * @param libname basename of the library
   * @throws UnsatisfiedLinkError if library cannot be founds
   */
  @throws(classOf[UnsatisfiedLinkError])
  def loadLibrary(libname: String) {
    val mappedLibname = System.mapLibraryName(libname)
    val loadLibname: String =
      if (mappedLibname.endsWith("dylib")) {
        logger.info("Replaced .dylib with .jnilib")
        mappedLibname.replace(".dylib", ".jnilib")
      } else {
        mappedLibname
      }
    logger.info(s"Attempting to load $loadLibname")
    extractResourceFileToTempDir(loadLibname, target => {
      logger.info("Loading library from {}", target.getPath)
      System.load(target.getPath)
    })
  }

  /**
   * Translate all those Windows to "Windows". ("Windows XP", "Windows Vista", "Windows 7", etc.)
   */
  private def unifyOSName(osname: String): String = {
    if (osname.startsWith("Windows")) {
      "Windows"
    }
    osname
  }

  private def getUnifiedOSName: String = {
    unifyOSName(System.getProperty("os.name"))
  }

  @throws(classOf[IOException])
  private def createTempFile(name: String): File = {
    new File(_tempDir + File.separator + name)
  }

  /**
   * Copies the resource file to a temp file and do an action.
   * @param filename source file name (in lib/native).
   * @param action callback function to deal with the copied file.
   */
  def extractResourceFileToTempDir(filename: String, action: File => Unit): Unit = {
    val libFileInJar = libPathInJar + filename
    val is: InputStream = getClass.getResourceAsStream(libFileInJar)
    if (is == null) {
      throw new UnsatisfiedLinkError(s"Couldn't find the resource $filename")
    }
    logger.info(s"Loading $filename from $libPathInJar")
    try {
      val tempfile = createTempFile(filename)
      logger.debug("tempfile.getPath() = {}", tempfile.getPath)
      val os: OutputStream = new FileOutputStream(tempfile)
      val savedTime: Long = System.currentTimeMillis
      val buf: Array[Byte] = new Array[Byte](8192)
      var len: Int = is.read(buf)
      while (len > 0) {
        os.write(buf, 0, len)
        len = is.read(buf)
      }
      os.flush()
      val lock = new FileInputStream(tempfile)
      os.close()
      val seconds = (System.currentTimeMillis - savedTime).toDouble / 1e3
      logger.info(s"Copying took $seconds seconds.")
      action(tempfile)
      lock.close()
    } catch {
      case io: IOException =>
        logger.error("Could not create the temp file: {}", io.toString)
      case ule: UnsatisfiedLinkError =>
        logger.error("Couldn't load copied link file: {}", ule.toString)
        throw ule
      case e: Throwable =>
        logger.error(e.getMessage, e)
        throw e
    }
  }
}
