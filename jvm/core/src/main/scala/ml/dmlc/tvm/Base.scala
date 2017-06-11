package ml.dmlc.tvm

import ml.dmlc.tvm.types.TVMValue

// import org.slf4j.{Logger, LoggerFactory}

private[tvm] object Base {
  // private val logger: Logger = LoggerFactory.getLogger("TVM_JVM")

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)
  class RefTVMValue(val value: TVMValue = null)

  type CPtrAddress = Long
  type FunctionHandle = CPtrAddress
  type TVMArrayHandle = CPtrAddress

  // FIXME
  val baseDir = System.getProperty("user.dir") + "/jvm/native"
  System.load(s"$baseDir/osx-x86_64-cpu/target/libtvm-osx-x86_64-cpu.jnilib")

  val _LIB = new LibInfo
  checkCall(_LIB.nativeLibInit())

  // TODO: shutdown hook won't work on Windows
  Runtime.getRuntime.addShutdownHook(new Thread() {
    override def run(): Unit = {
      notifyShutdown()
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

  // Notify MXNet about a shutdown
  private def notifyShutdown(): Unit = {
    // checkCall(_LIB.mxNotifyShutdown())
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
