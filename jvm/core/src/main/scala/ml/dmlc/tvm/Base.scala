package ml.dmlc.tvm

import ml.dmlc.tvm.types.TVMValue

private[tvm] object Base {
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
      tryLoadLibraryOS("tvm-jvm")
    } catch {
      case e: UnsatisfiedLinkError =>
        Console.err.println("[WARN] TVM native library not found in path. " +
          "Copying native library from the archive. " +
          "Consider installing the library somewhere in the path " +
          "(for Windows: PATH, for Linux: LD_LIBRARY_PATH), " +
          "or specifying by Java cmd option -Djava.library.path=[lib path].")
        NativeLibraryLoader.loadLibrary("tvm-jvm")
    }
  } catch {
    case e: UnsatisfiedLinkError =>
      Console.err.println("[ERROR] Couldn't find native library tvm")
      throw e
  }

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadLibraryOS(libname: String): Unit = {
    try {
      Console.err.println(s"Try loading $libname from native path.")
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
      Console.err.println(s"Try loading $libname-$arch-gpu from native path.")
      System.loadLibrary(s"$libname-$arch-gpu")
    } catch {
      case e: UnsatisfiedLinkError =>
        Console.err.println(s"Try loading $libname-$arch-cpu from native path.")
        System.loadLibrary(s"$libname-$arch-cpu")
    }
  }

  val _LIB = new LibInfo
  checkCall(_LIB.nativeLibInit())

  Runtime.getRuntime.addShutdownHook(new Thread() {
    override def run(): Unit = {
      // TODO
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
