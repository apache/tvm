package ml.dmlc.tvm

object HelloWorld {
  def main(args: Array[String]): Unit = {
    val baseDir = System.getProperty("user.dir") + "/native"
    System.load(s"$baseDir/osx-x86_64-cpu/target/libtvm-osx-x86_64-cpu.jnilib")
    // System.loadLibrary("jniExampleNative")
    val libInfo = new LibInfo
    libInfo.nativeLibInit()
  }
}
