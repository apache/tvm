package ml.dmlc.tvm

import ml.dmlc.tvm.types._
import ml.dmlc.tvm.Base._

object HelloWorld {
  def main(args: Array[String]): Unit = {
    // Function.listGlobalFuncNames().foreach(println)
    checkCall(_LIB.tvmFuncCall(1L, Array(new TVMValueLong(1L), new TVMValueDouble(2.7)), null))
  }
}
