package ml.dmlc.tvm

import ml.dmlc.tvm.types._
import ml.dmlc.tvm.types.TVMValue._
import ml.dmlc.tvm.Base._

object HelloWorld {
  def main(args: Array[String]): Unit = {
    // Function.listGlobalFuncNames().foreach(println)
    // checkCall(_LIB.tvmFuncCall(1L, Array(new TVMValueLong(1L), new TVMValueDouble(2.7)), null))
    Function.functions.foreach { case (k, v) => println(s"$k: $v") }
    val ret = Function.functions("module._LoadFromFile")("myadd.so", "")
    println("Return type id = " + ret.argTypeId)
  }
}
