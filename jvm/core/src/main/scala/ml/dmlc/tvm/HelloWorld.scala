package ml.dmlc.tvm

object HelloWorld {
  def main(args: Array[String]): Unit = {
    Function.listGlobalFuncNames().foreach(println)
  }
}
