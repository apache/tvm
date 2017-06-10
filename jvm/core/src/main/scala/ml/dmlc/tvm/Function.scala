package ml.dmlc.tvm

import ml.dmlc.tvm.Base.{FunctionHandle, checkCall, _LIB}
import ml.dmlc.tvm.FunctionArgType.FunctionArgType

import scala.collection.mutable.ArrayBuffer

object Function {
  /**
   * Get list of global functions registered.
   * @return List of global functions names.
   */
  private[tvm] def listGlobalFuncNames(): IndexedSeq[String] = {
    val names = ArrayBuffer.empty[String]
    checkCall(_LIB.tvmFuncListGlobalNames(names))
    names.toIndexedSeq
  }
}

/**
 * Initialize the function with handle
 * @param handle the handle to the underlying function.
 * @param isGlobal Whether this is a global function in python
 */
class FunctionBase(private val handle: FunctionHandle, private val isGlobal: Boolean) {
  override protected def finalize(): Unit = {
    if (!isGlobal) {
      checkCall(_LIB.tvmFuncFree(handle))
    }
  }

  def apply(args: FunctionArg*) = ???
}

object FunctionArgType extends Enumeration {
  type FunctionArgType = Value
  val Float32 = Value(0, "float32")
  val Float64 = Value(1, "float64")
  val Float16 = Value(2, "float16")
  val UInt8 = Value(3, "uint8")
  val Int32 = Value(4, "int32")
}

private[tvm] object FunctionArg {
  implicit def fromInt(x: Int): FunctionArg = ???
  implicit def fromDouble(x: Double): FunctionArg = ???
  implicit def fromFloat(x: Float): FunctionArg = ???
}

private[tvm] class FunctionArg(ftype: FunctionArgType)  {
}
