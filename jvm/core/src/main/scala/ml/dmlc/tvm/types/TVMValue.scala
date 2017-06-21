package ml.dmlc.tvm.types

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.{NDArray, Module}
import ml.dmlc.tvm.types.TypeCode._

private[tvm] object TVMValue {
  implicit def fromInt(x: Int): TVMValue = new TVMValueLong(x)
  implicit def fromLong(x: Long): TVMValue = new TVMValueLong(x)
  implicit def fromDouble(x: Double): TVMValue = new TVMValueDouble(x)
  implicit def fromFloat(x: Float): TVMValue = new TVMValueDouble(x)
  implicit def fromString(x: String): TVMValue = new TVMValueString(x)
  implicit def fromModule(x: Module): TVMValue = new TVMValueModuleHandle(x.handle)
  implicit def fromNDArray(x: NDArray): TVMValue = new TVMValueNDArrayHandle(x.handle)
}

private[tvm] sealed class TVMValue(val argType: TypeCode) {
  // easy for JNI to use
  val argTypeId = argType.id
}

private[tvm] sealed class TVMValueLong(val value: Long) extends TVMValue(INT)
private[tvm] sealed class TVMValueDouble(val value: Double) extends TVMValue(FLOAT)
private[tvm] sealed class TVMValueString(val value: String) extends TVMValue(STR)
private[tvm] sealed class TVMValueModuleHandle(
  val value: ModuleHandle) extends TVMValue(MODULE_HANDLE)
private[tvm] sealed class TVMValueNDArrayHandle(
  val value: TVMArrayHandle) extends TVMValue(ARRAY_HANDLE)
private[tvm] sealed class TVMValueNull extends TVMValue(NULL)
