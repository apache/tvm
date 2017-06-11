package ml.dmlc.tvm.types

// Type code used in API calls
object TypeCode extends Enumeration {
  type TypeCode = Value
  val INT = Value(0)
  val UINT = Value(1)
  val FLOAT = Value(2)
  val HANDLE = Value(3)
  val NULL = Value(4)
  val TVM_TYPE = Value(5)
  val TVM_CONTEXT = Value(6)
  val ARRAY_HANDLE = Value(7)
  val NODE_HANDLE = Value(8)
  val MODULE_HANDLE = Value(9)
  val FUNC_HANDLE = Value(10)
  val STR = Value(11)
  val BYTES = Value(12)
}
