package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.TVMValue

import scala.collection.mutable.ArrayBuffer

object Function {
  val functions = initFunctionAPI()

  /**
   * Get list of global functions registered.
   * @return List of global functions names.
   */
  private[tvm] def listGlobalFuncNames(): IndexedSeq[String] = {
    val names = ArrayBuffer.empty[String]
    checkCall(_LIB.tvmFuncListGlobalNames(names))
    names.toIndexedSeq
  }

  /**
   * Get a global function by name.
   * @param name The name of the global function.
   * @param isGlobal Whether it is a global function. FIXME: not a good name
   * @param allowMissing Whether allow missing function or raise an error.
   * @return The function to be returned, None if function is missing.
   */
  private def getGlobalFunc(name: String, isGlobal: Boolean = false,
                            allowMissing: Boolean = false): Option[Function] = {
    val handle = new RefFunctionHandle()
    checkCall(_LIB.tvmFuncGetGlobal(name, handle))
    if (handle.value != 0) {
      Some(new Function(handle.value, isGlobal))
    } else {
      if (allowMissing) {
        None
      } else {
        throw new IllegalArgumentException("Cannot find global function " + name)
      }
    }
  }


  private def initFunctionAPI(): Map[String, Function] = {
    listGlobalFuncNames().map { funcName =>
      (funcName, getGlobalFunc(funcName, isGlobal = true).get)
    }.toMap
  }
}

/**
 * Initialize the function with handle
 * @param handle the handle to the underlying function.
 * @param isGlobal Whether this is a global function in python
 */
class Function(private val handle: FunctionHandle, private val isGlobal: Boolean) {
  override protected def finalize(): Unit = {
    if (!isGlobal) {
      checkCall(_LIB.tvmFuncFree(handle))
    }
  }

  def apply(args: TVMValue*): TVMValue = {
    val ret = new RefTVMValue()
    checkCall(_LIB.tvmFuncCall(handle, args.toArray, ret))
    ret.value
    /*
    temp_args = []
    values, tcodes, num_args = _make_tvm_args(args, temp_args)
    ret_val = TVMValue()
    ret_tcode = ctypes.c_int()
    check_call(_LIB.TVMFuncCall(
      self.handle, values, tcodes, ctypes.c_int(num_args),
      ctypes.byref(ret_val), ctypes.byref(ret_tcode)))
    _ = temp_args
    _ = args
    return RETURN_SWITCH[ret_tcode.value](ret_val)
    */
  }
}

object FunctionArgType extends Enumeration {
  type FunctionArgType = Value
  val Float32 = Value(0, "float32")
  val Float64 = Value(1, "float64")
  val Float16 = Value(2, "float16")
  val UInt8 = Value(3, "uint8")
  val Int32 = Value(4, "int32")
}
