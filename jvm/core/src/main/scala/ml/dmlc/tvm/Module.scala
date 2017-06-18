package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.{TVMValueModuleHandle, TVMValue}
import ml.dmlc.tvm.types.TypeCode._

class Module(private val handle: ModuleHandle) {
  private var entry: Function = null
  private val entryName = "__tvm_main__"

  override protected def finalize(): Unit = {
    checkCall(_LIB.tvmModFree(handle))
  }

  /**
   * Get the entry function
   * @return The entry function if exist
   */
  def entryFunc: Function = {
    if (entry == null) {
      entry = getFunction(entryName)
    }
    entry
  }

  /**
   * Get function from the module.
   * @param name The name of the function.
   * @param queryImports Whether also query modules imported by this module.
   * @return The result function.
   */
  def getFunction(name: String, queryImports: Boolean = false): Function = {
    val retHandle = new RefFunctionHandle()
    checkCall(_LIB.tvmModGetFunction(
      handle, name, if (queryImports) 1 else 0, retHandle))
    if (retHandle.value == 0) {
      throw new IllegalArgumentException("Module has no function " + name)
    }
    new Function(retHandle.value, false)
  }

  /**
   * Add module to the import list of current one.
   * @param module The other module.
   */
  def importModule(module: Module): Unit = {
    checkCall(_LIB.tvmModImport(handle, module.handle))
  }

  def apply(args: TVMValue*): TVMValue = {
    entryFunc(args: _*)
  }
}

object Module {
  /**
   * Load module from file
   * @param path The path to the module file.
   * @param fmt The format of the file,
   *            if not specified it will be inferred from suffix of the file.
   * @return The loaded module
   */
  def load(path: String, fmt: String = ""): Module = {
    // TODO
    val ret = Function.functions("module._LoadFromFile")("myadd.so", "")
    require(ret.argType == MODULE_HANDLE)
    val tvmValue = ret.asInstanceOf[TVMValueModuleHandle]
    new Module(tvmValue.value)
  }
}
