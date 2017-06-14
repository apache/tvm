package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.TVMValue

class Module(private val handle: ModuleHandle) {
  private var entry: Function = null
  private var entryName = "__tvm_main__"

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
