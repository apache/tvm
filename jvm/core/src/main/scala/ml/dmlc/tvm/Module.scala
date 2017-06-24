/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types._
import ml.dmlc.tvm.types.TypeCode._

// scalastyle:off finalize
class Module(private[tvm] val handle: ModuleHandle) {
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
// scalastyle:on finalize

object Module {
  private val function = Function.initAPI(
    name => name.startsWith("module."),
    name => name.substring("module.".length))

  def apply(name: String): Function = function(name)

  /**
   * Load module from file
   * @param path The path to the module file.
   * @param fmt The format of the file,
   *            if not specified it will be inferred from suffix of the file.
   * @return The loaded module
   */
  def load(path: String, fmt: String = ""): Module = {
    val ret = Module("_LoadFromFile")(path, fmt)
    require(ret.argType == MODULE_HANDLE)
    val tvmValue = ret.asInstanceOf[TVMValueModuleHandle]
    new Module(tvmValue.value)
  }

  /**
   * Whether module runtime is enabled for target,
   * e.g., The following code checks if gpu is enabled.
   * Module.enabled("gpu")
   * @param target The target device type.
   * @return Whether runtime is enabled.
   */
  def enabled(target: String): Boolean = {
    val ret = Module("_Enabled")(target)
    if (ret.asInstanceOf[TVMValueLong].value == 0) false else true
  }
}
