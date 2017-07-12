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

package ml.dmlc.tvm;

import ml.dmlc.tvm.Base.*;

import java.util.Map;

public class Module {
  private static final Map<String, Function> FUNCTIONS = Function.initAPI(
    new Function.InitAPINameFilter() {
      @Override public boolean accept(String name) {
        return name != null && name.startsWith("module.");
      }
    }, new Function.InitAPINameGenerator() {
      @Override public String generate(String name) {
        return name.substring("module.".length());
      }
    });

  public final long handle;
  public Module(long handle) {
    this.handle = handle;
  }

  private Function entry = null;
  private final String entryName = "__tvm_main__";

  @Override protected void finalize() {
    Base.checkCall(Base._LIB.tvmModFree(handle));
  }

  /**
   * Get the entry function
   * @return The entry function if exist
   */
  public Function entryFunc() {
    if (entry == null) {
      entry = getFunction(entryName);
    }
    return entry;
  }

  /**
   * Get function from the module.
   * @param name The name of the function.
   * @param queryImports Whether also query modules imported by this module.
   * @return The result function.
   */
  public Function getFunction(String name, boolean queryImports) {
    RefLong retHandle = new RefLong();
    Base.checkCall(Base._LIB.tvmModGetFunction(
      handle, name, queryImports ? 1 : 0, retHandle));
    if (retHandle.value == 0) {
      throw new IllegalArgumentException("Module has no function " + name);
    }
    return new Function(retHandle.value, false);
  }

  public Function getFunction(String name) {
    return getFunction(name, false);
  }

  /**
   * Add module to the import list of current one.
   * @param module The other module.
   */
  public void importModule(Module module) {
    Base.checkCall(Base._LIB.tvmModImport(handle, module.handle));
  }

  /**
   * Load module from file
   * @param path The path to the module file.
   * @param fmt The format of the file,
   *            if not specified it will be inferred from suffix of the file.
   * @return The loaded module
   */
  public static Module load(String path, String fmt) {
    TVMValue ret = FUNCTIONS.get("_LoadFromFile").pushArg(path).pushArg(fmt).invoke();
    assert(ret.typeCode == TypeCode.MODULE_HANDLE);
    return ret.asModule();
  }

  public static Module load(String path) {
    return load(path, "");
  }

  /**
   * Whether module runtime is enabled for target,
   * e.g., The following code checks if gpu is enabled.
   * Module.enabled("gpu")
   * @param target The target device type.
   * @return Whether runtime is enabled.
   */
  public static boolean enabled(String target) {
    TVMValue ret = FUNCTIONS.get("_Enabled").pushArg(target).invoke();
    return ret.asLong() != 0;
  }
}
