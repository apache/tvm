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

package org.apache.tvm;

import java.util.HashMap;
import java.util.Map;

/**
 * Container of compiled functions of TVM.
 */
public class Module extends TVMObject {
  private static ThreadLocal<Map<String, Function>> apiFuncs
      = new ThreadLocal<Map<String, Function>>() {
        @Override
        protected Map<String, Function> initialValue() {
          return new HashMap<String, Function>();
        }
      };

  private static Function getApi(String name) {
    Function func = apiFuncs.get().get(name);
    if (func == null) {
      func = Function.getFunction(name);
      apiFuncs.get().put(name, func);
    }
    return func;
  }

  Module(long handle) {
    super(handle, TypeIndex.kTVMFFIModule);
  }

  private Function entry = null;
  private final String entryName = "main";


  /**
   * Easy for user to get the instance from returned TVMValue.
   * @return this
   */
  @Override public Module asModule() {
    return this;
  }

  /**
   * Get the entry function.
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
    TVMValue ret = getApi("ffi.ModuleGetFunction")
        .pushArg(this).pushArg(name).pushArg(queryImports ? 1 : 0).invoke();
    return ret.asFunction();
  }

  public Function getFunction(String name) {
    return getFunction(name, false);
  }

  /**
   * Add module to the import list of current one.
   * @param module The other module.
   */
  public void importModule(Module module) {
    getApi("ffi.ModuleImportModule")
        .pushArg(this).pushArg(module).invoke();
  }

  /**
   * Get type key of the module.
   * @return type key of the module.
   */
  public String typeKey() {
    return getApi("ffi.ModuleGetTypeKind").pushArg(this).invoke().asString();
  }

  /**
   * Load module from file.
   * @param path The path to the module file.
   * @param fmt The format of the file,
   *            if not specified it will be inferred from suffix of the file.
   * @return The loaded module
   */
  public static Module load(String path, String fmt) {
    TVMValue ret = getApi("ffi.ModuleLoadFromFile").pushArg(path).pushArg(fmt).invoke();
    return ret.asModule();
  }

  public static Module load(String path) {
    return load(path, "");
  }

  /**
   * Whether module runtime is enabled for target,
   * e.g., The following code checks if cuda is enabled.
   * Module.enabled("cuda")
   * @param target The target device type.
   * @return Whether runtime is enabled.
   */
  public static boolean enabled(String target) {
    TVMValue ret = getApi("runtime.RuntimeEnabled").pushArg(target).invoke();
    return ret.asLong() != 0;
  }
}
