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
public class Module extends TVMValue {
  public final long handle;
  private boolean isReleased = false;

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
      func = Function.getFunction("runtime." + name);
      apiFuncs.get().put(name, func);
    }
    return func;
  }

  Module(long handle) {
    super(ArgTypeCode.MODULE_HANDLE);
    this.handle = handle;
  }

  private Function entry = null;
  private final String entryName = "__tvm_main__";

  @Override protected void finalize() throws Throwable {
    release();
    super.finalize();
  }

  /**
   * Easy for user to get the instance from returned TVMValue.
   * @return this
   */
  @Override public Module asModule() {
    return this;
  }

  @Override long asHandle() {
    return handle;
  }

  /**
   * Release the Module.
   * <p>
   * We highly recommend you to do this manually since the GC strategy is lazy.
   * </p>
   */
  @Override public void release() {
    if (!isReleased) {
      Base.checkCall(Base._LIB.tvmModFree(handle));
      isReleased = true;
    }
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
    Base.RefLong retHandle = new Base.RefLong();
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
   * Get type key of the module.
   * @return type key of the module.
   */
  public String typeKey() {
    return getApi("ModuleGetTypeKey").pushArg(this).invoke().asString();
  }

  /**
   * Load module from file.
   * @param path The path to the module file.
   * @param fmt The format of the file,
   *            if not specified it will be inferred from suffix of the file.
   * @return The loaded module
   */
  public static Module load(String path, String fmt) {
    TVMValue ret = getApi("ModuleLoadFromFile").pushArg(path).pushArg(fmt).invoke();
    assert ret.typeCode == ArgTypeCode.MODULE_HANDLE;
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
    TVMValue ret = getApi("RuntimeEnabled").pushArg(target).invoke();
    return ret.asLong() != 0;
  }
}
