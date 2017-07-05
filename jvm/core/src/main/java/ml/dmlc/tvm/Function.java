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

import ml.dmlc.tvm.types.TVMValue;
import ml.dmlc.tvm.types.TypeCode;

import java.util.*;

public class Function {
  public final long handle;
  public final boolean isResident;

  static interface InitAPINameFilter {
    public boolean accept(String name);
  }

  static interface InitAPINameGenerator {
    public String generate(String name);
  }

  static Map<String, Function> initAPI(InitAPINameFilter filter, InitAPINameGenerator generator) {
    Map<String, Function> functions = new HashMap<String, Function>();
    for (String fullName : listGlobalFuncNames()) {
      if (filter.accept(fullName)) {
        String funcName = generator.generate(fullName);
        functions.put(funcName, getGlobalFunc(fullName, true, false));
      }
    }
    return Collections.unmodifiableMap(functions);
  }

  /**
   * Get list of global functions registered.
   * @return List of global functions names.
   */
  private static List<String> listGlobalFuncNames() {
    List<String> names = new ArrayList<String>();
    Base.checkCall(Base._LIB.tvmFuncListGlobalNames(names));
    return Collections.unmodifiableList(names);
  }

  /**
   * Get a global function by name.
   * @param name The name of the function.
   * @param isResident Whether it is a global 'resident' function.
   * @param allowMissing Whether allow missing function or raise an error.
   * @return The function to be returned, None if function is missing.
   */
  private static Function getGlobalFunc(String name, boolean isResident, boolean allowMissing) {
    Base.RefLong handle = new Base.RefLong();
    Base.checkCall(Base._LIB.tvmFuncGetGlobal(name, handle));
    if (handle.value != 0) {
      return new Function(handle.value, isResident);
    } else {
      if (allowMissing) {
        return null;
      } else {
        throw new IllegalArgumentException("Cannot find global function " + name);
      }
    }
  }

  /**
   * Initialize the function with handle
   * @param handle the handle to the underlying function.
   * @param isResident Whether this is a resident function in jvm
   */
  public Function(long handle, boolean isResident) {
    this.handle = handle;
    this.isResident = isResident;
  }

  @Override protected void finalize() {
    if (!isResident) {
      Base.checkCall(Base._LIB.tvmFuncFree(handle));
    }
  }

  public TVMValue invoke() {
    Base.RefTVMValue ret = new Base.RefTVMValue();
    Base.checkCall(Base._LIB.tvmFuncCall(handle, ret));
    return ret.value;
  }

  public Function pushArg(int arg) {
    Base._LIB.tvmFuncPushArgLong(arg);
    return this;
  }

  public Function pushArg(long arg) {
    Base._LIB.tvmFuncPushArgLong(arg);
    return this;
  }

  public Function pushArg(float arg) {
    Base._LIB.tvmFuncPushArgDouble(arg);
    return this;
  }

  public Function pushArg(double arg) {
    Base._LIB.tvmFuncPushArgDouble(arg);
    return this;
  }

  public Function pushArg(String arg) {
    Base._LIB.tvmFuncPushArgString(arg);
    return this;
  }

  public Function pushArg(NDArray arg) {
    Base._LIB.tvmFuncPushArgHandle(arg.handle, TypeCode.ARRAY_HANDLE.id);
    return this;
  }
}
