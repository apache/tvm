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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Function {
  final long handle;
  public final boolean isResident;
  private boolean isReleased = false;

  private static ThreadLocal<Map<String, Function>> apiFuncs
      = new ThreadLocal<Map<String, Function>>() {
          @Override
          protected Map<String, Function> initialValue() {
            return new HashMap<String, Function>();
          }
      };

  /**
   * Get registered function.
   * @param name full function name.
   * @return TVM function.
   */
  static Function getFunction(final String name) {
    Function fun = apiFuncs.get().get(name);
    if (fun == null) {
      for (String fullName : listGlobalFuncNames()) {
        if (fullName.equals(name)) {
          fun = getGlobalFunc(fullName, true, false);
          apiFuncs.get().put(name, fun);
        }
      }
    }
    return fun;
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

  @Override protected void finalize() throws Throwable {
    release();
    super.finalize();
  }

  /**
   * Release the Function.
   * <p>
   * We highly recommend you to do this manually since the GC strategy is lazy
   * and `finalize()` is not guaranteed to be called when GC happens.
   * </p>
   */
  public void release() {
    if (!isReleased) {
      if (!isResident) {
        Base.checkCall(Base._LIB.tvmFuncFree(handle));
        isReleased = true;
      }
    }
  }

  /**
   * Invoke the function.
   * @return the result.
   */
  public TVMValue invoke() {
    Base.RefTVMValue ret = new Base.RefTVMValue();
    Base.checkCall(Base._LIB.tvmFuncCall(handle, ret));
    return ret.value;
  }

  /**
   * Push argument to the function.
   * @param arg int argument.
   * @return this
   */
  public Function pushArg(int arg) {
    Base._LIB.tvmFuncPushArgLong(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg long argument.
   * @return this
   */
  public Function pushArg(long arg) {
    Base._LIB.tvmFuncPushArgLong(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg float argument.
   * @return this
   */
  public Function pushArg(float arg) {
    Base._LIB.tvmFuncPushArgDouble(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg double argument.
   * @return this
   */
  public Function pushArg(double arg) {
    Base._LIB.tvmFuncPushArgDouble(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg String argument.
   * @return this
   */
  public Function pushArg(String arg) {
    Base._LIB.tvmFuncPushArgString(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg NDArray.
   * @return this
   */
  public Function pushArg(NDArray arg) {
    Base._LIB.tvmFuncPushArgHandle(arg.handle, TypeCode.ARRAY_HANDLE.id);
    return this;
  }

  /**
   * Invoke function with arguments.
   * @param args Can be Integer, Long, Float, Double, String, NDArray.
   * @return the result.
   */
  public TVMValue call(Object... args) {
    for (Object arg : args) {
      if (arg instanceof Integer) {
        pushArg((Integer) arg);
      } else if (arg instanceof Long) {
        pushArg((Long) arg);
      } else if (arg instanceof Float) {
        pushArg((Float) arg);
      } else if (arg instanceof Double) {
        pushArg((Double) arg);
      } else if (arg instanceof String) {
        pushArg((String) arg);
      } else if (arg instanceof NDArray) {
        pushArg((NDArray) arg);
      } else {
        throw new IllegalArgumentException("Invalid argument: " + arg);
      }
    }
    return invoke();
  }
}
