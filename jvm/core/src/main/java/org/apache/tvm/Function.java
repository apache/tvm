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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * TVM Packed Function.
 */
public class Function extends TVMValue {
  final long handle;
  public final boolean isResident;
  private boolean isReleased = false;

  /**
   * Get registered function.
   * @param name full function name.
   * @return TVM function.
   */
  public static Function getFunction(final String name) {
    for (String fullName : listGlobalFuncNames()) {
      if (fullName.equals(name)) {
        return getGlobalFunc(fullName, true, false);
      }
    }
    return null;
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
   * Initialize the function with handle.
   * @param handle the handle to the underlying function.
   * @param isResident Whether this is a resident function in jvm
   */
  Function(long handle, boolean isResident) {
    super(ArgTypeCode.FUNC_HANDLE);
    this.handle = handle;
    this.isResident = isResident;
  }

  Function(long handle) {
    this(handle, false);
  }

  @Override protected void finalize() throws Throwable {
    release();
    super.finalize();
  }

  /**
   * Easy for user to get the instance from returned TVMValue.
   * @return this
   */
  @Override public Function asFunction() {
    return this;
  }

  @Override long asHandle() {
    return handle;
  }

  /**
   * Release the Function.
   * <p>
   * We highly recommend you to do this manually since the GC strategy is lazy.
   * </p>
   */
  @Override public void release() {
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
  public Function pushArg(NDArrayBase arg) {
    int id = arg.isView ? ArgTypeCode.ARRAY_HANDLE.id : ArgTypeCode.NDARRAY_CONTAINER.id;
    Base._LIB.tvmFuncPushArgHandle(arg.handle, id);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Module.
   * @return this
   */
  public Function pushArg(Module arg) {
    Base._LIB.tvmFuncPushArgHandle(arg.handle, ArgTypeCode.MODULE_HANDLE.id);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Function.
   * @return this
   */
  public Function pushArg(Function arg) {
    Base._LIB.tvmFuncPushArgHandle(arg.handle, ArgTypeCode.FUNC_HANDLE.id);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg bytes.
   * @return this
   */
  public Function pushArg(byte[] arg) {
    Base._LIB.tvmFuncPushArgBytes(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Device.
   * @return this
   */
  public Function pushArg(Device arg) {
    Base._LIB.tvmFuncPushArgDevice(arg);
    return this;
  }

  /**
   * Invoke function with arguments.
   * @param args Can be Integer, Long, Float, Double, String, NDArray.
   * @return the result.
   */
  public TVMValue call(Object... args) {
    for (Object arg : args) {
      pushArgToStack(arg);
    }
    return invoke();
  }

  private static void pushArgToStack(Object arg) {
    if (arg instanceof Integer) {
      Base._LIB.tvmFuncPushArgLong((Integer) arg);
    } else if (arg instanceof Long) {
      Base._LIB.tvmFuncPushArgLong((Long) arg);
    } else if (arg instanceof Float) {
      Base._LIB.tvmFuncPushArgDouble((Float) arg);
    } else if (arg instanceof Double) {
      Base._LIB.tvmFuncPushArgDouble((Double) arg);
    } else if (arg instanceof String) {
      Base._LIB.tvmFuncPushArgString((String) arg);
    } else if (arg instanceof byte[]) {
      Base._LIB.tvmFuncPushArgBytes((byte[]) arg);
    } else if (arg instanceof NDArrayBase) {
      NDArrayBase nd = (NDArrayBase) arg;
      int id = nd.isView ? ArgTypeCode.ARRAY_HANDLE.id : ArgTypeCode.NDARRAY_CONTAINER.id;
      Base._LIB.tvmFuncPushArgHandle(nd.handle, id);
    } else if (arg instanceof Module) {
      Base._LIB.tvmFuncPushArgHandle(((Module) arg).handle, ArgTypeCode.MODULE_HANDLE.id);
    } else if (arg instanceof Function) {
      Base._LIB.tvmFuncPushArgHandle(((Function) arg).handle, ArgTypeCode.FUNC_HANDLE.id);
    } else if (arg instanceof Device) {
      Base._LIB.tvmFuncPushArgDevice((Device) arg);
    } else if (arg instanceof TVMValue) {
      TVMValue tvmArg = (TVMValue) arg;
      switch (tvmArg.typeCode) {
        case UINT:
        case INT:
          Base._LIB.tvmFuncPushArgLong(tvmArg.asLong());
          break;
        case FLOAT:
          Base._LIB.tvmFuncPushArgDouble(tvmArg.asDouble());
          break;
        case STR:
          Base._LIB.tvmFuncPushArgString(tvmArg.asString());
          break;
        case BYTES:
          Base._LIB.tvmFuncPushArgBytes(tvmArg.asBytes());
          break;
        case HANDLE:
        case ARRAY_HANDLE:
        case MODULE_HANDLE:
        case FUNC_HANDLE:
          Base._LIB.tvmFuncPushArgHandle(tvmArg.asHandle(), tvmArg.typeCode.id);
          break;
        default:
          throw new IllegalArgumentException("Invalid argument: " + arg);
      }
    } else {
      throw new IllegalArgumentException("Invalid argument: " + arg);
    }
  }

  public static interface Callback {
    public Object invoke(TVMValue... args);
  }

  /**
   * Register user-defined global function.
   * @param name The function name.
   * @param function The function to be registered.
   * @param override Whether override existing entry.
   */
  public static void register(String name, Callback function, boolean override) {
    Base.RefLong createdFuncHandleRef = new Base.RefLong();
    Base.checkCall(Base._LIB.tvmFuncCreateFromCFunc(function, createdFuncHandleRef));
    int ioverride = override ? 1 : 0;
    Base.checkCall(Base._LIB.tvmFuncRegisterGlobal(name, createdFuncHandleRef.value, ioverride));
  }

  /**
   * Register user-defined global function, do not override existing entry.
   * @param name The function name.
   * @param function The function to be registered.
   */
  public static void register(String name, Callback function) {
    register(name, function, false);
  }

  /**
   * Convert a Java function to TVM function.
   * @param function Java function.
   * @return TVM function.
   */
  public static Function convertFunc(Callback function) {
    Base.RefLong createdFuncHandleRef = new Base.RefLong();
    Base.checkCall(Base._LIB.tvmFuncCreateFromCFunc(function, createdFuncHandleRef));
    return new Function(createdFuncHandleRef.value);
  }

  private static Object invokeRegisteredCbFunc(Callback cb, TVMValue[] args) {
    if (cb == null) {
      System.err.println("[ERROR] Failed to get registered function");
      return null;
    }
    return cb.invoke(args);
  }
}
