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
public class Function extends TVMObject {
  /**
   * Get registered function.
   * @param name full function name.
   * @return TVM function.
   */
  public static Function getFunction(final String name) {
    return getGlobalFunc(name, true);
  }

  /**
   * Get list of global functions registered.
   * @return List of global functions names.
   */
  private static List<String> listGlobalFuncNames() {
    List<String> names = new ArrayList<String>();
    Base.checkCall(Base._LIB.tvmFFIFunctionListGlobalNames(names));
    return Collections.unmodifiableList(names);
  }

  /**
   * Get a global function by name.
   * @param name The name of the function.
   * @param allowMissing Whether allow missing function or raise an error.
   * @return The function to be returned, None if function is missing.
   */
  private static Function getGlobalFunc(String name, boolean allowMissing) {
    Base.RefLong handle = new Base.RefLong();
    Base.checkCall(Base._LIB.tvmFFIFunctionGetGlobal(name, handle));
    if (handle.value != 0) {
      return new Function(handle.value);
    } else {
      if (allowMissing) {
        return null;
      } else {
        throw new IllegalArgumentException("Cannot find global function " + name);
      }
    }
  }

  Function(long handle) {
    super(handle, TypeIndex.kTVMFFIFunction);
  }

  /**
   * Easy for user to get the instance from returned TVMValue.
   * @return this
   */
  @Override public Function asFunction() {
    return this;
  }

  /**
   * Invoke the function.
   * @return the result.
   */
  public TVMValue invoke() {
    Base.RefTVMValue ret = new Base.RefTVMValue();
    Base.checkCall(Base._LIB.tvmFFIFunctionCall(handle, ret));
    return ret.value;
  }

  /**
   * Push argument to the function.
   * @param arg int argument.
   * @return this
   */
  public Function pushArg(int arg) {
    Base._LIB.tvmFFIFunctionPushArgLong(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg long argument.
   * @return this
   */
  public Function pushArg(long arg) {
    Base._LIB.tvmFFIFunctionPushArgLong(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg float argument.
   * @return this
   */
  public Function pushArg(float arg) {
    Base._LIB.tvmFFIFunctionPushArgDouble(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg double argument.
   * @return this
   */
  public Function pushArg(double arg) {
    Base._LIB.tvmFFIFunctionPushArgDouble(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg String argument.
   * @return this
   */
  public Function pushArg(String arg) {
    Base._LIB.tvmFFIFunctionPushArgString(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Tensor.
   * @return this
   */
  public Function pushArg(TensorBase arg) {
    if (arg instanceof Tensor) {
      Base._LIB.tvmFFIFunctionPushArgHandle(((Tensor) arg).handle, TypeIndex.kTVMFFITensor);
    } else {
      Base._LIB.tvmFFIFunctionPushArgHandle(arg.dltensorHandle, TypeIndex.kTVMFFIDLTensorPtr);
    }
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Module.
   * @return this
   */
  public Function pushArg(Module arg) {
    Base._LIB.tvmFFIFunctionPushArgHandle(arg.handle, TypeIndex.kTVMFFIModule);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Function.
   * @return this
   */
  public Function pushArg(Function arg) {
    Base._LIB.tvmFFIFunctionPushArgHandle(arg.handle, TypeIndex.kTVMFFIFunction);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg bytes.
   * @return this
   */
  public Function pushArg(byte[] arg) {
    Base._LIB.tvmFFIFunctionPushArgBytes(arg);
    return this;
  }

  /**
   * Push argument to the function.
   * @param arg Device.
   * @return this
   */
  public Function pushArg(Device arg) {
    Base._LIB.tvmFFIFunctionPushArgDevice(arg);
    return this;
  }

  /**
   * Invoke function with arguments.
   * @param args Can be Integer, Long, Float, Double, String, Tensor.
   * @return the result.
   */
  public TVMValue call(Object... args) {
    for (Object arg : args) {
      pushArgToStack(arg);
    }
    return invoke();
  }

  private static void pushArgToStack(Object arg) {
    if (arg instanceof TensorBase) {
      TensorBase nd = (TensorBase) arg;
      if (nd instanceof Tensor) {
        Base._LIB.tvmFFIFunctionPushArgHandle(((Tensor) nd).handle, TypeIndex.kTVMFFITensor);
      } else {
        Base._LIB.tvmFFIFunctionPushArgHandle(nd.dltensorHandle, TypeIndex.kTVMFFIDLTensorPtr);
      }
    } else if (arg instanceof TVMObject) {
      TVMObject obj = (TVMObject) arg;
      Base._LIB.tvmFFIFunctionPushArgHandle(obj.handle, obj.typeIndex);
    } else if (arg instanceof Integer) {
      Base._LIB.tvmFFIFunctionPushArgLong((Integer) arg);
    } else if (arg instanceof Long) {
      Base._LIB.tvmFFIFunctionPushArgLong((Long) arg);
    } else if (arg instanceof Float) {
      Base._LIB.tvmFFIFunctionPushArgDouble((Float) arg);
    } else if (arg instanceof Double) {
      Base._LIB.tvmFFIFunctionPushArgDouble((Double) arg);
    } else if (arg instanceof String) {
      Base._LIB.tvmFFIFunctionPushArgString((String) arg);
    } else if (arg instanceof byte[]) {
      Base._LIB.tvmFFIFunctionPushArgBytes((byte[]) arg);
    } else if (arg instanceof Device) {
      Base._LIB.tvmFFIFunctionPushArgDevice((Device) arg);
    } else if (arg instanceof TVMValueBytes) {
      byte[] bytes = ((TVMValueBytes) arg).value;
      Base._LIB.tvmFFIFunctionPushArgBytes(bytes);
    } else if (arg instanceof TVMValueString) {
      String str = ((TVMValueString) arg).value;
      Base._LIB.tvmFFIFunctionPushArgString(str);
    } else if (arg instanceof TVMValueDouble) {
      double value = ((TVMValueDouble) arg).value;
      Base._LIB.tvmFFIFunctionPushArgDouble(value);
    } else if (arg instanceof TVMValueLong) {
      long value = ((TVMValueLong) arg).value;
      Base._LIB.tvmFFIFunctionPushArgLong(value);
    } else if (arg instanceof TVMValueNull) {
      Base._LIB.tvmFFIFunctionPushArgHandle(0, TypeIndex.kTVMFFINone);
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
    Base.checkCall(Base._LIB.tvmFFIFunctionCreateFromCallback(function, createdFuncHandleRef));
    int ioverride = override ? 1 : 0;
    Base.checkCall(Base._LIB.tvmFFIFunctionSetGlobal(name, createdFuncHandleRef.value, ioverride));
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
    Base.checkCall(Base._LIB.tvmFFIFunctionCreateFromCallback(function, createdFuncHandleRef));
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
