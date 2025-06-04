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

import org.apache.tvm.NativeLibraryLoader.Action;

import java.io.File;
import java.io.IOException;

/**
 * Initializing methods and types.
 */
final class Base {

  /**
   * Hold Long reference for JNI.
   */
  public static class RefLong {
    public final long value;

    public RefLong(final long value) {
      this.value = value;
    }

    public RefLong() {
      this(0L);
    }
  }

  /**
   * Hold TVMValue reference for JNI.
   */
  public static class RefTVMValue {
    public final TVMValue value;

    public RefTVMValue(TVMValue value) {
      this.value = value;
    }

    public RefTVMValue() {
      this(null);
    }
  }

  public static final LibInfo _LIB = new LibInfo();

  static {
    boolean loadNativeRuntimeLib = true;
    try {
      try {
        tryLoadLibraryOS("tvm4j");
      } catch (UnsatisfiedLinkError e) {
        System.err.println("[WARN] TVM native library not found in path. "
            + "Copying native library from the archive. "
            + "Consider installing the library somewhere in the path "
            + "(for Windows: PATH, for Linux: LD_LIBRARY_PATH), "
            + "or specifying by Java cmd option -Djava.library.path=[lib path].");
        NativeLibraryLoader.loadLibrary("tvm4j");
      }
    } catch (Throwable e) {
      System.err.println("[WARN] Couldn't find native library tvm4j.");
      e.printStackTrace();
      System.err.println("Try to load tvm4j (runtime packed version) ...");
      try {
        System.loadLibrary("tvm4j_runtime_packed");
        // if tvm runtime is packed in libtvm4j, we do not need to dlopen libtvm_runtime.so.
        loadNativeRuntimeLib = false;
      } catch (UnsatisfiedLinkError errFull) {
        System.err.println("[ERROR] Couldn't find native library tvm4j_runtime_packed.");
        throw new RuntimeException(errFull);
      }
    }

    System.err.println("libtvm4j loads successfully.");
    // always use linked lib
    _LIB.nativeLibInit(null);
    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override public void run() {
        _LIB.shutdown();
      }
    });
  }

  /**
   * Load JNI for different OS.
   * @param libname library name.
   * @throws UnsatisfiedLinkError if loading fails.
   */
  private static void tryLoadLibraryOS(String libname) throws UnsatisfiedLinkError {
    try {
      System.err.println(String.format("Try loading %s from native path.", libname));
      System.loadLibrary(libname);
    } catch (UnsatisfiedLinkError e) {
      String os = System.getProperty("os.name");
      // ref: http://lopica.sourceforge.net/os.html
      if (os.startsWith("Linux")) {
        tryLoadLibraryXPU(libname, "linux-x86_64");
      } else if (os.startsWith("Mac")) {
        tryLoadLibraryXPU(libname, "osx-x86_64");
      } else {
        // TODO(yizhi) support windows later
        throw new UnsatisfiedLinkError("Windows not supported currently");
      }
    }
  }

  /**
   * Load native library for different architectures.
   * @param libname library name.
   * @param arch architecture.
   * @throws UnsatisfiedLinkError if loading fails
   */
  private static void tryLoadLibraryXPU(String libname, String arch) throws UnsatisfiedLinkError {
    System.err.println(String.format("Try loading %s-%s from native path.", libname, arch));
    System.loadLibrary(String.format("%s-%s", libname, arch));
  }

  // helper function definitions
  /**
   * Check the return value of C API call.
   * <p>
   * This function will raise exception when error occurs.
   * Wrap every API call with this function
   * </p>
   * @param ret return value from API calls
   */
  public static void checkCall(int ret) throws TVMError {
    if (ret != 0) {
      throw new TVMError(_LIB.tvmFFIGetLastError());
    }
  }

  /**
   * TVM Runtime error.
   */
  static class TVMError extends RuntimeException {
    public TVMError(String err) {
      super(err);
    }
  }

  /**
   * Cannot be instantiated.
   */
  private Base() {
  }
}
