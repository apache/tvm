/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/**
 * Runtime environment that provide js libaries calls.
 */
import { Pointer } from "./ctypes";
import { LibraryProvider } from "./types";
import { assert } from "./support";
import * as ctypes from "./ctypes";

/**
 * Detect library provider from the importObject.
 *
 * @param importObject The import object.
 */
function detectLibraryProvider(
  importObject: Record<string, any>
): LibraryProvider | undefined {
  if (
    importObject["wasmLibraryProvider"] &&
    importObject["wasmLibraryProvider"]["start"] &&
    importObject["wasmLibraryProvider"]["imports"] !== undefined
  ) {
    const item = importObject as { wasmLibraryProvider: LibraryProvider };
    // create provider so that we capture imports in the provider.
    return {
      imports: item.wasmLibraryProvider.imports,
      start: (inst: WebAssembly.Instance): void => {
        item.wasmLibraryProvider.start(inst);
      },
    };
  } else if (importObject["imports"] && importObject["start"] !== undefined) {
    return importObject as LibraryProvider;
  } else if (importObject["wasiImport"] && importObject["start"] !== undefined) {
    // WASI
    return {
      imports: {
        "wasi_snapshot_preview1": importObject["wasiImport"],
      },
      start: (inst: WebAssembly.Instance): void => {
        importObject["start"](inst);
      }
    };
  } else {
    return undefined;
  }
}

/**
 * Environment to impelement most of the JS library functions.
 */
export class Environment implements LibraryProvider {
  logger: (msg: string) => void;
  imports: Record<string, any>;
  /**
   * Maintains a table of FTVMWasmPackedCFunc that the C part
   * can call via TVMWasmPackedCFunc.
   *
   * We maintain a separate table so that we can have un-limited amount
   * of functions that do not maps to the address space.
   */
  packedCFuncTable: Array<ctypes.FTVMWasmPackedCFunc | undefined> = [
    undefined,
  ];
  /**
   * Free table index that can be recycled.
   */
  packedCFuncTableFreeId: Array<number> = [];

  private libProvider?: LibraryProvider;

  constructor(
    importObject: Record<string, any> = {},
    logger: (msg: string) => void = console.log
  ) {
    this.logger = logger;
    this.libProvider = detectLibraryProvider(importObject);
    // get imports from the provider
    if (this.libProvider !== undefined) {
      this.imports = this.libProvider.imports;
    } else {
      this.imports = importObject;
    }
    // update with more functions
    this.imports.env = this.environment(this.imports.env);
  }

  /** Mark the start of the instance. */
  start(inst: WebAssembly.Instance): void {
    if (this.libProvider !== undefined) {
      this.libProvider.start(inst);
    }
  }

  private environment(initEnv: Record<string, any>): Record<string, any> {
    // default env can be be overriden by libraries.
    const defaultEnv = {
      "__cxa_thread_atexit": (): void => {},
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      "emscripten_notify_memory_growth": (index: number): void => {}
    };
    const wasmPackedCFunc: ctypes.FTVMWasmPackedCFunc = (
      args: Pointer,
      typeCodes: Pointer,
      nargs: number,
      ret: Pointer,
      resourceHandle: Pointer
    ): number => {
      const cfunc = this.packedCFuncTable[resourceHandle];
      assert(cfunc !== undefined);
      return cfunc(args, typeCodes, nargs, ret, resourceHandle);
    };

    const wasmPackedCFuncFinalizer: ctypes.FTVMWasmPackedCFuncFinalizer = (
      resourceHandle: Pointer
    ): void => {
      this.packedCFuncTable[resourceHandle] = undefined;
      this.packedCFuncTableFreeId.push(resourceHandle);
    };

    const newEnv = {
      TVMWasmPackedCFunc: wasmPackedCFunc,
      TVMWasmPackedCFuncFinalizer: wasmPackedCFuncFinalizer,
      "__console_log": (msg: string): void => {
        this.logger(msg);
      }
    };
    return Object.assign(defaultEnv, initEnv, newEnv);
  }
}
