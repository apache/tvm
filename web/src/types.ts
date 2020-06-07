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
/** Common type definitions. */

/**
 * Library interface provider that can provide
 * syslibs(e.g. libs provided by WASI and beyond) for the Wasm runtime.
 *
 * It can be viewed as a generalization of imports used in WebAssembly instance creation.
 *
 * The {@link LibraryProvider.start} callback will be called
 * to allow the library provider to initialize related resources during startup time.
 *
 * We can use Emscripten generated js Module as a { wasmLibraryProvider: LibraryProvider }.
 */
export interface LibraryProvider {
  /** The imports that can be passed to WebAssembly instance creation. */
  imports: Record<string, any>;
  /**
   * Callback function to notify the provider the created instance.
   * @param inst The created instance.
   */
  start: (inst: WebAssembly.Instance) => void;
}

/**
 * Disposable classes that contains resources (WasmMemory, GPU buffer)
 * which needs to be explicitly disposed.
 */
export interface Disposable {
  /**
   * Dispose the internal resource
   * This function can be called multiple times,
   * only the first call will take effect.
   */
  dispose: () => void;
}
