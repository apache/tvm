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
const fs = require("fs");
const ts = require("typescript");

const previousTypeScriptLoader = require.extensions[".ts"];
require.extensions[".ts"] = (module, filename) => {
  const source = fs.readFileSync(filename, "utf8");
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2018 },
    fileName: filename,
  }).outputText;
  module._compile(output, filename);
};

const { CacheState } = require("../../src/cache_state.ts");

if (previousTypeScriptLoader === undefined) {
  delete require.extensions[".ts"];
} else {
  require.extensions[".ts"] = previousTypeScriptLoader;
}

test("keeps an evicted shape tuple alive until cache state disposal", () => {
  const cacheState = new CacheState(1);
  const first = { dispose: jest.fn() };
  const second = { dispose: jest.fn() };

  cacheState.shapeCache.get("first", () => first);
  cacheState.shapeCache.get("second", () => second);

  expect(first.dispose).not.toHaveBeenCalled();

  cacheState.dispose();

  expect(first.dispose).toHaveBeenCalledTimes(1);
  expect(second.dispose).toHaveBeenCalledTimes(1);
});
