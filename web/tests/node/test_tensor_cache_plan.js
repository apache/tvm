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
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const ts = require("typescript");

const plannerPath = path.join(__dirname, "../../src/tensor_cache_plan.ts");
const plannerSource = fs.readFileSync(plannerPath, "utf8");
const plannerJavaScript = ts.transpileModule(plannerSource, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2018,
  },
  fileName: plannerPath,
}).outputText;
const plannerModule = { exports: {} };
new Function("module", "exports", plannerJavaScript)(
  plannerModule,
  plannerModule.exports
);
const { planTensorCacheChunks } = plannerModule.exports;

const MiB = 1024 * 1024;

function assertValidPlan(plan, sourceBytes, targetBytes, maxChunkBytes, alignment) {
  assert(plan !== undefined);
  let sourceOffset = 0;
  let targetOffset = 0;
  for (const chunk of plan.chunks) {
    assert.strictEqual(chunk.sourceByteOffset, sourceOffset);
    assert.strictEqual(chunk.targetByteOffset, targetOffset);
    assert(chunk.sourceByteLength <= maxChunkBytes);
    assert(chunk.targetByteLength <= maxChunkBytes);
    assert.strictEqual(chunk.targetByteOffset % alignment, 0);
    assert.strictEqual(chunk.targetByteLength % alignment, 0);
    sourceOffset += chunk.sourceByteLength;
    targetOffset += chunk.targetByteLength;
  }
  assert.strictEqual(sourceOffset, sourceBytes);
  assert.strictEqual(targetOffset, targetBytes);
}

test("tensor-cache chunks keep WebGPU offsets aligned", () => {
  const plan = planTensorCacheChunks(
    [50000000, 3],
    150000000,
    150000000,
    128 * MiB,
    4
  );

  assertValidPlan(plan, 150000000, 150000000, 128 * MiB, 4);
  assert(plan.chunks.length > 1);
});

test("tensor-cache chunks keep the final WebGPU copy aligned", () => {
  const plan = planTensorCacheChunks([6, 1], 12, 12, 8, 4);

  assertValidPlan(plan, 12, 12, 8, 4);
  assert.deepStrictEqual(
    plan.chunks.map((chunk) => chunk.targetByteLength),
    [8, 4]
  );
});

test("tensor-cache chunks reject an unaligned WebGPU total", () => {
  const plan = planTensorCacheChunks([5, 1], 10, 10, 8, 4);

  assert.strictEqual(plan, undefined);
});

test("tensor-cache CPU decode does not depend on WebGPU copy alignment", () => {
  const recordBytes = 128 * MiB + 1;
  const decodePlan = planTensorCacheChunks(
    [recordBytes, 1],
    recordBytes,
    recordBytes,
    128 * MiB
  );
  const gpuCopyPlan = planTensorCacheChunks(
    [recordBytes, 1],
    recordBytes,
    recordBytes,
    128 * MiB,
    4
  );

  assertValidPlan(decodePlan, recordBytes, recordBytes, 128 * MiB, 1);
  assert.strictEqual(decodePlan.chunks.length, 2);
  assert.strictEqual(gpuCopyPlan, undefined);
});

test("tensor-cache chunks bound a large raw record", () => {
  const plan = planTensorCacheChunks(
    [262144, 1120],
    1174405120,
    1174405120,
    128 * MiB,
    4
  );

  assertValidPlan(plan, 1174405120, 1174405120, 128 * MiB, 4);
  assert.strictEqual(plan.chunks.length, 9);
});

test("tensor-cache chunks bound encoded and decoded BF16 sizes", () => {
  const plan = planTensorCacheChunks(
    [262144, 1120],
    587202560,
    1174405120,
    128 * MiB,
    4
  );

  assertValidPlan(plan, 587202560, 1174405120, 128 * MiB, 4);
  assert.strictEqual(plan.chunks.length, 9);
});

test("tensor-cache chunks reject an outer row above the cap", () => {
  const plan = planTensorCacheChunks(
    [2, 40 * MiB],
    160 * MiB,
    320 * MiB,
    128 * MiB,
    4
  );
  assert.strictEqual(plan, undefined);
});

test("tensor-cache chunks reject targets at the wasm32 address-space limit", () => {
  for (const targetBytes of [0x100000000, 0x100000000 + 1]) {
    const plan = planTensorCacheChunks(
      [targetBytes],
      targetBytes,
      targetBytes,
      128 * MiB
    );

    assert.strictEqual(plan, undefined);
  }
});
