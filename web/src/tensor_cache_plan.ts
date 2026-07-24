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

export interface TensorCacheChunk {
  outerCount: number;
  sourceByteOffset: number;
  sourceByteLength: number;
  targetByteOffset: number;
  targetByteLength: number;
}

export interface TensorCacheChunkPlan {
  chunks: Array<TensorCacheChunk>;
}

// Tensor view offsets are currently marshalled into TVM JS's wasm32 runtime.
const wasm32AddressSpaceBytes = 0x100000000;

function greatestCommonDivisor(lhs: number, rhs: number): number {
  while (rhs !== 0) {
    const remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

/**
 * Plan outer-dimension chunks whose encoded and decoded sizes stay bounded.
 *
 * Returns undefined when outer-dimension chunking cannot satisfy the size and
 * target-alignment constraints. Callers can then retain the full-record path.
 */
export function planTensorCacheChunks(
  shape: Array<number>,
  sourceBytes: number,
  targetBytes: number,
  maxChunkBytes: number,
  targetAlignmentBytes = 1,
): TensorCacheChunkPlan | undefined {
  const outerDim = shape[0];
  if (
    shape.length === 0 ||
    !Number.isSafeInteger(outerDim) ||
    outerDim <= 0 ||
    !Number.isSafeInteger(sourceBytes) ||
    sourceBytes <= 0 ||
    !Number.isSafeInteger(targetBytes) ||
    targetBytes <= 0 ||
    targetBytes >= wasm32AddressSpaceBytes ||
    !Number.isSafeInteger(maxChunkBytes) ||
    maxChunkBytes <= 0 ||
    !Number.isSafeInteger(targetAlignmentBytes) ||
    targetAlignmentBytes <= 0 ||
    sourceBytes % outerDim !== 0 ||
    targetBytes % outerDim !== 0 ||
    targetBytes % targetAlignmentBytes !== 0
  ) {
    return undefined;
  }

  const sourceStrideBytes = sourceBytes / outerDim;
  const targetStrideBytes = targetBytes / outerDim;
  const maxStrideBytes = Math.max(sourceStrideBytes, targetStrideBytes);
  if (maxStrideBytes > maxChunkBytes) {
    return undefined;
  }

  const rowsPerAlignment =
    targetAlignmentBytes /
    greatestCommonDivisor(targetStrideBytes, targetAlignmentBytes);
  const maxOuterCount = Math.floor(maxChunkBytes / maxStrideBytes);
  const chunkOuterCount =
    maxOuterCount - (maxOuterCount % rowsPerAlignment);
  if (chunkOuterCount <= 0) {
    return undefined;
  }

  const chunks = new Array<TensorCacheChunk>();
  for (let outerOffset = 0; outerOffset < outerDim; outerOffset += chunkOuterCount) {
    const outerCount = Math.min(chunkOuterCount, outerDim - outerOffset);
    chunks.push({
      outerCount,
      sourceByteOffset: outerOffset * sourceStrideBytes,
      sourceByteLength: outerCount * sourceStrideBytes,
      targetByteOffset: outerOffset * targetStrideBytes,
      targetByteLength: outerCount * targetStrideBytes,
    });
  }

  return { chunks };
}
