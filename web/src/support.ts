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
 * Check if value is a promise type
 *
 * @param value The input value
 * @returns Whether value is promise
 */
export function isPromise(value: any): boolean {
  return value !== undefined && (
    typeof value == "object" || typeof value == "function"
  ) && typeof value.then == "function";
}
/**
 * Convert string to Uint8array.
 * @param str The string.
 * @returns The corresponding Uint8Array.
 */
export function StringToUint8Array(str: string): Uint8Array {
  const arr: Uint8Array = new TextEncoder().encode(str);
  const resArr = new Uint8Array(arr.length + 1);
  for (let i = 0; i < arr.length; ++i) {
    resArr[i] = arr[i];
  }
  resArr[arr.length] = 0;
  return resArr;
}

/**
 * Convert Uint8array to string.
 * @param array The array.
 * @returns The corresponding string.
 */
export function Uint8ArrayToString(arr: Uint8Array): string {
  const ret = [];
  for (const ch of arr) {
    ret.push(String.fromCharCode(ch));
  }
  return ret.join("");
}

/**
 * Internal assert helper
 * @param condition The condition to fail.
 * @param msg The message.
 */
export function assert(condition: boolean, msg?: string): asserts condition {
  if (!condition) {
    throw new Error("AssertError:" + (msg || ""));
  }
}

/**
 * Get the path to the wasm library in nodejs.
 * @return The wasm path.
 */
export function wasmPath(): string {
  return __dirname + "/wasm";
}

/**
 * Linear congruential generator for random number generating that can be seeded.
 *
 * Follows the implementation of `include/tvm/support/random_engine.h`, which follows the
 * sepcification in https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine.
 *
 * Note `Number.MAX_SAFE_INTEGER = 2^53 - 1`, and our intermediates are strictly less than 2^48.
 */

export class LinearCongruentialGenerator {
  readonly modulus: number;
  readonly multiplier: number;
  readonly increment: number;
  // Always within the range (0, 2^32 - 1) non-inclusive; if 0, will forever generate 0.
  private rand_state: number;

  /**
   * Set modulus, multiplier, and increment. Initialize `rand_state` according to `Date.now()`.
   */
  constructor() {
    this.modulus = 2147483647;  // 2^32 - 1
    this.multiplier = 48271;  // between 2^15 and 2^16
    this.increment = 0;
    this.setSeed(Date.now());
  }

  /**
   * Sets `rand_state` after normalized with `modulus` to ensure that it is within range.
   * @param seed Any integer. Used to set `rand_state` after normalized with `modulus`.
   *
   * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
   */
  setSeed(seed: number) {
    if (!Number.isInteger(seed)) {
      throw new Error("Seed should be an integer.");
    }
    this.rand_state = seed % this.modulus;
    if (this.rand_state == 0) {
      this.rand_state = 1;
    }
    this.checkRandState();
  }

  /**
   * Generate the next integer in the range (0, this.modulus) non-inclusive, updating `rand_state`.
   *
   * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
   */
  nextInt(): number {
    // `intermediate` is always < 2^48, hence less than `Number.MAX_SAFE_INTEGER` due to the
    // invariants as commented in the constructor.
    const intermediate = this.multiplier * this.rand_state + this.increment;
    this.rand_state = intermediate % this.modulus;
    this.checkRandState();
    return this.rand_state;
  }

  /**
   * Generates random float between (0, 1) non-inclusive, updating `rand_state`.
   *
   * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
   */
  randomFloat(): number {
    return this.nextInt() / this.modulus;
  }

  private checkRandState(): void {
    if (this.rand_state <= 0) {
      throw new Error("Random state is unexpectedly not strictly positive.");
    }
    if (!Number.isInteger(this.rand_state)) {
      throw new Error("Random state is unexpectedly not an integer.");
    }
  }
}
