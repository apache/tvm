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
/* eslint-disable no-undef */

const tvmjs = require("../../dist");

test("Test coverage of [0,100] inclusive", () => {
  const covered = Array(100);
  const rng = new tvmjs.LinearCongruentialGenerator();
  for (let i = 0; i < 100000; i++) {
    covered[rng.nextInt() % 100] = true;
  }
  const notCovered = [];
  for (let i = 0; i < 100; i++) {
    if (!covered[i]) {
      notCovered.push(i);
    }
  }
  expect(notCovered).toEqual([]);
});

test("Test whether the same seed make two RNGs generate same results", () => {
  const rng1 = new tvmjs.LinearCongruentialGenerator();
  const rng2 = new tvmjs.LinearCongruentialGenerator();
  rng1.setSeed(42);
  rng2.setSeed(42);

  for (let i = 0; i < 100; i++) {
    expect(rng1.randomFloat()).toBeCloseTo(rng2.randomFloat());
  }
});

test("Test two RNGs with different seeds generate different results", () => {
  const rng1 = new tvmjs.LinearCongruentialGenerator();
  const rng2 = new tvmjs.LinearCongruentialGenerator();
  rng1.setSeed(41);
  rng2.setSeed(42);
  let numSame = 0;
  const numTest = 100;

  // Generate `numTest` random numbers, make sure not all are the same.
  for (let i = 0; i < numTest; i++) {
    if (rng1.nextInt() === rng2.nextInt()) {
      numSame += 1;
    }
  }
  expect(numSame < numTest).toBe(true);
});

test('Illegal argument to `setSeed()`', () => {
  expect(() => {
    const rng1 = new tvmjs.LinearCongruentialGenerator();
    rng1.setSeed(42.5);
  }).toThrow("Seed should be an integer.");
});
