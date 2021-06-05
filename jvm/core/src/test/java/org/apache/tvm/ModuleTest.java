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

import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Random;

public class ModuleTest {
  private final Logger logger = LoggerFactory.getLogger(ModuleTest.class);
  private static String loadingDir;

  @BeforeClass
  public static void beforeClass() {
    loadingDir = System.getProperty("test.tempdir");
  }

  @Test
  public void test_load_add_func_cpu() {
    Module fadd = Module.load(loadingDir + File.separator + "add_cpu.so");

    Device dev = new Device("cpu", 0);
    long[] shape = new long[]{2};
    NDArray arr = NDArray.empty(shape, dev);

    arr.copyFrom(new float[]{3f, 4f});

    NDArray res = NDArray.empty(shape, dev);

    fadd.entryFunc().pushArg(arr).pushArg(arr).pushArg(res).invoke();
    assertArrayEquals(new float[]{6f, 8f}, res.asFloatArray(), 1e-3f);

    // test call() api
    fadd.entryFunc().call(arr, arr, res);
    assertArrayEquals(new float[]{6f, 8f}, res.asFloatArray(), 1e-3f);

    arr.release();
    res.release();
    fadd.release();
  }

  @Test
  public void test_load_add_func_cuda() {
    final Random RND = new Random(0);

    Device dev = new Device("cuda", 0);
    if (!dev.exist()) {
      logger.warn("CUDA GPU does not exist. Skip the test.");
      return;
    }

    Module fadd = Module.load(loadingDir + File.separator + "add_cuda.so");
    Module faddDev = Module.load(loadingDir + File.separator + "add_cuda.ptx");
    fadd.importModule(faddDev);

    final int dim = 100;
    long[] shape = new long[]{dim};
    NDArray arr = NDArray.empty(shape, dev);

    float[] data = new float[dim];
    float[] dataX2 = new float[dim];
    for (int i = 0; i < dim; ++i) {
      data[i] = RND.nextFloat();
      dataX2[i] = data[i] * 2;
    }
    arr.copyFrom(data);

    NDArray res = NDArray.empty(shape, dev);
    fadd.entryFunc().pushArg(arr).pushArg(arr).pushArg(res).invoke();

    assertArrayEquals(dataX2, res.asFloatArray(), 1e-3f);

    arr.release();
    res.release();
    faddDev.release();
    fadd.release();
  }
}
