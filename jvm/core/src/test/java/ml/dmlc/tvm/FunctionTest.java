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

package ml.dmlc.tvm;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class FunctionTest {
  @Test
  public void test_reg_sum_number() {
    Function.register("sum_number", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        long res = 0L;
        for (TVMValue arg : args) {
          res += arg.asLong();
        }
        return res;
      }
    });
    Function func = Function.getFunction("sum_number");
    TVMValue res = func.pushArg(10).pushArg(20).invoke();
    assertEquals(30, res.asLong());
    res.release();
    func.release();
  }

  @Test
  public void test_reg_add_string() {
    Function.register("add_string", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        String res = "";
        for (TVMValue arg : args) {
          res += arg.asString();
        }
        return res;
      }
    });
    Function func = Function.getFunction("add_string");
    TVMValue res = func.pushArg("Hello").pushArg(" ").pushArg("World!").invoke();
    assertEquals("Hello World!", res.asString());
    res.release();
    func.release();
  }

  @Test
  public void test_reg_sum_first_byte() {
    Function.register("sum_first_byte", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        byte[] bt = new byte[1];
        for (TVMValue arg : args) {
          bt[0] += arg.asBytes()[0];
        }
        return bt;
      }
    });
    Function func = Function.getFunction("sum_first_byte");
    TVMValue res = func.pushArg(new byte[]{1}).pushArg(new byte[]{2, 3}).invoke();
    assertArrayEquals(new byte[]{3}, res.asBytes());
    res.release();
    func.release();
  }

  @Test
  public void test_reg_sum_ndarray() {
    final long[] shape = new long[]{2, 1};
    Function.register("sum_ndarray", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        double sum = 0.0;
        for (TVMValue arg : args) {
          NDArray arr = NDArray.empty(shape, new TVMType("float32"));
          arg.asNDArray().copyTo(arr);
          float[] nativeArr = arr.asFloatArray();
          for (int i = 0; i < nativeArr.length; ++i) {
            sum += nativeArr[i];
          }
          arr.release();
        }
        return sum;
      }
    });
    Function func = Function.getFunction("sum_ndarray");
    NDArray arr = NDArray.empty(shape, new TVMType("float32"));
    arr.copyFrom(new float[]{2f, 3f});
    TVMValue res = func.pushArg(arr).pushArg(arr).invoke();
    assertEquals(10.0, res.asDouble(), 1e-3);
    res.release();
    func.release();
  }
}
