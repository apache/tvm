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

import java.util.Map;

/**
 * TVM API functions.
 */
public final class API {
  /**
   * Predefined functions.
   */
  private static final Map<String, Function> FUNCTIONS = Function.initAPI(
      new Function.InitAPINameFilter() {
        @Override public boolean accept(final String name) {
          return name != null && name.indexOf('.') == -1 && !name.startsWith("_");
        }
      }, new Function.InitAPINameGenerator() {
        @Override public String generate(final String name) {
          return name;
        }
      });

  /**
   * Get a tvm api function according by name.
   * @param name function name.
   * @return a TVM Function.
   */
  public static Function get(final String name) {
    return FUNCTIONS.get(name);
  }

  /**
   * Cannot be instantiated.
   */
  private API() {
  }
}
