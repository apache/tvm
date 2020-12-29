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

/**
 * Base class of NDArray. To handle callback array.
 * Only deep-copy supported.
 */
public class NDArrayBase extends TVMValue {
  protected final long handle;
  protected final boolean isView;
  private boolean isReleased = false;

  NDArrayBase(long handle, boolean isView) {
    super(ArgTypeCode.ARRAY_HANDLE);
    this.handle = handle;
    this.isView = isView;
  }

  NDArrayBase(long handle) {
    this(handle, true);
  }

  @Override public NDArrayBase asNDArray() {
    return this;
  }

  @Override long asHandle() {
    return handle;
  }

  /**
   * Copy array to target.
   * @param target The target array to be copied, must have same shape as this array.
   * @return target
   */
  public NDArrayBase copyTo(NDArrayBase target) {
    Base.checkCall(Base._LIB.tvmArrayCopyFromTo(handle, target.handle));
    return target;
  }

  /**
   * Release the NDArray memory.
   * <p>
   * We highly recommend you to do this manually since the GC strategy is lazy.
   * </p>
   */
  public void release() {
    if (!isReleased) {
      if (!isView) {
        Base.checkCall(Base._LIB.tvmArrayFree(handle));
        isReleased = true;
      }
    }
  }

  @Override protected void finalize() throws Throwable {
    release();
    super.finalize();
  }
}
