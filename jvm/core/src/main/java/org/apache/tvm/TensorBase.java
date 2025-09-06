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
 * Base class of Tensor. To handle callback array.
 * Only deep-copy supported.
 */
public class TensorBase extends TVMValue {
  protected long handle;
  public final boolean isView;
  protected final long dltensorHandle;

  TensorBase(long handle, boolean isView) {
    this.dltensorHandle = isView ? handle : handle + 8 * 2;
    this.handle = isView ? 0 : handle;
    this.isView = isView;
  }

  @Override public TensorBase asTensor() {
    return this;
  }

  /**
   * Release the Tensor.
   */
  public void release() {
    if (this.handle != 0) {
      Base.checkCall(Base._LIB.tvmFFIObjectFree(this.handle));
      this.handle = 0;
    }
  }

  @Override protected void finalize() throws Throwable {
    release();
    super.finalize();
  }

  /**
   * Copy array to target.
   * @param target The target array to be copied, must have same shape as this array.
   * @return target
   */
  public TensorBase copyTo(TensorBase target) {
    Base.checkCall(Base._LIB.tvmFFIDLTensorCopyFromTo(this.dltensorHandle, target.dltensorHandle));
    return target;
  }
}
