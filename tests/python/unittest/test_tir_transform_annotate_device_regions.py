# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import tvm.testing
from tvm.script import tir as T, ir as I


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.AnnotateDeviceRegions()


class TestAnnotateThreadExtent(BaseCompare):
    """Annotation inserted at the "thread_extent" attribute"""

    def before(A: T.Buffer(16, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        i = T.launch_thread("threadIdx.x", 16)
        A[i] = 0.0

    def expected(A: T.Buffer(16, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        T.attr(T.target("cuda"), "target", 0)
        i = T.launch_thread("threadIdx.x", 16)
        A[i] = 0.0


class TestAnnotateDeviceScope(BaseCompare):
    """Annotation inserted at the "device_scope" attribute"""

    def before(A: T.Buffer(1, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        T.attr(0, "device_scope", 0)
        A[0] = 0.0

    def expected(A: T.Buffer(1, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        T.attr(T.target("cuda"), "target", 0)
        T.attr(0, "device_scope", 0)
        A[0] = 0.0


if __name__ == "__main__":
    tvm.testing.main()
