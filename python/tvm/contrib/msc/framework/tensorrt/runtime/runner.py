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
"""tvm.contrib.msc.framework.tensorrt.runtime.runner"""

from tvm.contrib.msc.core.runtime import BYOCRunner
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.framework.tensorrt.frontend import partition_for_tensorrt
from tvm.contrib.msc.framework.tensorrt.codegen import to_tensorrt


class TensorRTRunner(BYOCRunner):
    """Runner of tensorrt"""

    def setup(self):
        """Setup the runner"""

        super().setup()
        if not self._device.startswith("cuda"):
            self._device = "cuda"

    @property
    def codegen_func(self):
        return to_tensorrt

    @property
    def partition_func(self):
        return partition_for_tensorrt

    @property
    def framework(self):
        return MSCFramework.TENSORRT
