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
import os
from unittest import mock
import tempfile

import tvm

def test_empty_library_export():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "tmp_lib")
        print(temp_file_path)
        with mock.patch.object(tvm.runtime.Module, "is_empty") as is_empty_mock:
            is_empty_mock.return_value = True
            module = tvm.runtime.Module(None)
            module.export_library(temp_file_path)
        assert(os.path.isfile(temp_file_path))
       

if __name__ == "__main__":
    test_empty_library_export()
