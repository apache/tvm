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
"""Configure pytest"""
import sys

COLLECT_IGNORE = []
if sys.platform.startswith("win"):
    COLLECT_IGNORE.append("frontend/coreml")
    COLLECT_IGNORE.append("frontend/keras")
    COLLECT_IGNORE.append("frontend/pytorch")
    COLLECT_IGNORE.append("frontend/tensorflow")
    COLLECT_IGNORE.append("frontend/tflite")
    COLLECT_IGNORE.append("frontend/onnx")

    COLLECT_IGNORE.append("tir_base/test_tir_intrin.py")
