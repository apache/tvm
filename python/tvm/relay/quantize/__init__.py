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
# pylint: disable=wildcard-import, redefined-builtin
"""Automatic quantization utilities."""
from __future__ import absolute_import as _abs

from .quantize import *
from ._collect import collect
from ._collectwj import collect_wj
from ._collectwj_1 import collect_wj_1
from ._partition import register_partition_function
from ._collectwj_2 import collect_wj_2
from ._annotate import register_annotate_inference_function_dict
from ._collectwj_3 import collect_wj_3
from .model_output import print_model
from ._collectwj_4 import collect_wj_4
from ._collectwj_5 import collect_wj_5
from ._collectwj_6 import collect_wj_6
from ._collectwj_7 import collect_wj_7