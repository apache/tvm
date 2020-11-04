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
#pylint: disable=wildcard-import, redefined-builtin
"""Automatic quantization utilities."""
from __future__ import absolute_import as _abs

from . import _op_attrs
from .base import qconfig, current_qconfig
from .analysis import inspect_graph_statistic
from .hardware import Hardware, OpDesc, create_accelerator_description
from .search import generate_search_space, search_quantize_strategy
from .search import DefaultSetting, RandomSearchTuner, GreedySearchTuner, BatchedGreedySearchTuner
from .search import serialize, deserialize
from .quantize import CalibrationDataset, prerequisite_optimize, create_quantizer
from .record import serialize, deserialize, load_from_file, pick_best
