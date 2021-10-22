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

"""
Common helper services to manage the STM.AI type
"""


def stm_ai_error_to_str(err_code, err_type):
    """Return a human readable description of the ai run-time error

    see <AICore>::EmbedNets git (src/api/ai_platform.h file)
    """

    type_switcher = {
        0x00: "None",
        0x01: "Tools Platform Api Mismatch",
        0x02: "Types Mismatch",
        0x10: "Invalid handle",
        0x11: "Invalid State",
        0x12: "Invalid Input",
        0x13: "Invalid Output",
        0x14: "Invalid Param",
        0x30: "Init Failed",
        0x31: "Allocation Failed",
        0x32: "Deallocation Failed",
        0x33: "Create Failed",
    }

    code_switcher = {
        0x0000: "None",
        0x0010: "Network",
        0x0011: "Network Params",
        0x0012: "Network Weights",
        0x0013: "Network Activations",
        0x0014: "Layer",
        0x0015: "Tensor",
        0x0016: "Array",
        0x0017: "Invalid Ptr",
        0x0018: "Invalid Size",
        0x0020: "Invalid Format",
        0x0021: "Invalid Batch",
        0x0030: "Missed Init",
        0x0040: "In Use",
    }

    desc_ = 'type=0x{:x}("{}")'.format(err_type, type_switcher.get(err_type, str(err_type)))
    desc_ += '/code=0x{:x}("{}")'.format(err_code, code_switcher.get(err_code, str(err_code)))

    return desc_


def stm_ai_node_type_to_str(n_type, full=True):
    """Return a human readable description of a CNode / Clayer

    see <AICore>::EmbedNets git (src/layers/layers_list.h file)
    """
    base = 0x100  # stateless operator
    base_2 = 0x180  # stateful operator
    n_type = n_type & 0xFFFF
    ls_switcher = {
        0: "Output",
        base: "Base",
        base + 1: "Add",
        base + 2: "BN",
        base + 3: "Conv2D",
        base + 4: "Dense",
        base + 5: "GRU",
        base + 6: "LRN",
        base + 7: "NL",
        base + 8: "Norm",
        base + 9: "Conv2dPool",
        base + 10: "Transpose",
        base + 11: "Pool",
        base + 12: "Softmax",
        base + 13: "Split",
        base + 14: "TimeDelay",
        base + 15: "TimeDistributed",
        base + 16: "Concat",
        base + 17: "GEMM",
        base + 18: "Upsample",
        base + 19: "Eltwise",
        base + 20: "EltwiseInt",
        base + 21: "InstNorm",
        base + 22: "Pad",
        base + 23: "Slice",
        base + 24: "Tile",
        base + 25: "Reduce",
        base + 26: "RNN",
        base + 27: "Resize",
        base + 28: "Gather",
        base + 29: "Pack",
        base + 30: "UnPack",
        base + 31: "Container",
        base + 32: "Lambda",
        base_2: "Stateful",
        base_2 + 1: "LSTM",
        base_2 + 2: "Custom",
    }
    desc_ = "{}".format(ls_switcher.get(n_type, str(n_type)))
    if full:
        desc_ += " (0x{:x})".format(n_type)
    return desc_
