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
Specialized attributes that can be added to schedules to alter
the behaviour of AArch64 codegen.
"""


class SMEAttributes:
    """
    This class serves as a convenience wrapper for processor state annotations
    relating to the Scalable Matrix Extension (SME). Processor state annotations
    are inserted at compile time and alter some global state of the processor
    during execution. For example, the streaming mode attribute can be used to
    transfer some vector operations to a separate processing element. These
    attributes can be added to block-level annotations in AArch64 schedules to
    define a desired state.

    Please refer to the following pages for more information regarding the SME
    attributes and their behaviours:
     - https://arm-software.github.io/acle/main/acle.html#markdown-toc-sme-attributes
     - https://llvm.org/docs/AArch64SME.html

    Attributes
    ----------
    STREAMING_MODE : str
        Whether execution should occur in regular mode or streaming mode. When
        enabled, some vector operations may be transferred to a separate processing
        element.
    ZA_STORAGE : str
        Defines how the ZA area of storage provided by the SME extension should be
        utilized.
    """

    STREAMING_MODE = "pragma_aarch64_pstate_sm"

    class StreamingModeValues:
        """
        Streaming mode attribute values. By default, a function is considered
        'non-streaming' (often referred to as 'regular').

        Attributes
        ----------
        ENABLED : str
            The processor state must be in streaming mode before executing the marked function.
        COMPATIBLE : str
            The marked function can be run in either streaming or non-streaming mode.
        """

        ENABLED = "enabled"
        COMPATIBLE = "compatible"

    ZA_STORAGE = "pragma_aarch64_pstate_za"

    class ZAStorageValues:
        """
        ZA Storage attribure values. By default, a function has no ZA state. In other words, it
        does not use the ZA storage.

        Attributes
        ----------
        NEW : str
            A new ZA state is created "from scratch".
        SHARED : str
            The ZA state is shared with the calling function.
        """

        NEW = "new"
        SHARED = "shared"
