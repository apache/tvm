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
"""Utilities that enable analyze Relay and get mappings for
the unique identifier of the Relay line to the tuple of
compiler name, composite name and composite/function identifier."""
import re

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor


class AnalyzeOperationsDistribution(ExprVisitor):
    """A visitor pass that maintains the dictionary unique_op_ids where
    the tuple (compiler name, composite name) corresponds to the unique
    identifier of the Relay line. The identifier will allow us to link
    the lines of the initial Relay with the information about operators
    offloading, which is present in the partitioned Relay
    TVMC compiler adds a unique Relay line identifier as a suffix to the
    call span field using the tag_suffixes pass if the --dump-offloads
    option is specified.

    Attributes
    ----------
    unique_op_ids : Dict[str, str]
        Mapping the unique identifier of the Relay line obtained from
        the "span" field of the Call and the tuple of compiler name,
        composite name.
    func_name : str
        The name of the composite in the partitioned Relay or
        'generic' in case the Call has not been included in any composite.
    compiler_name : str
        A name of the compiler (e.g. 'ethos-u' or 'cmsis-nn') or 'generic'
        in case the Call has not been included in any composite.
    """

    def __init__(self):
        self.unique_op_ids = {}
        self.func_name = ""
        self.compiler_name = ""
        super().__init__()

    def extract(self, call: relay.Call):
        self.compiler_name = "generic"
        self.func_name = "generic"
        if "Compiler" in call.attrs:
            self.compiler_name = call.attrs["Compiler"]
        self.visit(call)

    def visit_call(self, call: relay.Call):
        if isinstance(call.op, tvm.ir.Op):
            if call.span:
                src = call.span.source_name.name
                suffix = tvm.relay.transform.suffixes.SUFFIX_STRING
                result = re.search(r"(.*)(" + suffix + r")(.*)", src)
                res = result.group(1)
                self.unique_op_ids[res] = [self.compiler_name, self.func_name]
        if isinstance(call.op, relay.Function):
            self.func_name = call.op.attrs["Composite"]
        super().visit_call(call)


def analyze_operations_distribution(mod):
    """Traverses the partitioned graph to get the unique identifier
    of the Relay line from the Call's span field.
    The result is maintained in the dictionary unique_op_ids where
    the unique indicator obtained from the op's span corresponds to
    the tuple (compiler name, composite name).
    With this information we can annotate the textual representation
    of the initial Relay by indicating into which target composite
    and function the operators are converted

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The partitioned Relay graph usually obtained with
        partition_for_<target> function

    Returns
    -------
    unique_op_ids : Dict[str, str]
        Mapping from the unique identifier of the Relay line to the tuple of
        compiler name, composite name.
    """
    analyze = AnalyzeOperationsDistribution()
    for _, func in mod.functions.items():
        analyze.extract(func)
    return analyze.unique_op_ids
