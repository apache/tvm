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
"Add suffix to the relay.Call's span fields"
from collections import defaultdict

import tvm

from ..expr_functor import ExprMutator
from .. import expr as _expr

SUFFIX_STRING = r"_PART_"


class _SuffixTagger(ExprMutator):
    """A pass to traverse the Relay graph to add suffix to the call's span fields.
    This making span an unique indicator of a Relay line and we can use it to
    obtain the mapping between the Relay that gets generated from a relay frontend
    and the Relay after partitioning.
    """

    def __init__(self):
        ExprMutator.__init__(self)
        # key: span or source name, value: counter, indexed from 0
        self.lookup = defaultdict(int)
        self.suffix = SUFFIX_STRING
        # a set to record hashes of an expressions which spans have been already rewritten
        self.hashes = set()

    def _tag_suffix(self, span):
        # To avoid error once we introduce the SequentialSpan in the future
        """https://discuss.tvm.apache.org/
        t/pre-rfc-tvm-explorer-infrastructure/13457#pass-source-information-builder-6
        """
        # Don't need this if currently
        if isinstance(span, tvm.relay.Span):
            ori_name = span.source_name.name
            new_name = ori_name + self.suffix + str(self.lookup[ori_name])
            self.lookup[ori_name] += 1
            return tvm.relay.Span(
                tvm.relay.SourceName(new_name),
                span.line,
                span.end_line,
                span.column,
                span.end_column,
            )
        return span

    def visit(self, expr):
        if hasattr(expr, "span"):
            return super().visit(expr)
        return expr

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_op = self.visit(call.op)
        if tvm.ir.structural_hash(call) not in self.hashes:
            self.hashes.add(tvm.ir.structural_hash(call))
            expr__ = _expr.CallWithFields(
                call,
                new_op,
                new_args,
                call.attrs,
                call.type_args,
                None,
                self._tag_suffix(call.span),
            )
        else:
            expr__ = _expr.CallWithFields(
                call, new_op, new_args, call.attrs, call.type_args, None, call.span
            )
        return expr__


def tag_suffixes(mod):
    """Traverses the Relay graph to add suffix to the call's span fields.
    That making span as an unique indicator of a Relay call and we can use it to
    obtain the mapping between the offloaded result and the frontend operators.

    Parameters
    ----------
    tvm.ir.IRModule
        The IRModule that gets generated from a relay frontend.

    Returns
    -------
    tvm.ir.IRModule
        The IRModule with call's span fields tagged with suffixes.
    """
    tagger = _SuffixTagger()
    for global_var, func in mod.functions.items():
        func = tagger.visit(func)
        mod.update_func(global_var, func)
    return mod
