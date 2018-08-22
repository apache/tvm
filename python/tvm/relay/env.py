# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, wildcard-import
"""A global environment storing everything needed to interpret or compile a Realy program."""
from typing import Union, List
from relay.ir import register_relay_node, NodeBase
from relay.ir import GlobalId, OperatorId, Item, FileId, Span, ShapeExtension
from relay.ir import Operator, Defn
from relay._env import *
import tvm

# Move me to C++ if possible.
__tgt_host__ = __tgt__ = "llvm"
__relay_tvm_context__ = tvm.cpu()

ADD_ID = "__add__"
SUB_ID = "__sub__"
MUL_ID = "__mul__"
DIV_ID = "__div__"
NEG_ID = "__neg__"
LT_ID = "__lt__"
LE_ID = "__le__"
GT_ID = "__gt__"
GE_ID = "__ge__"
EQ_ID = "__eq__"
NE_ID = "__ne__"

@register_relay_node
class Environment(NodeBase):
    """The global Relay environment containing definitions,
       primitives, options, and more.
    """
    def add(self, item: Item) -> None:
        return Environment_add(self, item)

    def global_id(self, name: str) -> GlobalId:
        return Environment_global_id(self, name)

    def operator_id(self, name: str) -> OperatorId:
        return Environment_operator_id(self, name)

    def lookup(self, ident: Union[GlobalId, OperatorId]) -> Item:
        if isinstance(ident, OperatorId):
            return Environment_lookup_operator(self, ident)
        else:
            return Environment_lookup_global(self, ident)

    def add_source(self, file_name: str, source: str) -> FileId:
        return Environment_add_source(self, file_name, source)

    def report_error(self, message: str, span: Span) -> None:
        return Environment_report_error(self, message, span)

    def register_shape_ext(self, ext: ShapeExtension) -> None:
        return Environment_register_shape_ext(self, ext)

    def display_errors(self) -> None:
        return Environment_display_errors(self)

    def operators(self) -> List[Operator]:
        return Environment_get_operators(self)

    def defns(self) -> List[Defn]:
        return Environment_get_defns(self)

    def tvm_context(self):
        return __relay_tvm_context__

    def add_id(self) -> OperatorId:
        return self.operator_id(ADD_ID)

    def sub_id(self) -> OperatorId:
        return self.operator_id(SUB_ID)

    def mul_id(self) -> OperatorId:
        return self.operator_id(MUL_ID)

    def div_id(self) -> OperatorId:
        return self.operator_id(DIV_ID)

    def neg_id(self) -> OperatorId:
        return self.operator_id(NEG_ID)

    def lt_id(self) -> OperatorId:
        return self.operator_id(LT_ID)

    def le_id(self) -> OperatorId:
        return self.operator_id(LE_ID)

    def gt_id(self) -> OperatorId:
        return self.operator_id(GT_ID)

    def ge_id(self) -> OperatorId:
        return self.operator_id(GE_ID)

    def eq_id(self) -> OperatorId:
        return self.operator_id(EQ_ID)

    def ne_id(self) -> OperatorId:
        return self.operator_id(NE_ID)
