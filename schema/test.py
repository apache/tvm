import schema
from schema import *
from schema.typing import *

BaseExprNode = ObjectDef(name="BaseExprNode", type_key="BaseExpr", base=ObjectBase, fields=[])
BaseExpr = ObjectRefDef(name="BaseExpr", base=ObjectRefBase, internal_object=BaseExprNode)
print(generate(BaseExprNode))
print(generate(BaseExpr))

dtype = FieldDef("dtype", DataType)
PrimExprNode = ObjectDef(name="PrimExprNode", type_key="PrimExpr", base=BaseExprNode, fields=[dtype])
# not fully supported: customized methods
PrimExpr = ObjectRefDef(name="PrimExpr", base=BaseExpr, internal_object=PrimExprNode)
print(generate(PrimExprNode))
print(generate(PrimExpr))

# not fully supported: customized methods, `mutable` modifier
fields = [
    FieldDef("span", Span),
    FieldDef("checked_type_", Type),
]
RelayExprNode = ObjectDef(name="RelayExprNode", type_key="RelayExpr", base=BaseExprNode, fields=fields)
RelayExpr = ObjectRefDef(name="RelayExpr", base=BaseExpr, internal_object=RelayExprNode)
print(generate(RelayExprNode))
print(generate(RelayExpr))

# not fully supported: SEqualReduce, SHashReduce
name_hint = FieldDef("name_hint", String)
GlobalVarNode = ObjectDef(name="GlobalVarNode", type_key="GlobalVar", base=RelayExprNode, fields=[name_hint],
                          fvisit_attrs=True)
GlobalVar = ObjectRefDef(name="GlobalVar", base=RelayExpr, internal_object=GlobalVarNode)
print(generate(GlobalVarNode))
print(generate(GlobalVar))
register(GlobalVarNode)

fields = [
    FieldDef("value", int64_t),
]
IntImmNode = ObjectDef(name="IntImmNode", type_key="IntImm", base=PrimExprNode,
    fields=fields, fvisit_attrs=True, fsequal_reduce=True, fshash_reduce=True)
IntImm = ObjectRefDef(name="IntImm", base=PrimExpr, internal_object=IntImmNode)
print(generate(IntImmNode))
print(generate(IntImm))
register(IntImmNode)
register(IntImm)

print('\n')
process('./sample.h', './sample.h.gen')
