import tvm
from tvm import meta

object_ = meta.ObjectDef(name="Object", ref_name="ObjectRef", nmspace="tvm", base=None, variables=[])
print(object_)
base_expr = meta.ObjectDef(name="BaseExprNode", ref_name="BaseExpr", nmspace="tvm", base=object_, variables=[])
print(base_expr)
print()
print(meta.generate(base_expr))
