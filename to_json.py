import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprFunctor

import json

class ToTypeScriptJSON(ExprFunctor):
    def visit_function(self, func):
        json_params = [self.visit(param) for param in func.params]
        fn_json = { "type": "Function", "params": json_params,
           "body": self.visit(func.body)
        }
        return fn_json


    def visit_var(self, var):
        # TODO(@jroesch): we need to add a type field
        var_json = { "type": "Var", "name_hint": var.name_hint }
        return var_json

    def visit_call(self, call):
        return {}

x = relay.var('x', shape=(10, 5))
y = relay.var('x', shape=(10, 5))
f: relay.Function = relay.Function([x, y], x + y)

to_json = ToTypeScriptJSON()
program_ser = to_json.visit(f)
program_json = json.dumps(program_ser)

from tvm.relay.build_module import build

mod = tvm.IRModule.from_expr(f)

graph_json, rt_mod, params = build(mod, target='llvm')

import pdb; pdb.set_trace()

span_to_perf_data = { ... }
span_to_source_code = { ... }
span_to_graph = { ... }

a = g + h + j
if (x + y) {
    z + w + .........
} else {
    a + b + c + d ..
}


x  y
 \ /
  +
  |
  if
| ....| | ....|
    \      /
        |
