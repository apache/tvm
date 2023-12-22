from collections import OrderedDict
from typing import Dict, List

from tvm import arith


class Statement():
    def __init__(self, output: str, dependent_region: dict, var_map: OrderedDict, range_map: OrderedDict):
        self.output = output
        self.dependent_region = dependent_region
        self.var_map = var_map
        self.range_map = range_map

def _merge_two_bounds(x: arith.ConstIntBound, y: arith.ConstIntBound):
    return arith.ConstIntBound(min(x.min_value, y.min_value), max(x.max_value, y.max_value))

class InputShapeInference():
    def __init__(self, deps: List[Statement]):
        self.deps = deps

    def _infer(self, shape: Dict[str, List[arith.ConstIntBound]], rstep: Dict[str, int]):
        shape = shape.copy()
        ana = arith.Analyzer()
        for dep in reversed(self.deps):
            for var, bound in zip(dep.var_map.values(), shape[dep.output]):
                ana.update(var, bound)
            for var, bound in dep.range_map.items():
                if var.name in rstep:
                    bound = arith.ConstIntBound(0, min(bound.max_value, rstep[var.name] - 1))
                ana.update(var, bound)
            for name, regions in dep.dependent_region.items():
                for region in regions:
                    bounds = [ana.const_int_bound(index) for index in region]
                    if name in shape: # simply merge two bounds
                        bounds = [_merge_two_bounds(x, y) for x, y in zip(shape[name], bounds)]
                    shape[name] = bounds

        for name, bounds in shape.items():
            shape[name] = [c.max_value - c.min_value + 1 for c in bounds]
        return shape

    def infer(self, shape, rstep: Dict[str, int] = {}):
        if isinstance(shape, (list, tuple)):
            shape = {"output0" : [arith.ConstIntBound(0, val - 1) for val in shape]}
        shape = self._infer(shape, rstep)
        return shape

    def get_input_exprs(self, output_exprs):
        result = output_exprs.copy()
        ana = arith.Analyzer()
        for dep in reversed(self.deps):
            for var, expr in zip(dep.var_map.values(), result[dep.output]):
                ana.bind(var, expr)
            for var in dep.range_map:
                ana.bind(var, 0)
            for name, regions in dep.dependent_region.items():
                if name in result:
                    continue
                region = regions[0]
                input_expr = [ana.simplify(index) for index in region]
                result[name] = input_expr
        return result

