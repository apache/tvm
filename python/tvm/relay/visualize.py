from .expr_functor import ExprFunctor
from . import expr as _expr
import networkx as nx
from PIL import Image
import tempfile

class VisualizeExpr(ExprFunctor):
    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.counter = 0

    def viz(self, expr):
        #assert isinstance(expr, _expr.Function)
        for param in expr.params:
            self.visit(param)

        return self.visit(expr.body)

    def visit_var(self, var):
        name = var.name_hint
        self.graph.add_node(name)
        self.graph.nodes[name]['style'] = 'filled'
        self.graph.nodes[name]['fillcolor'] = 'mistyrose'
        return var.name_hint

    def visit_tuple_getitem(self, get_item):
        tuple = self.visit(get_item.tuple_value)
        # self.graph.nodes[tuple]
        index = get_item.index
        # import pdb; pdb.set_trace()
        return tuple

    def visit_tuple(self, tup):
        #tuple = self.visit(get_item.tuple_value)
        # self.graph.nodes[tuple]
        #index = get_item.index
        # import pdb; pdb.set_trace()
        return tup

    def visit_call(self, call):
        parents = []
        for arg in call.args:
            parents.append(self.visit(arg))
        # assert isinstance(call.op, _expr.Op)
        name = "{}({})".format(call.op.name, self.counter)
        self.counter += 1
        self.graph.add_node(name)
        self.graph.nodes[name]['style'] = 'filled'
        self.graph.nodes[name]['fillcolor'] = 'turquoise'
        self.graph.nodes[name]['shape'] = 'diamond'
        edges = []
        for i, parent in enumerate(parents):
            edges.append((parent, name, { 'label': 'arg{}'.format(i) }))
        self.graph.add_edges_from(edges)
        return name

def visualize(expr):
    viz_expr = VisualizeExpr()
    viz_expr.viz(expr)
    graph = viz_expr.graph

    name = "yolo_png"
    dotg = nx.nx_pydot.to_pydot(graph)
    dotg.write_png(name)
    img = Image.open(name)
    img.show()

    # with tempfile.NamedTemporaryFile(delete=False) as tf:
    #     dotg = nx.nx_pydot.to_pydot(graph)
    #     dotg.write_png(tf.name)
    #     img = Image.open(tf.name)
    #     print("11111111111111111111")
    #     img.show()