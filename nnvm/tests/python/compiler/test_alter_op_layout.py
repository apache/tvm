"""Unittest cases for AlterOpLayout pass"""
from nnvm import symbol as sym
from nnvm.compiler import graph_attr
from nnvm.top import registry as reg
import nnvm.graph as graph

def get_layouts(g):
    ldict = {}
    vlayout = g.json_attr("layout")
    entry_ptr = g.index.entry_ptr
    for i, n in enumerate(g.index.nodes):
        begin, end = entry_ptr[i], entry_ptr[i + 1]
        ldict[n["name"]] = vlayout[begin:end]
    return ldict


def test_alter_conv2d_layout():
    data = sym.Variable("data", shape=(1, 32, 512, 512))
    conv = sym.conv2d(data, name="conv", channels=16,
                      kernel_size=(3,3), padding=(1,1),
                      use_bias=False, layout="NCHW")
    # split here
    convs = sym.split(conv, indices_or_sections=2)
    relus = [sym.relu(x, name="relu") for x in convs]
    relu = sym.concatenate(*relus)
    flatten = sym.flatten(relu, name="flatten")
    softmax = sym.softmax(flatten, name="softmax")
    g = graph.create(softmax)

    g = g.apply("CorrectLayout")
    g = graph_attr.set_dtype_inputs(g, "float32")
    g = g.apply(["InferShape", "InferType"])
    layouts_origin = get_layouts(g)

    @reg.register_alter_op_layout("conv2d", level=100)
    def alter_conv2d_layout(attrs, inputs, tinfos):
        new_attrs = {k : attrs[k] for k in attrs.keys()}
        new_attrs["layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "NCHW16c"
        new_attrs["name"] = "conv_alter"
        return sym.conv2d(inputs[0], inputs[1], **new_attrs)

    g = g.apply("AlterOpLayout")
    layouts = get_layouts(g)

    # check copy layouts
    for node in ["data", "relu", "flatten", "softmax", "conv_weight"]:
        assert(layouts[node] == layouts_origin[node])
    assert(layouts["conv_alter"] == layouts_origin["conv"])


if __name__ == "__main__":
    test_alter_conv2d_layout()
