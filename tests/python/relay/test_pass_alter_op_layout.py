"""Test alter op layout pass"""

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay.ir_pass import *

def test_alter_op():
    """Test directly replacing an operator with a new one"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    @register_alter_op_layout("nn.conv2d", level=100)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, relay.multiply(weight, relay.const(2.0, "float32")),
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    a = before()
    a = infer_type(a)
    a = alter_op_layout(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))


def test_alter_return_none():
    """Test doing nothing by returning 'None' """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    @register_alter_op_layout("nn.global_max_pool2d", level=101)
    def alter_conv2d(attrs, inputs, tinfos):
        called[0] = True
        return None

    a = before()
    a = infer_type(a)
    a = alter_op_layout(a)

    b = before()
    b = infer_type(b)
    assert(alpha_equal(a, b))
    assert(called[0])


def test_alter_layout():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias")
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.bias_add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.cast(y, 'int32')
        y = relay.nn.batch_flatten(y)
        y = relay.Function(free_vars(y), y)
        return y

    @register_alter_op_layout("nn.conv2d", level=102)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        new_attrs['kernel_layout'] = 'OIHW16i'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(64, 64, 3, 3))

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        w = relay.layout_transform(weight, "OIHW", "OIHW16i")
        y = relay.nn.conv2d(y, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            kernel_layout="OIHW16i",
                            data_layout="NCHW16c")
        b = relay.expand_dims(bias, axis=1, num_newaxis=2)
        b = relay.layout_transform(b, "CHW", "CHW16c")
        y = relay.add(y, b)

        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NCHW16c")
        y = relay.cast(y, 'int32')
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(free_vars(y), y)
        return y

    a = before()
    a = infer_type(a)
    a = canonicalize_ops(a)
    a = infer_type(a)
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))


def test_alter_layout_dual_path():
    """
    Test alternating the layout with two outputs.
    One path continues to use the new layout while one path fall backs to old layout.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.batch_flatten(y)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(free_vars(ret), ret)
        return y

    @register_alter_op_layout("nn.conv2d", level=103)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NCHW16c')
        y1 = relay.nn.relu(y1)
        y1 = relay.layout_transform(y1, "NCHW16c", "NCHW")
        y2 = relay.layout_transform(y, "NCHW16c", "NCHW")
        y2 = relay.nn.batch_flatten(y2)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(free_vars(ret), ret)
        return y

    a = before()
    a = infer_type(a)
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))

def test_alter_layout_resnet():
    """Test alternating the layout of a residual block
    This also tests the elimination of duplicated transformation.
    If a same transformation applies to a same node twice, only one transformation will be created.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2,
                             channels=32,
                             kernel_size=(1, 1))
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y)
        return relay.Function(free_vars(y), y)

    @register_alter_op_layout("nn.conv2d", level=104)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2,
                             channels=32,
                             kernel_size=(1, 1),
                             data_layout='NCHW16c')
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        return relay.Function(free_vars(y), y)

    a = before()
    a = infer_type(a)
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))


def test_alter_layout_broadcast_op():
    """Test boradcast operators """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        scale = relay.var("scale", shape=(64, 1, 1))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.bias_add(y, bias) # test broadcasting to lhs
        y = relay.multiply(scale, y)         # test broadcasting to rhs
        y = relay.Function(free_vars(y), y)
        return y

    @register_alter_op_layout("nn.conv2d", level=105)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        scale = relay.var("scale", shape=(64, 1, 1))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        bias = relay.expand_dims(bias, 1, 2)
        bias = relay.layout_transform(bias, "CHW", "CHW16c")
        scale = relay.layout_transform(scale, "CHW", "CHW16c")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.add(y, bias)          # test broadcasting to lhs
        y = relay.multiply(scale, y)      # test broadcasting to rhs
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(free_vars(y), y)
        return y

    a = before()
    a = infer_type(a)
    a = canonicalize_ops(a)
    a = infer_type(a)
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))

def test_alter_layout_scalar():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.add(y, relay.const(1, "float32"))
        y = relay.Function(free_vars(y), y)
        return y

    @register_alter_op_layout("nn.conv2d", level=106)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight")

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.add(y, relay.const(1.0, "float32"))

        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(free_vars(y), y)
        return y

    a = before()
    a = infer_type(a)
    a = canonicalize_ops(a)
    a = infer_type(a)
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))

def test_alter_layout_concatenate():
    """ """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        ret = relay.concatenate([y, y1], axis=1)
        y = relay.Function(free_vars(ret), ret)
        return y

    @register_alter_op_layout("nn.conv2d", level=107)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NCHW16c')
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW16c", "NCHW")
        y = relay.Function(free_vars(ret), ret)
        return y

    a = before()
    a = infer_type(a)
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))


def test_alter_layout_nchw_upsamping_op():
    """Test upsamping operators """
    def before():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var('weight', shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.upsampling(y, scale=2)
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2))
        y = relay.Function(free_vars(y), y)
        return y

    @register_alter_op_layout("nn.conv2d", level=108)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.nn.upsampling(y, scale=2, layout="NCHW16c")
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2), layout='NCHW16c')
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(free_vars(y), y)
        return y

    a = before()
    a = infer_type(a)
    a = canonicalize_ops(a)
    a = infer_type(a)
    
    a = alter_op_layout(a)
    a = infer_type(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))


if __name__ == "__main__":
    test_alter_op()
    test_alter_return_none()
    test_alter_layout()
    test_alter_layout_dual_path()
    test_alter_layout_resnet()
    test_alter_layout_broadcast_op()
    test_alter_layout_scalar()
    test_alter_layout_concatenate()
    test_alter_layout_nchw_upsamping_op()
