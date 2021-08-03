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
# pylint: disable=broad-except
"""Common utilities"""
from __future__ import absolute_import as _abs
import logging
import numpy as np

import tvm
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from .. import expr as _expr
from .. import function as _function
from .. import transform as _transform
from .. import op as _op
from .. import analysis

# pylint: disable=invalid-name
logger = logging.getLogger("Common")
# Uncomment below line to print all debug msgs
# logger.setLevel(logging.DEBUG)


class RequiredAttr(object):
    """Dummpy class to represent required attr"""


class StrAttrsDict(object):
    """Helper class to parse attrs stored as Dict[str, str].

    Parameters
    ----------
    attrs : Dict[str, str]
        The attributes to be used.
    """

    def __init__(self, attrs):
        self.attrs = attrs

    def has_attr(self, key):
        """Checks if a attribute is present in the map.

        Parameters
        ----------
        key : str
            The attribute key

        Returns
        -------
        bool : True if the key is present in the attributes else false.
        """
        return key in self.attrs

    def get_float(self, key, default=RequiredAttr()):
        """Get float attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            return float(self.attrs[key])
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_int(self, key, default=RequiredAttr()):
        """Get int attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            val = self.attrs[key]
            if val == "None":
                return None
            return int(val)
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_str(self, key, default=RequiredAttr()):
        """Get str attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            return self.attrs[key]
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_int_tuple(self, key, default=RequiredAttr()):
        """Get int tuple attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            tshape = self.attrs[key]
            return tuple(
                int(x) if x.strip("- ").isdigit() else None
                for x in tshape.strip("()[]").split(",")
                if x
            )
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_float_tuple(self, key, default=RequiredAttr()):
        """Get float tuple attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """

        if key in self.attrs:
            tshape = self.attrs[key]
            return tuple(float(x.strip()) for x in tshape.strip("()[]").split(","))
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_tuple_tuple_int(self, key, default=RequiredAttr()):
        """Get int list attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            value = self.attrs[key]
            seq = []
            for tup in value.strip("()").split("),"):
                tup = tup.strip("[]()")
                els = [int(x.strip("( ")) for x in tup.split(",")]
                seq.append(tuple(els))

            return tuple(seq)

        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_int_list(self, key, default=RequiredAttr()):
        """Get int list attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            tshape = self.attrs[key]
            return tuple(int(x.strip()) for x in tshape.strip("[]()").split(","))
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default

    def get_bool(self, key, default=RequiredAttr()):
        """Get bool tuple attribute

        Parameters
        ----------
        key : str
            The attribute key

        default : float
            The default value.

        Returns
        -------
        value : The result
        """
        if key in self.attrs:
            val = self.attrs[key]
            return val.strip().lower() in ["true", "1", "t", "y", "yes"]
        if isinstance(default, RequiredAttr):
            raise AttributeError("Required attribute {} not found.".format(key))
        return default


def get_relay_op(op_name):
    """Get the callable function from Relay based on operator name.
    Parameters
    ----------
    op_name : str
        The Relay operator name.
    """
    if "." in op_name:
        # explicit hierarchical modules
        op = _op
        try:
            for opn in op_name.split("."):
                op = getattr(op, opn)
        except AttributeError:
            op = None
    else:
        # try search op in various modules
        for candidate in (_op, _op.nn, _op.image, _op.vision, _op.contrib):
            op = getattr(candidate, op_name, None)
            if op is not None:
                break
    if not op:
        raise tvm.error.OpNotImplemented("Unable to map op_name {} to relay".format(op_name))
    return op


class ExprTable(object):
    """Table storing Relay expressions by names."""

    def __init__(self):
        self.exprs = {}
        self.params = {}
        self.const_ctr = 1
        self.in_padding = False

    def new_const(self, value, shape=None, dtype="float32"):
        name = "_param_%d" % (self.const_ctr)
        if hasattr(value, "shape"):
            shape = value.shape
        self.const_ctr += 1
        self.params[name] = value
        self.exprs[name] = _expr.var(name_hint=name, shape=shape, dtype=dtype)
        return self.exprs[name]

    def get_expr(self, name):
        return self.exprs[name]

    def set_expr(self, name, expr, force_override=False):
        assert isinstance(expr, _expr.Expr)
        # if name exists, we should override the value
        # otherwise, we can not get like x = func(x) work.
        # One example is CoreML preprocess, which will override
        # the same name of input.
        # However, according to git log, Find keras frontend depends
        # on this property, so we add one force_override to control it.
        if name not in self.exprs or force_override:
            self.exprs[name] = expr

    def has_expr(self, name):
        return name in self.exprs

    def set_padding(self, paddings):
        self.paddings = paddings
        self.in_padding = True

    def clear_padding(self):
        self.in_padding = False


class AttrCvt(object):
    """Common attribute converter. An AttrConverter instance is a callable:
    ```
    attr_converter = AttrConverter(op_name, transforms={'a':'b', 'c':('d', 1)})
    new_op_name, new_attr = attr_converter(attrs)
    ```

    Parameters
    ----------
    op_name : str or callable
        If set as str, returned operator name is the str.
        If set as callable, returned operator is the str returned by calling:
        `op_name = func(attr)`

    transforms : dict of `new_name, or (new_name, default_value, transform function)`
        If only a new_name is provided, it's like renaming the attribute name.
        If default_value if provided, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.

    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occurred.

    disables : list
        A list of attributes that is disabled in relay. Log warnings.

    ignores : list
        A list of attributes that is ignored in relay. Debug level logging.

    extras : dict
        A series of additional attributes should be added anyway to the returned
        attribute dict.

    custom_check : callable
        A custom function takes attribute, and return True/False.
        Raise RuntimeError if not bool(True) returned.
    """

    def __init__(
        self,
        op_name,
        transforms=None,
        excludes=None,
        disables=None,
        ignores=None,
        extras=None,
        custom_check=None,
    ):
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, attrs, *args):
        self._ignores.append("_output_shapes")
        self._ignores.append("_input_shapes")
        self._ignores.append("T")
        self._ignores.append("use_cudnn_on_gpu")
        self._ignores.append("_node_name")
        self._ignores.append("is_training")
        self._ignores.append("_target_layout")

        # apply custom check
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError("Check failed: {}".format(msg))
        # get new op_name
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)

        # ignore 'tvm_custom' always
        self._ignores.append("tvm_custom")

        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError(
                    "Attribute %s in operator %s is not" + " supported.", k, op_name
                )
            if k in self._disables:
                logger.debug("Attribute %s is disabled in relay.sym.%s", k, op_name)
            elif k in self._ignores:
                if k != "tvm_custom":
                    logger.debug("Attribute %s is ignored in relay.sym.%s", k, op_name)
            elif k in self._transforms:
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                if defaults is None:
                    new_attr = self._required_attr(attrs, k)
                else:
                    new_attr = attrs.get(k, None)
                if new_attr is None:
                    new_attrs[new_name] = defaults
                else:
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                new_attrs[k] = attrs[k]
        # add extras
        new_attrs.update(self._extras)
        return get_relay_op(op_name)(*inputs, **new_attrs)

    def _parse_default(self, target):
        """Helper function to parse default values."""
        if not isinstance(target, (list, tuple)):
            k, v, t = target, None, lambda x: x
        elif len(target) == 1:
            k, v, t = target[0], None, lambda x: x
        elif len(target) == 2:
            k, v, t = target[0], target[1], lambda x: x
        elif len(target) > 2:
            k, v, t = target[0], target[1], target[2]
        else:
            k = None  # should raise
        if not isinstance(k, str):
            msg = "{} is not a valid target, (name, default) expected.".format(target)
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, str):
            return value.strip().lower() in ["true", "1", "t", "y", "yes"]
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError("Required attribute {} not found.".format(key))
        return attr[key]


def get_name(node):
    name = ""
    if hasattr(node, "name_hint"):
        name = node.name_hint
    return name


def infer_type(node, mod=None):
    """A method to infer the type of an intermediate node in the relay graph."""
    if isinstance(mod, IRModule):
        mod["main"] = _function.Function(tvm.relay.analysis.free_vars(node), node)
        mod = _transform.InferType()(mod)
        entry = mod["main"]
        ret = entry.body
    else:
        new_mod = IRModule.from_expr(node)
        if mod is not None:
            new_mod.update(mod)

        new_mod = _transform.InferType()(new_mod)
        entry = new_mod["main"]
        ret = entry if isinstance(node, _function.Function) else entry.body

    return ret


def fold_constant(node, mod=None):
    if mod is None:
        mod = IRModule()
    return _transform.FoldConstantExpr(node, mod)


def infer_channels(inputs, transpose=False):
    """A hack for getting 'channels' or 'units' since caffe2 does not provide
    these attributes. We check the shape of weights provided to get the number.
    """
    out_type = infer_type(inputs)
    out_shapes = [get_const_tuple(out_type.checked_type.shape)]
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels


def infer_shape(inputs, mod=None):
    """A method to get the output type of an intermediate node in the graph."""
    out_type = infer_type(inputs, mod=mod)
    checked_type = out_type.checked_type
    if hasattr(checked_type, "shape"):
        # Regular operator that outputs tensors
        return get_const_tuple(checked_type.shape)
    # The return type is not a tensor, for example List
    return checked_type


def infer_value(input_val, params, mod=None):
    """A hack for getting the value of an expression by evaluating a
    portion of the relay graph. This is often needed for functions that
    whose output shape depends on the value of a tensor.
    """
    # Check that all free variables have associated parameters.
    assert all(
        var.name_hint in params.keys() for var in analysis.free_vars(input_val)
    ), "All inputs to infer must be available in params."
    assert tvm.runtime.enabled("llvm"), "LLVM must be enabled to infer value."
    try:
        # TODO(kevinthesun): Use VM for all cases.
        # pylint: disable=import-outside-toplevel
        from tvm.contrib import graph_executor

        func = _function.Function(analysis.free_vars(input_val), input_val)
        with tvm.transform.PassContext(opt_level=0):
            lib = tvm.relay.build(func, target="llvm", params=params)
        dev = tvm.cpu(0)
        m = graph_executor.GraphModule(lib["default"](dev))
        m.run()
        return m.get_output(0)
    except Exception:
        if isinstance(mod, IRModule):
            mod["main"] = _function.Function(analysis.free_vars(input_val), input_val)
        else:
            mod = IRModule.from_expr(input_val)
        exc = tvm.relay.create_executor("debug", mod=mod, device=tvm.cpu(), target="llvm")
        inputs = []
        for param in mod["main"].params:
            inputs.append(params[param.name_hint])
        result = exc.evaluate()(*inputs)
        return result


def infer_value_simulated(input_val, params):
    """Extension to infer_value that can be used when some input
    values are missing. This function creates dummy inputs with the same
    shape and random values then calls infer_value. This is helpful when
    implementing certain onnx operators where we need to evaluate the graph
    to determine a static shape.
    """
    fake_params = []
    # Add a fake copy of all missing params.
    for free_param in analysis.free_vars(input_val):
        if free_param.name_hint not in params:
            fp_dtype = free_param.type_annotation.dtype
            fp_shape = [s.value for s in free_param.type_annotation.shape]
            fake_params.append(free_param)
            params[free_param.name_hint] = tvm.nd.array(np.random.rand(*fp_shape).astype(fp_dtype))
    # Now infer the value.
    output_value = infer_value(input_val, params)
    # Clean fake params out of param dictionary.
    for fake_p in fake_params:
        params.pop(fake_p.name_hint, None)
    return output_value


def try_infer_value(val, on_success=None, on_failure=None):
    """Try running infer_value on the input val, and if successful, return the inferred value or
    pass it to on_success callback if provided. Otherwise, run on_failure callback if it is
    provided, or return the input val as output. In each case, the second return value
    indicates whether infer_value has succeeded or not.
    """
    try:
        ret = infer_value(val, {}).numpy()
        if on_success:
            return on_success(ret), True
        return ret, True
    except Exception:
        if on_failure:
            return on_failure(), False
        return val, False


def new_var(name_hint, type_annotation=None, shape=None, dtype="float32"):
    return _expr.var(name_hint, type_annotation, shape, dtype)


class Renamer(object):
    """A simply renamer for operators.

    Parameters
    ----------
    new_name : str
        The new name for the operator
    """

    def __init__(self, new_name):
        self._new_name = new_name

    def __call__(self, inputs, attrs, *args):
        if "tvm_custom" in attrs:
            attrs.pop("tvm_custom")
        return get_relay_op(self._new_name)(*inputs, **attrs)


def to_int_list(np_array):
    """Convert a np array to a python int list.

    Note: This function converts np.int32 to python's int.
    If we don't do this conversion, numpy's automatic upcast will make
    the shape / parameters be converted to int64 IntImm in relay and
    cause problems in relay/TOPI.
    """
    return [int(x) for x in np_array]


def unbind(data, axis=0):
    """
    Unbind was gotten from pytorch.py and modified. The operation removes a tensor dimension
    and returns a tuple of all slices along a given dimension, already without it.
    TODO (vvchernov): It needs such operation on relay side to reduce time consumption
    on squeeze operation.

    Parameters
    ----------
    data : relay.Expr
        Input tensor
    axis : int
        Axis along which tensor is splited. Tensors in the list do not have this axis.
    Returns
    -------
    result : List[relay.Expr]
        The sequence of computed tensors
    """
    shape = infer_shape(data)
    if axis >= len(shape):
        msg = "Please check input dim, it shouldn't be greater than or equal to rank."
        raise AttributeError(msg)

    selections = shape[axis]
    res_split = _op.split(data, selections, axis)
    ret = []
    for i in range(selections):
        ret.append(_op.squeeze(res_split[i], axis=[axis]))
    ret = _expr.TupleWrapper(_expr.Tuple(ret), selections)
    return ret


def lstm_cell(
    input_seqs,
    hidden_state,
    cell_state,
    w_inp,
    w_hid,
    b_inp=None,
    b_hid=None,
    proj=None,
    p_i=None,
    p_f=None,
    p_o=None,
    f_act=_op.sigmoid,
    g_act=_op.tanh,
    h_act=_op.tanh,
    backwards=False,
):
    """
    Common implementation of LSTM cell for all frontends of TVM
    TODO (vvchernov): currently it is used by onnx and pytorch. Extend for other frontends

    Parameters
    ----------
    input_seqs : List[relay.Expr]
        The sequence of input tensors
        Input tensor should be 2d while issue #8412 is not resolved
        Shape = (batch, feature_size)
    hidden_state : relay.Expr
        Hidden state. shape = (batch, hidden_size)
    cell_state : relay.Expr
        Cell state. shape = (batch, hidden_size)
    w_inp, w_hid : relay.Expr
        weight matrices. wi shape = (4 * hidden_size, feature_size)
        wh shape = (4 * hidden_size, hidden_size or proj_size)
        NOTE: wi = (w_ii|w_if|w_ig|w_io) for input, forget, cell and output gates.
        The order is important for correct LSTM calculation!
    b_inp, b_hid : relay.Expr
        bias matrices. The same order of internal parts as for weights. shape = (4 * hidden_size)
    proj : relay.Expr
        projection matrix. shape = (proj_size, hidden_size)
    p_i, p_f, p_o : relay.Expr
        peephole LSTM matrices. shape = (batch, hidden_size)
    f_act, g_act, h_act : relay.op
        activation funtions
    backwards : bool
        Flag for reverse pass of LSTM

    Returns
    -------
    result : List[relay.Expr], relay.Expr, relay.Expr
        The sequence of computed result, final hidden and cell state
    """

    outputs_list = []
    for x_t in input_seqs if not backwards else reversed(input_seqs):
        # x_t shape = (batch, feature size), step shape = (batch, feature size + hidden_size)
        step = _op.concatenate([x_t, hidden_state], axis=1)
        cat_w = _op.concatenate([w_inp, w_hid], axis=1)
        # Instead of nn.dense(x_t, w_inp) + nn.dense(hidden_state, w_hid)
        # the nn.dense(step, cat_w) is used
        # gates shape = (batch, 4 * hidden_size)
        gates = _op.nn.dense(step, cat_w)
        # Add biases
        if b_inp is not None:
            gates += b_inp
        if b_hid is not None:
            gates += b_hid
        # any gate shape = (batch, hidden_size)
        inp_gate, fgt_gate, cell_gate, otp_gate = _op.split(gates, 4, axis=-1)

        if p_i is not None and p_f is not None:
            inp_gate = f_act(inp_gate + p_i * cell_state)
            fgt_gate = f_act(fgt_gate + p_f * cell_state)
        else:
            inp_gate = f_act(inp_gate)
            fgt_gate = f_act(fgt_gate)

        cell_gate = g_act(cell_gate)
        cell_state = fgt_gate * cell_state + inp_gate * cell_gate
        if p_o is not None:
            otp_gate = f_act(otp_gate + p_o * cell_state)
        else:
            otp_gate = f_act(otp_gate)

        hidden_state = otp_gate * h_act(cell_state)

        if proj is not None:
            hidden_state = _op.nn.dense(hidden_state, proj)

        outputs_list.append(hidden_state)  # [seq_num, (batch, hidden_size)]

    return outputs_list, hidden_state, cell_state
