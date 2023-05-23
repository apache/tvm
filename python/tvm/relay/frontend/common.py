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

from ..expr_functor import ExprMutator
from .. import expr as _expr
from .. import function as _function
from .. import transform as _transform
from .. import op as _op
from .. import ty as _ty
from .. import analysis


class DuplicateFilter:
    """A log filter that only prints the same message once."""

    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        self.msgs.add(record.msg)
        return record.msg not in self.msgs


# pylint: disable=invalid-name
logger = logging.getLogger("Frontend")
logger.addFilter(DuplicateFilter())
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
            raise AttributeError(f"Required attribute {key} not found.")
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
        raise tvm.error.OpNotImplemented(f"Unable to map op_name {op_name} to relay")
    return op


class ExprTable(object):
    """Table storing Relay expressions by names."""

    def __init__(self):
        self.exprs = {}
        self.params = {}
        self.const_ctr = 1
        self.in_padding = False

    def new_const(self, value, shape=None, dtype="float32", source_name=None):
        """Construct a new var expr and add to exprs dictionary"""
        name = f"_param_{self.const_ctr}"
        if hasattr(value, "shape"):
            shape = value.shape
        self.const_ctr += 1
        self.params[name] = value
        self.exprs[name] = _expr.var(name_hint=name, shape=shape, dtype=dtype)
        if source_name:
            self.exprs[name] = set_span(self.exprs[name], source_name)
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
                raise RuntimeError(f"Check failed: {msg}")
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
            msg = f"{target} is not a valid target, (name, default) expected."
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
            raise AttributeError(f"Required attribute {key} not found.")
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
        inputs = []
        for param in mod["main"].params:
            inputs.append(params[param.name_hint])
        result = tvm.relay.create_executor(
            "debug", mod=mod, device=tvm.cpu(), target="llvm"
        ).evaluate()(*inputs)
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


def try_infer_value(val, on_success=None, on_failure=None, parameters=None):
    """Try running infer_value on the input val, and if successful, return the inferred value or
    pass it to on_success callback if provided. Otherwise, run on_failure callback if it is
    provided, or return the input val as output. In each case, the second return value
    indicates whether infer_value has succeeded or not.
    """
    try:
        params = parameters if parameters is not None else {}
        ret = infer_value(val, params).numpy()
        if on_success:
            return on_success(ret), True
        return ret, True
    except Exception:
        if on_failure:
            return on_failure(), False
        return val, False


def shape_of(x, dtype="int64", start=None, end=None):
    """Get shape of a tensor."""

    ttype = infer_type(x).checked_type
    if not _ty.is_dynamic(ttype):
        shape = list(ttype.shape)
        start = start or 0  # default to first
        end = end or len(shape)  # default to last
        shape_sliced = shape[start:end]
        return _expr.const(shape_sliced, dtype)
    return _op.shape_of(x, dtype)


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
    Unbind was taken from Pytorch frontend. The operation removes a tensor dimension
    and returns a tuple of all slices along a given dimension, with specified axis removed.
    TODO (vvchernov): It needs such operation on relay side to reduce time consumption
    on squeeze operation.

    Parameters
    ----------
    data : relay.Expr
        Input tensor
    axis : int
        Axis along which tensor is split.
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
    return _expr.TupleWrapper(_expr.Tuple(ret), selections)


def rnn_cell(
    input_seqs, hidden_state, w_inp, w_hid, b_inp=None, b_hid=None, backwards=False, act=_op.tanh
):
    """
    Common implementation of RNN cell for all frontends of TVM

    Parameters
    ----------
    input_seqs : List[relay.Expr]
        The sequence of input tensors
        Input tensor should be 2d while issue #8412 is not resolved
        Shape = (batch, feature_size)
    hidden_state : relay.Expr
        Hidden state. shape = (batch_size, hidden_size)
    w_inp, w_hid: relay.Expr
        weight matrices. shape = (hidden_size, feature_size), (hidden_size, feature_size)
    b_inp, b_hid : relay.Expr
        bias matrices. The same order of internal parts as for weights. shape = (1 * hidden_size)
    backwards : bool
        Flag for reverse pass of RNN
    act : relay.op
        activation function. It is tanh by default.

    Returns
    -------
    result : List[relay.Expr], relay.Expr, relay.Expr
        The sequence of computed result, final hidden and cell state
    """
    outputs_list = []
    for x_t in input_seqs if not backwards else reversed(input_seqs):
        xwt = _op.nn.dense(x_t, w_inp)
        hwt = _op.nn.dense(hidden_state, w_hid)
        if b_inp is not None and b_hid is not None:
            xwt += b_inp
            hwt += b_hid
        hidden_state = act(xwt + hwt)
        outputs_list.append(hidden_state)  # [seq_num, (batch, hidden_size)]
    return outputs_list, hidden_state


def gru_cell(
    input_seqs,
    hidden_state,
    w_inp,
    w_hid,
    b_inp=None,
    b_hid=None,
    rz_act=_op.sigmoid,
    n_act=_op.tanh,
    backwards=False,
    linear_before_reset=True,
    sequence_lens=None,
):
    """
    Common implementation of GRU cell for all frontends of TVM
    TODO(vvchernov): currently it is used by pytorch and ONNX. Extend for other frontends

    Parameters
    ----------
    input_seqs : List[relay.Expr]
        The sequence of input tensors
        Input tensor should be 2d while issue #8412 is not resolved
        Shape = (batch, feature_size)
    hidden_state : relay.Expr
        Hidden state. shape = (batch_size, hidden_size)
    w_inp, w_hid : relay.Expr
        weight matrices. wi shape = (3 * hidden_size, feature_size)
        wh shape = (3 * hidden_size, hidden_size)
        NOTE: wi = (w_ir|w_iz|w_in) for reset, update and new gates.
        The order is important for correct GRU calculation!
    b_inp, b_hid : relay.Expr
        bias matrices. The same order of internal parts as for weights. shape = (3 * hidden_size)
    r_act : relay.op
        activation function for reset gate. it is sigmoid by default
    z_act : relay.op
        activation function for update gate. it is sigmoid by default
    n_act : relay.op
        activation function for new gate. it is tanh by default
    backwards : bool
        Flag for reverse pass of GRU
    linear_before_reset : bool
        Flag for applying the linear transformation before multiplying by the output of the reset
        gate.
    sequence_lens : relay.op
        Tensor specifying lengths of the sequences in a batch.
        Shape = (batch_size)
    Returns
    -------
    result : List[relay.Expr], relay.Expr, relay.Expr
        The sequence of computed result, final hidden and cell state
    """

    outputs_list = []

    seq_len = len(input_seqs)
    input_dtype = infer_type(input_seqs[0]).checked_type.dtype

    if sequence_lens is not None:
        shape = infer_shape(sequence_lens)
        dtype = infer_type(sequence_lens).checked_type.dtype

        arange = _op.arange(_op.const(0), _op.const(seq_len), dtype=dtype)
        arange = _op.expand_dims(arange, 1)
        sequence_lens = _op.broadcast_to(sequence_lens, [seq_len, shape[0]])

        # cast to data dtype
        mask = _op.less(arange, sequence_lens)
        mask = _op.cast(mask, dtype=input_dtype)
        mask = _op.expand_dims(mask, 2)
        mask_seqs = unbind(mask)

        res_mask = _op.greater_equal(arange, sequence_lens)
        res_mask = _op.cast(res_mask, dtype=input_dtype)
        res_mask = _op.expand_dims(res_mask, 2)
        res_mask_seqs = unbind(res_mask)

        if backwards:
            # need a mask to keep intial_h_B correct
            initial_h = hidden_state
            initial_h_mask = _op.equal(arange, sequence_lens)
            initial_h_mask = _op.cast(initial_h_mask, dtype=input_dtype)
            initial_h_mask = _op.expand_dims(initial_h_mask, 2)
            initial_h_mask_seqs = unbind(initial_h_mask)

    output = _op.zeros(infer_shape(hidden_state), input_dtype)
    for i in range(seq_len) if not backwards else reversed(range(seq_len)):
        x_t = input_seqs[i]
        xwt = _op.nn.dense(x_t, w_inp)
        if linear_before_reset:
            hwt = _op.nn.dense(hidden_state, w_hid)
            if b_inp is not None and b_hid is not None:
                xwt += b_inp
                hwt += b_hid
            i_r, i_z, i_n = _op.split(xwt, 3, axis=-1)
            h_r, h_z, h_n = _op.split(hwt, 3, axis=-1)
            r_gate = rz_act(i_r + h_r)
            z_gate = rz_act(i_z + h_z)
            n_gate = n_act(i_n + r_gate * h_n)
        else:
            i_r, i_z, i_n = _op.split(xwt, 3, axis=1)
            w_hr, w_hz, w_hn = _op.split(w_hid, 3, axis=0)
            r_gate = i_r + _op.nn.dense(hidden_state, w_hr)
            z_gate = i_z + _op.nn.dense(hidden_state, w_hz)
            if b_inp is not None and b_hid is not None:
                b_ir, b_iz, b_in = _op.split(b_inp, 3, axis=-1)
                b_hr, b_hz, b_hn = _op.split(b_hid, 3, axis=-1)
                r_gate += b_ir + b_hr
                r_gate = rz_act(r_gate)
                z_gate += b_iz + b_hz
                i_n += b_in
                h_n = _op.nn.dense((r_gate * hidden_state), w_hn) + b_hn
            else:
                r_gate = rz_act(r_gate)
                h_n = _op.nn.dense((r_gate * hidden_state), w_hn)
            z_gate = rz_act(z_gate)
            n_gate = n_act(i_n + h_n)

        hidden_state = (hidden_state - n_gate) * z_gate + n_gate

        if sequence_lens is not None:
            hidden_state = hidden_state * mask_seqs[i]

        outputs_list.append(hidden_state)  # [seq_num, (batch, hidden_size)]

        if sequence_lens is not None:
            output = output * res_mask_seqs[i] + hidden_state
        else:
            output = hidden_state

        # make sure initial_h_B correct
        if backwards and sequence_lens is not None:
            hidden_state = hidden_state + initial_h * initial_h_mask_seqs[i]

    return outputs_list, output


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
        activation functions
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
        # nn.dense(step, cat_w) is used
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


def autopad(
    data,
    strides,
    kernel_shape,
    dilations=(1, 1),
    pad_type="constant",
    deconv=False,
    mode="SAME_UPPER",
    pad_value=0.0,
):
    """
    Perform autopadding with dynamic input shapes
    """
    # get attributes as constants
    strides = _op.const(np.array(strides), dtype="int64")
    dilated_kernel_shape = _op.const(
        np.array(
            [(kernel - 1) * dilation + 1 for kernel, dilation in zip(kernel_shape, dilations)]
        ),
        dtype="int64",
    )
    # get input shape
    ndim = len(infer_shape(data))
    shape = _op.strided_slice(shape_of(data, dtype="int64"), [2], [ndim])

    # set up integer constants
    zero = _op.const(0, dtype="int64")
    one = _op.const(1, dtype="int64")
    two = _op.const(2, dtype="int64")

    # Calculate total padding
    mod = _op.mod(shape, strides)

    left = _op.maximum(dilated_kernel_shape - strides, zero)
    right = _op.maximum(dilated_kernel_shape - mod, zero)

    total_pad = _op.where(_op.equal(mod, zero), left, right)
    if deconv:
        total_pad = _op.const(np.array(kernel_shape), dtype="int64") - one - total_pad

    # split total padding into before and after
    pad_before = _op.floor_divide(total_pad, two)
    pad_after = total_pad - pad_before

    # combine
    if "LOWER" in mode:
        pad = _op.concatenate(
            [_op.reshape(pad_after, [-1, 1]), _op.reshape(pad_before, [-1, 1])], axis=1
        )
    else:
        pad = _op.concatenate(
            [_op.reshape(pad_before, [-1, 1]), _op.reshape(pad_after, [-1, 1])], axis=1
        )

    # pad N and C with zeros
    pad = _op.concatenate([_op.const(np.zeros([2, 2], dtype="int64"), dtype="int64"), pad], axis=0)

    if isinstance(pad_value, (float, int)):
        pad_value = _op.const(pad_value)

    return _op.nn.pad(data, fold_constant(pad), pad_value, pad_type)


def ensure_scalar_shape(x):
    """
    Assume that `x` is a tensor with one element (regardless of tensor rank).
    Return a version of that tensor with rank 0.
    """
    x_shape = infer_shape(x)
    x_rank = len(x_shape)

    if x_rank == 0:
        return x

    num_elem = np.prod(x_shape)
    assert num_elem == 1, f"Cannot squeeze tensor shape {x_shape} to scalar form."

    return _op.squeeze(x)


def try_resolve_var_to_const(x, graph_params):
    """
    Try to resolve the value of tensor `x` to a specific value.
    If successful, return a Const op with that value.
    If unsuccessful, simply return `x`.
    """
    if isinstance(x, _expr.Var) and x.name_hint in graph_params:
        value = graph_params[x.name_hint].numpy()
        dtype = infer_type(x).checked_type.dtype
        return _op.const(value, dtype)

    return x


class _SpanFiller(ExprMutator):
    """SpanFiller"""

    def __init__(self, span):
        ExprMutator.__init__(self)
        if isinstance(span, tvm.relay.Span):
            self._span = span
        elif isinstance(span, str):
            self._span = tvm.relay.Span(tvm.relay.SourceName(span), 0, 0, 0, 0)
        elif isinstance(span, bytes):
            self._span = tvm.relay.Span(tvm.relay.SourceName(span.decode("utf-8")), 0, 0, 0, 0)
        else:
            assert False, f"unsupported span type: {type(span)}"

    def visit(self, expr):
        if hasattr(expr, "span") and expr.span:
            return expr

        return super().visit(expr)

    def visit_function(self, fn):
        new_params = [self.visit(x) for x in fn.params]
        new_body = self.visit(fn.body)
        return _function.FunctionWithFields(
            fn, list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs, None, self._span
        )

    def visit_let(self, let):
        new_variable = self.visit(let.var)
        new_value = self.visit(let.value)
        new_body = self.visit(let.body)
        return _expr.LetWithFields(let, new_variable, new_value, new_body, None, self._span)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        # call.op might be RelayExpr or Op type
        # ExprMutator will return directly if subject belongs to Op type
        new_op = self.visit(call.op)
        return _expr.CallWithFields(
            call, new_op, new_args, call.attrs, call.type_args, None, self._span
        )

    def visit_var(self, var):
        return _expr.VarWithFields(var, var.vid, var.type_annotation, None, self._span)

    def visit_if(self, ite):
        return _expr.IfWithFields(
            ite,
            self.visit(ite.cond),
            self.visit(ite.true_branch),
            self.visit(ite.false_branch),
            None,
            self._span,
        )

    def visit_tuple(self, tup):
        return _expr.TupleWithFields(
            tup, [self.visit(field) for field in tup.fields], None, self._span
        )

    def visit_tuple_getitem(self, op):
        return _expr.TupleGetItemWithFields(
            op, self.visit(op.tuple_value), op.index, None, self._span
        )

    def visit_constant(self, const):
        return _expr.ConstantWithFields(const, const.data, None, self._span)

    # TODO: Frontend model translation could not use following relay expressions so far,
    #       enable them when new models/impls leverage these kinds of relay expressions.
    def visit_ref_create(self, _):
        raise NotImplementedError()

    def visit_ref_write(self, _):
        raise NotImplementedError()

    def visit_ref_read(self, _):
        raise NotImplementedError()

    def visit_match(self, _):
        raise NotImplementedError()

    def fill(self, sym):
        """Fill span to sym when it is an expr, or return it without change

        Parameters
        ----------
        sym :
            A symbol which is generated from the conversion of a frontend operator.

        Returns
        -------
        sym:
            A expr with span-filled or the original sym.
        """
        if isinstance(sym, _expr.TupleWrapper):
            return _expr.TupleWrapper(self.visit(sym.tuple_value), sym.size)
        elif isinstance(sym, _expr.RelayExpr):
            return self.visit(sym)
        elif isinstance(sym, list):
            assert all(
                isinstance(expr, _expr.RelayExpr) for expr in sym
            ), f"unexpected relay expressions in {sym}"
            return [self.visit(expr) for expr in sym]
        elif isinstance(sym, tuple):
            # some op conversion may return dummy elements
            # e.g. op in frontend/pytorch.py: min_max_common
            assert all(
                isinstance(expr, (_expr.RelayExpr, type(None))) for expr in sym
            ), f"unexpected relay expressions in {sym}"
            return tuple(self.visit(expr) if expr else None for expr in sym)
        elif isinstance(sym, (float, int)):
            return sym
        elif isinstance(sym, np.ndarray):
            return sym
        elif not sym:
            # some op conversion may return None
            # e.g. op in frontend/pytorch.py: prim::device
            return sym

        raise RuntimeError(f"unsupported type {type(sym)}")


def set_span(sym, span):
    """
    Recursively tag the span to the symbol. Stop when it encounters a span-tagged expr. Disabled
    when setting the "relay.frontend.fill_span" as False to the config of PassContext

    Parameters
    ----------
    sym :
        A symbol is generated from the conversion of a frontend operator. Raise an error when the
        type of the symbol is not supported.

    span : String, Span, or bytes
        The source information of the corresponding symbol.

    Returns
    -------
    result :
        The symbol tagged with span.

    Examples
    --------
    .. code-block:: python

      x = set_span(relay.var("x", shape=(1, 64, 56, 56)), "x_var")
      w = relay.const(np.ones([64, 64, 3, 3]), dtype="int64")
      y = set_span(
          relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1)), "conv2d"
      )
      print(relay.Function([x], y))

      #fn (%x: Tensor[(1, 64, 56, 56), float32] /* span=x_var:0:0 */) {
      #  nn.conv2d(%x, meta[relay.Constant][0] /* span=conv2d:0:0 */, ...) /* span=conv2d:0:0 */
      #}
    """

    if tvm.transform.PassContext.current().config.get("relay.frontend.fill_span", True):
        return _SpanFiller(span).fill(sym)
    return sym
