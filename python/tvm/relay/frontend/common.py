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
"""Common utilities"""
from __future__ import absolute_import as _abs
import logging
from topi.util import get_const_tuple
from .. import expr as _expr
from .. import module as _module
from .. import transform as _transform
from .. import op as _op


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
            return tuple(int(x.strip()) for x in tshape.strip('()[]').split(',') if x)
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
            return tuple(float(x.strip()) for x in
                         tshape.strip('()[]').split(','))
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
            for tup in value.strip('()').split('),'):
                tup = tup.strip('[]()')
                els = [int(x.strip('( ')) for x in tup.split(',')]
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
            return tuple(int(x.strip()) for x in tshape.strip('[]()').split(','))
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
            return val.strip().lower() in ['true', '1', 't', 'y', 'yes']
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
    if '.' in op_name:
        # explicit hierachical modules
        op = _op
        try:
            for opn in op_name.split('.'):
                op = getattr(op, opn)
        except AttributeError:
            op = None
    else:
        # try search op in various modules
        for candidate in (_op, _op.nn, _op.image, _op.vision):
            op = getattr(candidate, op_name, None)
            if op is not None:
                break
    if not op:
        raise RuntimeError("Unable to map op_name {} to relay".format(op_name))
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

    def set_expr(self, name, expr):
        assert isinstance(expr, _expr.Expr)
        if name not in self.exprs:
            self.exprs[name] = expr

    def has_expr(self, name):
        return True if name in self.exprs else False

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
    def __init__(self, op_name, transforms=None,
                 excludes=None, disables=None, ignores=None,
                 extras=None, custom_check=None):
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, attrs, *args):
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
        self._ignores.append('tvm_custom')

        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError("Attribute {} not supported yet.".format(k))
            elif k in self._disables:
                logging.warning("Attribute %s is disabled in relay.sym.%s", k, op_name)
            elif k in self._ignores:
                if k != 'tvm_custom':
                    logging.warning("Attribute %s is ignored in relay.sym.%s", k, op_name)
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
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError("Required attribute {} not found.".format(key))
        return attr[key]

def get_name(node):
    name = ''
    if hasattr(node, "name_hint"):
        name = node.name_hint
    return name


def infer_type(node):
    """A method to infer the type of an intermediate node in the relay graph."""
    mod = _module.Module.from_expr(node)
    mod = _transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(node, _expr.Function) else entry.body

def infer_shape(inputs):
    """A method to get the output shape of an intermediate node in the graph."""
    out_type = infer_type(inputs)
    out_shapes = get_const_tuple(out_type.checked_type.shape)
    return out_shapes

def infer_channels(inputs, transpose=False):
    """A hack for getting 'channels' or 'units' since caffe2 does not provide
    these attributes. We check the shape of weights provided to get the number.
    """
    out_type = infer_type(inputs)
    out_shapes = [get_const_tuple(out_type.checked_type.shape)]
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels

def new_var(name_hint,
            type_annotation=None,
            shape=None,
            dtype="float32"):
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
        if 'tvm_custom' in attrs:
            attrs.pop('tvm_custom')
        return get_relay_op(self._new_name)(*inputs, **attrs)
