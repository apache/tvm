"""Shared functions and classes for frontends."""
from __future__ import absolute_import as _abs
import logging
from nnvm import sym as _sym
from .._base import string_types

def get_nnvm_op(op_name):
    op = getattr(_sym, op_name)
    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op

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
        return get_nnvm_op(self._new_name)(*inputs, **attrs)


class AttrConverter(object):
    """Common attribute conveter. An AttrConverter instance is a callable:
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
        If default_value if provded, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.
    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occured.
    disables : list
        A list of attributes that is disabled in nnvm. Log warnings.
    ignores : list
        A list of attributes that is ignored in nnvm. Debug level logging.
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
        if isinstance(self._op_name, string_types):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)
        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError("Attribute {} not supported yet.".format(k))
            elif k in self._disables:
                logging.warning("Attribute %s is disabled in nnvm.sym.%s", k, op_name)
            elif k in self._ignores:
                logging.debug("Attribute %s is ignored in nnvm.sym.%s", k, op_name)
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
        return get_nnvm_op(op_name)(*inputs, **new_attrs)

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
        if not isinstance(k, string_types):
            msg = "{} is not a valid target, (name, default) expected.".format(target)
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, string_types):
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError("Required attribute {} not found.".format(key))
        return attr[key]


class SymbolTable(object):
    """Table storing symbols by names."""
    def __init__(self):
        self.vars = {}
        self.params = {}
        self.const_ctr = 1
        self.in_padding = False
        self.paddings = [0, 0]

    def new_const(self, value):
        name = "_param_%d" % (self.const_ctr)
        self.const_ctr += 1
        self.params[name] = value
        self.vars[name] = _sym.Variable(name=name)
        return self.vars[name]

    def get_var(self, name, must_contain=True):
        if must_contain:
            assert name in self.vars
        if name not in self.vars:
            self.vars[name] = _sym.Variable(name=name)
        return self.vars[name]

    def set_var(self, name, sym):
        assert isinstance(sym, _sym.Symbol)
        self.vars[name] = sym

    def set_padding(self, paddings):
        self.paddings = paddings
        self.in_padding = True

    def clear_padding(self):
        self.in_padding = False
