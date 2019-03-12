import warnings
import traceback
import sys

def _excepthook(type, value, tb):
    print(''.join(traceback.format_exception(type, value, tb)))

sys.excepthook = _excepthook

class OperatorError(Exception):
    pass

def _raise_error_helper(exception, msg, *args):
    raise exception(msg.format(*args))

def raise_attribute_required(key, op_name):
    class OperatorAttributeRequired(OperatorError):
        pass
    msg = 'Required attribute {} not found in operator {}.'
    _raise_error_helper(OperatorAttributeRequired, msg, key, op_name)

def raise_attribute_invalid(val, attr, op_name):
    class OperatorAttributeValueNotValid(OperatorError):
        pass
    msg = 'Value {} in attr {} is not valid in operator {}.'
    _raise_error_helper(OperatorAttributeValueNotValid, msg, val, attr,
                        op_name)

def raise_operator_unimplemented(*missing_ops):
    class OperatorNotImplemented(OperatorError):
        pass
    missing_ops = str(missing_ops).strip('(,)')
    msg = 'The following operators are not supported: {}.'
    _raise_error_helper(OperatorNotImplemented, msg, missing_ops)

def raise_attribute_unimplemented(key, op_name):
    class OperatorAttributeNotImplemented(OperatorError):
        pass
    msg = 'Attribute {} is not supported in operator {}.'
    _raise_error_helper(OperatorAttributeNotImplemented, msg, key, op_name)

def warn_not_used(attr, op_name):
    msg = '{} is ignored in {}.'.format(attr, op_name)
    warnings.warn(msg)
