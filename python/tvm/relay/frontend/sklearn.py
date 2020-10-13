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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel

import numpy as np
import tvm
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import vision as _vision

from ..function import Function
from ..expr import Call, Let
from ..expr import If, Tuple, TupleGetItem
from ..expr import RefCreate, RefRead, RefWrite
from ..expr_functor import ExprFunctor
from ..adt import Match, Clause

from .common import AttrCvt, Renamer, ExprTable
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, get_name
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated


def _SimpleImputer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer: 
    Imputation transformer for completing missing values.
    """
    boolean_mask = _op.isnan(inexpr)
    fill_col = _op.const(np.array(op.statistics_, dtype=dtype))
    input_shape = _op.shape_of(inexpr)
    reps = _op.take(input_shape, _op.const([0]))
    reps = _op.concatenate([reps, _op.const([1])], axis=0)

    fill_val = _op.tile(fill_col, reps=reps)
    indices =_op.const(np.arange(len(op.statistics_)))
    fill_val = _op.take(fill_val, indices=indices, axis=1)

    ret = _op.where(boolean_mask,
                    fill_val,
                    inexpr)
    
    return ret

def _RobustImputer(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer: 
    Imputation transformer for completing missing values with multi-column support.
    """
    if columns: 
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    if op.mask_function is not None:
        inf_mask = _op.isinf(inexpr)
        nan_val = _op.full_like(inexpr, _op.const(np.array(np.nan, dtype=dtype)))
        inexpr = _op.where(inf_mask, nan_val, inexpr) 
    ret = _SimpleImputer(op.simple_imputer_, inexpr, dshape, dtype, columns)

    return ret 
    
def _ThresholdOneHotEncoder(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer: 
    Encode categorical integer features as a one-hot numeric array, with optional restrictions on
    feature encoding.
    """
    if columns: 
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    num_cat = len(op.categories_)
    cols = _op.split(inexpr, num_cat, axis=1)

    out = [] 
    for i in range(num_cat):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot_mask = _op.equal(tiled_col, cat_tensor)
        one_hot = _op.cast(one_hot_mask, dtype)
        out.append(one_hot)

    ret = _op.concatenate(out, axis=1) 
    return ret

def _RobustStandardScaler(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer: 
    Standardize features by removing the mean and scaling to unit variance
    """
    scaler = op.scaler_
    ret = _op.subtract(inexpr, _op.const(np.array(scaler.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(scaler.scale_, dtype), dtype))
    return ret

def _ColumnTransformer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Compose: 
    Applies transformers to columns of an array 
    """
    out = []
    for _, pipe, cols in op.transformers_:
        mod = pipe.steps[0][1]
        out.append(sklearn_op_to_relay(mod, inexpr, dshape, dtype, cols))
    
    return _op.concatenate(out, axis=1)

_convert_map = {
    'ColumnTransformer':_ColumnTransformer,
    'SimpleImputer': _SimpleImputer,
    'RobustImputer': _RobustImputer,
    'RobustStandardScaler': _RobustStandardScaler,
    'ThresholdOneHotEncoder': _ThresholdOneHotEncoder
}

def sklearn_op_to_relay(op, inexpr, dshape, dtype, columns=None):
    classname = type(op).__name__
    return _convert_map[classname](op, inexpr, dshape, dtype, columns)

def from_sklearn(model,
                 shape=None,
                 dtype="float32",
                 columns=None):

    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            "Unable to import scikit-learn which is required {}".format(e))
    
    inexpr = _expr.var('input', shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype, columns)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    return IRModule.from_expr(func), []

def from_auto_ml(model,
                shape=None,
                dtype="float32"):

    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            "Unable to import scikit-learn which is required {}".format(e))

    outexpr = _expr.var('input', shape=shape, dtype=dtype)
    for _, transformer in model.feature_transformer.steps:
        outexpr = sklearn_op_to_relay(transformer, outexpr, shape, dtype, None)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    return IRModule.from_expr(func), []
