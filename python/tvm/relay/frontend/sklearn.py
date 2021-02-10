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
# pylint: disable=import-outside-toplevel, used-before-assignment, unused-import
"""Scikit-learn frontend."""
import numpy as np
import tvm
from tvm import relay
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .common import infer_type as _infer_type
from .common import infer_value as _infer_value


def _SimpleImputer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer:
    Imputation transformer for completing missing values.
    """
    boolean_mask = _op.logical_or(_op.isnan(inexpr), _op.isinf(inexpr))
    fill_col = _op.const(np.array(op.statistics_, dtype=dtype))
    input_shape = _op.shape_of(inexpr)
    reps = _op.take(input_shape, _op.const([0]))
    reps = _op.concatenate([reps, _op.const([1])], axis=0)

    fill_val = _op.tile(fill_col, reps=reps)
    indices = _op.const(np.arange(len(op.statistics_)))
    fill_val = _op.take(fill_val, indices=indices, axis=1)

    ret = _op.where(boolean_mask, fill_val, inexpr)

    return ret


def _RobustImputer(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Imputation transformer for completing missing values with multi-column support.
    """
    if columns:
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    ret = _SimpleImputer(op.simple_imputer_, inexpr, dshape, dtype, columns)

    return ret


def _RobustMissingIndicator(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Imputation transformer for completing missing values with multi-column support.
    """
    if columns:
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    ret = _op.logical_or(_op.isnan(inexpr), _op.isinf(inexpr))

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
    Standardize features by removing the mean and scaling to unit variance.
    """
    scaler = op.scaler_
    ret = _op.subtract(inexpr, _op.const(np.array(scaler.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(scaler.scale_, dtype), dtype))
    return ret


def _FeatureUnion(op, inexpr, dshape, dtype, func_name, columns=None):
    """
    Scikit-Learn Pipeline:
    Concatenates results of multiple transformer objects.
    """
    out = []
    for _, mod in op.transformer_list:
        out.append(sklearn_op_to_relay(mod, inexpr, dshape, dtype, func_name, None))

    return _op.concatenate(out, axis=1)


def _Pipeline(op, inexpr, dshape, dtype, func_name, columns=None):
    """
    Scikit-Learn Pipeline:
    Pipeline of transforms with a final estimator.
    """
    for _, mod in op.steps:
        inexpr = sklearn_op_to_relay(mod, inexpr, dshape, dtype, func_name, None)
    return inexpr


def _ColumnTransformer(op, inexpr, dshape, dtype, func_name, columns=None):
    """
    Scikit-Learn Compose:
    Applies transformers to columns of an array
    """
    out = []
    for _, pipe, cols in op.transformers_:
        if pipe == "drop":
            continue
        mod = pipe.steps[0][1]
        op_type = column_transformer_op_types[type(mod).__name__]
        out.append(sklearn_op_to_relay(pipe, inexpr[op_type], dshape, dtype, func_name, cols))

    return _op.concatenate(out, axis=1)


def _InverseLabelTransformer(op, inexpr, dshape, dtype, columns=None):
    """
    Identity transformation of the label data. The conversion to string happens in runtime.
    """
    if len(dshape) > 2:
        raise ValueError(
            "Dim of Input for InverseLabelTransformer should be 1 or 2, {} is given".format(
                len(dshape)
            )
        )

    if len(dshape) == 1:
        ret = _op.cast(_op.greater(inexpr, _op.const(0.5)), "int32")
    else:
        ret = _op.argmax(inexpr, axis=1)

    return ret


def _RobustOrdinalEncoder(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Encode categorical features as an integer array additional feature of handling unseen values.
    The input to this transformer should be an array-like of integers or strings, denoting the
    values taken on by categorical (discrete) features. The features are converted to ordinal
    integers. This results in a single column of integers (0 to n_categories - 1) per feature.
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

        offset = _op.const(np.arange(-1, len(category) - 1, dtype=dtype))
        zeros = _op.full_like(one_hot, _op.const(0, dtype=dtype))
        ordinal_col = _op.where(one_hot_mask, _op.add(one_hot, offset), zeros)
        ordinal = _op.expand_dims(_op.sum(ordinal_col, axis=1), -1)

        seen_mask = _op.cast(_op.sum(one_hot, axis=1), dtype="bool")
        seen_mask = _op.expand_dims(seen_mask, -1)
        extra_class = _op.full_like(ordinal, _op.const(len(category), dtype=dtype))
        robust_ordinal = _op.where(seen_mask, ordinal, extra_class)
        out.append(robust_ordinal)

    ret = _op.concatenate(out, axis=1)
    return ret


def _RobustLabelEncoder(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Encode target labels with value between 0 and n_classes-1.
    """
    if columns:
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    class_mask = []
    for i in range(len(op.classes_)):
        val = _op.const(np.array(op.classes_[i], dtype), dtype)
        class_mask.append(_op.equal(inexpr, val))
    for i in range(len(op.classes_)):
        if is_inverse:
            label_mask = _op.full_like(
                inexpr, _op.const(np.array(op.classes_[i], dtype), dtype=dtype)
            )
        else:
            label_mask = _op.full_like(inexpr, _op.const(i, dtype=dtype))

        if i == 0:
            out = _op.where(class_mask[i], label_mask, inexpr)
            continue
        out = _op.where(class_mask[i], label_mask, out)

    if op.fill_unseen_labels:
        unseen_mask = class_mask[0]
        for mask in class_mask[1:]:
            unseen_mask = _op.logical_or(unseen_mask, mask)
        unseen_mask = _op.logical_not(unseen_mask)
        unseen_label = (
            _op.const(-1, dtype=dtype)
            if is_inverse
            else _op.const(np.array(len(op.classes_)), dtype=dtype)
        )
        label_mask = _op.full_like(inexpr, unseen_label)
        out = _op.where(unseen_mask, label_mask, out)

    return out


def _NALabelEncoder(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Encoder for transforming labels to NA values which encode all non-float and non-finite values
    as NA values.
    """
    if columns:
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    flattened_inexpr = _op.reshape(inexpr, newshape=(-1, 1))
    # Hardcoded flattened shape to be (?, 1)
    flattened_dshape = (relay.Any(), 1)
    ri_out = _RobustImputer(op.model_, flattened_inexpr, flattened_dshape, dtype)
    ret = _op.reshape(ri_out, newshape=-1)
    return ret


def _RobustStandardScaler(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Standardize features by removing the mean and scaling to unit variance.
    """
    if columns:
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    scaler = op.scaler_
    ret = _op.subtract(inexpr, _op.const(np.array(scaler.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(scaler.scale_, dtype), dtype))
    return ret


def _KBinsDiscretizer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer:
    Bin continuous data into intervals.
    """
    if columns:
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    bin_edges = np.transpose(np.vstack(op.bin_edges_))
    out = _op.full_like(inexpr, _op.const(0, dtype=dtype))

    for i in range(1, len(bin_edges) - 1):
        indices_mask = _op.full_like(inexpr, _op.const(i, dtype=dtype))
        bin_edge = _op.const(bin_edges[i])
        bin_mask = _op.greater_equal(inexpr, bin_edge)
        out = _op.where(bin_mask, indices_mask, out)

    return out


def _TfidfVectorizer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer:
    Transform a count matrix to a normalized tf or tf-idf representation.
    """
    if op.use_idf:
        idf = _op.const(np.array(op.idf_, dtype=dtype), dtype=dtype)
        tfidf = _op.multiply(idf, inexpr)
        if op.sublinear_tf:
            tfidf = _op.add(tfidf, _op.const(1, dtype))
        ret = _op.nn.l2_normalize(tfidf, eps=0.0001, axis=[1])
    else:
        ret = _op.nn.l2_normalize(inexpr, eps=0.0001, axis=[1])

    return ret


def _RobustPCA(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer:
    PCA transformation with existing eigen vector.
    """
    eigvec = _op.const(np.array(op.robust_pca_.components_, dtype))

    if type(op.robust_pca_).__name__ == "PCA":
        mean = _op.const(np.array(op.robust_pca_.mean_, dtype))
        inexpr = _op.subtract(inexpr, mean)

    ret = _op.nn.dense(inexpr, eigvec)

    return ret


def _qt_transform_col(X_col, quantiles, inverse, qt, references):
    """
    Column transformation for Quantile-type Transformers
    """
    x_shape_n = _op.shape_of(X_col)

    if not inverse:
        lower_bound_x = _op.take(quantiles, indices=_expr.const(0))
        upper_bound_x = _op.take(
            quantiles,
            indices=_op.subtract(_op.take(x_shape_n, indices=_expr.const([0])), _expr.const(1)),
        )
        lower_bound_y = _expr.const(0)
        upper_bound_y = _expr.const(1)
    else:
        lower_bound_y = _op.take(quantiles, indices=_expr.const(0))
        upper_bound_y = _op.take(quantiles, indices=_expr.const(1))
        lower_bound_x = _expr.const(0)
        upper_bound_x = _expr.const(1)

    lower_bounds_idx = _op.equal(
        _op.cast(X_col, "float32"), _op.broadcast_to(lower_bound_x, x_shape_n)
    )
    upper_bounds_idx = _op.equal(
        _op.cast(X_col, "float32"), _op.broadcast_to(upper_bound_x, x_shape_n)
    )

    isfinite_mask = _op.logical_not(_op.isnan(X_col))

    X_col_finite = _op.reshape(
        _op.multiply(X_col, _op.cast(isfinite_mask, "float32")), newshape=(-1)
    )

    if not inverse:
        interp1 = _op.interpolate(
            X_col_finite,
            _op.reshape(quantiles, qt.quantiles_.shape[0]),
            _op.cast(references, "float32"),
        )

        interp2 = _op.interpolate(
            _op.negative(X_col_finite),
            _op.reverse(_op.negative(_op.reshape(quantiles, qt.quantiles_.shape[0])), axis=0),
            _op.reverse(_op.negative(_op.cast(references, "float32")), axis=0),
        )

        mul1 = _op.subtract(interp1, interp2)
        mul_out = _op.multiply(_op.cast(_expr.const(0.5), "float"), _op.cast(mul1, "float"))
    else:
        interp_out = _op.interpolate(X_col_finite, qt.references_, quantiles)
        mul_out = interp_out

    out = _op.where(_op.reshape(isfinite_mask, newshape=(-1)), mul_out, X_col_finite)

    upper_bound_y = _op.full(upper_bound_y, x_shape_n, dtype="float32")
    out = _op.where(
        _op.reshape(upper_bounds_idx, newshape=-(1)), _op.reshape(upper_bound_y, newshape=(-1)), out
    )

    lower_bound_y = _op.full(lower_bound_y, x_shape_n, dtype="float32")
    out = _op.where(
        _op.reshape(lower_bounds_idx, newshape=-(1)), _op.reshape(lower_bound_y, newshape=(-1)), out
    )

    return _op.reshape(out, newshape=(-1, 1))


def _QuantileTransformer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer:
    Transform features using quantiles information.
    """
    out = []
    inverse = False
    quantiles = op.quantiles_
    features = quantiles.shape[1]

    input_cols = _op.split(inexpr, features, axis=1)
    q_cols = _op.split(_expr.const(quantiles), features, axis=1)

    for feature_idx in range(features):
        feature = _qt_transform_col(
            input_cols[feature_idx],
            q_cols[feature_idx],
            inverse,
            op,
            _expr.const(op.references_),
        )

        out.append(feature)

    return _op.reshape(_op.concatenate(out, 1), _op.shape_of(inexpr))


def _QuantileExtremeValuesTransformer(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Transform features that contain "extreme" values using quantiles information.
    """
    out = []
    inverse = False

    columns_to_transform = op.cols_to_transform_
    quantiles = op.quantile_transformer_.quantiles_
    features = quantiles.shape[1]

    input_cols = _op.split(inexpr, features, axis=1)
    q_cols = _op.split(_expr.const(quantiles), features, axis=1)

    for feature_idx in range(features):
        if feature_idx in columns_to_transform:
            references = np.linspace(0, 1, quantiles.shape[0], endpoint=True)
            feature = _qt_transform_col(
                input_cols[feature_idx],
                q_cols[feature_idx],
                inverse,
                op.quantile_transformer_,
                _expr.const(tvm.nd.array(references)),
            )
        else:
            feature = input_cols[feature_idx]

        out.append(feature)

    return _op.reshape(_op.concatenate(tuple(out), 1), _op.shape_of(inexpr))


def _LogExtremeValuesTransformer(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Stateful log transformer for columns that contain "extreme" values
    """
    n_features = dshape[1]
    # if n_features != op.n_input_features_:
    #         raise ValueError("X shape does not match training shape.")
    out = []
    cols = _op.split(inexpr, n_features, axis=1)
    for j in range(n_features):
        if j in op.cols_to_transform_:
            if j in op.nonnegative_cols_:
                out.append(_op.log(_op.add(cols[j], _op.const(1, dtype))))
            else:
                sign_col = _op.sign(cols[j])
                out.append(
                    _op.multiply(sign_col, _op.log(_op.add(_op.abs(cols[j]), _op.const(1, dtype))))
                )
        else:
            out.append(cols[j])
    ret = _op.reshape(_op.stack(out, axis=1), newshape=(-1, n_features))
    return ret


_date_time_func_index = {
    "extract_weekday": 0,
    "extract_year": 1,
    "extract_hour": 2,
    "extract_minute": 3,
    "extract_second": 4,
    "extract_month": 5,
    "extract_week_of_year": 6,
}


def _cyclic_transform(data, low, high, dtype):
    normalized = _op.multiply(_op.subtract(data, low), _op.const(2 * np.pi, dtype))
    normalized = _op.divide(normalized, _op.add(_op.const(1, dtype), _op.subtract(high, low)))
    sin_values = _op.sin(normalized)
    cos_values = _op.cos(normalized)
    return sin_values, cos_values


def _DateTimeVectorizer(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer:
    Converts array-like data with datetime.datetime or strings describing datetime objects into
    numeric features
    """
    mins = []
    maxs = []
    cols = []
    cols_without_year = []  # year is not eligible for ordinal/cyclic transform

    for datetime_property in op.extract_:
        extract_func = datetime_property.extract_func.__name__
        cols.append(_date_time_func_index[extract_func])
        if datetime_property.min is not None:
            cols_without_year.append(_date_time_func_index[extract_func])
            mins.append(datetime_property.min)
            maxs.append(datetime_property.max)

    mins = np.array(mins, dtype=np.float32)
    maxs = np.array(maxs, dtype=np.float32)

    data = _op.take(inexpr, _op.const(cols_without_year), axis=1)
    year = _op.take(inexpr, _op.const([1]), axis=1)

    ordinal_values = _op.split(_op.subtract(data, _op.const(mins)), len(cols_without_year), axis=1)

    sin_values, cos_values = _cyclic_transform(data, _op.const(mins), _op.const(maxs), dtype)
    sin_values = _op.split(sin_values, len(cols_without_year), axis=1)
    cos_values = _op.split(cos_values, len(cols_without_year), axis=1)

    out, i = [], 0
    for col in cols:
        if col == 1:
            out.append(year)
        else:
            if op.mode == "ordinal":
                out.append(ordinal_values[i])
            elif op.mode == "cyclic":
                out.append(sin_values[i])
                out.append(cos_values[i])
            i += 1

    ret = _op.concatenate(out, axis=1)

    return ret


_convert_map = {
    "ColumnTransformer": {"transform": _ColumnTransformer},
    "SimpleImputer": {"transform": _SimpleImputer},
    "RobustImputer": {"transform": _RobustImputer},
    "RobustStandardScaler": {"transform": _RobustStandardScaler},
    "ThresholdOneHotEncoder": {"transform": _ThresholdOneHotEncoder},
    "NALabelEncoder": {"transform": _NALabelEncoder, "inverse_transform": _InverseLabelTransformer},
    "RobustLabelEncoder": {"inverse_transform": _InverseLabelTransformer},
    "RobustOrdinalEncoder": {"transform": _RobustOrdinalEncoder},
    "KBinsDiscretizer": {"transform": _KBinsDiscretizer},
    "TfidfVectorizer": {"transform": _TfidfVectorizer},
    "RobustMissingIndicator": {"transform": _RobustMissingIndicator},
    "RobustPCA": {"transform": _RobustPCA},
    "FeatureUnion": {"transform": _FeatureUnion},
    "DateTimeVectorizer": {"transform": _DateTimeVectorizer},
    "Pipeline": {"transform": _Pipeline},
    "QuantileTransformer": {"transform": _QuantileTransformer},
    "QuantileExtremeValuesTransformer": {"transform": _QuantileExtremeValuesTransformer},
    "LogExtremeValuesTransformer": {"transform": _LogExtremeValuesTransformer},
}

INPUT_FLOAT = 0
INPUT_STRING = 1
INPUT_DATETIME = 2

column_transformer_op_types = {
    "RobustImputer": INPUT_FLOAT,
    "RobustMissingIndicator": INPUT_FLOAT,
    "FeatureUnion": INPUT_FLOAT,
    "RobustStandardScaler": INPUT_FLOAT,
    "RobustOrdinalEncoder": INPUT_STRING,
    "ThresholdOneHotEncoder": INPUT_STRING,
    "DateTimeVectorizer": INPUT_DATETIME,
}


def sklearn_op_to_relay(op, inexpr, dshape, dtype, func_name, columns=None):
    """
    Convert Sklearn Ops to Relay Ops.
    """
    classname = type(op).__name__

    if classname not in _convert_map:
        raise NameError("Model {} not supported in scikit-learn frontend".format(classname))
    if func_name not in _convert_map[classname]:
        raise NameError(
            "Function {} of Model {} not supported in scikit-learn frontend".format(
                func_name, classname
            )
        )

    if classname in ["ColumnTransformer", "Pipeline", "FeatureUnion"]:
        return _convert_map[classname][func_name](op, inexpr, dshape, dtype, func_name, columns)

    return _convert_map[classname][func_name](op, inexpr, dshape, dtype, columns)


def from_sklearn(model, shape=None, dtype="float32", func_name="transform", columns=None):
    """
    Import scikit-learn model to Relay.
    """
    try:
        import sklearn  # pylint: disable=unused-import
    except ImportError as e:
        raise ImportError("Unable to import scikit-learn which is required {}".format(e))

    if type(model).__name__ == "ColumnTransformer":
        raise NameError("ColumnTransformer is not supported for single op compilation.")

    inexpr = _expr.var("input", shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype, func_name, columns)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    return IRModule.from_expr(func), []


def from_auto_ml(model, shape=None, dtype="float32", func_name="transform"):
    """
    Import scikit-learn model to Relay.
    """
    try:
        import sklearn  # pylint: disable=unused-import
    except ImportError as e:
        raise ImportError("Unable to import scikit-learn which is required {}".format(e))

    if func_name == "transform":
        inexpr_float = _expr.var("input_float", shape=shape, dtype=dtype)
        inexpr_string = _expr.var("input_string", shape=shape, dtype=dtype)
        inexpr_datetime = _expr.var("input_datetime", shape=shape, dtype=dtype)
        inexpr = [inexpr_float, inexpr_string, inexpr_datetime]

        if type(model.feature_transformer.steps[0][1]).__name__ != "ColumnTransformer":
            raise NameError(
                "The First Transformer must be an ColumnTransformer, but {} is given".format(
                    type(model.feature_transformer.steps[0][1]).__name__
                )
            )

        outexpr = inexpr
        for _, transformer in model.feature_transformer.steps:
            outexpr = sklearn_op_to_relay(transformer, outexpr, shape, dtype, func_name, None)
    else:
        inexpr = _expr.var("input", shape=shape, dtype=dtype)
        transformer = model.target_transformer
        outexpr = sklearn_op_to_relay(transformer, inexpr, shape, dtype, func_name, None)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    return IRModule.from_expr(func), []
