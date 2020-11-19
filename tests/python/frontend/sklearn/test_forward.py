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
# pylint: disable=used-before-assignment
import numpy as np

from scipy.sparse import random as sparse_random
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sagemaker_sklearn_extension.externals import AutoMLTransformer
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.impute import RobustImputer, RobustMissingIndicator
from sagemaker_sklearn_extension.decomposition import RobustPCA
from sagemaker_sklearn_extension.preprocessing import (
    RobustStandardScaler,
    ThresholdOneHotEncoder,
    RobustLabelEncoder,
    RobustOrdinalEncoder,
    NALabelEncoder,
)

from tvm import topi
import tvm.topi.testing
import tvm
import tvm.testing
from tvm import te
from tvm import relay
import tvm.testing


class SklearnTestHelper:
    def __init__(self, target="llvm", ctx=tvm.cpu(0)):
        self.compiled_model = None
        self.target = target
        self.ctx = ctx

    def compile(self, model, dshape, dtype, func_name, columns=None, auto_ml=False):
        if auto_ml:
            mod, _ = relay.frontend.from_auto_ml(model, dshape, dtype, func_name)
        else:
            mod, _ = relay.frontend.from_sklearn(model, dshape, dtype, func_name, columns)

        self.ex = relay.create_executor("vm", mod=mod, ctx=self.ctx, target=self.target)

    def run(self, data):
        result = self.ex.evaluate()(data)
        return result.asnumpy()


def _test_model_impl(helper, model, dshape, input_data, auto_ml=False):
    helper.compile(model, dshape, "float32", "transform", None, auto_ml)
    tvm_out = helper.run(input_data)
    if auto_ml:
        sklearn_out = model.feature_transformer.transform(input_data)
    elif type(model).__name__ == "ThresholdOneHotEncoder":
        sklearn_out = model.transform(input_data).toarray()
    else:
        sklearn_out = model.transform(input_data)

    tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_simple_imputer():
    st_helper = SklearnTestHelper()
    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.nan, 13, 14, 15]],
        dtype=np.float32,
    )

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="median")
    imp_mean.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, imp_mean, dshape, data)


def test_robust_imputer():
    st_helper = SklearnTestHelper()
    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.inf, 13, 14, 15]],
        dtype=np.float32,
    )

    ri = RobustImputer(dtype=None, strategy="constant", fill_values=np.nan, mask_function=None)
    ri.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, ri, dshape, data)


def test_robust_missing_indicator():
    st_helper = SklearnTestHelper()
    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.inf, 13, 14, 15]],
        dtype=np.float32,
    )

    rmi = RobustMissingIndicator()
    rmi.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, rmi, dshape, data)


def test_robust_scaler():
    st_helper = SklearnTestHelper()
    rss = RobustStandardScaler()

    data = np.array([[-1, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
    rss.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, rss, dshape, data)


def test_threshold_onehot_encoder():
    st_helper = SklearnTestHelper()
    tohe = ThresholdOneHotEncoder()

    data = np.array([[10, 1, 7], [11, 3, 8], [11, 2, 9]], dtype=np.float32)
    tohe.fit(data)
    tohe.categories_ = [[10, 11], [1, 2, 3], [7, 8, 9]]

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, tohe, dshape, data)


def test_robust_ordinal_encoder():
    st_helper = SklearnTestHelper()
    roe = RobustOrdinalEncoder()
    data = np.array([[0, 1], [0, 4], [1, 2], [1, 10]], dtype=np.float32)
    roe.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, roe, dshape, data)


def test_na_label_encoder():
    st_helper = SklearnTestHelper()
    nle = NALabelEncoder()
    i_put = np.array([[1, 2, 2, 6]], dtype=np.float32)
    nle.fit(i_put)
    data = np.array([[np.nan, 0, 1, 2, 6]], dtype=np.float32)
    dshape = (relay.Any(), len(data))
    _test_model_impl(st_helper, nle, dshape, data)


def test_kbins_discretizer():
    st_helper = SklearnTestHelper()
    kd = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
    data = np.array(
        [[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]], dtype=np.float32
    )
    kd.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, kd, dshape, data)


# def test_tfidf_vectorizer():
#     st_helper = SklearnTestHelper()
#     tiv = TfidfVectorizer()
#     data = [
#         'This is the first document.',
#         'This document is the second document.',
#         'And this is the third one.',
#         'Is this the first document?',
#     ]

#     dshape = (relay.Any(), len(data))
#     st_helper.compile(tiv, dshape, 'int32')
#     sklearn_out = tiv.fit_transform(data).toarray()
#     tvm_out = st_helper.run(data)
#     tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_pca():
    st_helper = SklearnTestHelper()
    pca = PCA(n_components=2)
    rpca = RobustPCA()
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32)
    pca.fit(data)
    rpca.robust_pca_ = pca
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, rpca, dshape, data)

    tSVD = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    data = sparse_random(
        100, 100, density=0.01, format="csr", dtype="float32", random_state=42
    ).toarray()
    tSVD.fit(data)
    rpca.robust_pca_ = tSVD
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, rpca, dshape, data)


def test_automl():
    st_helper = SklearnTestHelper()

    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.nan, 13, 14, 15]],
        dtype=np.float32,
    )

    pipeline = Pipeline(
        steps=[("robustimputer", RobustImputer(fill_values=np.nan, strategy="constant"))]
    )

    ct = ColumnTransformer(transformers=[("numeric_processing", pipeline, [0, 1, 2, 3])])
    ct.fit(data)

    pipeline = Pipeline(steps=[("column_transformer", ct)])
    header = Header(column_names=["x1", "x2", "x3", "class"], target_column_name="class")

    na = NALabelEncoder()
    na.fit(data)

    automl_transformer = AutoMLTransformer(header, pipeline, na)

    dshape = (relay.Any(), relay.Any())
    _test_model_impl(st_helper, automl_transformer, dshape, data, auto_ml=True)


def test_feature_union():
    st_helper = SklearnTestHelper()
    rPCA = RobustPCA(n_components=2)
    tSVD = RobustPCA(n_components=1)
    tSVD.robust_pca_ = TruncatedSVD(n_components=1)
    union = FeatureUnion([("pca", rPCA), ("svd", tSVD)])
    data = np.array([[0.0, 1.0, 3], [2.0, 2.0, 5]], dtype=np.float32)
    union.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, union, dshape, data)


def test_pipeline():
    st_helper = SklearnTestHelper()
    pipe = Pipeline([("imputer", RobustImputer()), ("scaler", RobustStandardScaler())])
    data = np.array([[0.0, 1.0, 3], [2.0, 2.0, 5]], dtype=np.float32)
    pipe.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, pipe, dshape, data)


def _test_quantile_transformer(shape, n_quantiles):
    from sklearn.preprocessing import QuantileTransformer

    st_helper = SklearnTestHelper()

    rng = np.random.RandomState(0)
    data = np.sort(rng.normal(loc=0.5, scale=0.25, size=shape), axis=0)

    qt = QuantileTransformer(n_quantiles=n_quantiles, random_state=0)
    qt.fit_transform(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, qt, dshape, data.astype("float32"))


def test_quantile_transformer():
    _test_quantile_transformer((25, 1), 10)
    _test_quantile_transformer((25, 1), 30)
    _test_quantile_transformer((25, 1), 98)
    _test_quantile_transformer((12, 3), 10)
    _test_quantile_transformer((12, 3), 30)
    _test_quantile_transformer((12, 3), 98)


def test_quantile_extremevalues_transformer():
    from sagemaker_sklearn_extension.preprocessing import QuantileExtremeValuesTransformer

    st_helper = SklearnTestHelper()

    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, 1.0],
            [-2.0, 2.0, 2.0],
            [-3.0, 3.0, 3.0],
            [-4.0, 4.0, 4.0],
            [-5.0, 5.0, 5.0],
            [-6.0, 6.0, 6.0],
            [-7.0, 7.0, 7.0],
            [-8.0, 8.0, 8.0],
            [-9.0, 9.0, 9.0],
            [-10.0, 10.0, 10.0],
            [-1e5, 1e6, 11.0],
        ]
    )

    qt = QuantileExtremeValuesTransformer(threshold_std=2.0)
    qt.fit_transform(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, qt, dshape, data.astype("float32"))


def test_inverse_label_transformer():
    st_helper = SklearnTestHelper()
    rle = RobustLabelEncoder()

    # Binary Classification
    data = np.random.random_sample((10,)).astype(np.float32)
    dshape = (relay.Any(),)
    st_helper.compile(rle, dshape, "float32", "inverse_transform")
    python_out = (data > 0.5).astype(int)
    tvm_out = st_helper.run(data)
    tvm.testing.assert_allclose(python_out, tvm_out, rtol=1e-5, atol=1e-5)

    # Multiclass Classification
    data = np.random.random_sample((10, 5)).astype(np.float32)
    dshape = (relay.Any(), 5)
    st_helper.compile(rle, dshape, "float32", "inverse_transform")
    python_out = np.argmax(data, axis=1)
    tvm_out = st_helper.run(data)
    tvm.testing.assert_allclose(python_out, tvm_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_simple_imputer()
    test_robust_imputer()
    test_robust_missing_indicator
    test_robust_scaler()
    test_threshold_onehot_encoder()
    test_robust_ordinal_encoder()
    test_na_label_encoder()
    test_kbins_discretizer()
    # test_tfidf_vectorizer()
    test_pca()
    test_automl()
    test_feature_union()
    test_inverse_label_transformer()
    test_quantile_transformer()
    test_quantile_extremevalues_transformer()
