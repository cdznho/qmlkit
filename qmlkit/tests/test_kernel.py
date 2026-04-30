import numpy as np
import pytest

from qmlkit import QuantumFeatureMap, QuantumKernel


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("method", ["iqp", "random", "tensor"])
def test_quantum_kernel_fit_transform_shape(method):
    X, _ = _data()
    features = QuantumKernel(method=method, random_state=7).fit_transform(X)
    assert features.ndim == 2
    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_quantum_feature_map_public_name_matches_kernel_behavior():
    X, _ = _data()
    feature_map = QuantumFeatureMap(method="tensor", n_components=5)
    features = feature_map.fit_transform(X)
    assert features.shape == (X.shape[0], 5)
    assert isinstance(QuantumKernel(method="iqp"), QuantumFeatureMap)


def test_quantum_kernel_is_deterministic_for_fixed_random_state():
    X, _ = _data()
    first = QuantumKernel(method="random", random_state=11).fit_transform(X)
    second = QuantumKernel(method="random", random_state=11).fit_transform(X)
    np.testing.assert_allclose(first, second)


def test_quantum_kernel_cache_returns_copy():
    X, _ = _data()
    kernel = QuantumKernel(method="iqp", cache=True).fit(X)
    first = kernel.transform(X)
    first[0, 0] = 999.0
    second = kernel.transform(X)
    assert second[0, 0] != 999.0


def test_quantum_kernel_matrix_shape():
    X, _ = _data()
    kernel = QuantumKernel(method="tensor").fit(X)
    K = kernel.kernel_matrix(X[:5], X[5:9])
    assert K.shape == (5, 4)


def test_quantum_kernel_matrix_is_symmetric_for_single_input():
    X, _ = _data()
    K = QuantumKernel(method="iqp").kernel_matrix(X)
    assert K.shape == (X.shape[0], X.shape[0])
    np.testing.assert_allclose(K, K.T)


def test_quantum_kernel_rejects_invalid_method():
    X, _ = _data()
    with pytest.raises(ValueError, match="Unsupported method"):
        QuantumKernel(method="bad").fit(X)


def test_quantum_kernel_rejects_feature_width_mismatch():
    X, _ = _data()
    kernel = QuantumKernel(method="iqp").fit(X)
    with pytest.raises(ValueError, match="feature width"):
        kernel.transform(X[:, :2])


def test_quantum_kernel_sklearn_pipeline_fit_predict():
    sklearn = pytest.importorskip("sklearn")
    assert sklearn is not None
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    X, y = _data()
    model = Pipeline([
        ("embed", QuantumKernel(method="iqp")),
        ("clf", SVC()),
    ])
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
