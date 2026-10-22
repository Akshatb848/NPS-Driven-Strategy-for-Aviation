"""The committed model artefact loads and behaves as the README describes."""
import pathlib
import zipfile

import joblib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_saved_random_forest_and_scaler(tmp_path):
    with zipfile.ZipFile(ROOT / "best_rf_model.zip") as zf:
        zf.extractall(tmp_path)
    model = joblib.load(tmp_path / "best_rf_model.pkl")
    scaler = joblib.load(tmp_path / "scaler.pkl")

    assert type(model).__name__ == "RandomForestRegressor"
    params = model.get_params()
    assert params["n_estimators"] == 100 and len(model.estimators_) == 100
    assert params["max_depth"] == 30
    assert params["min_samples_split"] == 10
    assert params["min_samples_leaf"] == 4
    assert params["max_features"] == "sqrt"
    assert params["bootstrap"] is False
    assert model.n_features_in_ == 10
    assert scaler.n_features_in_ == 10
    assert int(scaler.n_samples_seen_) == 58718  # 80% of 73,398 rows

    ratings = np.array([[1] * 10, [3] * 10, [5] * 10], dtype=float)  # all-low, all-mid, all-high touchpoint ratings
    pred = model.predict(scaler.transform(ratings))
    assert np.all(np.isfinite(pred))
    assert np.all((pred >= -100) & (pred <= 100))
