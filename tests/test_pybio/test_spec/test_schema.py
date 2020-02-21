from pybio.spec import schema


def test_prediction_schema_allows_for_missing_weights():
    loaded = schema.Prediction().load({})
    assert loaded.weights is None


def test_prediction_schema_allows_for_null_weights():
    loaded = schema.Prediction().load({"weights": None})
    assert loaded.weights is None
