import numpy as np
from SPOT.training.train import get_model_metrics




def test_get_model_metrics():

    class MockModel:

        @staticmethod
        def predict(data):
            return ([0.0])

    X_test = np.array([3, 4]).reshape(-1, 1)
    y_test = np.array([0])
    data = {"test": {"X": X_test, "y": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'accuracy' in metrics
    accuracy = metrics['accuracy']
    np.testing.assert_almost_equal(accuracy, 1.0)
