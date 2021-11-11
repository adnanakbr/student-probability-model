import numpy as np
from SPOT.training.train import train_model, get_model_metrics


# def test_train_model():
#     X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
#     y_train = np.array([10, 9, 8, 8, 6, 5])
#     data = {"train": {"X": X_train, "y": y_train}}
#
#     reg_model = train_model(data, {"alpha": 1.2})
#
#     preds = reg_model.predict([[1], [2]])
#     np.testing.assert_almost_equal(preds, [9.93939393939394, 9.03030303030303])


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
