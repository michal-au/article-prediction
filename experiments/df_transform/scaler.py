"""
NOT NEEDED
"""
from sklearn.preprocessing import MaxAbsScaler


def scale(train_X, test_X):
    scaler = MaxAbsScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    return train_X, test_X
