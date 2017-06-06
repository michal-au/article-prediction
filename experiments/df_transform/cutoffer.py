"""
NOT NEEDED
"""
import numpy as np

CUTOFF = 5


def cutoff_feature(train_X, feature):
    train_X[feature].fillna('missing_value', inplace=True)
    vc = train_X[feature].value_counts()
    train_X[feature] = np.where(vc[train_X[feature]] < CUTOFF, '-OOV-', train_X[feature])
    return train_X


def cutoff_test_feature(train_X, test_X, feature):
    test_X[feature].fillna('missing_value', inplace=True)
    vc = train_X[feature].value_counts()
    test_X[feature] = np.where(vc[test_X[feature]], test_X[feature], '-OOV-')
    return test_X
