import numpy as np  # type: ignore
import pandas  # type: ignore
import matplotlib.pyplot as plot  # type: ignore
import gc
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import os
from cPickle import dump
import scipy

from sklearn.linear_model import LogisticRegression, SGDClassifier

from ...lib.utils import read_settings, load_csr_matrix
"""
from ..extract_features.list_to_binary_features import resolve_list_features, remove_nans_from_dicts, prepare_for_vectorizer, convert_bools
from .plot import log_result, export_errors, create_result_df, get_accuracy
from .utils import create_dir_for_file, read_settings
from ..extract_features.df_transform import convert_list_vector, convert_nominal_vector, get_feature_value_counts, LIST_FEATURES
from ..experiments.df_transform.embeddings import process_embeddings
from ..experiments.df_transform.scaler import scale
from ..experiments.df_transform.cutoffer import cutoff_feature, cutoff_test_feature
"""
TRAIN_NAME = 'bnc-train1000000'
TEST_NAME = 'bnc-test271385'

SETTINGS = read_settings()

train_mtrx_path = os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), TRAIN_NAME)
test_mtrx_path = os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), TEST_NAME)


def load_data():

    """
    train_X = load_csr_matrix(train_mtrx_path + '.npz')
    train_Y = np.load(train_mtrx_path + '-Y.npy')
    """
    test_X = load_csr_matrix(test_mtrx_path + '.npz')
    test_Y = np.load(test_mtrx_path + '-Y.npy')

    print "READING DATASET.... a"
    a = load_csr_matrix(os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), 'bnc-train-0-1000000.npz'))
    print "READING DATASET.... b"
    b = load_csr_matrix(os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), 'bnc-train-1000000-1999995.npz'))
    print "CONCAT TRAINING DATA...."
    train_X = scipy.sparse.vstack([a, b])

    a_Y = np.load(os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), 'bnc-train-0-1000000' + '-Y.npy'))
    b_Y = np.load(os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), 'bnc-train-1000000-1999995' + '-Y.npy'))
    #a_Y = np.load(os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), 'bnc-train-0-100' + '-Y.npy'))
    #b_Y = np.load(os.path.join(SETTINGS.get('paths', 'dataFeaturesSparseBnc'), 'bnc-train-100-199' + '-Y.npy'))
    train_Y = np.concatenate((a_Y, b_Y))
    return train_X, train_Y, test_X, test_Y


def save_model(model):
    print 'saving model ...'
    dump(model, open(os.path.join(SETTINGS.get('paths', 'model'), 'model-logReg.pkl'), 'wb'))
    print '... done'


def train_model(model_call):
    train_X, train_Y, test_X, test_Y = load_data()
    train_X, train_Y = train_X[:1000000], train_Y[:1000000]

    model = eval(model_call)
    print 'fitting the model ...'
    model.fit(train_X, train_Y)
    print '... done'
    save_model(model)

    score = model.score(test_X, test_Y)
    train_score = model.score(train_X, train_Y)
    print 'SCORE: {}, (train: {})'.format(score, train_score)


def grid_search(model_call):
    train_X, train_Y, test_X, test_Y = load_data()
    model = eval(model_call)

    #X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.9, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.05, train_size=0.1, random_state=430)
    tuned_parameters = [
        #{'loss': ['modified_huber'], 'penalty': ['l1', 'l2']},
        #{'loss': ['log'], 'penalty': ['l1', 'l2']}

        #{'loss': ['modified_huber'], 'penalty': ['elasticnet'], 'l1_ratio': [0, .2, .4, .6, .8, 1], 'n_iter': [1, 5, 10, 15]},
        #{'loss': ['log'], 'penalty': ['elasticnet'], 'l1_ratio': [0, .2, .4, .6, .8, 1], 'n_iter': [1, 5, 10, 15]}

        # best so far::: loss: log, penalty: l2, n_iter: 10?
        #{'loss': ['log'], 'penalty': ['l2'], 'n_iter': [10, 15, 20], 'alpha': [0.000001, 0.00001, 0.0001, 0.001]}
    ]

    clf = GridSearchCV(model, tuned_parameters, cv=5)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
