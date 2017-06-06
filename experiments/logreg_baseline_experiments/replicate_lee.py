from sklearn.linear_model import LogisticRegression
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from code.lib.utils import load_postprocessed_feature_data, logreg_gridsearch
import time
import os
import pandas as pd
import scipy
import numpy as np
from code.lib.utils import read_settings
from code.experiments.features import get_features_for_set_names
import pickle
import argparse


#FEATURES = [
#    'a_hypernyms', 'a_head_form', 'a_head_number', 'a_non_article_det', 'a_parent', 'a_referent', 'b_pos_after_head_as_list', 'b_pos_before_head_as_list',
#    'b_words_after_head_as_list', 'b_words_after_np_as_list', 'b_words_before_head_as_list', 'b_words_before_np_as_list'
#]
TRAIN_SET_NAME = 'train'
TEST_SET_NAME = 'test'
SETTINGS = read_settings()
CV_RESULTS_LOG_PATH = os.path.join(SETTINGS.get('paths', 'logModelResults'), 'penn', 'logreg_lee_replicate')

DATA_SIZE = 263088

DEFAULT_MODEL_PARAMS = model_params = {
    'multi_class': 'multinomial',
    'random_state': 42,
    'penalty': 'l2',
    'solver': 'lbfgs',
}


def gridsearch(data, model_name):
    print("Gridsearch for {}".format(model_name))
    tuning_params = {
        'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }
    logreg_gridsearch(
        data['train_x'],
        data['train_y'],
        os.path.join(CV_RESULTS_LOG_PATH, model_name + '.csv'),
        DEFAULT_MODEL_PARAMS,
        tuning_params,
    )


def train_and_eval(data, model_name, regularization_parameter):
    model_params = dict(DEFAULT_MODEL_PARAMS, C=regularization_parameter)
    model = LogisticRegression(**model_params)
    model.fit(data['train_x'], data['train_y'])
    train_predictions = model.predict(data['train_x'])
    test_predictions = model.predict(data['test_x'])
    with open(os.path.join(CV_RESULTS_LOG_PATH, model_name + '.txt'), 'a+') as f:
        f.write(str(model.get_params()) + '\n')
        f.write("Accuracy ({}): {:.4f}\n".format(TRAIN_SET_NAME, metrics.accuracy_score(data['train_y'], train_predictions)))
        f.write("Accuracy ({}): {:.4f}\n".format(TEST_SET_NAME, metrics.accuracy_score(data['test_y'], test_predictions)))


def load_data(cutoff_value=5):
    data_path = None  # cutoff_value == 5 je defaultni hodnota, cesta bude zjistena automaticky
    if cutoff_value in (0, 3):
        data_path = SETTINGS.get('paths', 'dataFeaturesPennPostprocessed{}'.format(cutoff_value))
    features = get_features_for_set_names(('orig',))
    data = {}
    data['train_x'], data['train_y'] = load_postprocessed_feature_data(features, TRAIN_SET_NAME, data_path=data_path)
    assert data['train_x'].shape[0] == DATA_SIZE

    data['test_x'], data['test_y'] = load_postprocessed_feature_data(features, TEST_SET_NAME, data_path=data_path)
    assert data['train_x'].shape[1] == data['test_x'].shape[1]
    return data


if __name__ == '__main__':
    #gridsearch(load_data(cutoff_value=0), 'multinom-lists-cutoff0')
    train_and_eval(load_data(cutoff_value=0), 'multinom-lists-cutoff0-c04', 0.4)